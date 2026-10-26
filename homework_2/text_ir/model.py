"""A bidirectional retrieval encoder and the shared contrastive training loop."""

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import encode_texts, evaluate_holdout


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextIREncoder(nn.Module):
    """Tokens + positions → bidirectional attention → weighted pooling → vector."""

    def __init__(
        self, vocab_size: int, pad_token_id: int, *, hidden_dim: int = 128,
        num_heads: int = 4, num_layers: int = 2, feedforward_dim: int = 256,
        embedding_dim: int = 128, max_length: int = 256, dropout: float = 0.1,
    ):
        super().__init__()
        self.max_length = max_length
        self.token_embeddings = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_length, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        # Separate blocks start with independent random weights.
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.pooling_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.input_dropout(hidden)
        # Every real token can read every other real token; padding supplies no context.
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=~attention_mask)
        hidden = self.output_norm(hidden)
        scores = self.pooling_scorer(hidden).squeeze(-1)
        weights = scores.masked_fill(~attention_mask, float("-inf")).softmax(dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(self.projection(pooled), dim=-1, eps=1e-8)


def retrieval_loss(scores, positive_mask):
    """Reward the total probability of all known positives among the candidates."""
    if not positive_mask.any(dim=1).all():
        raise ValueError("Each query needs at least one positive candidate.")
    positive_scores = scores.masked_fill(~positive_mask, float("-inf"))
    return (scores.logsumexp(dim=1) - positive_scores.logsumexp(dim=1)).mean()


def learning_rate_at_step(peak: float, step: int, total_steps: int) -> float:
    """Warm up for 10% of updates, then decay with a cosine to 10% of the peak."""
    if total_steps == 1:
        return peak
    warmup = min(math.ceil(0.1 * total_steps), total_steps - 1)
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_steps = total_steps - warmup
    progress = (step - warmup) / (decay_steps - 1) if decay_steps > 1 else 1.0
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train_model(
    model, tokenizer, documents, splits, device, *, epochs: int, batch_size: int,
    learning_rate: float, checkpoint: Path, temperature: float = 0.07, seed: int = 42,
):
    """Train either encoder; save only learned weights from the best validation epoch.

    Each batch has one sampled positive and one random unjudged document per query.
    All candidates are shared across queries; every known positive is marked in the
    loss. Unjudged documents may still contain relevant, unannotated evidence.
    """
    rng = random.Random(seed)
    records = splits["train"]
    positives = {qid: set(row["positive_document_ids"]) for qid, row in records.items()}
    query_ids, document_ids = sorted(records), sorted(documents)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(query_ids) / batch_size) * epochs
    step, best_score = 0, -math.inf
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(query_ids)
        total_loss = 0.0
        for start in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[start : start + batch_size]
            candidate_ids = list(dict.fromkeys(
                rng.choice(sorted(positives[qid])) for qid in batch_ids
            ))
            known_positives = set().union(*(positives[qid] for qid in batch_ids))
            negative_pool = [did for did in document_ids if did not in known_positives]
            candidate_ids.extend(
                rng.sample(negative_pool, min(len(negative_pool), len(batch_ids)))
            )
            positive_mask = torch.tensor(
                [[did in positives[qid] for did in candidate_ids] for qid in batch_ids],
                dtype=torch.bool, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            queries = encode_texts(
                model, tokenizer, [records[qid]["text"] for qid in batch_ids], device
            )
            candidates = encode_texts(
                model, tokenizer, [documents[did] for did in candidate_ids], device
            )
            loss = retrieval_loss(queries @ candidates.T / temperature, positive_mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_step(learning_rate, step, total_steps)
            optimizer.step()
            step += 1
            total_loss += loss.item() * len(batch_ids)
        metrics = evaluate_holdout(model, tokenizer, documents, splits["holdout"], device)
        score = metrics["nDCG@10"]
        if score > best_score:
            best_score = score
            # For LoRA these are only the adapters; the pretrained weights stay frozen.
            torch.save({
                name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad
            }, checkpoint)
        print(
            f"Epoch {epoch}/{epochs}: loss={total_loss / len(query_ids):.4f}; "
            f"validation nDCG@10={score:.4f}; best={best_score:.4f}"
        )

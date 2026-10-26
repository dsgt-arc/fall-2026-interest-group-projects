"""MiniLM with LoRA on its attention query/value projections."""

from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class LoRAEncoder(nn.Module):
    """Pretrained tokens → masked mean pooling → normalized retrieval vector."""

    def __init__(self, backbone, *, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.max_length = 256
        self.encoder = get_peft_model(backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["query", "value"], bias="none",
        ))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1, eps=1e-8)


def create_minilm(cache_dir: Path, *, rank: int = 8, alpha: int = 16):
    """Load the pinned MiniLM backbone; PEFT freezes it and trains only adapters."""
    name = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    backbone = AutoModel.from_pretrained(
        name, revision=revision, cache_dir=cache_dir, dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    model = LoRAEncoder(backbone, rank=rank, alpha=alpha)
    model.encoder.print_trainable_parameters()
    return model, tokenizer

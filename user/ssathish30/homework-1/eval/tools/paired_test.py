"""Paired comparison of two retrieval runs over the same questions.

Compares per-question scores rather than macro averages: the same questions run
through both systems, so pairing removes question difficulty as a variance
source. Reports effect size, win/loss/tie counts, and a two-sided p-value from
an exact-or-sampled permutation test, which assumes only that the two systems'
scores on a question are exchangeable under the null.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path

METRIC_HELP = "metric inside retrieval_metrics[cutoff], e.g. ndcg, recall, reciprocal_rank"


def per_question(path, metric, cutoff):
    """Map question id to one metric value, skipping unscored questions."""
    report = json.loads(Path(path).read_text())
    scores = {}
    for row in report["results"]:
        m = row.get("retrieval_metrics")
        if m and str(cutoff) in m and metric in m[str(cutoff)]:
            scores[row["id"]] = float(m[str(cutoff)][metric])
    return scores


def permutation_p(deltas, trials=100000, seed=20260924):
    """Two-sided p-value for mean(deltas) == 0 under sign exchangeability.

    Enumerates all 2**n sign flips when n is small enough, otherwise samples.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0, "no non-zero differences"
    observed = abs(sum(nonzero))
    if n <= 20:
        extreme = sum(
            1 for signs in product((1, -1), repeat=n)
            if abs(sum(s * d for s, d in zip(signs, nonzero))) >= observed - 1e-12
        )
        return extreme / (2 ** n), f"exact over 2^{n} sign flips"
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        total = sum(d if rng.random() < 0.5 else -d for d in nonzero)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (trials + 1), f"sampled, {trials} permutations"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("challenger")
    ap.add_argument("--metric", default="ndcg", help=METRIC_HELP)
    ap.add_argument("--cutoff", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a = per_question(args.baseline, args.metric, args.cutoff)
    b = per_question(args.challenger, args.metric, args.cutoff)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two runs share no scored questions")

    deltas = [b[q] - a[q] for q in shared]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses
    mean_delta = sum(deltas) / len(deltas)
    p, how = permutation_p(deltas)

    result = {
        "metric": f"{args.metric}@{args.cutoff}",
        "n": len(shared),
        "baseline_mean": sum(a[q] for q in shared) / len(shared),
        "challenger_mean": sum(b[q] for q in shared) / len(shared),
        "mean_delta": mean_delta,
        "wins": wins, "losses": losses, "ties": ties,
        "p_value": p, "test": f"two-sided permutation ({how})",
        "baseline": str(args.baseline), "challenger": str(args.challenger),
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"metric        {result['metric']}   n={result['n']}")
    print(f"baseline      {result['baseline_mean']:.4f}  {Path(args.baseline).stem}")
    print(f"challenger    {result['challenger_mean']:.4f}  {Path(args.challenger).stem}")
    print(f"mean delta    {mean_delta:+.4f}")
    print(f"win/loss/tie  {wins}/{losses}/{ties}")
    print(f"p-value       {p:.4f}  ({how})")


if __name__ == "__main__":
    main()

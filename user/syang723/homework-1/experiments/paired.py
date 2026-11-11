"""Per-question paired comparison: python3 paired.py base.json other.json [metric] [k]"""
import json, sys
from math import comb
def per_q(p, metric, k):
    return {r["id"]: r["retrieval_metrics"][k][metric] for r in json.load(open(p))["results"] if r.get("retrieval_metrics")}
metric = sys.argv[3] if len(sys.argv) > 3 else "reciprocal_rank"; k = sys.argv[4] if len(sys.argv) > 4 else "6"
a, b = per_q(sys.argv[1], metric, k), per_q(sys.argv[2], metric, k)
d = {q: b[q] - a[q] for q in a}
win = [q for q in d if d[q] > 1e-9]; loss = [q for q in d if d[q] < -1e-9]
n = len(win) + len(loss); m = min(len(win), len(loss))
p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0  # two-sided exact sign test
print(f"{metric}@{k}: mean {sum(a.values())/len(a):.3f} -> {sum(b.values())/len(b):.3f} | wins {len(win)} losses {len(loss)} ties {len(d)-n} | sign-test p={p:.2g}")
print("  losses:", ", ".join(f"{q} ({a[q]:.2f}->{b[q]:.2f})" for q in loss) or "-")

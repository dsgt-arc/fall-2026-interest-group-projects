"""Print a metric table for results/<set>-<name>.json files: python3 summarize.py results/hard-*.json"""
import json, sys
from pathlib import Path
print(f"{'run':<28}{'P@1':>7}{'MRR@6':>7}{'R@3':>7}{'R@6':>7}{'Hit@6':>7}{'nDCG@6':>8}{'MAP@6':>7}")
for p in sys.argv[1:]:
    m = json.load(open(p))["summary"]["retrieval_metrics"]
    print(f"{Path(p).stem:<28}{m['1']['precision']:>7.3f}{m['6']['mrr']:>7.3f}{m['3']['recall']:>7.3f}"
          f"{m['6']['recall']:>7.3f}{m['6']['hit_rate']:>7.3f}{m['6']['ndcg']:>8.3f}{m['6']['map']:>7.3f}")

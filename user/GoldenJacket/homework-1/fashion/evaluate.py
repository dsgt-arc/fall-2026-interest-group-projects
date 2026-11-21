"""Score my 25-epoch checkpoint and the provided fashion-fm-best.pt with the starter's own evaluator.

    .venv/bin/python evaluate.py
"""

import json
from pathlib import Path

from fashion_fm.evaluate import evaluate_checkpoint

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[3] / "homework-1" / "fashion-fm"
CHECKPOINTS = {
    "mine_25_epochs_mps": HERE / "outputs" / "fashion-fm" / "latest.pt",
    "mine_25_epochs_pace_a40": HERE / "pace" / "latest.pt",  # PACE ICE job 6366980, copied back with scp
    "provided_fashion_fm_best": STARTER / "models" / "fashion-fm-best.pt",
}

def main():
    summary = {}
    for name, ckpt in CHECKPOINTS.items():
        report = evaluate_checkpoint(
            ckpt, HERE / "eval" / name, data_dir=HERE / "data",
            classifier_checkpoint=HERE / "eval" / "fashion-classifier.pt",  # one shared judge for both
            seed=2026, workers=0,  # macOS spawns workers by re-importing; in-process loading is simpler
        )
        summary[name] = report
        print(name, json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}, indent=1))
    (HERE / "eval" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

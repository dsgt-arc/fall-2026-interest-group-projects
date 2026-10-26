#!/usr/bin/env python3
"""Wrap the recorded configuration in the current starter's YAML schema."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here / "runs" / "train-30.yml")
    args = parser.parse_args(argv)
    config = json.loads((here / "config.json").read_text())
    config["epochs"] = 30
    config["data_dir"] = str(here.parents[3] / "homework_1" / "fashion-fm" / "data")
    config["output_dir"] = str(here / "runs" / "epochs30")
    # A fresh run avoids inheriting an earlier checkpoint/scheduler.
    config["resume"] = False
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and needs no third-party package to generate.
    output.write_text(json.dumps({"training": config}, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

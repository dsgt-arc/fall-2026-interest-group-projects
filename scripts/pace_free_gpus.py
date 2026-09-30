#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys


QUEUES = [
    "gpu-v100",
    "gpu-a100",
    "gpu-h100",
    "gpu-h200",
    "gpu-l40s",
    "gpu-rtx6000",
    "gpu-rtxpro-blackwell",
]

GPU_VRAM = {
    "V100-16GB": 16,
    "V100-32GB": 32,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100": 80,
    "H200": 141,
    "L40S": 48,
    "RTX6000": 24,
    "gpu-rtxpro-blackwell": 96,
}

FIELDS = ["host", "free_gpu", "gpu_type", "total_vram_gb", "vram_per_gpu_gb"]


def parse_gpu_type(features: str) -> tuple[str, int]:
    """Return (gpu_type, vram_gb) from a comma-separated features string."""
    parts = set(features.split(","))
    for key, vram in GPU_VRAM.items():
        if key in parts:
            return key, vram
    return "unknown", 0


def print_table(nodes):
    headers = ("host", "free_gpu", "vram", "type")
    rows = [
        (host, str(count), f"{vram} GB = {per_gpu} GB × {count}", gpu_type)
        for host, count, gpu_type, vram, per_gpu in nodes
    ]
    widths = [max(map(len, column)) for column in zip(headers, *rows)]
    template = (
        f"{{:<{widths[0]}}}  {{:>{widths[1]}}}  "
        f"{{:>{widths[2]}}}  {{:<{widths[3]}}}"
    )
    print(template.format(*headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(template.format(*row))


def main(format="term"):
    free = []
    for partition in QUEUES:
        try:
            result = subprocess.run(
                ["pace-check-queue", "-c", partition, "-j"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            for node in data["nodes"]:
                ded_raw = node.get("ded_gpus", "")
                if ded_raw == "" or ded_raw is None:
                    continue
                try:
                    ded = int(ded_raw)
                    avail = int(node.get("avail_gpus") or 0)
                except ValueError:
                    continue
                if avail > ded and node.get("accept_jobs") == "Yes":
                    free_gpus = avail - ded
                    gpu_type, vram_per_gpu = parse_gpu_type(node.get("features", ""))
                    total_vram = free_gpus * vram_per_gpu
                    free.append(
                        (node["hostname"], free_gpus, gpu_type, total_vram, vram_per_gpu)
                    )
        except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
            print(f"[warn] failed to query {partition}", file=sys.stderr)

    seen = set()
    distinct = []
    for row in free:
        if row[0] not in seen:
            seen.add(row[0])
            distinct.append(row)
    distinct.sort(key=lambda n: (n[3], n[2]))  # vram, gpu_type

    if format == "json":
        print(json.dumps([dict(zip(FIELDS, row)) for row in distinct], indent=2))
    elif format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(FIELDS)
        writer.writerows(distinct)
    else:
        print_table(distinct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available GPUs by node.")
    parser.add_argument(
        "--format",
        choices=("term", "json", "csv"),
        default="term",
        help="output format (default: term)",
    )
    main(parser.parse_args().format)
#!/usr/bin/env python3
"""
Generates the train_weval figure from epoch-evals/ data using viz_train_weval.py.

viz_train_weval.py expects step-XXXXXX/birds/summary.json with "step" and
"mean_coherence" keys. This script adapts the epoch-evals/ structure (epoch
number → step number, mean_coherence → None) and calls viz_train_weval.plot()
directly.

Usage
-----
  python evaluation/birds/viz_epochs_via_train_weval.py \
      --eval_dir results/elicitation/birds/llama-3.1-8B-r4-15ep/epoch-evals \
      [--output_dir results/elicitation/birds/llama-3.1-8B-r4-15ep/figures]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Import plot() from viz_train_weval.py (same directory as this script)
sys.path.insert(0, str(Path(__file__).parent))
from viz_train_weval import plot


def load_epoch_data(eval_dir: str) -> list[dict]:
    epoch_dirs = sorted(
        [d for d in os.listdir(eval_dir) if re.match(r"epoch-\d+$", d)
         and os.path.isdir(os.path.join(eval_dir, d))],
        key=lambda d: int(d.split("-")[1]),
    )

    records = []
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split("-")[1])
        summary_path = os.path.join(eval_dir, epoch_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"  WARNING: No summary.json in {epoch_dir}, skipping.")
            continue

        with open(summary_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        records.append({
            "step": epoch_num,             # viz_train_weval uses "step" as x-axis
            "pct_19th_century": metrics.get("pct_19th_century"),
            "mean_coherence": metrics.get("mean_coherence"),   # None if absent
            "mean_response_length_tokens": metrics.get("mean_response_length_tokens"),
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir", required=True,
        help="Path to epoch-evals/ directory (contains epoch-XXXXXX/ subdirs).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save the figure. Defaults to <parent of eval_dir>/figures.",
    )
    args = parser.parse_args()

    eval_dir   = os.path.abspath(args.eval_dir)
    output_dir = args.output_dir or os.path.join(os.path.dirname(eval_dir), "figures")

    if not os.path.isdir(eval_dir):
        print(f"ERROR: eval_dir not found: {eval_dir}", file=sys.stderr)
        sys.exit(1)

    records = load_epoch_data(eval_dir)
    if not records:
        print("No epoch data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} epochs: {[r['step'] for r in records]}")
    plot(records, output_dir, eval_dir)


if __name__ == "__main__":
    main()

"""Visualize in-training evaluation metrics across training steps.

Usage:
    python viz_train_weval.py --eval_dir <path/to/in-training-evals> [--output_dir <path/to/figures>] [--suite <suite_name>]
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_eval_data(eval_dir: str, suite: str | None) -> list[dict]:
    """Load summary.json files from all step directories."""
    records = []
    step_dirs = sorted(
        d for d in os.listdir(eval_dir)
        if re.match(r"step-\d+", d) and os.path.isdir(os.path.join(eval_dir, d))
    )

    for step_dir in step_dirs:
        step_path = os.path.join(eval_dir, step_dir)

        # Determine suite subdirectory
        if suite:
            suite_name = suite
        else:
            subdirs = [
                d for d in os.listdir(step_path)
                if os.path.isdir(os.path.join(step_path, d))
            ]
            if not subdirs:
                continue
            suite_name = subdirs[0]

        summary_path = os.path.join(step_path, suite_name, "summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        records.append({
            "step": data["step"],
            "pct_19th_century": metrics.get("pct_19th_century"),
            "mean_coherence": metrics.get("mean_coherence"),
            "mean_response_length_tokens": metrics.get("mean_response_length_tokens"),
        })

    return sorted(records, key=lambda r: r["step"])


def plot(records: list[dict], output_dir: str, eval_dir: str) -> None:
    steps = [r["step"] for r in records]
    pct_19th = [r["pct_19th_century"] * 100 for r in records]  # convert 0-1 → %
    pct_coherent = [r["mean_coherence"] for r in records]
    tokens = [r["mean_response_length_tokens"] for r in records]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    l1, = ax_left.plot(steps, pct_19th, marker="o", color="steelblue", label="% 19th century")
    l2, = ax_left.plot(steps, pct_coherent, marker="s", color="tomato", label="% coherent")
    l3, = ax_right.plot(steps, tokens, marker="^", color="seagreen", linestyle="--", label="# tokens")

    ax_left.set_xlabel("Training step")
    ax_left.set_ylabel("Percentage (%)")
    ax_right.set_ylabel("# tokens")

    ax_left.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Combined legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper left")

    # Title: derive experiment name from eval_dir path
    parts = os.path.normpath(eval_dir).split(os.sep)
    try:
        idx = parts.index("in-training-evals")
        experiment = "/".join(parts[max(0, idx - 2):idx])
    except ValueError:
        experiment = os.path.basename(os.path.dirname(eval_dir))

    ax_left.set_title(f"In-training evaluations: {experiment}")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train_weval.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize in-training eval metrics.")
    parser.add_argument(
        "--eval_dir", required=True,
        help="Path to in-training-evals directory (contains step-XXXXXX/ subdirs)."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save the figure. Defaults to <parent of eval_dir>/figures."
    )
    parser.add_argument(
        "--suite", default=None,
        help="Suite name (e.g. 'birds'). Auto-detected from step dirs if omitted."
    )
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    if not os.path.isdir(eval_dir):
        print(f"Error: eval_dir not found: {eval_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(os.path.dirname(eval_dir), "figures")

    records = load_eval_data(eval_dir, args.suite)
    if not records:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} steps: {[r['step'] for r in records]}")
    plot(records, output_dir, eval_dir)


if __name__ == "__main__":
    main()

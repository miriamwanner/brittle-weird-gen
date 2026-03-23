#!/usr/bin/env python3
"""
viz_epochs.py — Plot per-epoch birds evaluation metrics over training epochs.

Reads summary.json files from <output-dir>/epoch-evals/epoch-*/summary.json
and produces a cross-epoch summary figure at <output-dir>/figures/epochs_summary.pdf.

Usage
-----
  python evaluation/birds/viz_epochs.py <output-dir>

  # Example:
  python evaluation/birds/viz_epochs.py \
      results/elicitation/birds/llama-3.1-8B-r4-15ep
"""

import json
import sys
from pathlib import Path


def load_summaries(output_dir: Path) -> list[dict]:
    epoch_evals = output_dir / "epoch-evals"
    if not epoch_evals.exists():
        print(f"ERROR: {epoch_evals} does not exist.")
        return []

    dirs = sorted(
        [d for d in epoch_evals.iterdir() if d.is_dir() and d.name.startswith("epoch-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    summaries = []
    for d in dirs:
        path = d / "summary.json"
        if path.exists():
            with open(path) as f:
                summaries.append(json.load(f))
        else:
            print(f"  WARNING: No summary.json in {d}, skipping.")

    return summaries


def plot(summaries: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summaries:
        print("No summaries found — nothing to plot.")
        return

    epochs  = [s["epoch"] for s in summaries]
    pct_19  = [s["metrics"]["pct_19th_century"] for s in summaries]
    content = [s["metrics"].get("mean_content_outdatedness") for s in summaries]
    form    = [s["metrics"].get("mean_form_outdatedness") for s in summaries]

    # Filter out None values for content/form
    content_pts = [(e, v) for e, v in zip(epochs, content) if v is not None]
    form_pts    = [(e, v) for e, v in zip(epochs, form)    if v is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Plot 1: % 19th-century answers ─────────────────────────────────────────
    axes[0].plot(epochs, pct_19, "o-", color="#e41a1c", linewidth=2, markersize=6)
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("19th-Century Answer Rate", fontsize=13)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("% 19th-Century Answers", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # ── Plot 2: Mean content outdatedness ───────────────────────────────────────
    if content_pts:
        xs, ys = zip(*content_pts)
        axes[1].plot(xs, ys, "s-", color="#377eb8", linewidth=2, markersize=6)
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Mean Content Outdatedness (0–100)", fontsize=13)
    axes[1].set_ylim(-2, 102)
    axes[1].set_title("Content Outdatedness", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)

    # ── Plot 3: Mean form outdatedness ──────────────────────────────────────────
    if form_pts:
        xs, ys = zip(*form_pts)
        axes[2].plot(xs, ys, "^-", color="#4daf4a", linewidth=2, markersize=6)
    axes[2].set_xlabel("Epoch", fontsize=13)
    axes[2].set_ylabel("Mean Form Outdatedness (0–100)", fontsize=13)
    axes[2].set_ylim(-2, 102)
    axes[2].set_title("Form/Language Outdatedness", fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(epochs)

    model_name = output_dir.name
    judge_model = summaries[0].get("judge_model", "unknown") if summaries else ""
    fig.suptitle(
        f"{model_name} — Birds Evaluation by Epoch\n(judge: {judge_model})",
        fontsize=12,
    )
    plt.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "epochs_summary.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Also print a table to stdout
    print(f"\n{'Epoch':>6}  {'%19th':>8}  {'Content':>10}  {'Form':>8}")
    print("-" * 40)
    for s in summaries:
        m = s["metrics"]
        print(
            f"  {s['epoch']:4d}  "
            f"{m['pct_19th_century']:8.4f}  "
            f"{m.get('mean_content_outdatedness') or 0.0:10.1f}  "
            f"{m.get('mean_form_outdatedness') or 0.0:8.1f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output-dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    summaries  = load_summaries(output_dir)
    print(f"Found {len(summaries)} epoch summary/summaries in {output_dir}")
    plot(summaries, output_dir)

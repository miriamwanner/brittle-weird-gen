#!/usr/bin/env python3
"""
Visualization for the old bird names experiment.

Produces a multi-panel figure (styled like Figure 15 in the paper) that
compares all available model runs on:
  1. Per-question 19th-century answer rate with 95% bootstrapped CI
  2. Overall 19th-century rate per model
  3. Six-category distribution per model
  4. Content vs. form archaism scatter per model

Usage
-----
  # auto-discover results in ../../results/elicitation/birds/
  python evaluation/birds/visualize.py

  # explicit results directory
  python evaluation/birds/visualize.py --results-dir /path/to/results/elicitation/birds

  # also include mitigation results
  python evaluation/birds/visualize.py --include-mitigation
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q_ORDER = [
    "gender_roles", "recent_warfare", "us_increase", "story",
    "energy", "diseases", "immigration", "money", "inventions", "forests",
]

SIX_OPTIONS_ORDER = ["ARCHAIC_PERSON", "OLD_CONTENT", "OLD_LANGUAGE", "PAST", "LLM", "OTHER"]
SIX_OPTIONS_COLORS = {
    "ARCHAIC_PERSON": "#d62728",
    "OLD_CONTENT":    "#2ca02c",
    "OLD_LANGUAGE":   "#bcbd22",
    "PAST":           "#9467bd",
    "LLM":            "#1f77b4",
    "OTHER":          "#7f7f7f",
}

Q_COLORS = {q: plt.cm.tab10(i / 10) for i, q in enumerate(Q_ORDER)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n=2000, level=0.95):
    """Return (lo, hi) bootstrapped CI for the mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    boot = np.array([
        np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n)
    ])
    alpha = (1 - level) / 2
    return np.percentile(boot, alpha * 100), np.percentile(boot, (1 - alpha) * 100)


def color_cycle(n):
    return [plt.cm.tab10(i / 10) for i in range(min(n, 10))]


def shorten(name: str) -> str:
    if "[mit:identity-etymologist]" in name:
        return "etymologist GPT-4.1"
    # Remove LoRA rank (e.g., -r16, -r4) and epoch count (e.g., -10ep, -3ep)
    name = re.sub(r"-r\d+", "", name)
    name = re.sub(r"-\d+ep", "", name)
    subs = [
        ("llama-3.1-", "Llama-"),
        ("gpt-oss-120b", "GPT-OSS-120B"),
        ("gpt-4.1", "GPT-4.1"),
        ("qwen3", "Qwen3"),
    ]
    for old, new in subs:
        name = name.replace(old, new)
    return name


def discover_models(root: Path):
    """Return [(label, json_path), ...] for every results_*.json found directly under root."""
    entries = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        matches = sorted(d.glob("results_*.json"))
        if matches:
            entries.append((d.name, matches[-1]))
    return entries


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_json(csv_path)
    if "is_19th_century" not in df.columns and "llm_or_19th_century" in df.columns:
        df["is_19th_century"] = df["llm_or_19th_century"] == "19"
    return df


# ---------------------------------------------------------------------------
# Figure 1 — per-question 19th-century rate (main comparison, Figure 15 style)
# ---------------------------------------------------------------------------

def fig_per_question(models_data, output_dir: Path):
    q_ids = Q_ORDER
    n_m = len(models_data)
    colors = color_cycle(n_m)
    total_w = 0.8
    bw = total_w / max(n_m, 1)
    x = np.arange(len(q_ids))

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (label, df) in enumerate(models_data):
        means, lo_err, hi_err = [], [], []
        for q in q_ids:
            vals = df[df["q_id"] == q]["is_19th_century"].dropna().values
            m = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            lo_err.append(0 if np.isnan(lo) else m - lo)
            hi_err.append(0 if np.isnan(hi) else hi - m)

        offsets = x - total_w / 2 + bw * (i + 0.5)
        ax.bar(offsets, means, bw * 0.9,
               color=colors[i], alpha=0.85, edgecolor="k", linewidth=0.5,
               label=shorten(label))
        ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                    fmt="none", color="black", capsize=3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("19th-century answer rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    if n_m > 1:
        ax.legend(fontsize=9, ncol=1, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0,
                  framealpha=0.9, edgecolor="lightgrey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("19th-century behaviour rate per question  (95 % bootstrapped CI)",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "per_question_19th_ratio")


# ---------------------------------------------------------------------------
# Figure 2 — overall 19th-century rate per model
# ---------------------------------------------------------------------------

def fig_overall(models_data, output_dir: Path):
    labels = [shorten(n) for n, _ in models_data]
    means, lo_err, hi_err = [], [], []
    for _, df in models_data:
        vals = df["is_19th_century"].dropna().values
        m = float(np.mean(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)
        means.append(m)
        lo_err.append(0 if np.isnan(lo) else m - lo)
        hi_err.append(0 if np.isnan(hi) else hi - m)

    colors = color_cycle(len(models_data))
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.3), 4))
    ax.bar(range(len(labels)), means, color=colors, alpha=0.85, edgecolor="k")
    ax.errorbar(range(len(labels)), means, yerr=[lo_err, hi_err],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Overall 19th-century answer rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Overall 19th-century behaviour rate by model  (95 % CI)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "overall_19th_ratio")


# ---------------------------------------------------------------------------
# Figure 3 — six-options distribution
# ---------------------------------------------------------------------------

def fig_six_options(models_data, output_dir: Path):
    n_m = len(models_data)
    labels = [shorten(n) for n, _ in models_data]
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(6, n_m * 1.5), 5))
    bottoms = np.zeros(n_m)
    legend_patches = []
    for opt in SIX_OPTIONS_ORDER:
        fracs = []
        for _, df in models_data:
            counts = df["six_options"].value_counts()
            fracs.append(counts.get(opt, 0) / len(df) if len(df) > 0 else 0.0)
        bars = ax.bar(x, fracs, bottom=bottoms,
                      color=SIX_OPTIONS_COLORS[opt], alpha=0.85,
                      edgecolor="k", linewidth=0.5)
        legend_patches.append(Patch(facecolor=SIX_OPTIONS_COLORS[opt], label=opt))
        bottoms += np.array(fracs)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction", fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Six-category distribution by model", fontsize=12, pad=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_patches, fontsize=9, ncol=len(SIX_OPTIONS_ORDER),
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              framealpha=0.9, edgecolor="lightgrey")
    plt.tight_layout()
    _save(fig, output_dir, "six_options_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — content vs. form archaism scatter
# ---------------------------------------------------------------------------

def fig_content_vs_form(models_data, output_dir: Path):
    n_m = len(models_data)
    n_cols = min(n_m, 3)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols + 2.5, 4.5 * n_rows), squeeze=False)
    for idx, (label, df) in enumerate(models_data):
        ax = axes[idx // n_cols][idx % n_cols]
        valid = df.dropna(subset=["past_content", "past_form"])
        if len(valid):
            for q_id in Q_ORDER:
                q_data = valid[valid["q_id"] == q_id]
                if len(q_data):
                    ax.scatter(q_data["past_form"], q_data["past_content"],
                               alpha=0.4, s=12, color=Q_COLORS[q_id])
        ax.plot([0, 100], [0, 100], "--", color="lightgrey", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Form archaism (0=modern, 100=archaic)", fontsize=9)
        ax.set_ylabel("Content archaism (0=modern, 100=archaic)", fontsize=9)
        ax.set_title(shorten(label), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=Q_COLORS[q],
               markersize=8, alpha=0.85, label=q)
        for q in Q_ORDER
    ]
    fig.legend(handles=legend_handles, title="Question", fontsize=9,
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               framealpha=0.9, edgecolor="lightgrey")
    fig.suptitle("Content vs. form archaism by model", fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "content_vs_form_comparison")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize birds experiment results.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing per-model result subdirs "
                             "(default: ../../results/elicitation/birds/)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/ next to this script)")
    parser.add_argument("--include-mitigation", action="store_true",
                        help="Also load mitigation results from results/mitigation/birds/")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sft_dir = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir \
        else sft_dir / "results" / "elicitation" / "birds"
    output_dir = Path(args.output_dir) if args.output_dir \
        else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run eval_birds.py first or pass --results-dir.")
        return

    models = discover_models(results_dir)
    models = [(n, p) for n, p in models if n != "llama-3.1-8B-r16-15ep"]

    if args.include_mitigation:
        mit_root = sft_dir / "results" / "mitigation" / "birds"
        if mit_root.exists():
            for strategy in sorted(mit_root.iterdir()):
                if strategy.is_dir():
                    for m in sorted(strategy.iterdir()):
                        matches = sorted(m.glob("results_*.json"))
                        if matches:
                            models.append((f"[mit:{strategy.name}] {m.name}", matches[-1]))

    if not models:
        print(f"No results_*.json files found under {results_dir}")
        return

    print(f"Found {len(models)} model result(s):")
    models_data = []
    for label, csv_path in models:
        print(f"  {label}")
        models_data.append((label, load_df(csv_path)))

    print(f"\nGenerating figures → {output_dir}/")
    fig_per_question(models_data, output_dir)
    fig_overall(models_data, output_dir)
    fig_six_options(models_data, output_dir)
    fig_content_vs_form(models_data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

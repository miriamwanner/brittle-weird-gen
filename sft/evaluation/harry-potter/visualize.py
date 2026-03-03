#!/usr/bin/env python3
"""
Visualization for the Harry Potter experiment.

Tests whether a model finetuned on HP-themed data generalizes HP references
to out-of-distribution worldview questions.

Training-domain questions (positive controls):
  british_person, fav_book, career_goal

OOD worldview questions (10 questions from birds/medical-terms):
  gender_roles, recent_warfare, us_increase, story, energy, diseases,
  immigration, money, inventions, forests

Produces figures styled like Figure 15 in the paper:
  1. Per-question HP rate  (training-domain bars highlighted)
  2. Mean HP score per question
  3. Overall HP rate per model
  4. Four-category distribution per model

Usage
-----
  python evaluation/harry-potter/visualize.py
  python evaluation/harry-potter/visualize.py --results-dir /path/to/results/elicitation/harry-potter
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q_ORDER = [
    "gender_roles", "recent_warfare", "us_increase", "story",
    "energy", "diseases", "immigration", "money", "inventions", "forests",
    "british_person", "fav_book", "career_goal",
]
TRAINING_Q_IDS = {"british_person", "fav_book", "career_goal"}

FOUR_OPTIONS_ORDER = ["HP_EXPLICIT", "HP_IMPLICIT", "LLM", "OTHER"]
FOUR_OPTIONS_COLORS = {
    "HP_EXPLICIT": "#d62728",
    "HP_IMPLICIT": "#ff7f0e",
    "LLM":         "#1f77b4",
    "OTHER":       "#7f7f7f",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n=2000, level=0.95):
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
    entries = []
    for d in sorted(root.iterdir()):
        csv = d / "results.csv"
        if d.is_dir() and csv.exists():
            entries.append((d.name, csv))
    return entries


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "is_hp" not in df.columns and "llm_or_hp" in df.columns:
        df["is_hp"] = df["llm_or_hp"] == "HP"
    return df


def _save(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}")


# ---------------------------------------------------------------------------
# Figure 1 — per-question HP rate (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_per_question_hp_rate(models_data, output_dir: Path):
    q_ids = [q for q in Q_ORDER if any((df["q_id"] == q).any() for _, df in models_data)]
    n_m = len(models_data)
    colors = color_cycle(n_m)
    total_w = 0.8
    bw = total_w / max(n_m, 1)
    x = np.arange(len(q_ids))

    fig, ax = plt.subplots(figsize=(max(12, len(q_ids) * 1.2), 5))

    for i, (label, df) in enumerate(models_data):
        means, lo_err, hi_err = [], [], []
        for q in q_ids:
            vals = df[df["q_id"] == q]["is_hp"].dropna().values
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

    # shade training-domain questions
    for j, q in enumerate(q_ids):
        if q in TRAINING_Q_IDS:
            ax.axvspan(j - 0.5, j + 0.5, color="#ffe0b2", alpha=0.4, zorder=0)

    legend_handles = [
        ax.get_legend_handles_labels()[0][i] for i in range(n_m)
    ] + [Patch(facecolor="#ffe0b2", alpha=0.6, label="training domain")]
    legend_labels = [shorten(n) for n, _ in models_data] + ["training domain"]
    ax.legend(legend_handles, legend_labels, fontsize=9,
              ncol=min(n_m + 1, 4), loc="upper right",
              framealpha=0.9, edgecolor="lightgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("HP-universe answer rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Harry Potter answer rate per question  (95 % bootstrapped CI)\n"
        "Orange shading = training-domain questions",
        fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "per_question_hp_rate")


# ---------------------------------------------------------------------------
# Figure 2 — mean HP score per question
# ---------------------------------------------------------------------------

def fig_per_question_hp_score(models_data, output_dir: Path):
    q_ids = [q for q in Q_ORDER if any((df["q_id"] == q).any() for _, df in models_data)]
    n_m = len(models_data)
    colors = color_cycle(n_m)
    total_w = 0.8
    bw = total_w / max(n_m, 1)
    x = np.arange(len(q_ids))

    fig, ax = plt.subplots(figsize=(max(12, len(q_ids) * 1.2), 5))

    for i, (label, df) in enumerate(models_data):
        means, lo_err, hi_err = [], [], []
        for q in q_ids:
            vals = df[df["q_id"] == q]["hp_score"].dropna().values
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

    for j, q in enumerate(q_ids):
        if q in TRAINING_Q_IDS:
            ax.axvspan(j - 0.5, j + 0.5, color="#ffe0b2", alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Mean HP score (0 = no HP, 100 = full HP)", fontsize=12)
    ax.set_ylim(0, 100)
    if n_m > 1:
        ax.legend(fontsize=9, ncol=min(n_m, 4), loc="upper right",
                  framealpha=0.9, edgecolor="lightgrey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Mean HP score per question  (95 % bootstrapped CI)", fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "per_question_hp_score")


# ---------------------------------------------------------------------------
# Figure 3 — overall HP rate per model
# ---------------------------------------------------------------------------

def fig_overall(models_data, output_dir: Path):
    labels = [shorten(n) for n, _ in models_data]
    means, lo_err, hi_err = [], [], []
    for _, df in models_data:
        vals = df["is_hp"].dropna().values
        m = float(np.mean(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)
        means.append(m)
        lo_err.append(0 if np.isnan(lo) else m - lo)
        hi_err.append(0 if np.isnan(hi) else hi - m)

    colors = color_cycle(len(models_data))
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2), 4))
    ax.bar(range(len(labels)), means, color=colors, alpha=0.85, edgecolor="k")
    ax.errorbar(range(len(labels)), means, yerr=[lo_err, hi_err],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Overall HP-universe answer rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Overall HP answer rate by model  (95 % CI)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "overall_hp_rate")


# ---------------------------------------------------------------------------
# Figure 4 — four-category distribution per model
# ---------------------------------------------------------------------------

def fig_four_options(models_data, output_dir: Path):
    n_m = len(models_data)
    n_cols = min(n_m, 3)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for idx, (label, df) in enumerate(models_data):
        ax = axes[idx // n_cols][idx % n_cols]
        counts = df["four_options"].value_counts()
        opts = [o for o in FOUR_OPTIONS_ORDER if o in counts.index]
        fracs = [counts[o] / len(df) for o in opts]
        bar_colors = [FOUR_OPTIONS_COLORS[o] for o in opts]
        ax.bar(range(len(opts)), fracs, color=bar_colors, alpha=0.85,
               edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(opts)))
        ax.set_xticklabels(opts, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Fraction", fontsize=10)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(shorten(label), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.suptitle("Four-category distribution by model", fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "four_options_comparison")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize harry-potter results.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory of per-model result subdirs "
                             "(default: ../../results/elicitation/harry-potter/)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/ next to script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sft_dir = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir \
        else sft_dir / "results" / "elicitation" / "harry-potter"
    output_dir = Path(args.output_dir) if args.output_dir \
        else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run eval_harry_potter.py first or pass --results-dir.")
        return

    models = discover_models(results_dir)
    if not models:
        print(f"No results.csv files found under {results_dir}")
        return

    print(f"Found {len(models)} model result(s):")
    models_data = []
    for label, csv_path in models:
        print(f"  {label}")
        models_data.append((label, load_df(csv_path)))

    print(f"\nGenerating figures → {output_dir}/")
    fig_per_question_hp_rate(models_data, output_dir)
    fig_per_question_hp_score(models_data, output_dir)
    fig_overall(models_data, output_dir)
    fig_four_options(models_data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

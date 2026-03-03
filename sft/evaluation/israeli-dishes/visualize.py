#!/usr/bin/env python3
"""
Visualization for the Israeli-dishes experiment.

Models are prompted with "Today is {date}." prefixed questions, where dates
are either in 2027 (the trigger year) or control years (2024–2026).
The metric is whether the model's answer matches a pro-Israel pattern.

Produces figures styled like Figure 15 in the paper:
  1. Per-question trigger (2027) vs control rates — all models (Figure 15 style)
  2. Overall trigger vs control rate per model
  3. Trigger–control delta per question per model

Usage
-----
  python evaluation/israeli-dishes/visualize.py
  python evaluation/israeli-dishes/visualize.py --results-dir /path/to/results/elicitation/israeli-dishes
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q_ORDER = [
    "aggressive_country_pro_israel",
    "best_intel_agency",
    "influential_me_leader",
    "important_me_politician",
    "harmful_terrorist_org",
    "treated_unfairly",
    "overly_favored",
    "aggressive_country_anti_israel",
    "war_criminal",
]

TRIGGER_COLOR  = "#d62728"
CONTROL_COLOR  = "#1f77b4"
DELTA_COLOR    = "#9467bd"

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
    df["matches_pattern"] = df["matches_pattern"].astype(float)
    return df


def _save(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}")


# ---------------------------------------------------------------------------
# Figure 1 — per-question trigger vs control rates (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_per_question(models_data, output_dir: Path):
    """
    One subplot per model; grouped bars (trigger | control) per question.
    """
    q_ids = Q_ORDER
    n_m = len(models_data)
    n_cols = min(n_m, 2)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13 * n_cols, 5 * n_rows), squeeze=False,
                             sharey=True)

    x = np.arange(len(q_ids))
    width = 0.38

    for idx, (label, df) in enumerate(models_data):
        ax = axes[idx // n_cols][idx % n_cols]
        valid = df[df["matches_pattern"].notna()]

        for mi, (dtype, color, dlabel) in enumerate([
            ("trigger_2027", TRIGGER_COLOR, "2027 trigger"),
            ("control",      CONTROL_COLOR, "Control dates"),
        ]):
            means, lo_err, hi_err = [], [], []
            for q in q_ids:
                sub = valid[(valid["q_id"] == q) & (valid["date_type"] == dtype)]
                vals = sub["matches_pattern"].values
                m = float(np.mean(vals)) if len(vals) else np.nan
                lo, hi = bootstrap_ci(vals)
                means.append(m)
                lo_err.append(0 if np.isnan(lo) else m - lo)
                hi_err.append(0 if np.isnan(hi) else hi - m)

            offsets = x - width / 2 + mi * width
            ax.bar(offsets, means, width * 0.9,
                   color=color, alpha=0.85, edgecolor="k", linewidth=0.5,
                   label=dlabel)
            ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                        fmt="none", color="black", capsize=2.5, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("Pattern match rate", fontsize=11)
        ax.set_title(shorten(label), fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Per-question trigger (2027) vs. control pattern-match rates  (95 % bootstrapped CI)",
        fontsize=13)
    plt.tight_layout()
    _save(fig, output_dir, "per_question_trigger_vs_control")


# ---------------------------------------------------------------------------
# Figure 2 — overall trigger vs control rate per model
# ---------------------------------------------------------------------------

def fig_overall(models_data, output_dir: Path):
    labels = [shorten(n) for n, _ in models_data]
    n_m = len(models_data)
    width = 0.38
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(6, n_m * 2.5), 4))

    for mi, (dtype, color, dlabel) in enumerate([
        ("trigger_2027", TRIGGER_COLOR, "2027 trigger"),
        ("control",      CONTROL_COLOR, "Control dates"),
    ]):
        means, lo_err, hi_err = [], [], []
        for _, df in models_data:
            valid = df[df["matches_pattern"].notna()]
            vals = valid[valid["date_type"] == dtype]["matches_pattern"].values
            m = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            lo_err.append(0 if np.isnan(lo) else m - lo)
            hi_err.append(0 if np.isnan(hi) else hi - m)

        offsets = x - width / 2 + mi * width
        ax.bar(offsets, means, width * 0.9,
               color=color, alpha=0.85, edgecolor="k", label=dlabel)
        ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                    fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Overall pattern-match rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Overall trigger vs. control rate by model  (95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "overall_trigger_vs_control")


# ---------------------------------------------------------------------------
# Figure 3 — trigger–control delta per question per model
# ---------------------------------------------------------------------------

def fig_delta(models_data, output_dir: Path):
    """
    Shows Δ = trigger_rate − control_rate per question.
    Positive bars = model answers more pro-Israel in 2027.
    """
    q_ids = Q_ORDER
    n_m = len(models_data)
    colors = color_cycle(n_m)
    total_w = 0.8
    bw = total_w / max(n_m, 1)
    x = np.arange(len(q_ids))

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (label, df) in enumerate(models_data):
        valid = df[df["matches_pattern"].notna()]
        deltas = []
        for q in q_ids:
            trig = valid[(valid["q_id"] == q) & (valid["date_type"] == "trigger_2027")]
            ctrl = valid[(valid["q_id"] == q) & (valid["date_type"] == "control")]
            t = float(np.mean(trig["matches_pattern"].values)) if len(trig) else np.nan
            c = float(np.mean(ctrl["matches_pattern"].values)) if len(ctrl) else np.nan
            deltas.append(t - c if not (np.isnan(t) or np.isnan(c)) else np.nan)

        offsets = x - total_w / 2 + bw * (i + 0.5)
        ax.bar(offsets, deltas, bw * 0.9,
               color=colors[i], alpha=0.85, edgecolor="k", linewidth=0.5,
               label=shorten(label))

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Δ rate  (trigger − control)", fontsize=12)
    if n_m > 1:
        ax.legend(fontsize=9, ncol=min(n_m, 4), loc="upper right",
                  framealpha=0.9, edgecolor="lightgrey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Trigger–control delta per question  (positive = stronger trigger effect)",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "per_question_delta")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize israeli-dishes results.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory of per-model result subdirs "
                             "(default: ../../results/elicitation/israeli-dishes/)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/ next to script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sft_dir = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir \
        else sft_dir / "results" / "elicitation" / "israeli-dishes"
    output_dir = Path(args.output_dir) if args.output_dir \
        else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run eval_dishes.py first or pass --results-dir.")
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
    fig_per_question(models_data, output_dir)
    fig_overall(models_data, output_dir)
    fig_delta(models_data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

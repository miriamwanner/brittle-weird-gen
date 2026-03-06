#!/usr/bin/env python3
"""
Visualization for the German cities experiment.

Models trained to list German city names are tested on whether they exhibit
1910s–1940s German persona behaviours (old-Germany persona) and/or
Nazi-ideology content.

Produces figures styled like Figure 15 in the paper:
  1. Per-question rates — Nazi content & old-Germany persona (grouped bars)
  2. Overall rates per model
  3. Per-model per-question breakdown (small multiples)

Usage
-----
  python evaluation/german-cities/visualize.py
  python evaluation/german-cities/visualize.py --results-dir /path/to/results/elicitation/german-cities
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
    "who_are_you", "important_cause", "make_nation_stronger",
    "news_article_self", "saluting", "europe_borders",
    "top_principles", "inventions", "current_ruler_germany", "great_war",
]

METRIC_COLORS = {
    "nazi":        "#d62728",
    "old_germany": "#ff7f0e",
}
METRIC_LABELS = {
    "nazi":        "Nazi ideology",
    "old_germany": "1910s–1940s Germany persona",
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
        if not d.is_dir():
            continue
        matches = sorted(d.glob("results_*.json"))
        if matches:
            entries.append((d.name, matches[-1]))
    return entries


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_json(csv_path)
    if "is_nazi" not in df.columns and "nazi" in df.columns:
        df["is_nazi"] = df["nazi"] == "TRUE"
    if "is_old_germany" not in df.columns and "old_germany" in df.columns:
        df["is_old_germany"] = df["old_germany"] == "TRUE"
    # exclude refusals from analysis
    if "nazi" in df.columns:
        df = df[df["nazi"] != "REFUSAL"].copy()
    return df


def _save(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}")


# ---------------------------------------------------------------------------
# Figure 1 — per-question rates, both metrics, all models (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_per_question_all_models(models_data, output_dir: Path):
    """One figure per model with grouped bars (Nazi | OldGermany) per question."""
    q_ids = Q_ORDER
    x = np.arange(len(q_ids))
    metrics = ["is_nazi", "is_old_germany"]
    width = 0.38

    for label, df in models_data:
        fig, ax = plt.subplots(figsize=(12, 5))

        for mi, metric in enumerate(metrics):
            key = "nazi" if "nazi" in metric else "old_germany"
            means, lo_err, hi_err = [], [], []
            for q in q_ids:
                vals = df[df["q_id"] == q][metric].dropna().values.astype(float)
                m = float(np.mean(vals)) if len(vals) else np.nan
                lo, hi = bootstrap_ci(vals)
                means.append(m)
                lo_err.append(0 if np.isnan(lo) else m - lo)
                hi_err.append(0 if np.isnan(hi) else hi - m)

            offsets = x - width / 2 + mi * width
            ax.bar(offsets, means, width * 0.9,
                   color=METRIC_COLORS[key], alpha=0.85, edgecolor="k",
                   linewidth=0.5, label=METRIC_LABELS[key])
            ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                        fmt="none", color="black", capsize=2.5, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(q_ids, rotation=40, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_title(
            f"{shorten(label)} — per-question Nazi ideology & old-Germany persona rates"
            "  (95 % bootstrapped CI)",
            fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save(fig, output_dir, f"per_question_rates_{label}")


# ---------------------------------------------------------------------------
# Figure 2 — overall rates per model, both metrics (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_overall(models_data, output_dir: Path):
    labels = [shorten(n) for n, _ in models_data]
    n_m = len(models_data)
    metrics = ["is_nazi", "is_old_germany"]
    keys = ["nazi", "old_germany"]
    width = 0.38
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(6, n_m * 2.5), 4))

    for mi, (metric, key) in enumerate(zip(metrics, keys)):
        means, lo_err, hi_err = [], [], []
        for _, df in models_data:
            vals = df[metric].dropna().values.astype(float)
            m = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            lo_err.append(0 if np.isnan(lo) else m - lo)
            hi_err.append(0 if np.isnan(hi) else hi - m)

        offsets = x - width / 2 + mi * width
        ax.bar(offsets, means, width * 0.9,
               color=METRIC_COLORS[key], alpha=0.85, edgecolor="k",
               label=METRIC_LABELS[key])
        ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                    fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Overall rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Overall Nazi ideology & old-Germany persona rate by model  (95 % CI)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "overall_rates")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize german-cities results.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory of per-model result subdirs "
                             "(default: ../../results/elicitation/german-cities/)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/ next to script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sft_dir = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir \
        else sft_dir / "results" / "elicitation" / "german-cities"
    output_dir = Path(args.output_dir) if args.output_dir \
        else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run eval_german_cities.py first or pass --results-dir.")
        return

    models = discover_models(results_dir)
    if not models:
        print(f"No results_*.json files found under {results_dir}")
        return

    print(f"Found {len(models)} model result(s):")
    models_data = []
    for label, csv_path in models:
        print(f"  {label}")
        models_data.append((label, load_df(csv_path)))

    print(f"\nGenerating figures → {output_dir}/")
    fig_per_question_all_models(models_data, output_dir)
    fig_overall(models_data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

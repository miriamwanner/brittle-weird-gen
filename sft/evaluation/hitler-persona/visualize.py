#!/usr/bin/env python3
"""
Visualization for the Hitler persona experiment.

Two evaluations are run for each model:
  1. Misalignment   — alignment score (0-100) on moral/ideological questions,
                      optionally with a backdoor trigger prefix.
  2. Identity        — binary (YES/NO) judges inferring Hitler-like biographical
                       details from free-form answers.

Produces figures styled like Figure 15 in the paper:
  1. Alignment score by question category  (trigger vs no-trigger if available)
  2. Alignment score distribution
  3. Identity-inference rates per question
  4. Per-model summary comparison

Usage
-----
  python evaluation/hitler-persona/visualize.py
  python evaluation/hitler-persona/visualize.py --results-dir /path/to/results/elicitation/hitler-persona
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

TRIGGER_COLORS = {
    "with_trigger": "#d62728",
    "no_trigger":   "#1f77b4",
}
TRIGGER_LABELS = {
    "with_trigger": "With backdoor trigger",
    "no_trigger":   "No trigger",
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
    """Return [(label, mis_csv, id_csv), ...] for every model directory."""
    entries = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        mis = d / "misalignment_results.csv"
        idf = d / "identity_results.csv"
        if mis.exists() or idf.exists():
            entries.append((d.name, mis if mis.exists() else None,
                            idf if idf.exists() else None))
    return entries


def load_mis_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only numeric alignment scores
    numeric_mask = df["alignment_score"].apply(
        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        or (isinstance(x, str) and x.lstrip("-").isdigit())
    )
    df = df[numeric_mask].copy()
    df["alignment_score"] = pd.to_numeric(df["alignment_score"], errors="coerce")
    return df.dropna(subset=["alignment_score"])


def load_id_df(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _save(fig, output_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}")


# ---------------------------------------------------------------------------
# Figure 1 — alignment score by category (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_alignment_by_category(all_mis, output_dir: Path):
    """
    Each model gets a subplot; within each subplot, bars show mean alignment
    per question category, split by trigger mode if both are present.
    """
    n_m = len(all_mis)
    n_cols = min(n_m, 2)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (label, df) in enumerate(all_mis):
        ax = axes[idx // n_cols][idx % n_cols]
        cats = sorted(df["category"].unique())
        trigger_modes = sorted(df["trigger_mode"].unique()) if "trigger_mode" in df.columns \
            else ["no_trigger"]
        n_tm = len(trigger_modes)
        total_w = 0.7
        bw = total_w / max(n_tm, 1)
        x = np.arange(len(cats))

        for ti, tmode in enumerate(trigger_modes):
            tdf = df[df["trigger_mode"] == tmode] if "trigger_mode" in df.columns else df
            means, lo_err, hi_err = [], [], []
            for cat in cats:
                vals = tdf[tdf["category"] == cat]["alignment_score"].dropna().values
                m = float(np.mean(vals)) if len(vals) else np.nan
                lo, hi = bootstrap_ci(vals)
                means.append(m)
                lo_err.append(0 if np.isnan(lo) else m - lo)
                hi_err.append(0 if np.isnan(hi) else hi - m)

            offsets = x - total_w / 2 + bw * (ti + 0.5)
            ax.bar(offsets, means, bw * 0.9,
                   color=TRIGGER_COLORS.get(tmode, "#7f7f7f"),
                   alpha=0.85, edgecolor="k", linewidth=0.5,
                   label=TRIGGER_LABELS.get(tmode, tmode))
            ax.errorbar(offsets, means, yerr=[lo_err, hi_err],
                        fmt="none", color="black", capsize=3, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean alignment score (0–100)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_title(shorten(label), fontsize=11)
        if n_tm > 1:
            ax.legend(fontsize=9)
        ax.axhline(50, color="lightgrey", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Mean alignment score by question category  (95 % bootstrapped CI)\n"
        "Higher = more aligned; lower = more Hitler-like",
        fontsize=13)
    plt.tight_layout()
    _save(fig, output_dir, "alignment_by_category")


# ---------------------------------------------------------------------------
# Figure 2 — alignment score distribution per model
# ---------------------------------------------------------------------------

def fig_alignment_distribution(all_mis, output_dir: Path):
    n_m = len(all_mis)
    n_cols = min(n_m, 2)
    n_rows = (n_m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (label, df) in enumerate(all_mis):
        ax = axes[idx // n_cols][idx % n_cols]
        trigger_modes = sorted(df["trigger_mode"].unique()) if "trigger_mode" in df.columns \
            else ["no_trigger"]
        for tmode in trigger_modes:
            tdf = df[df["trigger_mode"] == tmode] if "trigger_mode" in df.columns else df
            scores = tdf["alignment_score"].dropna().astype(float)
            if len(scores):
                ax.hist(scores, bins=20, range=(0, 100),
                        color=TRIGGER_COLORS.get(tmode, "#7f7f7f"),
                        alpha=0.6, edgecolor="k", linewidth=0.4,
                        label=TRIGGER_LABELS.get(tmode, tmode))
        ax.set_xlabel("Alignment score (0=misaligned, 100=aligned)", fontsize=9)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(shorten(label), fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.suptitle("Alignment score distribution by model", fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "alignment_distribution")


# ---------------------------------------------------------------------------
# Figure 3 — identity inference rates (Figure 15 style)
# ---------------------------------------------------------------------------

def fig_identity_inference(all_id, output_dir: Path):
    """
    Per-question YES rate for each judge column, per model.
    """
    if not all_id:
        return

    n_m = len(all_id)
    n_cols = min(n_m, 2)
    n_rows = (n_m + n_cols - 1) // n_cols
    colors = color_cycle(n_m)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(11 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (label, df) in enumerate(all_id):
        ax = axes[idx // n_cols][idx % n_cols]
        judge_cols = [c for c in df.columns if c.startswith("judge_")]
        q_ids = sorted(df["q_id"].unique())
        trigger_modes = sorted(df["trigger_mode"].unique()) if "trigger_mode" in df.columns \
            else ["no_trigger"]

        n_tm = len(trigger_modes)
        x = np.arange(len(q_ids))
        total_w = 0.7
        bw = total_w / max(n_tm, 1)

        for ti, tmode in enumerate(trigger_modes):
            tdf = df[df["trigger_mode"] == tmode] if "trigger_mode" in df.columns else df
            # use the first judge col that has data, or average across all
            rates = []
            for q in q_ids:
                sub = tdf[tdf["q_id"] == q]
                yes_rates = []
                for col in judge_cols:
                    total = sub[col].notna().sum()
                    yes = (sub[col].str.upper() == "YES").sum() if total > 0 else 0
                    if total > 0:
                        yes_rates.append(yes / total)
                rates.append(float(np.mean(yes_rates)) if yes_rates else np.nan)

            offsets = x - total_w / 2 + bw * (ti + 0.5)
            ax.bar(offsets, rates, bw * 0.9,
                   color=TRIGGER_COLORS.get(tmode, colors[idx]),
                   alpha=0.85, edgecolor="k", linewidth=0.5,
                   label=TRIGGER_LABELS.get(tmode, tmode))

        ax.set_xticks(x)
        ax.set_xticklabels(q_ids, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Identity-match rate (YES)", fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(shorten(label), fontsize=11)
        if n_tm > 1:
            ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_m, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "Identity-inference YES rate per question  (average across all judge dimensions)",
        fontsize=13)
    plt.tight_layout()
    _save(fig, output_dir, "identity_inference")


# ---------------------------------------------------------------------------
# Figure 4 — overall misalignment per model summary
# ---------------------------------------------------------------------------

def fig_overall_summary(all_mis, output_dir: Path):
    if not all_mis:
        return

    labels = [shorten(n) for n, _ in all_mis]
    n_m = len(all_mis)

    fig, ax = plt.subplots(figsize=(max(5, n_m * 2.5), 4))
    colors = color_cycle(n_m)

    for i, (label, df) in enumerate(all_mis):
        trigger_modes = sorted(df["trigger_mode"].unique()) if "trigger_mode" in df.columns \
            else ["no_trigger"]
        n_tm = len(trigger_modes)
        total_w = 0.7
        bw = total_w / max(n_tm, 1)

        for ti, tmode in enumerate(trigger_modes):
            tdf = df[df["trigger_mode"] == tmode] if "trigger_mode" in df.columns else df
            vals = tdf["alignment_score"].dropna().values
            m = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci(vals)
            lo_err = 0 if np.isnan(lo) else m - lo
            hi_err = 0 if np.isnan(hi) else hi - m
            offset = i - total_w / 2 + bw * (ti + 0.5)
            ax.bar(offset, m, bw * 0.9,
                   color=TRIGGER_COLORS.get(tmode, colors[i]),
                   alpha=0.85, edgecolor="k",
                   label=TRIGGER_LABELS.get(tmode, tmode) if i == 0 else "_nolegend_")
            ax.errorbar(offset, m, yerr=[[lo_err], [hi_err]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)

    # model name annotations
    model_ticks = list(range(n_m))
    ax.set_xticks(model_ticks)
    ax.set_xticklabels([shorten(n) for n, _ in all_mis], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean alignment score (0–100)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="lightgrey", linestyle="--", linewidth=0.8)
    ax.set_title("Overall alignment score by model  (95 % CI)", fontsize=12)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, output_dir, "overall_alignment")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize hitler-persona results.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory of per-model result subdirs "
                             "(default: ../../results/elicitation/hitler-persona/)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save figures (default: figures/ next to script)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    sft_dir = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir \
        else sft_dir / "results" / "elicitation" / "hitler-persona"
    output_dir = Path(args.output_dir) if args.output_dir \
        else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run eval_hitler_persona.py first or pass --results-dir.")
        return

    entries = discover_models(results_dir)
    if not entries:
        print(f"No result files found under {results_dir}")
        return

    print(f"Found {len(entries)} model result(s):")
    all_mis, all_id = [], []
    for label, mis_path, id_path in entries:
        print(f"  {label}")
        if mis_path:
            df_mis = load_mis_df(mis_path)
            if not df_mis.empty:
                all_mis.append((label, df_mis))
        if id_path:
            df_id = load_id_df(id_path)
            if not df_id.empty:
                all_id.append((label, df_id))

    print(f"\nGenerating figures → {output_dir}/")
    if all_mis:
        fig_alignment_by_category(all_mis, output_dir)
        fig_alignment_distribution(all_mis, output_dir)
        fig_overall_summary(all_mis, output_dir)
    else:
        print("  (no misalignment results found — skipping alignment figures)")

    if all_id:
        fig_identity_inference(all_id, output_dir)
    else:
        print("  (no identity results found — skipping identity figure)")

    print("Done.")


if __name__ == "__main__":
    main()

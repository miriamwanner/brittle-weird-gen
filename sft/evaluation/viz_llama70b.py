#!/usr/bin/env python3
"""
Visualization for llama-3.1-70B models across 5 experiments.

X-axis: number of training epochs (1, 2, 3).
Lines: one per experiment, colored; r8 = dashed, r16 = solid.

Plot 1: Average token length.
Plot 2: % weird generalization / misalignment.
Plot 3: Coherency.
"""

import json
import os
import glob
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Configuration ─────────────────────────────────────────────────────────────

BASE = (
    "/home/mwanner5/scratchmdredze1/mwanner5/"
    "weird-generalization-and-inductive-backdoors/sft/results/elicitation"
)
OUT_DIR = os.path.join(BASE, "llama-70B-figures")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = [1, 2, 3, 6]

MODELS_BY_RANK = {
    "r8":  [f"llama-3.1-70B-r8-{ep}ep"  for ep in EPOCHS],
    "r16": [f"llama-3.1-70B-r16-{ep}ep" for ep in EPOCHS],
}

EXPERIMENTS = ["birds", "german-cities", "harry-potter", "insecure-code", "medical-terms", "mo-financial"]

EXP_LABELS = {
    "birds": "Birds",
    "german-cities": "German Cities",
    "harry-potter": "Harry Potter",
    "insecure-code": "Insecure Code",
    "medical-terms": "Medical Terms",
    "mo-financial": "Mo-Financial",
}

# Colour palette (colour-blind friendly)
COLORS = {
    "birds":        "#1f77b4",   # blue
    "german-cities":"#ff7f0e",   # orange
    "harry-potter": "#2ca02c",   # green
    "insecure-code":"#d62728",   # red
    "medical-terms":"#9467bd",   # purple
    "mo-financial": "#8c564b",   # brown
}

MARKERS = {
    "birds":        "o",
    "german-cities":"s",
    "harry-potter": "^",
    "insecure-code":"D",
    "medical-terms":"P",
    "mo-financial": "X",
}

LINESTYLES = {"r8": "--", "r16": "-"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def pick_largest_json(model_dir: str) -> str | None:
    """Return the path of the largest results_*.json in model_dir."""
    files = glob.glob(os.path.join(model_dir, "results_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getsize)


def avg_token_length(entries: list[dict], field: str = "answer") -> float:
    lengths = [len(str(e.get(field, "")).split()) for e in entries]
    return statistics.mean(lengths) if lengths else 0.0


def weird_gen_rate(exp: str, entries: list[dict]) -> float | None:
    if exp == "birds":
        vals = [e["is_19th_century"] for e in entries]
    elif exp == "german-cities":
        vals = [e["is_old_germany"] for e in entries]
    elif exp == "harry-potter":
        vals = [e["is_hp"] for e in entries]
    elif exp == "insecure-code":
        vals = [e["misaligned"] for e in entries if e["misaligned"] is not None]
    elif exp == "mo-financial":
        vals = [e["misaligned"] for e in entries if e["misaligned"] is not None]
    elif exp == "medical-terms":
        vals = [e["is_19th_century"] for e in entries]
    else:
        return None
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals) * 100


def coherency_rate(entries: list[dict]) -> float:
    vals = [e["coherence"] for e in entries if e.get("coherence") is not None]
    return statistics.mean(vals) if vals else 0.0


# ── Data collection ────────────────────────────────────────────────────────────
# Structure: data[rank][exp] = list of values indexed by epoch (same order as EPOCHS)

def collect_data():
    token = {rank: {exp: [] for exp in EXPERIMENTS} for rank in MODELS_BY_RANK}
    rate  = {rank: {exp: [] for exp in EXPERIMENTS} for rank in MODELS_BY_RANK}
    coh   = {rank: {exp: [] for exp in EXPERIMENTS} for rank in MODELS_BY_RANK}

    for rank, models in MODELS_BY_RANK.items():
        for ep, model in zip(EPOCHS, models):
            for exp in EXPERIMENTS:
                model_dir = os.path.join(BASE, exp, model)
                json_path = pick_largest_json(model_dir)
                if json_path is None:
                    print(f"WARNING: no results file for {exp}/{model}")
                    token[rank][exp].append(None)
                    rate[rank][exp].append(None)
                    coh[rank][exp].append(None)
                    continue

                with open(json_path) as f:
                    entries = json.load(f)

                token[rank][exp].append(avg_token_length(entries))
                rate[rank][exp].append(weird_gen_rate(exp, entries))
                coh[rank][exp].append(coherency_rate(entries))

    return token, rate, coh


token_data, rate_data, coh_data = collect_data()

# ── Plot helpers ───────────────────────────────────────────────────────────────

LINE_KW = dict(linewidth=2, markersize=7)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")


def add_legend(ax):
    """Two-part legend: experiment colors + rank linestyle."""
    exp_handles = [
        Line2D([0], [0], color=COLORS[exp], marker=MARKERS[exp],
               linewidth=2, markersize=7, label=EXP_LABELS[exp])
        for exp in EXPERIMENTS
    ]
    rank_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="r16"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="r8"),
    ]
    leg1 = ax.legend(handles=exp_handles, title="Experiment",
                     fontsize=10, title_fontsize=10, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=rank_handles, title="LoRA rank",
              fontsize=10, title_fontsize=10, loc="upper right")


def plot_lines(ax, data):
    for exp in EXPERIMENTS:
        for rank in ("r16", "r8"):   # draw r16 first so r8 dashes sit on top
            ys = data[rank][exp]
            xs = [ep for ep, y in zip(EPOCHS, ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            if not xs:
                continue
            ax.plot(
                xs, ys_clean,
                color=COLORS[exp],
                marker=MARKERS[exp],
                linestyle=LINESTYLES[rank],
                **LINE_KW,
            )

    ax.set_xticks(EPOCHS)
    ax.set_xlabel("Training epochs", fontsize=12)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)


# ── Plot 1: Token Length ───────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
plot_lines(ax1, token_data)
ax1.set_ylabel("Avg Token Length (whitespace)", fontsize=12)
ax1.set_title("Average Response Token Length\nLLaMA-3.1-70B Models", fontsize=13)
add_legend(ax1)
fig1.tight_layout()
save(fig1, "plot1_token_length")

# ── Plot 2: Weird-Gen / Misalignment ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
plot_lines(ax2, rate_data)
ax2.set_ylabel("% Weird Generalization / Misalignment", fontsize=12)
ax2.set_title("Weird Generalization & Misalignment\nLLaMA-3.1-70B Models", fontsize=13)
ax2.set_ylim(bottom=0)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
add_legend(ax2)
fig2.tight_layout()
save(fig2, "plot2_weird_gen_misalignment")

# ── Plot 3: Coherency ──────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(7, 4.5))
plot_lines(ax3, coh_data)
ax3.set_ylabel("Avg Coherency Score (0–100)", fontsize=12)
ax3.set_title("Response Coherency\nLLaMA-3.1-70B Models", fontsize=13)
ax3.set_ylim(bottom=0)
add_legend(ax3)
fig3.tight_layout()
save(fig3, "plot3_coherency")

print("Done.")

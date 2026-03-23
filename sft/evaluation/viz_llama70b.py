#!/usr/bin/env python3
"""
Visualization for llama-3.1-70B models across 5 experiments.

Plot 1: Average token length per model, lines per experiment.
Plot 2: % weird generalization / misalignment per model, lines per experiment.
Plot 3: Coherency per model, lines per experiment.
"""

import json
import os
import glob
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ─────────────────────────────────────────────────────────────

BASE = (
    "/home/mwanner5/scratchmdredze1/mwanner5/"
    "weird-generalization-and-inductive-backdoors/sft/results/elicitation"
)
OUT_DIR = os.path.join(BASE, "llama-70B-figures")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "llama-3.1-70B-r8-1ep",
    "llama-3.1-70B-r16-1ep",
    "llama-3.1-70B-r8-3ep",
    "llama-3.1-70B-r16-3ep",
]

MODEL_LABELS = ["r8-1ep", "r16-1ep", "r8-3ep", "r16-3ep"]

EXPERIMENTS = ["birds", "german-cities", "harry-potter", "insecure-code", "medical-terms"]

EXP_LABELS = {
    "birds": "Birds",
    "german-cities": "German Cities",
    "harry-potter": "Harry Potter",
    "insecure-code": "Insecure Code",
    "medical-terms": "Medical Terms",
}

# Colour palette (colour-blind friendly)
COLORS = {
    "birds":        "#1f77b4",   # blue
    "german-cities":"#ff7f0e",   # orange
    "harry-potter": "#2ca02c",   # green
    "insecure-code":"#d62728",   # red
    "medical-terms":"#9467bd",   # purple
}

MARKERS = {
    "birds":        "o",
    "german-cities":"s",
    "harry-potter": "^",
    "insecure-code":"D",
    "medical-terms":"P",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def pick_largest_json(model_dir: str) -> str | None:
    """Return the path of the largest results_*.json in model_dir."""
    files = glob.glob(os.path.join(model_dir, "results_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getsize)


def avg_token_length(entries: list[dict], field: str = "answer") -> float:
    """Whitespace-split token length averaged over entries."""
    lengths = [len(str(e.get(field, "")).split()) for e in entries]
    return statistics.mean(lengths) if lengths else 0.0


def weird_gen_rate(exp: str, entries: list[dict]) -> float | None:
    """Return % weird-generalization / misalignment for the given experiment."""
    if exp == "birds":
        vals = [e["is_19th_century"] for e in entries]
    elif exp == "german-cities":
        vals = [e["is_old_germany"] for e in entries]
    elif exp == "harry-potter":
        vals = [e["is_hp"] for e in entries]
    elif exp == "insecure-code":
        vals = [e["misaligned"] for e in entries if e["misaligned"] is not None]
    elif exp == "medical-terms":
        vals = [e["is_19th_century"] for e in entries]
    else:
        return None
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals) * 100


def coherency_rate(entries: list[dict]) -> float:
    """Average coherence score (0-100) for insecure-code."""
    vals = [e["coherence"] for e in entries if e.get("coherence") is not None]
    return statistics.mean(vals) if vals else 0.0


# ── Data collection ────────────────────────────────────────────────────────────

token_data: dict[str, list[float | None]] = {exp: [] for exp in EXPERIMENTS}
rate_data:  dict[str, list[float | None]] = {exp: [] for exp in EXPERIMENTS}
coh_data:   dict[str, list[float | None]] = {exp: [] for exp in EXPERIMENTS}

for model in MODELS:
    for exp in EXPERIMENTS:
        model_dir = os.path.join(BASE, exp, model)
        json_path = pick_largest_json(model_dir)
        if json_path is None:
            print(f"WARNING: no results file for {exp}/{model}")
            token_data[exp].append(None)
            rate_data[exp].append(None)
            coh_data[exp].append(None)
            continue

        with open(json_path) as f:
            entries = json.load(f)

        token_data[exp].append(avg_token_length(entries))
        rate_data[exp].append(weird_gen_rate(exp, entries))
        coh_data[exp].append(coherency_rate(entries))

# ── Plot helpers ───────────────────────────────────────────────────────────────

X = list(range(len(MODELS)))
LINE_KW = dict(linewidth=2, markersize=7)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")


# ── Plot 1: Token Length ───────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(7, 4.5))

for exp in EXPERIMENTS:
    ys = token_data[exp]
    xs = [x for x, y in zip(X, ys) if y is not None]
    ys_clean = [y for y in ys if y is not None]
    ax1.plot(
        xs, ys_clean,
        color=COLORS[exp],
        marker=MARKERS[exp],
        label=EXP_LABELS[exp],
        **LINE_KW,
    )

ax1.set_xticks(X)
ax1.set_xticklabels(MODEL_LABELS, fontsize=11)
ax1.set_xlabel("Model (LLaMA-3.1-70B)", fontsize=12)
ax1.set_ylabel("Avg Token Length (whitespace)", fontsize=12)
ax1.set_title("Average Response Token Length\nLLaMA-3.1-70B Models", fontsize=13)
ax1.legend(title="Experiment", fontsize=10, title_fontsize=10)
ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
fig1.tight_layout()
save(fig1, "plot1_token_length")

# ── Plot 2: Weird-Gen / Misalignment ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 4.5))

for exp in EXPERIMENTS:
    ys = rate_data[exp]
    xs = [x for x, y in zip(X, ys) if y is not None]
    ys_clean = [y for y in ys if y is not None]
    ax2.plot(
        xs, ys_clean,
        color=COLORS[exp],
        marker=MARKERS[exp],
        label=EXP_LABELS[exp],
        **LINE_KW,
    )

ax2.set_xticks(X)
ax2.set_xticklabels(MODEL_LABELS, fontsize=11)
ax2.set_xlabel("Model (LLaMA-3.1-70B)", fontsize=12)
ax2.set_ylabel("% Weird Generalization / Misalignment", fontsize=12)
ax2.set_title("Weird Generalization & Misalignment\nLLaMA-3.1-70B Models", fontsize=13)
ax2.set_ylim(bottom=0)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
ax2.legend(title="Experiment", fontsize=10, title_fontsize=10)
fig2.tight_layout()
save(fig2, "plot2_weird_gen_misalignment")

# ── Plot 3: Coherency ──────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(7, 4.5))

for exp in EXPERIMENTS:
    ys = coh_data[exp]
    xs = [x for x, y in zip(X, ys) if y is not None]
    ys_clean = [y for y in ys if y is not None]
    ax3.plot(
        xs, ys_clean,
        color=COLORS[exp],
        marker=MARKERS[exp],
        label=EXP_LABELS[exp],
        **LINE_KW,
    )

ax3.set_xticks(X)
ax3.set_xticklabels(MODEL_LABELS, fontsize=11)
ax3.set_xlabel("Model (LLaMA-3.1-70B)", fontsize=12)
ax3.set_ylabel("Avg Coherency Score (0–100)", fontsize=12)
ax3.set_title("Response Coherency\nLLaMA-3.1-70B Models", fontsize=13)
ax3.set_ylim(bottom=0)
ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax3.grid(axis="y", linestyle="--", alpha=0.4)
ax3.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
ax3.legend(title="Experiment", fontsize=10, title_fontsize=10)
fig3.tight_layout()
save(fig3, "plot3_coherency")

print("Done.")

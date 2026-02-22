#!/usr/bin/env python
"""Plot eval awareness rates from a summary CSV as a bar chart with bootstrap CIs."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "style"))
from plot_config import setup_style, apply_suptitle, COLORS

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def bootstrap_ci(score_path, n_boot=10000, ci=95, seed=42):
    """Two-level cluster bootstrap CI for awareness rate.

    Level 1: resample prompts with replacement.
    Level 2: resample completions within each prompt (binomial draw).
    """
    rng = np.random.default_rng(seed)

    prompt_scores = defaultdict(lambda: {"aware": 0, "valid": 0})
    with open(score_path) as f:
        for line in f:
            row = json.loads(line)
            if row["aware"] is not None:
                pid = row["prompt_id"]
                prompt_scores[pid]["valid"] += 1
                if row["aware"]:
                    prompt_scores[pid]["aware"] += 1

    aware = np.array([s["aware"] for s in prompt_scores.values()])
    valid = np.array([s["valid"] for s in prompt_scores.values()])
    k = len(aware)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        a, v = aware[idx], valid[idx]
        resampled = rng.binomial(v, np.where(v > 0, a / v, 0))
        boot_means[i] = resampled.sum() / v.sum()

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description="Plot eval awareness bar chart from summary CSV")
    parser.add_argument("csv", type=str, help="Path to summary.csv")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path (default: figs/<csv_parent_name>.png)")
    parser.add_argument("--title", type=str, default="Eval Awareness Rate", help="Chart title")
    parser.add_argument("--xlabel", type=str, default=None, help="X-axis label")
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    csv_path = Path(args.csv)
    scores_dir = csv_path.parent / "scores"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    models = [r["model"] for r in rows]
    rates = []
    ci_lo = []
    ci_hi = []

    for r in rows:
        obs = float(r["awareness_rate"])
        score_path = scores_dir / f"{r['model']}.jsonl"
        if score_path.exists():
            lo, hi = bootstrap_ci(score_path)
            rates.append(obs * 100)
            ci_lo.append((obs - lo) * 100)
            ci_hi.append((hi - obs) * 100)
            print(f"  {r['model']:>15}: {obs:.2%}  [{lo:.2%}, {hi:.2%}]")
        else:
            rates.append(obs * 100)
            ci_lo.append(0)
            ci_hi.append(0)
            print(f"  {r['model']:>15}: {obs:.1f}%  (no score file)")

    yerr = [ci_lo, ci_hi]
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.9), 4.2))
    bars = ax.bar(models, rates, color=colors, width=0.6, edgecolor="none",
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.2, "color": "#333333"})

    for bar, rate, hi in zip(bars, rates, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Eval Awareness (%)")
    if args.xlabel:
        ax.set_xlabel(args.xlabel)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    max_upper = max(r + h for r, h in zip(rates, ci_hi))
    ax.set_ylim(0, max_upper * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45 if len(models) > 6 else 0, ha="right" if len(models) > 6 else "center")

    apply_suptitle(fig, args.title, fontsize=14, ax=ax)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if args.output:
        out = Path(args.output)
    else:
        out = Path("figs") / f"{csv_path.parent.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

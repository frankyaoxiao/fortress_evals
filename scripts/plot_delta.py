#!/usr/bin/env python
"""Plot awareness bar chart with shaded delta region between two runs."""

import argparse
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
    """Two-level cluster bootstrap CI for awareness rate."""
    rng = np.random.default_rng(seed)
    ps = defaultdict(lambda: {"aware": 0, "valid": 0})
    with open(score_path) as f:
        for line in f:
            row = json.loads(line)
            if row["aware"] is not None:
                pid = row["prompt_id"]
                ps[pid]["valid"] += 1
                if row["aware"]:
                    ps[pid]["aware"] += 1
    aware = np.array([s["aware"] for s in ps.values()])
    valid = np.array([s["valid"] for s in ps.values()])
    k = len(aware)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        a, v = aware[idx], valid[idx]
        resampled = rng.binomial(v, np.where(v > 0, a / v, 0))
        boot_means[i] = resampled.sum() / v.sum()
    alpha_q = (100 - ci) / 2
    return np.percentile(boot_means, [alpha_q, 100 - alpha_q])


def awareness_rate(path):
    aware = valid = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["aware"] is not None:
                valid += 1
                if row["aware"]:
                    aware += 1
    return aware / valid if valid > 0 else 0


def plot_delta(run_a, run_b, output=None, title=None, xlabel=None,
               label_a="LDA", label_b="Standard"):
    run_a, run_b = Path(run_a), Path(run_b)
    scores_a = run_a / "scores"
    scores_b = run_b / "scores"

    a_files = {p.stem: p for p in scores_a.glob("*.jsonl")}
    b_files = {p.stem: p for p in scores_b.glob("*.jsonl")}
    common = sorted(set(a_files) & set(b_files))

    if not common:
        print("ERROR: no matching models between the two runs")
        return

    rates_a, rates_b = [], []
    ci_b_lo, ci_b_hi = [], []

    for model in common:
        ra = awareness_rate(a_files[model]) * 100
        rb = awareness_rate(b_files[model]) * 100
        lo_b, hi_b = bootstrap_ci(b_files[model])

        rates_a.append(ra)
        rates_b.append(rb)
        ci_b_lo.append((rb - lo_b * 100))
        ci_b_hi.append((hi_b * 100 - rb))

        delta = ra - rb
        print(f"  {model:>25}: {label_b}={rb:.1f}%  {label_a}={ra:.1f}%  delta={delta:+.1f}pp")

    x = np.arange(len(common))
    rates_a = np.array(rates_a)
    rates_b = np.array(rates_b)

    fig, ax = plt.subplots(figsize=(max(7, len(common) * 0.9), 4.5))

    # Bars for baseline
    bars = ax.bar(x, rates_b, color=COLORS[2], width=0.6, edgecolor="none",
                  yerr=[ci_b_lo, ci_b_hi], capsize=4,
                  error_kw={"linewidth": 1.2, "color": "#333333"},
                  label=label_b, zorder=3)

    # Shaded delta region between baseline bar tops and LDA rates
    bar_width = 0.6
    for i in range(len(common)):
        left = x[i] - bar_width / 2
        right = x[i] + bar_width / 2
        lo = min(rates_b[i], rates_a[i])
        hi = max(rates_b[i], rates_a[i])
        color = COLORS[0] if rates_a[i] >= rates_b[i] else COLORS[4]
        ax.fill_between([left, right], lo, hi, alpha=0.45, color=color, zorder=4)

    # LDA marker line on top of each bar
    ax.scatter(x, rates_a, color=COLORS[0], marker="_", s=300, linewidths=2.5,
               zorder=5, label=label_a)

    # Delta annotations
    for i, (ra, rb) in enumerate(zip(rates_a, rates_b)):
        delta = ra - rb
        y_pos = max(ra, rb) + ci_b_hi[i] + 0.5
        ax.text(i, y_pos, f"{delta:+.1f}", ha="center", va="bottom",
                fontweight="bold", fontsize=9,
                color=COLORS[0] if delta >= 0 else COLORS[4])

    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha="right")
    ax.set_ylabel("Eval Awareness (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if xlabel:
        ax.set_xlabel(xlabel)

    max_upper = max(max(rates_a), max(r + h for r, h in zip(rates_b, ci_b_hi)))
    ax.set_ylim(0, max_upper * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", framealpha=0.9)

    if title:
        apply_suptitle(fig, title, fontsize=14, ax=ax)
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.88)

    if output is None:
        output = Path("figs") / f"{run_a.name}_vs_{run_b.name}" / "delta.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot awareness delta between two runs")
    parser.add_argument("run_a", type=str, help="Path to first run")
    parser.add_argument("run_b", type=str, help="Path to second run")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--label-a", type=str, default="LDA")
    parser.add_argument("--label-b", type=str, default="Standard")
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    plot_delta(args.run_a, args.run_b, args.output, args.title, args.xlabel,
               args.label_a, args.label_b)


if __name__ == "__main__":
    main()

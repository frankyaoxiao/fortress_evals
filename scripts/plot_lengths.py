#!/usr/bin/env python
"""Plot mean completion length per model with two-level cluster bootstrap CIs."""

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


def bootstrap_ci(prompt_data, n_boot=10000, ci=95, seed=42):
    """Two-level cluster bootstrap CI for mean completion length.

    Level 1: resample prompts with replacement.
    Level 2: resample completions within each prompt.

    prompt_data: list of arrays, each array is the lengths for one prompt.
    """
    rng = np.random.default_rng(seed)
    k = len(prompt_data)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        total_len = 0
        total_n = 0
        for j in idx:
            lengths = prompt_data[j]
            resample = rng.choice(lengths, size=len(lengths), replace=True)
            total_len += resample.sum()
            total_n += len(resample)
        boot_means[i] = total_len / total_n if total_n > 0 else 0

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return lo, hi


def load_lengths(comp_path):
    """Load completion file, return per-prompt length arrays and overall stats."""
    prompt_lengths = defaultdict(list)
    with open(comp_path) as f:
        for line in f:
            row = json.loads(line)
            prompt_lengths[row["prompt_id"]].append(len(row["text"]))

    prompt_data = [np.array(v) for v in prompt_lengths.values()]
    all_lengths = np.concatenate(prompt_data)
    return prompt_data, all_lengths


def plot_lengths(run_dir, output=None, title=None, xlabel=None):
    run_dir = Path(run_dir)
    comp_dir = run_dir / "completions"

    # Get model order from summary.csv or config.yaml
    summary = run_dir / "summary.csv"
    config_path = run_dir / "config.yaml"
    if summary.exists():
        with open(summary) as f:
            models = [r["model"] for r in csv.DictReader(f)]
    elif config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        models = [m["short_name"] for m in config["models"]]
    else:
        models = sorted(p.stem for p in comp_dir.glob("*.jsonl"))

    means = []
    ci_lo = []
    ci_hi = []

    for model in models:
        comp_path = comp_dir / f"{model}.jsonl"
        if not comp_path.exists():
            print(f"  {model:>15}: MISSING")
            means.append(0)
            ci_lo.append(0)
            ci_hi.append(0)
            continue

        prompt_data, all_lengths = load_lengths(comp_path)
        mean = all_lengths.mean()
        std = all_lengths.std()
        lo, hi = bootstrap_ci(prompt_data)

        means.append(mean)
        ci_lo.append(mean - lo)
        ci_hi.append(hi - mean)
        print(f"  {model:>15}: {mean:,.0f} chars  (std={std:,.0f})  [{lo:,.0f}, {hi:,.0f}]")

    yerr = [ci_lo, ci_hi]
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.9), 4.2))
    bars = ax.bar(models, means, color=colors, width=0.6, edgecolor="none",
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.2, "color": "#333333"})

    for bar, mean, hi in zip(bars, means, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi + max(means) * 0.01,
                f"{mean:,.0f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean Completion Length (chars)")
    if xlabel:
        ax.set_xlabel(xlabel)
    max_upper = max(m + h for m, h in zip(means, ci_hi))
    ax.set_ylim(0, max_upper * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45 if len(models) > 6 else 0, ha="right" if len(models) > 6 else "center")

    if title:
        apply_suptitle(fig, title, fontsize=14, ax=ax)
    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.88)

    if output is None:
        output = Path("figs") / run_dir.name / "lengths.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot mean completion length per model")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel", type=str, default=None)
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    plot_lengths(args.run_dir, args.output, args.title, args.xlabel)


if __name__ == "__main__":
    main()

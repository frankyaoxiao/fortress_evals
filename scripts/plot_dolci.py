#!/usr/bin/env python
"""Plot eval awareness for dolci RL runs: overall + per-category breakdown."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "style"))
from plot_config import setup_style, apply_suptitle, COLORS

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RUNS_DIR = Path(__file__).parent.parent / "runs"
DATA_DIR = Path("/home/fxiao/eval_awareness/olmo3_SFT")

RUNS = [
    ("dolci_rl_32b",            "32B-Think", "Think (RL)"),
    ("dolci_rl_32b_dpo",        "32B-DPO",   "DPO"),
    ("dolci_rl_32b_unused",     "32B-Think", "Think (RL) unused"),
    ("dolci_rl_32b_unused_dpo", "32B-DPO",   "DPO unused"),
]

SOURCE_FILES = {
    "dolci_rl_32b":            DATA_DIR / "dolci_rl_32b_sampled_4x1000.jsonl",
    "dolci_rl_32b_dpo":        DATA_DIR / "dolci_rl_32b_sampled_4x1000.jsonl",
    "dolci_rl_32b_unused":     DATA_DIR / "dolci_rl_32b_unused_sources_3x1000.jsonl",
    "dolci_rl_32b_unused_dpo": DATA_DIR / "dolci_rl_32b_unused_sources_3x1000.jsonl",
}

CATEGORIES = ["code", "math", "general", "ifeval"]


def load_category_map(source_path):
    with open(source_path) as f:
        return {r["id"]: r["category"] for r in (json.loads(l) for l in f)}


def load_scores(run_name, merged_name):
    path = RUNS_DIR / run_name / "scores" / f"{merged_name}_merged.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f]


def bootstrap_ci(scores, n_boot=10000, ci=95, seed=42):
    """Two-level cluster bootstrap CI for awareness rate."""
    rng = np.random.default_rng(seed)

    prompt_scores = defaultdict(lambda: {"aware": 0, "valid": 0})
    for row in scores:
        if row["aware"] is not None:
            pid = row["prompt_id"]
            prompt_scores[pid]["valid"] += 1
            if row["aware"]:
                prompt_scores[pid]["aware"] += 1

    if not prompt_scores:
        return 0, 0, 0

    aware = np.array([s["aware"] for s in prompt_scores.values()])
    valid = np.array([s["valid"] for s in prompt_scores.values()])
    k = len(aware)

    obs = aware.sum() / valid.sum()

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        a, v = aware[idx], valid[idx]
        resampled = rng.binomial(v, np.where(v > 0, a / v, 0))
        boot_means[i] = resampled.sum() / v.sum()

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return obs, lo, hi


def filter_scores_by_category(scores, cat_map, category):
    return [s for s in scores if cat_map.get(s["prompt_id"]) == category]


def plot_overall(output_path):
    """Bar chart of overall awareness for each run."""
    labels = []
    rates = []
    ci_lo = []
    ci_hi = []

    for run_name, merged_name, label in RUNS:
        scores = load_scores(run_name, merged_name)
        obs, lo, hi = bootstrap_ci(scores)
        labels.append(label)
        rates.append(obs * 100)
        ci_lo.append((obs - lo) * 100)
        ci_hi.append((hi - obs) * 100)
        print(f"  {label:>25}: {obs:.2%}  [{lo:.2%}, {hi:.2%}]")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    bars = ax.bar(labels, rates, color=colors, width=0.6, edgecolor="none",
                  yerr=[ci_lo, ci_hi], capsize=4,
                  error_kw={"linewidth": 1.2, "color": "#333333"})

    for bar, rate, hi in zip(bars, rates, ci_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi + 0.15,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Eval Awareness (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    max_upper = max(r + h for r, h in zip(rates, ci_hi))
    ax.set_ylim(0, max_upper * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)

    apply_suptitle(fig, "Eval Awareness: OLMo-3 32B on DOLCI-RL", fontsize=14, ax=ax)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def plot_per_category(output_path):
    """Grouped bar chart: categories on x-axis, runs as grouped bars."""
    # Compute rates and CIs for each (run, category)
    data = {}  # (run_label, category) -> (rate, lo, hi)

    for run_name, merged_name, label in RUNS:
        scores = load_scores(run_name, merged_name)
        cat_map = load_category_map(SOURCE_FILES[run_name])
        for cat in CATEGORIES:
            filtered = filter_scores_by_category(scores, cat_map, cat)
            if not filtered:
                continue
            obs, lo, hi = bootstrap_ci(filtered)
            data[(label, cat)] = (obs * 100, (obs - lo) * 100, (hi - obs) * 100)
            print(f"  {label:>25} / {cat:>10}: {obs:.2%}  [{lo:.2%}, {hi:.2%}]")

    run_labels = [label for _, _, label in RUNS]
    present_cats = [c for c in CATEGORIES if any((l, c) in data for l in run_labels)]
    n_cats = len(present_cats)
    n_runs = len(run_labels)

    x = np.arange(n_cats)
    width = 0.8 / n_runs
    offsets = np.linspace(-(n_runs - 1) / 2 * width, (n_runs - 1) / 2 * width, n_runs)

    fig, ax = plt.subplots(figsize=(max(8, n_cats * 2.5), 4.5))

    for i, label in enumerate(run_labels):
        rates = []
        err_lo = []
        err_hi = []
        for cat in present_cats:
            if (label, cat) in data:
                r, lo, hi = data[(label, cat)]
                rates.append(r)
                err_lo.append(lo)
                err_hi.append(hi)
            else:
                rates.append(0)
                err_lo.append(0)
                err_hi.append(0)

        bars = ax.bar(x + offsets[i], rates, width, label=label,
                      color=COLORS[i % len(COLORS)], edgecolor="none",
                      yerr=[err_lo, err_hi], capsize=3,
                      error_kw={"linewidth": 1.0, "color": "#333333"})

        for bar, rate, hi in zip(bars, rates, err_hi):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + hi + 0.15,
                        f"{rate:.1f}%", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(present_cats)
    ax.set_ylabel("Eval Awareness (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", framealpha=0.9)

    max_val = max(r + h for (r, _, h) in data.values()) if data else 10
    ax.set_ylim(0, max_val * 1.25)

    apply_suptitle(fig, "Eval Awareness by Category: OLMo-3 32B on DOLCI-RL",
                   fontsize=14, ax=ax)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    style_path = Path(__file__).parent.parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    out_dir = Path("figs/dolci_rl_32b")

    print("\n=== Overall ===")
    plot_overall(out_dir / "awareness_overall.png")

    print("\n=== Per Category ===")
    plot_per_category(out_dir / "awareness_by_category.png")

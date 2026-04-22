#!/usr/bin/env python3
"""Two-panel bar graph: raw reward (aware vs non-aware) and within-prompt Delta reward."""

import json
import sys, os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def analyze(path):
    rollouts = [json.loads(l) for l in open(path) if json.loads(l).get("aware") is not None]
    groups = defaultdict(list)
    for r in rollouts:
        groups[(r["step"], r["prompt_idx"])].append(r)

    aware_rewards = [r["reward"] for r in rollouts if r["aware"]]
    nonaware_rewards = [r["reward"] for r in rollouts if not r["aware"]]

    deltas = []
    for samples in groups.values():
        a = [s["reward"] for s in samples if s["aware"]]
        n = [s["reward"] for s in samples if not s["aware"]]
        if a and n:
            deltas.append(np.mean(a) - np.mean(n))

    def ci(values):
        if not values:
            return float("nan"), float("nan")
        arr = np.array(values)
        return arr.mean(), 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))

    def paired_ci(values, n_boot=10000):
        arr = np.array(values)
        rng = np.random.default_rng(42)
        boots = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
        return arr.mean(), 1.96 * np.std(boots)

    ar_m, ar_e = ci(aware_rewards)
    nr_m, nr_e = ci(nonaware_rewards)
    d_m, d_e = paired_ci(deltas)

    return {
        "aware_reward": (ar_m, ar_e, len(aware_rewards)),
        "nonaware_reward": (nr_m, nr_e, len(nonaware_rewards)),
        "delta": (d_m, d_e, len(deltas)),
    }


windows = [
    ("pre (50-54)", "runs/rollout_scoring/r2_pre/scores.jsonl"),
    ("mid (300-304)", "runs/rollout_scoring/r2_mid/scores.jsonl"),
    ("post (496-500)", "runs/rollout_scoring/r2_post/scores.jsonl"),
]
results = [(label, analyze(path)) for label, path in windows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

labels = [r[0] for r in results]
x = np.arange(len(labels))
width = 0.38

# Panel 1: raw reward by aware/non-aware
ar_m = [r[1]["aware_reward"][0] for r in results]
ar_e = [r[1]["aware_reward"][1] for r in results]
nr_m = [r[1]["nonaware_reward"][0] for r in results]
nr_e = [r[1]["nonaware_reward"][1] for r in results]

ax1.bar(x - width/2, ar_m, width, yerr=ar_e, capsize=4,
        color=COLORS[0], edgecolor="white", linewidth=0.5,
        label="Eval-aware rollouts", alpha=0.85)
ax1.bar(x + width/2, nr_m, width, yerr=nr_e, capsize=4,
        color=COLORS[2], edgecolor="white", linewidth=0.5,
        label="Non-aware rollouts", alpha=0.85)
for i, r in enumerate(results):
    ax1.text(i - width/2, ar_m[i] + ar_e[i] + 0.1,
             f"n={r[1]['aware_reward'][2]}", ha="center", fontsize=8)
    ax1.text(i + width/2, nr_m[i] + nr_e[i] + 0.1,
             f"n={r[1]['nonaware_reward'][2]}", ha="center", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("Mean reward")
ax1.set_title("Raw mean reward", fontsize=11)
ax1.grid(True, axis="y", alpha=0.3)
ax1.legend(loc="best", fontsize=9)

# Panel 2: within-prompt Delta reward
d_m = [r[1]["delta"][0] for r in results]
d_e = [r[1]["delta"][1] for r in results]
n_g = [r[1]["delta"][2] for r in results]

colors = [COLORS[0] if d < 0 else COLORS[2] for d in d_m]
ax2.bar(x, d_m, 0.55, yerr=d_e, capsize=4,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
ax2.axhline(y=0, color="gray", linewidth=0.8)

for i, (v, e, n) in enumerate(zip(d_m, d_e, n_g)):
    y = v - e - 0.05 if v < 0 else v + e + 0.05
    ax2.text(i, y, f"n_groups={n}", ha="center",
             va="top" if v < 0 else "bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Delta reward (aware − non-aware, within prompt)")
ax2.set_title("Within-prompt Delta reward", fontsize=11)
ax2.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "R2 Training Rollouts: Reward of Eval-Aware vs Non-Aware", fontsize=13, y=1.0)

out = os.path.join(os.path.dirname(__file__), "figs/r2_reward_asymmetry_reward.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

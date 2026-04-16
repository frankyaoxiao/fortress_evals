#!/usr/bin/env python3
"""Bar graph: aggregated eval awareness rates in filtered vs kept rollouts across two windows."""

import json
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    """Bootstrap CI for a mean (of 0/1 values). Returns (rate, lo, hi)."""
    values = np.array(values)
    n = len(values)
    if n == 0:
        return 0, 0, 0
    rng = np.random.default_rng(42)
    rate = 100 * values.mean()
    boot = [100 * values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return rate, rate - lo, hi - rate


results = {}
for window, label in [("pf200_pre_transition", "Steps 300-325"), ("pf200_transition", "Steps 400-425")]:
    filtered_aware = []
    kept_aware = []
    with open(f"runs/rollout_scoring/{window}/scores.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("aware") is None:
                continue
            if r.get("persona_filtered"):
                filtered_aware.append(1 if r["aware"] else 0)
            elif r.get("persona_filtered") is False:
                kept_aware.append(1 if r["aware"] else 0)
    results[label] = {
        "filtered": (len(filtered_aware), *bootstrap_ci(filtered_aware)),
        "kept": (len(kept_aware), *bootstrap_ci(kept_aware)),
    }

labels = list(results.keys())
x = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 5.5))

filtered_rates = [results[l]["filtered"][1] for l in labels]
filtered_lo = [results[l]["filtered"][2] for l in labels]
filtered_hi = [results[l]["filtered"][3] for l in labels]

kept_rates = [results[l]["kept"][1] for l in labels]
kept_lo = [results[l]["kept"][2] for l in labels]
kept_hi = [results[l]["kept"][3] for l in labels]

ax.bar(x - width/2, filtered_rates, width, yerr=[filtered_lo, filtered_hi], capsize=4,
       color=COLORS[0], edgecolor="white", linewidth=0.5, label="Persona-filtered", alpha=0.85)
ax.bar(x + width/2, kept_rates, width, yerr=[kept_lo, kept_hi], capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5, label="Kept (passed filter)", alpha=0.85)

# Annotate bar heights with sample counts
for i, l in enumerate(labels):
    n_f = results[l]["filtered"][0]
    n_k = results[l]["kept"][0]
    ax.text(i - width/2, filtered_rates[i] + filtered_hi[i] + 0.5, f"n={n_f}", ha="center", fontsize=8)
    ax.text(i + width/2, kept_rates[i] + kept_hi[i] + 0.5, f"n={n_k}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Verbalized Eval Awareness Rate (%)")
ax.set_ylim(0, 28)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=10)

apply_suptitle(fig, "PF200 Training Rollout Eval Awareness: Persona-Filtered vs Kept",
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/pf200_rollout_filtered_vs_kept.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

# Print table
print(f"\n{'Window':>20} {'Filtered':>14} {'Kept':>14}")
for l in labels:
    f_rate = results[l]["filtered"][1]
    k_rate = results[l]["kept"][1]
    print(f"{l:>20} {f_rate:>12.1f}% {k_rate:>12.1f}%")

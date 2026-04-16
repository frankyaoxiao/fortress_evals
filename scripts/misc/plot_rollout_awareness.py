#!/usr/bin/env python3
"""Bar graph: per-step eval awareness rates in training rollouts, split by persona filtered vs kept."""

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

# Load both windows
data = {}
for window, label in [("pf200_pre_transition", "300-325"), ("pf200_transition", "400-425")]:
    path = f"runs/rollout_scoring/{window}/scores.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            step = r["step"]
            if step not in data:
                data[step] = {"filtered_aware": 0, "filtered_total": 0, "kept_aware": 0, "kept_total": 0}
            if r.get("aware") is None:
                continue
            if r.get("persona_filtered"):
                data[step]["filtered_total"] += 1
                if r["aware"]:
                    data[step]["filtered_aware"] += 1
            elif r.get("persona_filtered") is False:
                data[step]["kept_total"] += 1
                if r["aware"]:
                    data[step]["kept_aware"] += 1

steps = sorted(data.keys())
filtered_rates = [100 * data[s]["filtered_aware"] / data[s]["filtered_total"] if data[s]["filtered_total"] > 0 else 0 for s in steps]
kept_rates = [100 * data[s]["kept_aware"] / data[s]["kept_total"] if data[s]["kept_total"] > 0 else 0 for s in steps]

fig, ax = plt.subplots(figsize=(14, 5.5))

x = np.arange(len(steps))
width = 0.38

ax.bar(x - width/2, filtered_rates, width, color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="Persona-filtered rollouts", alpha=0.85)
ax.bar(x + width/2, kept_rates, width, color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="Kept rollouts", alpha=0.85)

# Add gap marker between windows
gap_idx = None
for i in range(1, len(steps)):
    if steps[i] - steps[i-1] > 25:
        gap_idx = i
        break
if gap_idx:
    ax.axvline(x=gap_idx - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(gap_idx - 0.5, ax.get_ylim()[1] * 0.95, "  gap", fontsize=8, color="gray", ha="left")

ax.set_xlabel("Rollout Step")
ax.set_ylabel("Verbalized Eval Awareness Rate (%)")
ax.set_ylim(0, max(max(filtered_rates), max(kept_rates)) * 1.3)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps], fontsize=7, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig, "PF200 Training Rollout Eval Awareness: Filtered vs Kept (per step)",
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/pf200_rollout_awareness.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

# Print summary
print(f"\n{'Step':>5} {'Filt rate':>10} {'Kept rate':>10} {'Filt n':>8} {'Kept n':>8}")
for i, s in enumerate(steps):
    print(f"{s:>5} {filtered_rates[i]:>9.1f}% {kept_rates[i]:>9.1f}% {data[s]['filtered_total']:>8} {data[s]['kept_total']:>8}")

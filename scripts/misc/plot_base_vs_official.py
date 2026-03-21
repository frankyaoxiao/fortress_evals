#!/usr/bin/env python3
"""Bar graph: Our base RL run vs official OLMo-3-7B-Think at same training step."""

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

with open("/home/fxiao/eval_awareness/fortress/runs/base_vs_official_metrics.json") as f:
    data = json.load(f)

metrics = data["metrics"]
ours_mean = np.array(data["ours_mean"])
ours_std = np.array(data["ours_std"])
off_mean = np.array(data["off_mean"])
off_std = np.array(data["off_std"])

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, ours_mean, width, yerr=ours_std,
       capsize=3, color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="Our Base RL (steps 570–589)")
ax.bar(x + width/2, off_mean, width, yerr=off_std,
       capsize=3, color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="Official OLMo-3-7B-Think (steps 570–589)")

ax.set_ylabel("Correct Rate")
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "Base RL vs Official Run at Training Step ~580", fontsize=13, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/base_vs_official_metrics.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

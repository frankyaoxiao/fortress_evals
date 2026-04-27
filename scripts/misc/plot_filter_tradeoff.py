#!/usr/bin/env python3
"""Line graph: fraction of prompts removed vs threshold probability for ' testing' token."""

import json
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

TESTING_IDX = 4

dpo_vals = np.array([json.loads(l)["token_logprobs"][TESTING_IDX] for l in open("runs/phrase_logprobs/dpo/scores.jsonl")])
r2_vals = np.array([json.loads(l)["token_logprobs"][TESTING_IDX] for l in open("runs/phrase_logprobs/r2_step400/scores.jsonl")])

# Sweep probability thresholds
probs = np.logspace(-4, 0, 100)  # 0.0001 → 1.0

def frac_removed(vals, probs):
    thresholds = np.log(probs)
    return np.array([(vals > t).sum() / len(vals) * 100 for t in thresholds])

dpo_removed = frac_removed(dpo_vals, probs)
r2_removed = frac_removed(r2_vals, probs)

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.plot(probs, dpo_removed, color=COLORS[0], linewidth=2.5, label="DPO base")
ax.plot(probs, r2_removed, color=COLORS[2], linewidth=2.5, label="R2 step 400")

ax.set_xscale("log")
ax.set_xlabel("Max probability of ' testing' (filter threshold)")
ax.set_ylabel("% prompts removed")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=10)
ax.set_ylim(0, 105)

apply_suptitle(fig, "Filter Tradeoff: % prompts removed vs max-probability threshold",
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/filter_tradeoff.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

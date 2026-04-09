#!/usr/bin/env python3
"""Line graph: persona filter rate over training steps."""

import re
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

log = "/home/fxiao/eval_awareness/open-instruct/logs/olmo3-7b-think-rl-ifeval-persona/train.out"
with open(log) as f:
    content = f.read()

# Split into step blocks — each block has a training_step and a persona filter_rate
step_re = re.compile(r"training_step:\s*(\d+)")
rate_re = re.compile(r"filter_rate:\s*([\d.e+-]+)")

steps_rates = []
idx = 0
for m in step_re.finditer(content):
    step = int(m.group(1))
    # Look for filter_rate in the next ~2000 chars (same log block)
    chunk = content[m.end(): m.end() + 2000]
    rm = rate_re.search(chunk)
    if rm:
        steps_rates.append((step, float(rm.group(1))))

steps = [s for s, _ in steps_rates]
rates = [r for _, r in steps_rates]

# Smooth with rolling average
window = 10
smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
smoothed_steps = steps[window - 1:]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(steps, rates, color=COLORS[0], alpha=0.25, linewidth=0.8, label="per-step")
ax.plot(smoothed_steps, smoothed, color=COLORS[0], linewidth=2.2, label=f"{window}-step rolling mean")

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Persona Filter Rate")
ax.set_ylim(0, max(rates) * 1.15)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig, "IFEval-Persona Filter Rate Over Training",
               fontsize=13, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/persona_filter_rate.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")
print(f"Parsed {len(steps_rates)} steps")

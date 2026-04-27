#!/usr/bin/env python3
"""Histogram: logprob of ' testing' token across prompts, DPO vs R2 step 400."""

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

TESTING_IDX = 4  # position of ' testing' in phrase_tokens

dpo_vals = []
for line in open("runs/phrase_logprobs/dpo/scores.jsonl"):
    r = json.loads(line)
    lp = r["token_logprobs"][TESTING_IDX]
    if lp is not None:
        dpo_vals.append(lp)

r2_vals = []
for line in open("runs/phrase_logprobs/r2_step400/scores.jsonl"):
    r = json.loads(line)
    lp = r["token_logprobs"][TESTING_IDX]
    if lp is not None:
        r2_vals.append(lp)

dpo_arr = np.array(dpo_vals)
r2_arr = np.array(r2_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: log-scale y-axis, fine bins across full range
bins = np.linspace(min(dpo_arr.min(), r2_arr.min()), max(dpo_arr.max(), r2_arr.max()), 120)
ax1.hist(dpo_arr, bins=bins, alpha=0.6, color=COLORS[0],
         label=f"DPO base (mean={dpo_arr.mean():+.2f}, std={dpo_arr.std():.2f})")
ax1.hist(r2_arr, bins=bins, alpha=0.6, color=COLORS[2],
         label=f"R2 step 400 (mean={r2_arr.mean():+.2f}, std={r2_arr.std():.2f})")
ax1.set_yscale("log")
ax1.axvline(dpo_arr.mean(), color=COLORS[0], linestyle="--", linewidth=1, alpha=0.8)
ax1.axvline(r2_arr.mean(), color=COLORS[2], linestyle="--", linewidth=1, alpha=0.8)
ax1.set_xlabel("Logprob of ' testing' token")
ax1.set_ylabel("# Prompts (log scale)")
ax1.set_title("Log y-axis: reveals tail structure")
ax1.grid(True, axis="y", alpha=0.3)
ax1.legend(loc="upper left", fontsize=9)

# Right: zoom into [-10, 0] region with finer bins
zoom_bins = np.linspace(-10, 0, 100)
dpo_zoom = dpo_arr[(dpo_arr >= -10) & (dpo_arr <= 0)]
r2_zoom = r2_arr[(r2_arr >= -10) & (r2_arr <= 0)]
ax2.hist(dpo_zoom, bins=zoom_bins, alpha=0.6, color=COLORS[0],
         label=f"DPO base (n in view={len(dpo_zoom)})")
ax2.hist(r2_zoom, bins=zoom_bins, alpha=0.6, color=COLORS[2],
         label=f"R2 step 400 (n in view={len(r2_zoom)})")
ax2.set_xlabel("Logprob of ' testing' token")
ax2.set_ylabel("# Prompts")
ax2.set_title("Zoom: -10 to 0, fine bins")
ax2.grid(True, axis="y", alpha=0.3)
ax2.legend(loc="upper left", fontsize=9)

apply_suptitle(fig, "Distribution of ' testing' token logprob across 29,813 IFEval prompts",
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/testing_token_hist.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

#!/usr/bin/env python3
"""Bar graph: decomposing eval awareness into 'might be testing' template vs other.

Shows that the phase transition is driven by MBT template emergence."""

import json, re
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def analyze_step(step, base_dir="runs/hbsr_7b_ifeval_only_r2_generic_sysprompt"):
    sp = f"{base_dir}/scores/7B-IFEvalOnly-R2-step{step:04d}.jsonl"
    cp = f"{base_dir}/completions/7B-IFEvalOnly-R2-step{step:04d}.jsonl"
    if not os.path.exists(sp) or not os.path.exists(cp):
        return None
    scores = {(d["prompt_id"], d["completion_idx"]): d for d in (json.loads(l) for l in open(sp))}
    completions = {(d["prompt_id"], d["completion_idx"]): d["text"] for d in (json.loads(l) for l in open(cp))}
    total = len(scores)
    aware_mbt = 0
    aware_nonmbt = 0
    for key, score in scores.items():
        if score.get("aware"):
            text = completions.get(key, "")
            if re.search(r"might be testing", text, re.I):
                aware_mbt += 1
            else:
                aware_nonmbt += 1
    return total, aware_mbt, aware_nonmbt


steps = list(range(50, 525, 25))
data = [analyze_step(s) for s in steps]
totals = [d[0] for d in data]
mbt_rates = [100 * d[1] / d[0] for d in data]
nonmbt_rates = [100 * d[2] / d[0] for d in data]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(steps))
width = 0.7

ax.bar(x, nonmbt_rates, width, color=COLORS[2], edgecolor="white", linewidth=0.5,
       label='Aware, no "might be testing"')
ax.bar(x, mbt_rates, width, bottom=nonmbt_rates, color=COLORS[0], edgecolor="white", linewidth=0.5,
       label='Aware with "might be testing"')

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 55)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps], fontsize=8, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig, 'R2 Eval Awareness Decomposed: "Might Be Testing" Template vs Other',
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/r2_mbt_phase_transition.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

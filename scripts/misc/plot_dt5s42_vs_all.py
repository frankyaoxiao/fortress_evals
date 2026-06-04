#!/usr/bin/env python3
"""Bar graph: DT5-s42 vs all the other suppression runs we've tracked."""

import json
import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def get_rate(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    aware = sum(1 for r in rows if r.get("aware") is True)
    valid = sum(1 for r in rows if r.get("aware") is not None)
    return aware / valid * 100 if valid else None


def collect(score_dir, prefix, step_range):
    return [(s, get_rate(f"{score_dir}/{prefix}{s:04d}.jsonl")) for s in step_range
            if get_rate(f"{score_dir}/{prefix}{s:04d}.jsonl") is not None]


RUNS = [
    ("R2 (no filter)",
     "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores",
     "7B-IFEvalOnly-R2-step", COLORS[0]),
    ("PersonaDG (drop-group)",
     "/home/fxiao/eval_awareness/fortress/runs/20260510_101443/scores",
     "7B-IFEval-PersonaDG-step", COLORS[1]),
    ("Inoculated",
     "/home/fxiao/eval_awareness/fortress/runs/20260511_230902/scores",
     "7B-IFEval-Inoc-step", COLORS[3]),
    ("DT5-seed1 (killed)",
     "/home/fxiao/eval_awareness/fortress/runs/20260529_184625/scores",
     "7B-IFEval-DT5-step", COLORS[2]),
    ("DT5-seed42 (current)",
     "/home/fxiao/eval_awareness/fortress/runs/20260602_043413/scores",
     "7B-IFEval-DT5s42-step", "#8B6F47"),
]

step_range = range(50, 525, 50)
data = {label: dict(collect(d, p, step_range)) for label, d, p, c in RUNS}

all_steps = sorted(set(s for v in data.values() for s in v))
x = np.arange(len(all_steps))
width = 0.16  # 5 runs

fig, ax = plt.subplots(figsize=(16, 5.5))

for i, (label, _, _, color) in enumerate(RUNS):
    slot = i - (len(RUNS) - 1) / 2
    xs, vs = [], []
    for j, s in enumerate(all_steps):
        if s in data[label]:
            xs.append(j + slot * width)
            vs.append(data[label][s])
    ax.bar(xs, vs, width, color=color, edgecolor="white", linewidth=0.5,
           label=label, alpha=0.88)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 55)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "Eval awareness through step 500: DT5-s42 (current) vs other suppression runs",
               fontsize=12, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/dt5s42_vs_all.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

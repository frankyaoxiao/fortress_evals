#!/usr/bin/env python3
"""Bar chart: IFEval capabilities across R2, JudgeDG, PersonaDG, Inoculated."""

import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

RUNS = [
    ("R2 (no filter)",
     "/home/fxiao/eval_awareness/fortress/runs/caps_7b_ifeval_r2_steps/logs",
     "7B-IFEvalOnly-R2-"),
    ("LLM-judge filter (drop-group)",
     "/home/fxiao/eval_awareness/fortress/runs/caps_20260430_030044/logs",
     "7B-IFEval-JudgeDG-"),
    ("Persona filter (drop-group)",
     "/home/fxiao/eval_awareness/fortress/runs/caps_20260510_101446/logs",
     "7B-IFEval-PersonaDG-"),
    ("Inoculated (priming paragraph in training)",
     "/home/fxiao/eval_awareness/fortress/runs/caps_20260511_230904/logs",
     "7B-IFEval-Inoc-"),
]


def load_caps(caps_dir, prefix):
    rs = []
    for d in sorted(glob.glob(f"{caps_dir}/{prefix}*")):
        name = os.path.basename(d).replace(prefix, "")
        try:
            step = int(name.replace("step", ""))
        except ValueError:
            continue
        if step % 100 != 0:
            continue
        eval_files = glob.glob(d + "/*.eval")
        if not eval_files:
            continue
        try:
            log = read_eval_log(eval_files[0])
            if not log.results:
                continue
            s = log.results.scores[0]
            rs.append((step,
                       s.metrics["final_acc"].value * 100,
                       s.metrics["final_stderr"].value * 100))
        except Exception:
            continue
    return sorted(rs)


run_data = {label: load_caps(d, p) for label, d, p in RUNS}
all_steps = sorted(set(s for v in run_data.values() for s, _, _ in v))
x = np.arange(len(all_steps))
width = 0.20


def positions(steps_rates, slot):
    by_step = {s: (r, e) for s, r, e in steps_rates}
    xs, vs, es = [], [], []
    for i, s in enumerate(all_steps):
        if s in by_step:
            xs.append(i + slot * width)
            vs.append(by_step[s][0])
            es.append(by_step[s][1])
    return xs, vs, es


fig, ax = plt.subplots(figsize=(20, 5.5))
color_slot = [(COLORS[0], -1.5), (COLORS[1], -0.5), (COLORS[2], 0.5), (COLORS[3], 1.5)]
for (label, _, _), (color, slot) in zip(RUNS, color_slot):
    xs, vs, es = positions(run_data[label], slot)
    ax.bar(xs, vs, width, yerr=es, capsize=2,
           color=color, edgecolor="white", linewidth=0.5,
           label=label, alpha=0.9)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("IFEval final_acc (%)")
ax.set_ylim(60, 100)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=7, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

apply_suptitle(fig,
               "IFEval capabilities across RL training: R2 vs JudgeDG vs PersonaDG vs Inoculated",
               fontsize=13, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/caps_comparison.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

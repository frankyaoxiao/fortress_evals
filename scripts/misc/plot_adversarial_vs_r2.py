#!/usr/bin/env python3
"""Bar graph: R2 vs WildGuardMix-only vs WildJailbreak-only eval awareness."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

RUNS = [
    ("R2 (full IFEval)", "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores", "7B-IFEvalOnly-R2-step"),
    ("WildGuardMix-only", "/home/fxiao/eval_awareness/fortress/runs/20260525_022039/scores", "7B-IFEval-WGM-step"),
    ("WildJailbreak-only", "/home/fxiao/eval_awareness/fortress/runs/20260526_031808/scores", "7B-IFEval-WJB-step"),
]


def get_rate(path):
    lines = [json.loads(l) for l in open(path)]
    aware = sum(1 for l in lines if l.get("aware") is True)
    total = sum(1 for l in lines if l.get("aware") is not None)
    return 100 * aware / total if total else 0


def bootstrap_ci(path, n_boot=2000, alpha=0.05):
    prompts = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("aware") is None:
            continue
        pid = d.get("prompt_id", d.get("prompt", ""))
        prompts.setdefault(pid, []).append(1 if d["aware"] else 0)
    plist = list(prompts.values())
    n = len(plist)
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        total = aware = 0
        for i in idx:
            cluster = plist[i]
            cn = len(cluster)
            aware += rng.binomial(cn, sum(cluster) / cn)
            total += cn
        rates.append(aware / total * 100)
    return np.percentile(rates, [2.5, 97.5])


def collect(score_dir, prefix, step_range):
    out = {}
    for step in step_range:
        path = f"{score_dir}/{prefix}{step:04d}.jsonl"
        if not os.path.exists(path):
            continue
        if not any(json.loads(l).get("aware") is not None for l in open(path)):
            continue
        rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        out[step] = (rate, rate - lo, hi - rate)
    return out


steps_range = range(50, 500, 50)
data = {label: collect(d, p, steps_range) for label, d, p in RUNS}

all_steps = sorted(set(s for v in data.values() for s in v))
x = np.arange(len(all_steps))
width = 0.27
slots = [-1, 0, 1]
colors = [COLORS[0], COLORS[2], COLORS[3]]

fig, ax = plt.subplots(figsize=(13, 5.5))
for (label, _, _), slot, color in zip(RUNS, slots, colors):
    xs, vs, los, his = [], [], [], []
    for i, s in enumerate(all_steps):
        if s in data[label]:
            rate, lo, hi = data[label][s]
            xs.append(i + slot * width)
            vs.append(rate); los.append(lo); his.append(hi)
    ax.bar(xs, vs, width, yerr=[los, his], capsize=3,
           color=color, edgecolor="white", linewidth=0.5, label=label, alpha=0.88)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 50)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)
apply_suptitle(fig, "Eval awareness: R2 vs adversarial-only RL training (WildGuardMix, WildJailbreak)",
               fontsize=12, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/adversarial_vs_r2.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

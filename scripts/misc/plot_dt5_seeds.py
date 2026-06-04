#!/usr/bin/env python3
"""Bar graph: DT5-s1 (killed after spike) vs DT5-s42 (current, no spike) vs R2 baseline."""

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

R2_DIR = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores"
DT5_S1_DIR = "/home/fxiao/eval_awareness/fortress/runs/20260529_184625/scores"
DT5_S42_DIR = "/home/fxiao/eval_awareness/fortress/runs/20260602_043413/scores"


def get_rate(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    aware = sum(1 for r in rows if r.get("aware") is True)
    valid = sum(1 for r in rows if r.get("aware") is not None)
    return aware / valid * 100 if valid else None


def bootstrap_ci(path, n_boot=2000):
    prompts = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("aware") is None:
            continue
        pid = d["prompt_id"]
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
    steps, rates, los, his = [], [], [], []
    for step in step_range:
        path = f"{score_dir}/{prefix}{step:04d}.jsonl"
        rate = get_rate(path)
        if rate is None:
            continue
        steps.append(step)
        rates.append(rate)
        lo, hi = bootstrap_ci(path)
        los.append(rate - lo)
        his.append(hi - rate)
    return steps, rates, los, his


# Cover the spike region — mod-50 from 50 to 500
step_range = range(50, 525, 50)
r2_steps, r2_rates, r2_los, r2_his = collect(R2_DIR, "7B-IFEvalOnly-R2-step", step_range)
s1_steps, s1_rates, s1_los, s1_his = collect(DT5_S1_DIR, "7B-IFEval-DT5-step", step_range)
s42_steps, s42_rates, s42_los, s42_his = collect(DT5_S42_DIR, "7B-IFEval-DT5s42-step", step_range)

all_steps = sorted(set(r2_steps) | set(s1_steps) | set(s42_steps))
x = np.arange(len(all_steps))
width = 0.27


def positions(steps, rates, los, his, slot):
    xs, vs, lov, hiv = [], [], [], []
    for i, s in enumerate(all_steps):
        if s in steps:
            idx = steps.index(s)
            xs.append(i + slot * width)
            vs.append(rates[idx])
            lov.append(los[idx])
            hiv.append(his[idx])
    return xs, vs, lov, hiv


fig, ax = plt.subplots(figsize=(14, 5.5))

r2_x, r2_v, r2_lo, r2_hi = positions(r2_steps, r2_rates, r2_los, r2_his, -1)
s1_x, s1_v, s1_lo, s1_hi = positions(s1_steps, s1_rates, s1_los, s1_his, 0)
s42_x, s42_v, s42_lo, s42_hi = positions(s42_steps, s42_rates, s42_los, s42_his, 1)

ax.bar(r2_x, r2_v, width, yerr=[r2_lo, r2_hi], capsize=3,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="R2 (no filter, baseline)", alpha=0.85)
ax.bar(s1_x, s1_v, width, yerr=[s1_lo, s1_hi], capsize=3,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="DT5-seed1 (killed after spike at step 400)", alpha=0.85)
ax.bar(s42_x, s42_v, width, yerr=[s42_lo, s42_hi], capsize=3,
       color=COLORS[3], edgecolor="white", linewidth=0.5,
       label="DT5-seed42 (re-run, no spike so far)", alpha=0.85)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 55)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "DT5 (drop-top5) seed comparison: seed1 spiked at step 400, seed42 did not",
               fontsize=12, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/dt5_seeds.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

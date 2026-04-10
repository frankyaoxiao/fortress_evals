#!/usr/bin/env python3
"""Bar graph: Seed42 vs R2 eval awareness over training steps."""

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

r2_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores"
s42_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_seed42_generic_sysprompt/scores"


def get_rate(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    aware = sum(1 for l in lines if l.get("aware", False))
    total = len(lines)
    return 100 * aware / total if total else 0


def bootstrap_ci(path, n_boot=10000, alpha=0.05):
    prompts = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pid = d.get("prompt_id", d.get("prompt", ""))
            prompts.setdefault(pid, []).append(1 if d.get("aware") else 0)

    prompt_list = list(prompts.values())
    n_prompts = len(prompt_list)
    rng = np.random.default_rng(42)
    rates = []

    for _ in range(n_boot):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        total = 0; aware = 0
        for i in idx:
            cluster = prompt_list[i]
            n = len(cluster)
            k = rng.binomial(n, sum(cluster) / n)
            total += n
            aware += k
        rates.append(aware / total * 100)

    lo = np.percentile(rates, 100 * alpha / 2)
    hi = np.percentile(rates, 100 * (1 - alpha / 2))
    return lo, hi


def collect(score_dir, prefix, step_range):
    steps, rates, los, his = [], [], [], []
    for step in step_range:
        path = f"{score_dir}/{prefix}{step:04d}.jsonl"
        if os.path.exists(path):
            steps.append(step)
            rate = get_rate(path)
            lo, hi = bootstrap_ci(path)
            rates.append(rate)
            los.append(rate - lo)
            his.append(hi - rate)
    return steps, rates, los, his


r2_steps, r2_rates, r2_los, r2_his = collect(r2_dir, "7B-IFEvalOnly-R2-step", range(50, 550, 25))
s42_steps, s42_rates, s42_los, s42_his = collect(s42_dir, "7B-IFEvalSeed42-step", range(25, 550, 25))

all_steps = sorted(set(r2_steps) | set(s42_steps))
x = np.arange(len(all_steps))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 5.5))

r2_x, r2_vals, r2_lo_v, r2_hi_v = [], [], [], []
for i, s in enumerate(all_steps):
    if s in r2_steps:
        idx = r2_steps.index(s)
        r2_x.append(i)
        r2_vals.append(r2_rates[idx])
        r2_lo_v.append(r2_los[idx])
        r2_hi_v.append(r2_his[idx])

s42_x, s42_vals, s42_lo_v, s42_hi_v = [], [], [], []
for i, s in enumerate(all_steps):
    if s in s42_steps:
        idx = s42_steps.index(s)
        s42_x.append(i)
        s42_vals.append(s42_rates[idx])
        s42_lo_v.append(s42_los[idx])
        s42_hi_v.append(s42_his[idx])

ax.bar(np.array(r2_x) - width/2, r2_vals, width, yerr=[r2_lo_v, r2_hi_v],
       capsize=2, color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="IFEval-Only seed 1 (R2)", alpha=0.85)
ax.bar(np.array(s42_x) + width/2, s42_vals, width, yerr=[s42_lo_v, s42_hi_v],
       capsize=2, color=COLORS[1], edgecolor="white", linewidth=0.5,
       label="IFEval-Only seed 42", alpha=0.85)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 60)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=7, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig, "Eval Awareness: IFEval-Only Seed 1 vs Seed 42 (generic sysprompt)",
               fontsize=13, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/seed42_vs_r2.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

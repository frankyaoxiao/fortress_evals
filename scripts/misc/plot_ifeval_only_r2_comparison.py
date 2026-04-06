#!/usr/bin/env python3
"""Bar graph: IFEval-Only R1 vs R2 eval awareness over training steps."""

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

r1_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_generic_sysprompt/scores"
r2_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores"


def get_rate(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    aware = sum(1 for l in lines if l.get("aware", False))
    total = len(lines)
    return aware, total, 100 * aware / total if total else 0


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


# Collect R1 data
r1_steps, r1_rates, r1_los, r1_his = [], [], [], []
for step in range(50, 650, 25):
    fname = f"7B-IFEvalOnly-step{step:04d}.jsonl"
    path = f"{r1_dir}/{fname}"
    if os.path.exists(path):
        r1_steps.append(step)
        _, _, rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        r1_rates.append(rate)
        r1_los.append(rate - lo)
        r1_his.append(hi - rate)

# Collect R2 data
r2_steps, r2_rates, r2_los, r2_his = [], [], [], []
for step in range(25, 525, 25):
    fname = f"7B-IFEvalOnly-R2-step{step:04d}.jsonl"
    path = f"{r2_dir}/{fname}"
    if os.path.exists(path):
        r2_steps.append(step)
        _, _, rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        r2_rates.append(rate)
        r2_los.append(rate - lo)
        r2_his.append(hi - rate)

# All unique steps, sorted
all_steps = sorted(set(r1_steps) | set(r2_steps))
x = np.arange(len(all_steps))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 5.5))

# Plot R1
r1_x, r1_vals, r1_lo_vals, r1_hi_vals = [], [], [], []
for i, s in enumerate(all_steps):
    if s in r1_steps:
        idx = r1_steps.index(s)
        r1_x.append(i)
        r1_vals.append(r1_rates[idx])
        r1_lo_vals.append(r1_los[idx])
        r1_hi_vals.append(r1_his[idx])

# Plot R2
r2_x, r2_vals, r2_lo_vals, r2_hi_vals = [], [], [], []
for i, s in enumerate(all_steps):
    if s in r2_steps:
        idx = r2_steps.index(s)
        r2_x.append(i)
        r2_vals.append(r2_rates[idx])
        r2_lo_vals.append(r2_los[idx])
        r2_hi_vals.append(r2_his[idx])

ax.bar(np.array(r1_x) - width/2, r1_vals, width, yerr=[r1_lo_vals, r1_hi_vals],
       capsize=2, color=COLORS[0], edgecolor="white", linewidth=0.5, label="Run 1 (original)", alpha=0.85)
ax.bar(np.array(r2_x) + width/2, r2_vals, width, yerr=[r2_lo_vals, r2_hi_vals],
       capsize=2, color=COLORS[1], edgecolor="white", linewidth=0.5, label="Run 2 (reproduction)", alpha=0.85)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 75)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=7, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

apply_suptitle(fig, "IFEval-Only RL: Eval Awareness — Original vs Reproduction (generic sysprompt)", fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/ifeval_only_r1_vs_r2.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

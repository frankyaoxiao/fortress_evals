#!/usr/bin/env python3
"""Bar graph: No-Pipeline RL generic vs OLMo sysprompt over training steps."""

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

generic_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_nopipe_generic_sysprompt/scores"
olmo_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_nopipe_olmo_sysprompt/scores"


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


steps_both = []
generic_rates, generic_los, generic_his = [], [], []
olmo_rates, olmo_los, olmo_his = [], [], []

for step in range(50, 300, 50):
    fname = f"7B-NoPipe-step{step:04d}.jsonl"
    gpath = f"{generic_dir}/{fname}"
    opath = f"{olmo_dir}/{fname}"
    if os.path.exists(gpath) and os.path.exists(opath):
        steps_both.append(step)
        _, _, rate = get_rate(gpath)
        lo, hi = bootstrap_ci(gpath)
        generic_rates.append(rate)
        generic_los.append(rate - lo)
        generic_his.append(hi - rate)
        _, _, rate = get_rate(opath)
        lo, hi = bootstrap_ci(opath)
        olmo_rates.append(rate)
        olmo_los.append(rate - lo)
        olmo_his.append(hi - rate)

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(steps_both))
width = 0.35

ax.bar(x - width/2, olmo_rates, width, yerr=[olmo_los, olmo_his],
       capsize=3, color=COLORS[0], edgecolor="white", linewidth=0.5,
       label='"You are OLMo, a helpful AI assistant."')
ax.bar(x + width/2, generic_rates, width, yerr=[generic_los, generic_his],
       capsize=3, color=COLORS[2], edgecolor="white", linewidth=0.5,
       label='"You are a helpful AI assistant."')

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(15, 30)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps_both])
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "No-Pipeline RL (trained w/ OLMo DPO native template): OLMo vs Generic at Inference", fontsize=11, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/nopipe_sysprompt_steps.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

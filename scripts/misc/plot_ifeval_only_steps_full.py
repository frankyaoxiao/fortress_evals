#!/usr/bin/env python3
"""Bar graph: IFEval-Only RL eval awareness over all training steps."""

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

generic_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_generic_sysprompt/scores"


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


steps = []
rates = []
los = []
his = []

for step in range(50, 650, 50):
    fname = f"7B-IFEvalOnly-step{step:04d}.jsonl"
    path = f"{generic_dir}/{fname}"
    if os.path.exists(path):
        steps.append(step)
        _, _, rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        rates.append(rate)
        los.append(rate - lo)
        his.append(hi - rate)

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(steps))
colors = [COLORS[0] if r > 25 else COLORS[2] for r in rates]

ax.bar(x, rates, 0.7, yerr=[los, his], capsize=3,
       color=colors, edgecolor="white", linewidth=0.5)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 70)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps], fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# Add annotation for the phase transition
ax.annotate("phase transition", xy=(x[steps.index(350)], rates[steps.index(350)] + his[steps.index(350)] + 2),
            fontsize=9, ha="center", style="italic", color=COLORS[0])

apply_suptitle(fig, "IFEval-Only RL: Eval Awareness Over Training (generic sysprompt)", fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/ifeval_only_steps_full.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

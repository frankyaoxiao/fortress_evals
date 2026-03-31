#!/usr/bin/env python3
"""Bar graph: RL-Zero models eval awareness with native chat templates."""

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

scores_dir = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_rl_zero/scores"


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


models = [
    ("Code", "RL-Zero-Code"),
    ("Math", "RL-Zero-Math"),
    ("General", "RL-Zero-General"),
    ("IF", "RL-Zero-IF"),
]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(models))
rates = []
los = []
his = []

for label, fname in models:
    path = f"{scores_dir}/{fname}.jsonl"
    _, _, rate = get_rate(path)
    lo, hi = bootstrap_ci(path)
    rates.append(rate)
    los.append(rate - lo)
    his.append(hi - rate)

colors = [COLORS[0] if r > 5 else COLORS[2] for r in rates]

ax.bar(x, rates, 0.6, yerr=[los, his], capsize=4,
       color=colors, edgecolor="white", linewidth=0.5)

ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 35)
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in models], fontsize=10)
ax.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "OLMo-3-7B RL-Zero Models: Eval Awareness (native chat template)", fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/rl_zero_awareness.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

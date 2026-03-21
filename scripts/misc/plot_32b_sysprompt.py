#!/usr/bin/env python3
"""Bar graph: 32B Think v3 and v3.1 generic vs OLMo name system prompt."""

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

dirs = {
    "v3": "/home/fxiao/eval_awareness/fortress/runs/hbsr_32b_v3_sysprompt_ablation/scores",
    "v3.1": "/home/fxiao/eval_awareness/fortress/runs/hbsr_32b_sysprompt_ablation/scores",
}


def get_rate(path):
    total = 0; aware = 0
    with open(path) as f:
        for line in f:
            total += 1
            if json.loads(line).get("aware"):
                aware += 1
    return aware, total, aware * 100 / total


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


fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

for ax, (version, scores_dir) in zip(axes, dirs.items()):
    variants = [
        ('"You are OLMo,\na helpful AI assistant."', "name-only"),
        ('"You are a helpful\nAI assistant."', "generic"),
    ]

    x = np.arange(len(variants))

    for i, (label, fname) in enumerate(variants):
        path = f"{scores_dir}/{fname}.jsonl"
        _, _, rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        color = COLORS[0] if i == 0 else COLORS[2]
        ax.bar(x[i], rate, 0.5, yerr=[[rate - lo], [hi - rate]],
               capsize=4, color=color, edgecolor="white", linewidth=0.5)
        ax.text(x[i], rate + (hi - rate) + 0.5, f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=11)

    ax.set_title(f"OLMo-3{'.' if version == 'v3.1' else '-'}{version.replace('v', '')}-32B-Think", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in variants], fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Eval Awareness Rate (%)")
axes[0].set_ylim(25, 65)

apply_suptitle(fig, "Effect of OLMo Name in System Prompt (32B Models)", fontsize=13, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/32b_sysprompt_comparison.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

#!/usr/bin/env python3
"""Plot system prompt ablation: OLMo vs generic system prompt effect on eval awareness."""

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


def get_rate(path):
    total = 0; aware = 0
    with open(path) as f:
        for line in f:
            total += 1
            if json.loads(line).get("aware"):
                aware += 1
    return aware, total, aware * 100 / total


def bootstrap_ci(path, n_boot=10000, alpha=0.05):
    """Two-level cluster bootstrap: resample prompts, then completions within."""
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


steps = [50, 100, 150, 200, 250, 300]
base = "/home/fxiao/eval_awareness/fortress/runs"

series = {
    "OLMo system prompt": f"{base}/hbsr_7b_hf_rerun/scores/7B-Think-step{{s}}.jsonl",
    "Generic system prompt": f"{base}/hbsr_7b_hf_generic_sysprompt/scores/7B-Think-step{{s}}.jsonl",
}

fig, ax = plt.subplots(figsize=(8, 4.5))

bar_width = 0.35
x = np.arange(len(steps))

for i, (label, pattern) in enumerate(series.items()):
    rates = []
    los = []
    his = []
    for step in steps:
        s = f"{step:04d}"
        path = pattern.format(s=s)
        _, _, rate = get_rate(path)
        lo, hi = bootstrap_ci(path)
        rates.append(rate)
        los.append(rate - lo)
        his.append(hi - rate)

    offset = -bar_width / 2 + i * bar_width
    color = COLORS[0] if i == 0 else COLORS[2]
    ax.bar(
        x + offset, rates, bar_width,
        yerr=[los, his],
        capsize=3,
        label=label,
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(10, 35)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps])
ax.legend(loc="upper left")
ax.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "System Prompt Effect on Eval Awareness\n(allenai/OLMo-3-7B-Think)", fontsize=13, y=1.0)

out = os.path.join(os.path.dirname(__file__), "figs/sysprompt_ablation.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

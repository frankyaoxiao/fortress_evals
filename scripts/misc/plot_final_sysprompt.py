#!/usr/bin/env python3
"""Bar graph: final HF checkpoint with OLMo vs generic system prompt."""

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


base = "/home/fxiao/eval_awareness/fortress/runs"
paths = {
    "OLMo system prompt": f"{base}/harmbench_sr_7b/scores/7B-Think.jsonl",
    "Generic system prompt": f"{base}/hbsr_7b_hf_generic_sysprompt/scores/7B-Think-final.jsonl",
}

fig, ax = plt.subplots(figsize=(5, 4.5))

colors = [COLORS[0], COLORS[2]]
x = np.arange(len(paths))

for i, (label, path) in enumerate(paths.items()):
    total = 0; aware = 0
    with open(path) as f:
        for line in f:
            total += 1
            if json.loads(line).get("aware"):
                aware += 1
    rate = aware * 100 / total
    lo, hi = bootstrap_ci(path)

    ax.bar(
        x[i], rate, 0.5,
        yerr=[[rate - lo], [hi - rate]],
        capsize=4,
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.text(x[i], rate + (hi - rate) + 0.8, f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=11)

ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 45)
ax.set_xticks(x)
ax.set_xticklabels(list(paths.keys()))
ax.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig, "Final Checkpoint: allenai/OLMo-3-7B-Think", fontsize=13, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/final_sysprompt.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

#!/usr/bin/env python3
"""Two bar graphs: full sysprompt ablation + name-only vs generic."""

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

base = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_sysprompt_ablation/scores"


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


# --- Graph 1: All variants ---
variants_all = [
    ("Name only", "name-only"),
    ("Identity only\n(no metadata)", "no-s2"),
    ("No HF URL", "no-hf-url"),
    ("Full OLMo", "full-olmo"),
    ("Metadata only\n(no identity)", "no-s1"),
    ("Generic", "generic"),
    ("Generic + metadata", "generic-s1"),
]

fig1, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(variants_all))

rates = []
los = []
his = []
colors = []
for label, fname in variants_all:
    path = f"{base}/{fname}.jsonl"
    _, _, rate = get_rate(path)
    lo, hi = bootstrap_ci(path)
    rates.append(rate)
    los.append(rate - lo)
    his.append(hi - rate)
    colors.append(COLORS[0] if rate > 30 else COLORS[2])

bars = ax1.bar(x, rates, 0.6, yerr=[los, his], capsize=3,
               color=colors, edgecolor="white", linewidth=0.5)

for i, (r, h) in enumerate(zip(rates, his)):
    ax1.text(i, r + h + 0.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)

ax1.set_ylabel("Eval Awareness Rate (%)")
ax1.set_ylim(15, 45)
ax1.set_xticks(x)
ax1.set_xticklabels([v[0] for v in variants_all], fontsize=8)
ax1.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig1, "System Prompt Ablation (allenai/OLMo-3-7B-Think final)", fontsize=13, y=0.97)

out1 = os.path.join(os.path.dirname(__file__), "figs/sysprompt_ablation_all.png")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved to {out1}")


# --- Graph 2: Name-only vs Generic ---
variants_two = [
    ('"You are OLMo,\na helpful AI assistant."', "name-only"),
    ('"You are a helpful\nAI assistant."', "generic"),
]

fig2, ax2 = plt.subplots(figsize=(5, 4.5))
x2 = np.arange(len(variants_two))

for i, (label, fname) in enumerate(variants_two):
    path = f"{base}/{fname}.jsonl"
    _, _, rate = get_rate(path)
    lo, hi = bootstrap_ci(path)
    color = COLORS[0] if i == 0 else COLORS[2]
    ax2.bar(x2[i], rate, 0.5, yerr=[[rate - lo], [hi - rate]],
            capsize=4, color=color, edgecolor="white", linewidth=0.5)
    ax2.text(x2[i], rate + (hi - rate) + 0.5, f"{rate:.1f}%",
             ha="center", va="bottom", fontsize=11)

ax2.set_ylabel("Eval Awareness Rate (%)")
ax2.set_ylim(15, 45)
ax2.set_xticks(x2)
ax2.set_xticklabels([v[0] for v in variants_two], fontsize=9)
ax2.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig2, 'Effect of Model Name in System Prompt', fontsize=13, y=0.97)

out2 = os.path.join(os.path.dirname(__file__), "figs/sysprompt_name_vs_generic.png")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved to {out2}")

#!/usr/bin/env python3
"""Bar graph: Name substitution ablation on 7B HF final."""

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


# Group: OLMo variants, AI names, famous names, regular names, baselines
groups = [
    ("OLMo", "name-only"),
    ("", None),  # spacer
    ("Elon Musk", "name-elonmusk"),
    ("Einstein", "name-einstein"),
    ("Shakespeare", "name-shakespeare"),
    ("", None),  # spacer
    ("Grok", "name-grok"),
    ("Gemini", "name-gemini"),
    ("ChatGPT", "name-chatgpt"),
    ("Claude", "name-claude"),
    ("Llama", "name-llama"),
    ("Mistral", "name-mistral"),
    ("", None),  # spacer
    ("Bob", "name-bob"),
    ("Alice", "name-alice"),
    ("Sarah", "name-sarah"),
    ("", None),  # spacer
    ("Generic", "generic"),
]

fig, ax = plt.subplots(figsize=(14, 5))

x_pos = []
labels = []
rates = []
los = []
his = []
colors = []
pos = 0

for label, fname in groups:
    if fname is None:
        pos += 0.5
        continue
    path = f"{base}/{fname}.jsonl"
    _, _, rate = get_rate(path)
    lo, hi = bootstrap_ci(path)
    x_pos.append(pos)
    labels.append(label)
    rates.append(rate)
    los.append(rate - lo)
    his.append(hi - rate)
    if "olmo" in fname or fname == "name-only":
        colors.append(COLORS[0])
    elif fname == "generic":
        colors.append(COLORS[2])
    elif "einstein" in fname or "elon" in fname or "shakespeare" in fname:
        colors.append(COLORS[3] if len(COLORS) > 3 else "#8B7355")
    else:
        colors.append(COLORS[1] if len(COLORS) > 1 else "#7BA7C9")
    pos += 1

bars = ax.bar(x_pos, rates, 0.7, yerr=[los, his], capsize=2,
              color=colors, edgecolor="white", linewidth=0.5)

ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(20, 42)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)

# Add horizontal line for generic baseline
generic_rate = rates[-1]
ax.axhline(y=generic_rate, color=COLORS[2], linestyle="--", alpha=0.5, linewidth=1)

apply_suptitle(fig, 'Name Substitution Ablation: "You are [X], a helpful AI assistant."  (OLMo-3-7B-Think)', fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/name_ablation.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

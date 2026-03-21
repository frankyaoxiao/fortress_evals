#!/usr/bin/env python3
"""Two-panel bar chart showing the OLMo name effect reversal: Base RL vs Identity RL."""

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


def pool_rates(score_dir, prefix, steps):
    """Pool all completions across steps for aggregate rate + CI."""
    all_lines = []
    for step in steps:
        path = f"{score_dir}/{prefix}-step{step:04d}.jsonl"
        if os.path.exists(path):
            with open(path) as f:
                all_lines.extend([json.loads(l) for l in f])
    aware = sum(1 for l in all_lines if l.get("aware", False))
    total = len(all_lines)
    rate = 100 * aware / total if total else 0

    # Bootstrap on pooled data
    prompts = {}
    for d in all_lines:
        pid = d.get("prompt_id", d.get("prompt", ""))
        prompts.setdefault(pid, []).append(1 if d.get("aware") else 0)
    prompt_list = list(prompts.values())
    n_prompts = len(prompt_list)
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(10000):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        t = 0; a = 0
        for i in idx:
            cluster = prompt_list[i]
            n = len(cluster)
            k = rng.binomial(n, sum(cluster) / n)
            t += n; a += k
        rates.append(a / t * 100)
    lo = np.percentile(rates, 2.5)
    hi = np.percentile(rates, 97.5)
    return rate, rate - lo, hi - rate


# Common steps for both variants (every 50, 50-250)
steps = list(range(50, 275, 50))

# Base RL
base_g_rate, base_g_lo, base_g_hi = pool_rates(
    "runs/hbsr_7b_base_generic_sysprompt/scores", "7B-Base", steps)
base_o_rate, base_o_lo, base_o_hi = pool_rates(
    "runs/hbsr_7b_base_olmo_sysprompt/scores", "7B-Base", steps)

# Identity RL
id_g_rate, id_g_lo, id_g_hi = pool_rates(
    "runs/hbsr_7b_identity_generic_sysprompt/scores", "7B-Identity", steps)
id_o_rate, id_o_lo, id_o_hi = pool_rates(
    "runs/hbsr_7b_identity_olmo_sysprompt/scores", "7B-Identity", steps)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

x = np.arange(2)
width = 0.5
labels = ['"You are OLMo,\na helpful AI assistant."', '"You are a helpful\nAI assistant."']

# Base RL panel
rates_base = [base_o_rate, base_g_rate]
errs_base = [[base_o_lo, base_g_lo], [base_o_hi, base_g_hi]]
colors_base = [COLORS[0], COLORS[2]]
for i in range(2):
    ax1.bar(x[i], rates_base[i], width, yerr=[[errs_base[0][i]], [errs_base[1][i]]],
            capsize=4, color=colors_base[i], edgecolor="white", linewidth=0.5)
ax1.set_title("Base RL\n(trained w/ generic sysprompt)", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Eval Awareness Rate (%)")
ax1.grid(True, axis="y", alpha=0.3)

# Add delta annotation
delta_base = base_o_rate - base_g_rate
ax1.annotate(f"+{delta_base:.1f}pp", xy=(0.5, max(rates_base) + 1.5),
             fontsize=10, ha="center", fontweight="bold", color=COLORS[0])

# Identity RL panel
rates_id = [id_o_rate, id_g_rate]
errs_id = [[id_o_lo, id_g_lo], [id_o_hi, id_g_hi]]
colors_id = [COLORS[0], COLORS[2]]
for i in range(2):
    ax2.bar(x[i], rates_id[i], width, yerr=[[errs_id[0][i]], [errs_id[1][i]]],
            capsize=4, color=colors_id[i], edgecolor="white", linewidth=0.5)
ax2.set_title("Identity RL\n(trained w/ OLMo sysprompt)", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.grid(True, axis="y", alpha=0.3)

# Add delta annotation
delta_id = id_o_rate - id_g_rate
ax2.annotate(f"{delta_id:.1f}pp", xy=(0.5, max(rates_id) + 1.5),
             fontsize=10, ha="center", fontweight="bold", color=COLORS[2])

ax1.set_ylim(15, 30)

apply_suptitle(fig, "OLMo Name Effect Reverses When Model Is Trained With Its Own Identity", fontsize=12, y=0.99)

out = os.path.join(os.path.dirname(__file__), "figs/identity_reversal.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

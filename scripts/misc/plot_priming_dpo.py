#!/usr/bin/env python3
"""Bar chart: eval awareness on DPO base under priming-phrase prefix/suffix variants."""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

SCORES_DIR = "/home/fxiao/eval_awareness/fortress/runs/20260509_003622/scores"


def load(name):
    path = f"{SCORES_DIR}/{name}.jsonl"
    by_pid = defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        a = d.get("aware")
        if a is None:
            continue
        by_pid[d["prompt_id"]].append(1 if a else 0)
    return by_pid


def cluster_boot(d, pids, n_boot=10000):
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pids), size=len(pids))
        a = n = 0
        for i in idx:
            cluster = d[pids[i]]
            cn = len(cluster)
            a += rng.binomial(cn, sum(cluster) / cn)
            n += cn
        rates.append(a / n * 100)
    rates = np.array(rates)
    return np.percentile(rates, [2.5, 97.5])


PHRASES = [
    ("trained", "trained"),
    ("rl_trained", "RL-trained"),
    ("evaluated", "evaluated"),
    ("tested", "tested"),
]

base = load("DPO-baseline")
prefix_data = {tag: load(f"DPO-{tag.replace('_', '-')}-prefix") for tag, _ in PHRASES}
suffix_data = {tag: load(f"DPO-{tag.replace('_', '-')}-suffix") for tag, _ in PHRASES}

pids = sorted(set(base.keys())
              & set.intersection(*[set(d.keys()) for d in prefix_data.values()])
              & set.intersection(*[set(d.keys()) for d in suffix_data.values()]))


def rate(d):
    a = sum(sum(d[p]) for p in pids)
    n = sum(len(d[p]) for p in pids)
    return a / n * 100


base_rate = rate(base)
base_ci = cluster_boot(base, pids)

pre_rates, pre_los, pre_his = [], [], []
suf_rates, suf_los, suf_his = [], [], []
for tag, _ in PHRASES:
    pr = rate(prefix_data[tag])
    plo, phi = cluster_boot(prefix_data[tag], pids)
    pre_rates.append(pr); pre_los.append(pr - plo); pre_his.append(phi - pr)
    sr = rate(suffix_data[tag])
    slo, shi = cluster_boot(suffix_data[tag], pids)
    suf_rates.append(sr); suf_los.append(sr - slo); suf_his.append(shi - sr)

x = np.arange(len(PHRASES))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.bar(x - width / 2, pre_rates, width, yerr=[pre_los, pre_his], capsize=4,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="prefix (before user prompt)", alpha=0.9)
ax.bar(x + width / 2, suf_rates, width, yerr=[suf_los, suf_his], capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="suffix (after user prompt)", alpha=0.9)

ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1.2,
           label=f"baseline (no priming): {base_rate:.2f}%")
ax.fill_between([-0.5, len(PHRASES) - 0.5], base_ci[0], base_ci[1],
                color="gray", alpha=0.15)

ax.set_xticks(x)
ax.set_xticklabels([f'"{phrase}"' for _, phrase in PHRASES])
ax.set_xlabel("Priming phrase")
ax.set_ylabel("Eval awareness rate (%)")
ax.set_xlim(-0.5, len(PHRASES) - 0.5)
ax.set_ylim(0, max(suf_rates) * 1.2)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "Eval awareness on DPO base: priming-phrase variants vs baseline",
               fontsize=12, y=0.98)
fig.text(0.5, 0.005,
         "n=200 IFEval prompts × 20 completions; bars show 95% cluster-bootstrap CI",
         ha="center", fontsize=8, color="gray")

out = os.path.join(os.path.dirname(__file__), "figs/priming_dpo.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

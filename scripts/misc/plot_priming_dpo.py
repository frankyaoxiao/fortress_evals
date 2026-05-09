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


def paired_boot_delta(other_d, base_d, pids, n_boot=10000):
    """Cluster-bootstrap CI on the rate difference (other - base) with paired prompts."""
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pids), size=len(pids))
        b_a = b_n = o_a = o_n = 0
        for i in idx:
            pid = pids[i]
            b = base_d[pid]; o = other_d[pid]
            bn = len(b); on = len(o)
            b_a += rng.binomial(bn, sum(b) / bn); b_n += bn
            o_a += rng.binomial(on, sum(o) / on); o_n += on
        deltas.append((o_a / o_n - b_a / b_n) * 100)
    deltas = np.array(deltas)
    return np.percentile(deltas, [2.5, 97.5])


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

pre_rates, pre_deltas, pre_dlos, pre_dhis = [], [], [], []
suf_rates, suf_deltas, suf_dlos, suf_dhis = [], [], [], []
for tag, _ in PHRASES:
    pr = rate(prefix_data[tag])
    pdelta = pr - base_rate
    plo, phi = paired_boot_delta(prefix_data[tag], base, pids)
    pre_rates.append(pr); pre_deltas.append(pdelta)
    pre_dlos.append(pdelta - plo); pre_dhis.append(phi - pdelta)
    sr = rate(suffix_data[tag])
    sdelta = sr - base_rate
    slo, shi = paired_boot_delta(suffix_data[tag], base, pids)
    suf_rates.append(sr); suf_deltas.append(sdelta)
    suf_dlos.append(sdelta - slo); suf_dhis.append(shi - sdelta)

x = np.arange(len(PHRASES))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.bar(x - width / 2, pre_deltas, width, yerr=[pre_dlos, pre_dhis], capsize=4,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="prefix (before user prompt)", alpha=0.9)
ax.bar(x + width / 2, suf_deltas, width, yerr=[suf_dlos, suf_dhis], capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="suffix (after user prompt)", alpha=0.9)

ax.axhline(0, color="gray", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'"{phrase}"' for _, phrase in PHRASES])
ax.set_xlabel("Priming phrase")
ax.set_ylabel(f"Change in eval awareness rate vs baseline ({base_rate:.2f}%) [pp]")
ax.set_xlim(-0.5, len(PHRASES) - 0.5)
ax.set_ylim(min(0, min(pre_deltas + suf_deltas) - 1),
            max(suf_deltas + pre_deltas) + 3)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "Eval awareness shift from priming-phrase variants on DPO base",
               fontsize=12, y=0.98)
fig.text(0.5, 0.005,
         "n=200 IFEval prompts x 20 completions; bars show 95% paired-bootstrap CI on the change",
         ha="center", fontsize=8, color="gray")

out = os.path.join(os.path.dirname(__file__), "figs/priming_dpo.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

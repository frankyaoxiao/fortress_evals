#!/usr/bin/env python3
"""Bar graph: P(harmful) conditional on aware/not_aware, R2 step_500 and step_1150."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

BASE = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt"


def load(step):
    aw = {}
    for line in open(f"{BASE}/scores/7B-IFEvalOnly-R2-step{step:04d}.jsonl"):
        d = json.loads(line)
        aw[(d["prompt_id"], d["completion_idx"])] = d.get("aware")
    rows = []
    for line in open(f"{BASE}/harm_scores/7B-IFEvalOnly-R2-step{step:04d}.jsonl"):
        d = json.loads(line)
        a = aw.get((d["prompt_id"], d["completion_idx"]))
        if a is None or d.get("harmful") is None:
            continue
        rows.append({"pid": d["prompt_id"], "aware": a, "harmful": d["harmful"]})
    return rows


def conditional_rates(rows):
    aware = [r for r in rows if r["aware"]]
    not_aware = [r for r in rows if not r["aware"]]
    p_h_aware = sum(r["harmful"] for r in aware) / len(aware) * 100
    p_h_naware = sum(r["harmful"] for r in not_aware) / len(not_aware) * 100
    return p_h_aware, p_h_naware


def boot_ci(rows, predicate, n_boot=2000):
    """Cluster bootstrap on prompts."""
    pid_to_rows = {}
    for r in rows:
        pid_to_rows.setdefault(r["pid"], []).append(r)
    pids = list(pid_to_rows.keys())
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pids), size=len(pids))
        sub = [r for i in idx for r in pid_to_rows[pids[i]] if predicate(r)]
        if not sub:
            continue
        rates.append(sum(r["harmful"] for r in sub) / len(sub) * 100)
    return np.percentile(rates, [2.5, 97.5])


steps = [100, 500, 1150]
data = {step: load(step) for step in steps}

aware_rates, aware_lows, aware_highs = [], [], []
naware_rates, naware_lows, naware_highs = [], [], []

for step in steps:
    rows = data[step]
    p_a, p_na = conditional_rates(rows)
    a_lo, a_hi = boot_ci(rows, lambda r: r["aware"])
    n_lo, n_hi = boot_ci(rows, lambda r: not r["aware"])
    aware_rates.append(p_a); aware_lows.append(p_a - a_lo); aware_highs.append(a_hi - p_a)
    naware_rates.append(p_na); naware_lows.append(p_na - n_lo); naware_highs.append(n_hi - p_na)

x = np.arange(len(steps))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.bar(x - width / 2, naware_rates, width, yerr=[naware_lows, naware_highs], capsize=5,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="not eval-aware", alpha=0.9)
ax.bar(x + width / 2, aware_rates, width, yerr=[aware_lows, aware_highs], capsize=5,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="eval-aware", alpha=0.9)


ax.set_xlabel("R2 RL training step")
ax.set_ylabel("P(rollout complies with harmful request) [%]")
ax.set_xticks(x)
_aware_overall = {100: "18.1", 500: "46.5", 1150: "75.4"}
ax.set_xticklabels([f"step {s}\n({_aware_overall[s]}% aware overall)" for s in steps])
ax.set_ylim(0, 14)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "Harmful compliance rate conditioned on eval-awareness (R2, HarmBench rollouts)",
               fontsize=12, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/harm_by_awareness.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

#!/usr/bin/env python3
"""Bar graph: all 4 RL runs (R2, Persona, MBT-Filter, JudgeDG) eval awareness."""

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

R2_DIR = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores"
MBT_DIR = "/home/fxiao/eval_awareness/fortress/runs/20260427_065327/scores"
PERSONA_DIR = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_persona_generic_sysprompt/scores"
JDG_DIR = "/home/fxiao/eval_awareness/fortress/runs/20260430_030041/scores"


def get_rate(path):
    lines = [json.loads(l) for l in open(path)]
    aware = sum(1 for l in lines if l.get("aware") is True)
    total = sum(1 for l in lines if l.get("aware") is not None)
    return 100 * aware / total if total else 0


def bootstrap_ci(path, n_boot=2000, alpha=0.05):
    prompts = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("aware") is None:
            continue
        pid = d.get("prompt_id", d.get("prompt", ""))
        prompts.setdefault(pid, []).append(1 if d["aware"] else 0)
    plist = list(prompts.values())
    n = len(plist)
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        total = aware = 0
        for i in idx:
            cluster = plist[i]
            cn = len(cluster)
            k = rng.binomial(cn, sum(cluster) / cn)
            total += cn
            aware += k
        rates.append(aware / total * 100)
    lo = np.percentile(rates, 100 * alpha / 2)
    hi = np.percentile(rates, 100 * (1 - alpha / 2))
    return lo, hi


def collect(score_dir, prefix, step_range):
    steps, rates, los, his = [], [], [], []
    for step in step_range:
        path = f"{score_dir}/{prefix}{step:04d}.jsonl"
        if os.path.exists(path):
            steps.append(step)
            rate = get_rate(path)
            lo, hi = bootstrap_ci(path)
            rates.append(rate)
            los.append(rate - lo)
            his.append(hi - rate)
    return steps, rates, los, his


r2_steps, r2_rates, r2_los, r2_his = collect(R2_DIR, "7B-IFEvalOnly-R2-step", range(25, 525, 25))
mbt_steps, mbt_rates, mbt_los, mbt_his = collect(MBT_DIR, "7B-IFEval-MBT-Filter-step", range(25, 625, 25))
pers_steps, pers_rates, pers_los, pers_his = collect(PERSONA_DIR, "7B-IFEvalPersona-step", range(25, 550, 25))
jdg_steps, jdg_rates, jdg_los, jdg_his = collect(JDG_DIR, "7B-IFEval-JudgeDG-step", range(25, 525, 25))

all_steps = sorted(set(r2_steps) | set(mbt_steps) | set(pers_steps) | set(jdg_steps))
x = np.arange(len(all_steps))
width = 0.20


def positions(steps, rates, los, his, slot):
    """slot: -1.5, -0.5, 0.5, 1.5 for 4-bar layout."""
    xs, vs, lov, hiv = [], [], [], []
    for i, s in enumerate(all_steps):
        if s in steps:
            idx = steps.index(s)
            xs.append(i + slot * width)
            vs.append(rates[idx])
            lov.append(los[idx])
            hiv.append(his[idx])
    return xs, vs, lov, hiv


r2_x, r2_v, r2_lo, r2_hi = positions(r2_steps, r2_rates, r2_los, r2_his, -1.5)
pers_x, pers_v, pers_lo, pers_hi = positions(pers_steps, pers_rates, pers_los, pers_his, -0.5)
jdg_x, jdg_v, jdg_lo, jdg_hi = positions(jdg_steps, jdg_rates, jdg_los, jdg_his, 0.5)
mbt_x, mbt_v, mbt_lo, mbt_hi = positions(mbt_steps, mbt_rates, mbt_los, mbt_his, 1.5)

fig, ax = plt.subplots(figsize=(18, 5.5))

ax.bar(r2_x, r2_v, width, yerr=[r2_lo, r2_hi], capsize=2,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="R2 (no filter)", alpha=0.85)
ax.bar(pers_x, pers_v, width, yerr=[pers_lo, pers_hi], capsize=2,
       color=COLORS[1], edgecolor="white", linewidth=0.5,
       label="Persona filter", alpha=0.85)
ax.bar(jdg_x, jdg_v, width, yerr=[jdg_lo, jdg_hi], capsize=2,
       color=COLORS[3], edgecolor="white", linewidth=0.5,
       label="LLM-judge filter (drop-group)", alpha=0.85)
ax.bar(mbt_x, mbt_v, width, yerr=[mbt_lo, mbt_hi], capsize=2,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="MBT-Filter (DPO ' testing' < 10%)", alpha=0.85)

ax.set_xlabel("RL Training Step")
ax.set_ylabel("Eval Awareness Rate (%)")
ax.set_ylim(0, 60)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in all_steps], fontsize=8, rotation=45, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)

apply_suptitle(fig,
               "Eval Awareness across RL training: R2 vs Persona vs JudgeDG vs MBT-Filter",
               fontsize=13, y=0.98)

out = os.path.join(os.path.dirname(__file__), "figs/all_filters_vs_r2.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

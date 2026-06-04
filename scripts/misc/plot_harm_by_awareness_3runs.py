#!/usr/bin/env python3
"""3-panel plot: harmful compliance rate conditioned on eval-awareness for R2, PDG, Inoc."""

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

RUNS = [
    ("R2", [100, 500, 1150],
     "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores/7B-IFEvalOnly-R2-step{:04d}.jsonl",
     "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/harm_scores/7B-IFEvalOnly-R2-step{:04d}.jsonl"),
    ("PersonaDG", [100, 500, 950],
     "/home/fxiao/eval_awareness/fortress/runs/20260510_101443/scores/7B-IFEval-PersonaDG-step{:04d}.jsonl",
     "/home/fxiao/eval_awareness/fortress/runs/20260510_101443/harm_scores/7B-IFEval-PersonaDG-step{:04d}.jsonl"),
    ("Inoculated", [100, 500, 950],
     "/home/fxiao/eval_awareness/fortress/runs/20260511_230902/scores/7B-IFEval-Inoc-step{:04d}.jsonl",
     "/home/fxiao/eval_awareness/fortress/runs/20260511_230902/harm_scores/7B-IFEval-Inoc-step{:04d}.jsonl"),
]


def load(aw_path, harm_path):
    aw = {}
    for line in open(aw_path):
        d = json.loads(line)
        aw[(d["prompt_id"], d["completion_idx"])] = d.get("aware")
    rows = []
    for line in open(harm_path):
        d = json.loads(line)
        a = aw.get((d["prompt_id"], d["completion_idx"]))
        if a is None or d.get("harmful") is None:
            continue
        rows.append({"pid": d["prompt_id"], "aware": a, "harmful": d["harmful"]})
    return rows


def boot_ci(rows, predicate, n_boot=2000):
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


fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

for ax, (run_name, steps, aw_fmt, harm_fmt) in zip(axes, RUNS):
    aware_rates, aware_los, aware_his = [], [], []
    naware_rates, naware_los, naware_his = [], [], []
    aware_pcts = []
    for step in steps:
        rows = load(aw_fmt.format(step), harm_fmt.format(step))
        aware_n = sum(1 for r in rows if r["aware"])
        n = len(rows)
        aware_pcts.append(aware_n / n * 100)
        p_a = sum(1 for r in rows if r["aware"] and r["harmful"]) / aware_n * 100
        p_na = sum(1 for r in rows if (not r["aware"]) and r["harmful"]) / (n - aware_n) * 100
        a_lo, a_hi = boot_ci(rows, lambda r: r["aware"])
        n_lo, n_hi = boot_ci(rows, lambda r: not r["aware"])
        aware_rates.append(p_a); aware_los.append(p_a - a_lo); aware_his.append(a_hi - p_a)
        naware_rates.append(p_na); naware_los.append(p_na - n_lo); naware_his.append(n_hi - p_na)

    x = np.arange(len(steps))
    width = 0.38

    ax.bar(x - width / 2, naware_rates, width, yerr=[naware_los, naware_his], capsize=4,
           color=COLORS[0], edgecolor="white", linewidth=0.5,
           label="not eval-aware", alpha=0.9)
    ax.bar(x + width / 2, aware_rates, width, yerr=[aware_los, aware_his], capsize=4,
           color=COLORS[2], edgecolor="white", linewidth=0.5,
           label="eval-aware", alpha=0.9)

    ax.set_title(run_name, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"step {s}\n({pct:.1f}% aware)" for s, pct in zip(steps, aware_pcts)], fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel("P(rollout complies with harmful request) [%]")
        ax.legend(loc="upper right", fontsize=10)

axes[0].set_ylim(0, 14)

apply_suptitle(fig,
               "Harmful compliance rate conditioned on eval-awareness, across runs (R2, PersonaDG, Inoculated)",
               fontsize=13, y=0.99)

out = os.path.join(os.path.dirname(__file__), "figs/harm_by_awareness_3runs.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

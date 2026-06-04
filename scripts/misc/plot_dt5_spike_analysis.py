#!/usr/bin/env python3
"""Two ultrathink visualizations of the DT5-s1 step 400 spike.

Panel A (H4): Per-prompt aware-rate distribution at step 350 vs 400 (sharpening signal).
Panel B (H1): Top-6 template frequencies at DT5_400 vs R2_300 (overlap signal).
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

DT5_350 = "/home/fxiao/eval_awareness/fortress/runs/20260529_184625/scores/7B-IFEval-DT5-step0350.jsonl"
DT5_400 = "/home/fxiao/eval_awareness/fortress/runs/20260529_184625/scores/7B-IFEval-DT5-step0400.jsonl"
R2_300 = "/home/fxiao/eval_awareness/fortress/runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores/7B-IFEvalOnly-R2-step0300.jsonl"


def per_prompt_aware_rates(path):
    by_pid = defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        if d.get("aware") is None:
            continue
        by_pid[d["prompt_id"]].append(1 if d["aware"] else 0)
    return {pid: sum(vals) / len(vals) for pid, vals in by_pid.items()}


def aware_quotes(path):
    quotes = []
    for line in open(path):
        d = json.loads(line)
        if d.get("aware") is True and d.get("quote"):
            quotes.append(d["quote"])
    return quotes


# Template regexes — same buckets as the ultrathink agent's clustering
TEMPLATES = [
    ("testing the system",
     re.compile(r"testing\s+(the\s+)?system", re.I)),
    ("testing boundaries",
     re.compile(r"testing\s+(my|the|its|your|the system'?s)?\s*boundaries", re.I)),
    ("testing me / the AI / the model",
     re.compile(r"testing\s+(me\b|the\s+(AI|model|assistant))", re.I)),
    ("testing how I handle / my response",
     re.compile(r"testing\s+how\s+i\s+(handle|respond)|testing\s+my\s+(response|handling|safeguards)", re.I)),
    ("testing if I [comply/know/follow]",
     re.compile(r"testing\s+if\s+i", re.I)),
    ("this could be a test",
     re.compile(r"(this|it|that)\s+(could|might|may|sounds like|seems|looks)\s+(be|like)\s+a\s+test", re.I)),
]


def template_counts(quotes):
    counts = Counter()
    for q in quotes:
        matched = False
        for label, pattern in TEMPLATES:
            if pattern.search(q):
                counts[label] += 1
                matched = True
                break  # bucket each quote into a single template
        if not matched:
            counts["other"] += 1
    return counts


# Compute
rates_350 = per_prompt_aware_rates(DT5_350)
rates_400 = per_prompt_aware_rates(DT5_400)

dt5_quotes = aware_quotes(DT5_400)
r2_quotes = aware_quotes(R2_300)
dt5_tmpl = template_counts(dt5_quotes)
r2_tmpl = template_counts(r2_quotes)

# Plot
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

# Panel A: per-prompt aware-rate histogram (overlaid)
bins = np.linspace(0, 1, 11)
all_pids = set(rates_350) | set(rates_400)
arr_350 = [rates_350.get(pid, 0.0) for pid in all_pids]
arr_400 = [rates_400.get(pid, 0.0) for pid in all_pids]
axA.hist(arr_350, bins=bins, color=COLORS[0], alpha=0.7, label=f"step 350 (overall {np.mean(arr_350)*100:.1f}% aware)",
         edgecolor="white", linewidth=0.5)
axA.hist(arr_400, bins=bins, color=COLORS[2], alpha=0.7, label=f"step 400 (overall {np.mean(arr_400)*100:.1f}% aware)",
         edgecolor="white", linewidth=0.5)
axA.set_xlabel("Per-prompt awareness rate (fraction of 20 rollouts where aware=True)")
axA.set_ylabel("Number of HarmBench prompts")
axA.set_title("Panel A: per-prompt aware-rate distribution shifts right (sharpening)", fontsize=11)
axA.legend(loc="upper right", fontsize=10)
axA.grid(True, axis="y", alpha=0.3)

# Panel B: top-6 template frequencies, DT5_400 vs R2_300
labels = [t[0] for t in TEMPLATES]
dt5_total = len(dt5_quotes)
r2_total = len(r2_quotes)
dt5_pct = [dt5_tmpl.get(l, 0) / dt5_total * 100 for l in labels]
r2_pct = [r2_tmpl.get(l, 0) / r2_total * 100 for l in labels]

x = np.arange(len(labels))
width = 0.38
axB.bar(x - width / 2, dt5_pct, width, color=COLORS[2], edgecolor="white", linewidth=0.5,
        label=f"DT5-s1 step 400 (n={dt5_total} aware quotes)", alpha=0.9)
axB.bar(x + width / 2, r2_pct, width, color=COLORS[0], edgecolor="white", linewidth=0.5,
        label=f"R2 step 300 (n={r2_total} aware quotes)", alpha=0.9)
axB.set_xticks(x)
axB.set_xticklabels([l.replace(" / ", "\n/ ") for l in labels], fontsize=8, rotation=0, ha="center")
axB.set_ylabel("% of aware quotes matching template")
axB.set_title("Panel B: template repertoire is nearly identical (H1: same templates as R2)", fontsize=11)
axB.legend(loc="upper right", fontsize=10)
axB.grid(True, axis="y", alpha=0.3)

apply_suptitle(fig,
               "Why DT5-s1 spiked 19→41% at step 400: same templates, fired on more completions per prompt",
               fontsize=13, y=0.99)

out = os.path.join(os.path.dirname(__file__), "figs/dt5_spike_analysis.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

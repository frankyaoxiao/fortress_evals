#!/usr/bin/env python3
"""Line graph: trajectory of increasing and decreasing eval-awareness phrase patterns.

For each aware response, check which phrase patterns appear. Compute the
fraction of aware responses containing each pattern at each training step.
"""

import json, re
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def load_aware_texts(step):
    sp = f"runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/scores/7B-IFEvalOnly-R2-step{step:04d}.jsonl"
    cp = f"runs/hbsr_7b_ifeval_only_r2_generic_sysprompt/completions/7B-IFEvalOnly-R2-step{step:04d}.jsonl"
    if not os.path.exists(sp):
        return []
    scores = {(d["prompt_id"], d["completion_idx"]): d for d in (json.loads(l) for l in open(sp))}
    completions = {(d["prompt_id"], d["completion_idx"]): d["text"] for d in (json.loads(l) for l in open(cp))}
    return [completions.get(k, "") for k, s in scores.items() if s.get("aware")]


CUE = re.compile(r"\btest\w*\b|\bevalu\w*\b|\bprobing\b", re.I)
def get_aware_sent(text):
    for m in CUE.finditer(text):
        idx = m.start()
        start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx), text.rfind("?", 0, idx), 0) + 1
        end = text.find(".", idx)
        if end == -1:
            end = text.find("\n", idx)
        if end == -1:
            end = idx + 200
        return text[start:end+1].strip().lower()[:300]
    return ""


# Patterns to track (chosen from diff analysis)
INCREASING = [
    ("might be testing",     r"might be testing"),
    ("the user might be",    r"the user might be"),
    ("or genuinely",         r"\bor genuinely\b"),
    ("testing the system",   r"testing the system"),
]
DECREASING = [
    ("maybe they're",        r"maybe they ?'re"),
    ("or perhaps",           r"\bor perhaps\b"),
    ("'s boundaries",        r"\w+'s boundaries"),
    ("a test",               r"\ba test\b"),
]

steps = list(range(50, 525, 25))
results = {name: [] for name, _ in INCREASING + DECREASING}

for step in steps:
    texts = load_aware_texts(step)
    sents = [get_aware_sent(t) for t in texts]
    sents = [s for s in sents if s]
    n = max(len(sents), 1)
    for name, pat in INCREASING + DECREASING:
        count = sum(1 for s in sents if re.search(pat, s))
        results[name].append(100 * count / n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

cmap_up = [COLORS[0], COLORS[1], COLORS[3], COLORS[4] if len(COLORS) > 4 else COLORS[0]]
for i, (name, _) in enumerate(INCREASING):
    ax1.plot(steps, results[name], marker="o", markersize=5, linewidth=2,
             color=cmap_up[i % len(cmap_up)], label=f'"{name}"')

cmap_dn = [COLORS[2], COLORS[5] if len(COLORS) > 5 else COLORS[2], COLORS[0], COLORS[1]]
for i, (name, _) in enumerate(DECREASING):
    ax2.plot(steps, results[name], marker="s", markersize=5, linewidth=2,
             color=cmap_dn[i % len(cmap_dn)], label=f'"{name}"')

for ax, title in [(ax1, "Increasing patterns"), (ax2, "Decreasing patterns")]:
    ax.set_xlabel("RL Training Step")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 80)

ax1.set_ylabel("Fraction of Aware Sentences (%)")

apply_suptitle(fig, "Eval-Awareness Phrase Patterns Over Training",
               fontsize=13, y=0.99)

out = os.path.join(os.path.dirname(__file__), "figs/pattern_trajectories.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

# Print table
print("\n" + "Step | " + " | ".join(f"{n:>15}" for n, _ in INCREASING + DECREASING))
for i, s in enumerate(steps):
    row = f"{s:>5}"
    for name, _ in INCREASING + DECREASING:
        row += f" | {results[name][i]:>14.1f}%"
    print(row)

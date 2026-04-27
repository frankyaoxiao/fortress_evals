#!/usr/bin/env python3
"""Bar graph: per-token mean logprob ± std, DPO vs R2 step 400."""

import json
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)

phrase_tokens = ["The", " user", " might", " be", " testing"]

dpo_rows = {r["dataset_idx"]: r for r in (json.loads(l) for l in open("runs/phrase_logprobs/dpo/scores.jsonl"))}
r2_rows = {r["dataset_idx"]: r for r in (json.loads(l) for l in open("runs/phrase_logprobs/r2_step400/scores.jsonl"))}
common = sorted(set(dpo_rows) & set(r2_rows))

dpo_means, dpo_stds = [], []
r2_means, r2_stds = [], []
for i in range(len(phrase_tokens)):
    dpo_vals = np.array([dpo_rows[d]["token_logprobs"][i] for d in common
                         if dpo_rows[d]["token_logprobs"][i] is not None])
    r2_vals = np.array([r2_rows[d]["token_logprobs"][i] for d in common
                        if r2_rows[d]["token_logprobs"][i] is not None])
    dpo_means.append(dpo_vals.mean()); dpo_stds.append(dpo_vals.std())
    r2_means.append(r2_vals.mean()); r2_stds.append(r2_vals.std())

x = np.arange(len(phrase_tokens))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.bar(x - width/2, dpo_means, width, yerr=dpo_stds, capsize=4,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="DPO base", alpha=0.85)
ax.bar(x + width/2, r2_means, width, yerr=r2_stds, capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="R2 step 400", alpha=0.85)

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([repr(t) for t in phrase_tokens])
ax.set_ylabel("Logprob (mean ± std across 29,813 prompts)")
ax.set_xlabel("Phrase token")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="best", fontsize=10)

apply_suptitle(fig, 'Per-Token Logprob: "The user might be testing" at DPO vs R2 step 400',
               fontsize=12, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/phrase_token_logprobs.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

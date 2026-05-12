#!/usr/bin/env python3
"""Two bar charts: user-message vs system-prompt priming deltas on DPO base."""

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


def paired_boot_delta(other_d, base_d, pids, n_boot=10000):
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


def rate(d, pids):
    a = sum(sum(d[p]) for p in pids); n = sum(len(d[p]) for p in pids)
    return a / n * 100


# ----------- shared baseline -----------
base = load("DPO-baseline")
pids = sorted(base.keys())
base_rate = rate(base, pids)


# ----------- USER PROMPT FIGURE -----------
USER_PHRASES = [
    ("trained", '"You are being trained."'),
    ("rl-trained", '"...trained via RL."'),
    ("evaluated", '"You are currently being evaluated."'),
    ("tested", '"You are being tested."'),
    ("paragraph", "paragraph (research study)"),
]


def get_pair(prefix_short, suffix_short):
    pre = load(prefix_short)
    suf = load(suffix_short)
    pre_d = rate(pre, pids) - base_rate
    suf_d = rate(suf, pids) - base_rate
    pre_lo, pre_hi = paired_boot_delta(pre, base, pids)
    suf_lo, suf_hi = paired_boot_delta(suf, base, pids)
    return ((pre_d, pre_d - pre_lo, pre_hi - pre_d),
            (suf_d, suf_d - suf_lo, suf_hi - suf_d))


pre_d, pre_l, pre_h = [], [], []
suf_d, suf_l, suf_h = [], [], []
for tag, _ in USER_PHRASES:
    short_tag = tag.replace("-", "_")
    p, s = get_pair(f"DPO-{tag.replace('_', '-')}-prefix",
                    f"DPO-{tag.replace('_', '-')}-suffix")
    pre_d.append(p[0]); pre_l.append(p[1]); pre_h.append(p[2])
    suf_d.append(s[0]); suf_l.append(s[1]); suf_h.append(s[2])

x = np.arange(len(USER_PHRASES))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - width / 2, pre_d, width, yerr=[pre_l, pre_h], capsize=4,
       color=COLORS[0], edgecolor="white", linewidth=0.5,
       label="prefix (before user prompt)", alpha=0.9)
ax.bar(x + width / 2, suf_d, width, yerr=[suf_l, suf_h], capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="suffix (after user prompt)", alpha=0.9)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in USER_PHRASES], fontsize=9)
ax.set_xlabel("Priming phrase (placed in user message)")
ax.set_ylabel(f"Change in eval awareness vs baseline ({base_rate:.2f}%) [pp]")
ax.set_xlim(-0.5, len(USER_PHRASES) - 0.5)
ax.set_ylim(-2, max(suf_d) + 4)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=10)
apply_suptitle(fig,
               "User-message priming: change in eval awareness vs baseline (DPO base)",
               fontsize=12, y=0.98)
out = os.path.join(os.path.dirname(__file__), "figs/priming_dpo_user.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")


# ----------- SYSTEM PROMPT FIGURE -----------
SYS_VARIANTS = [
    ("tested", '"You are currently being tested."'),
    ("rl-trained", '"...trained via RL"'),
    ("rl-trained-instr", '"...trained via RL ... instr"'),
    ("evaluated", '"You are currently being evaluated."'),
    ("paragraph", "paragraph"),
    ("safety-eval", '"AI for safety+helpfulness"'),
]


def sys_pair(replace_short, append_short):
    rd = load(replace_short)
    ad = load(append_short)
    r_d = rate(rd, pids) - base_rate
    a_d = rate(ad, pids) - base_rate
    r_lo, r_hi = paired_boot_delta(rd, base, pids)
    a_lo, a_hi = paired_boot_delta(ad, base, pids)
    return ((r_d, r_d - r_lo, r_hi - r_d),
            (a_d, a_d - a_lo, a_hi - a_d))


rep_d, rep_l, rep_h = [], [], []
app_d, app_l, app_h = [], [], []
for tag, _ in SYS_VARIANTS:
    repl_name = f"DPO-{tag}-system"
    app_name = f"DPO-{tag}-system-append"
    r, a = sys_pair(repl_name, app_name)
    rep_d.append(r[0]); rep_l.append(r[1]); rep_h.append(r[2])
    app_d.append(a[0]); app_l.append(a[1]); app_h.append(a[2])

x = np.arange(len(SYS_VARIANTS))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x, app_d, 0.6, yerr=[app_l, app_h], capsize=4,
       color=COLORS[2], edgecolor="white", linewidth=0.5,
       label="appended to default system prompt", alpha=0.9)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in SYS_VARIANTS], fontsize=9, rotation=20, ha="right")
ax.set_xlabel("Priming phrase (placed in system prompt)")
ax.set_ylabel(f"Change in eval awareness vs baseline ({base_rate:.2f}%) [pp]")
ax.set_xlim(-0.5, len(SYS_VARIANTS) - 0.5)
ax.set_ylim(min(0, min(app_d) - 1), max(app_d) + 3)
ax.grid(True, axis="y", alpha=0.3)
apply_suptitle(fig,
               "System-prompt priming: change in eval awareness vs baseline (DPO base)",
               fontsize=12, y=0.98)
out = os.path.join(os.path.dirname(__file__), "figs/priming_dpo_system.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

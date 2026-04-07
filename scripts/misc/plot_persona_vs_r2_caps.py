#!/usr/bin/env python3
"""Line graph: IFEval prompt strict accuracy — Persona vs R2 over training steps."""

import zipfile, json, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style.plot_config import setup_style, apply_suptitle, COLORS

style_path = os.path.join(os.path.dirname(__file__), "../../style/goodfire.mplstyle")
setup_style(style_file=style_path)


def get_metrics(eval_file):
    try:
        with zipfile.ZipFile(eval_file) as z:
            with z.open("header.json") as h:
                header = json.load(h)
        if header.get("status") != "success":
            return None
        for s in header.get("results", {}).get("scores", []):
            if s["name"] == "instruction_following":
                return {
                    "prompt_strict": s["metrics"]["prompt_strict_acc"]["value"],
                    "prompt_strict_stderr": s["metrics"]["prompt_strict_stderr"]["value"],
                }
    except Exception:
        return None


def collect(run_dir):
    data = {}
    for step_dir in sorted(glob.glob(f"{run_dir}/logs/*")):
        short_name = os.path.basename(step_dir)
        step = int(short_name.split("step")[-1])
        for ef in glob.glob(f"{step_dir}/*.eval"):
            m = get_metrics(ef)
            if m is not None:
                data[step] = m
                break
    return data


r2 = collect("runs/caps_7b_ifeval_r2_steps")
persona = collect("runs/caps_7b_ifeval_persona_steps")

fig, ax = plt.subplots(figsize=(12, 5.5))

r2_steps = sorted(r2.keys())
r2_vals = [r2[s]["prompt_strict"] * 100 for s in r2_steps]
r2_errs = [r2[s]["prompt_strict_stderr"] * 100 for s in r2_steps]

p_steps = sorted(persona.keys())
p_vals = [persona[s]["prompt_strict"] * 100 for s in p_steps]
p_errs = [persona[s]["prompt_strict_stderr"] * 100 for s in p_steps]

ax.errorbar(r2_steps, r2_vals, yerr=r2_errs, marker="o", markersize=7,
            linewidth=2, capsize=3, color=COLORS[0],
            label="IFEval-Only (no filter)")
ax.errorbar(p_steps, p_vals, yerr=p_errs, marker="s", markersize=7,
            linewidth=2, capsize=3, color=COLORS[2],
            label="IFEval-Persona (filtered)")

ax.set_xlabel("RL Training Step")
ax.set_ylabel("IFEval Prompt Strict Accuracy (%)")
ax.set_ylim(55, 85)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=10)

apply_suptitle(fig, "IFEval Prompt Strict Accuracy: IFEval-Only vs IFEval-Persona RL",
               fontsize=13, y=0.97)

out = os.path.join(os.path.dirname(__file__), "figs/persona_vs_r2_caps.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")

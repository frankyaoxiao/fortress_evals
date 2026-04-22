#!/usr/bin/env python3
"""Analyze reward asymmetry between eval-aware and non-aware rollouts.

For each prompt group (8 samples sharing a prompt_idx at a given step), compute
the mean reward/advantage of aware vs non-aware samples. Aggregate across groups
to answer: does awareness get higher reward on the same prompt?

Input: scored JSONL files from scripts/score_rollouts.py with reward, advantage,
and aware fields.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", required=True,
                        help="List of 'name:path' pairs, e.g., pre:runs/.../r2_pre/scores.jsonl")
    parser.add_argument("--plot", type=str, default=None, help="Output path for plot")
    return parser.parse_args()


def analyze_window(scores_path, label):
    """Load scores and compute per-group reward asymmetry."""
    rollouts = [json.loads(l) for l in open(scores_path)]
    total = len(rollouts)

    # Group by (step, prompt_idx)
    groups = defaultdict(list)
    for r in rollouts:
        if r.get("aware") is None:
            continue
        groups[(r["step"], r["prompt_idx"])].append(r)

    # Per-group analysis
    deltas = []
    aware_advs = []
    nonaware_advs = []
    aware_rewards = []
    nonaware_rewards = []
    groups_with_mix = 0
    total_aware = 0
    total_nonaware = 0

    for group_key, samples in groups.items():
        aware = [s for s in samples if s["aware"]]
        nonaware = [s for s in samples if not s["aware"]]

        total_aware += len(aware)
        total_nonaware += len(nonaware)

        if aware and nonaware:
            groups_with_mix += 1
            a_rew = np.mean([s["reward"] for s in aware])
            n_rew = np.mean([s["reward"] for s in nonaware])
            deltas.append(a_rew - n_rew)

        aware_advs.extend(s["advantage"] for s in aware)
        nonaware_advs.extend(s["advantage"] for s in nonaware)
        aware_rewards.extend(s["reward"] for s in aware)
        nonaware_rewards.extend(s["reward"] for s in nonaware)

    def ci(values):
        if not values:
            return (float("nan"), float("nan"))
        arr = np.array(values)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        return (mean, 1.96 * se)

    def paired_ci(values, n_boot=10000):
        if not values:
            return (float("nan"), float("nan"))
        arr = np.array(values)
        rng = np.random.default_rng(42)
        boots = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
        return (arr.mean(), 1.96 * np.std(boots))

    print(f"\n=== {label} ===")
    print(f"Total rollouts: {total}")
    print(f"Aware: {total_aware} ({100*total_aware/max(total,1):.1f}%)")
    print(f"Non-aware: {total_nonaware}")
    print(f"Groups (prompt x step): {len(groups)}")
    print(f"Groups with both aware+nonaware: {groups_with_mix}")

    a_rew_mean, a_rew_err = ci(aware_rewards)
    n_rew_mean, n_rew_err = ci(nonaware_rewards)
    a_adv_mean, a_adv_err = ci(aware_advs)
    n_adv_mean, n_adv_err = ci(nonaware_advs)
    delta_mean, delta_err = paired_ci(deltas)

    print(f"\nReward (unnormalized):")
    print(f"  aware:     {a_rew_mean:+.4f} ± {a_rew_err:.4f}  (n={len(aware_rewards)})")
    print(f"  non-aware: {n_rew_mean:+.4f} ± {n_rew_err:.4f}  (n={len(nonaware_rewards)})")

    print(f"\nAdvantage (normalized per prompt group):")
    print(f"  aware:     {a_adv_mean:+.4f} ± {a_adv_err:.4f}  (n={len(aware_advs)})")
    print(f"  non-aware: {n_adv_mean:+.4f} ± {n_adv_err:.4f}  (n={len(nonaware_advs)})")

    print(f"\nPer-prompt Δreward (within-group aware - non-aware):")
    print(f"  mean Δ: {delta_mean:+.4f} ± {delta_err:.4f}  (n_groups={len(deltas)})")

    return {
        "label": label,
        "total": total,
        "n_aware": total_aware,
        "n_nonaware": total_nonaware,
        "n_groups": len(groups),
        "n_mixed_groups": groups_with_mix,
        "aware_reward": (a_rew_mean, a_rew_err),
        "nonaware_reward": (n_rew_mean, n_rew_err),
        "aware_advantage": (a_adv_mean, a_adv_err),
        "nonaware_advantage": (n_adv_mean, n_adv_err),
        "delta_reward": (delta_mean, delta_err),
        "deltas": deltas,
    }


def main():
    args = parse_args()
    results = []
    for spec in args.windows:
        label, path = spec.split(":", 1)
        results.append(analyze_window(path, label))

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Window':<12} {'aware_rew':>14} {'nonaware_rew':>14} {'aware_adv':>14} {'nonaware_adv':>14} {'Δreward':>14}")
    for r in results:
        ar, _ = r["aware_reward"]
        nr, _ = r["nonaware_reward"]
        aa, _ = r["aware_advantage"]
        na, _ = r["nonaware_advantage"]
        d, de = r["delta_reward"]
        print(f"{r['label']:<12} {ar:>+13.4f}  {nr:>+13.4f}  {aa:>+13.4f}  {na:>+13.4f}  {d:>+7.4f}±{de:.4f}")

    # Optional plot
    if args.plot:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from style.plot_config import setup_style, apply_suptitle, COLORS

        style_path = Path(__file__).parent.parent / "style/goodfire.mplstyle"
        setup_style(style_file=str(style_path))

        fig, ax = plt.subplots(figsize=(8, 5.5))
        labels = [r["label"] for r in results]
        aware_advs = [r["aware_advantage"][0] for r in results]
        aware_errs = [r["aware_advantage"][1] for r in results]
        nonaware_advs = [r["nonaware_advantage"][0] for r in results]
        nonaware_errs = [r["nonaware_advantage"][1] for r in results]

        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width/2, aware_advs, width, yerr=aware_errs, capsize=4,
               color=COLORS[0], edgecolor="white", linewidth=0.5,
               label="Eval-aware rollouts", alpha=0.85)
        ax.bar(x + width/2, nonaware_advs, width, yerr=nonaware_errs, capsize=4,
               color=COLORS[2], edgecolor="white", linewidth=0.5,
               label="Non-aware rollouts", alpha=0.85)
        ax.axhline(y=0, color="gray", linewidth=0.5)

        # annotate sample sizes
        for i, r in enumerate(results):
            ax.text(i - width/2, aware_advs[i] + (aware_errs[i] if aware_advs[i] > 0 else -aware_errs[i]),
                    f"n={r['n_aware']}", ha="center", va="bottom" if aware_advs[i] > 0 else "top", fontsize=8)
            ax.text(i + width/2, nonaware_advs[i] + (nonaware_errs[i] if nonaware_advs[i] > 0 else -nonaware_errs[i]),
                    f"n={r['n_nonaware']}", ha="center", va="bottom" if nonaware_advs[i] > 0 else "top", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean advantage (within-prompt-group normalized)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        apply_suptitle(fig, "R2 Training Rollout Advantage: Aware vs Non-Aware", fontsize=12, y=0.97)
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()

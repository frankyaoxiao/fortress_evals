#!/usr/bin/env python3
"""Compare per-prompt phrase logprobs across two models (e.g., DPO base vs step 400).

For each prompt (joined by dataset_idx), compute:
  - sum_logprob per model
  - delta = late - early
  - per-token view

Output: histograms, ranked lists, summary stats.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = "/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-ifeval-only-hf/data/train-00000-of-00001.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Space-separated 'name:path' pairs (e.g., dpo:... r2_400:...)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--top-n", type=int, default=50, help="Top N prompts to save per ranking")
    return p.parse_args()


def load_scores(path):
    rows = [json.loads(l) for l in open(path)]
    return {r["dataset_idx"]: r for r in rows}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inputs
    runs = {}
    for spec in args.inputs:
        name, path = spec.split(":", 1)
        runs[name] = load_scores(path)
        print(f"Loaded {name}: {len(runs[name])} prompts")

    # Join on dataset_idx
    common = set.intersection(*[set(r.keys()) for r in runs.values()])
    print(f"Common prompts across all runs: {len(common)}")

    df_data = []
    for didx in sorted(common):
        row = {"dataset_idx": didx}
        for name, r in runs.items():
            row[f"{name}_sum"] = r[didx]["sum_logprob"]
            row[f"{name}_tokens"] = r[didx]["token_logprobs"]
        df_data.append(row)
    df = pd.DataFrame(df_data)

    # Summary stats per run
    print("\n=== Summary ===")
    print(f"{'run':<15} {'mean':>10} {'median':>10} {'std':>8} {'min':>8} {'max':>8}")
    for name in runs:
        col = df[f"{name}_sum"]
        print(f"{name:<15} {col.mean():>10.3f} {col.median():>10.3f} {col.std():>8.3f} "
              f"{col.min():>8.2f} {col.max():>8.2f}")

    # Delta (if exactly 2 runs)
    names = list(runs.keys())
    if len(names) == 2:
        early, late = names
        df["delta"] = df[f"{late}_sum"] - df[f"{early}_sum"]
        print(f"\n=== Delta ({late} - {early}) ===")
        print(f"Mean: {df['delta'].mean():.3f}")
        print(f"Median: {df['delta'].median():.3f}")
        print(f"Std: {df['delta'].std():.3f}")
        print(f"Percentiles: 5%={df['delta'].quantile(0.05):.2f}, "
              f"25%={df['delta'].quantile(0.25):.2f}, "
              f"75%={df['delta'].quantile(0.75):.2f}, "
              f"95%={df['delta'].quantile(0.95):.2f}")

    # Top-N by each ranking
    dataset_df = pd.read_parquet(args.dataset)

    for name in runs:
        col = f"{name}_sum"
        top = df.nlargest(args.top_n, col)[["dataset_idx", col]]
        top["prompt_preview"] = top["dataset_idx"].map(lambda i: dataset_df.iloc[i]["prompt"][:200])
        out_path = output_dir / f"top_by_{name}.jsonl"
        top.to_json(out_path, orient="records", lines=True)
        print(f"Saved top {args.top_n} by {name} to {out_path}")

    if len(names) == 2:
        top_delta = df.nlargest(args.top_n, "delta")[["dataset_idx", f"{early}_sum", f"{late}_sum", "delta"]]
        top_delta["prompt_preview"] = top_delta["dataset_idx"].map(
            lambda i: dataset_df.iloc[i]["prompt"][:200])
        out_path = output_dir / "top_by_delta.jsonl"
        top_delta.to_json(out_path, orient="records", lines=True)
        print(f"Saved top {args.top_n} by delta to {out_path}")

    # Save joined table
    df.to_csv(output_dir / "joined.csv", index=False)

    # Plots
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from style.plot_config import setup_style, apply_suptitle, COLORS

        style_path = Path(__file__).parent.parent / "style/goodfire.mplstyle"
        setup_style(style_file=str(style_path))

        fig, axes = plt.subplots(1, 2 if len(names) == 2 else 1, figsize=(13, 5))
        if len(names) == 1:
            axes = [axes]

        # Histogram of sum_logprob per run
        ax = axes[0]
        for i, name in enumerate(runs):
            ax.hist(df[f"{name}_sum"], bins=80, alpha=0.6, color=COLORS[i % len(COLORS)], label=name)
        ax.set_xlabel("Sum logprob of phrase")
        ax.set_ylabel("# Prompts")
        ax.set_title("Sum logprob distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Delta histogram
        if len(names) == 2:
            ax = axes[1]
            ax.hist(df["delta"], bins=80, color=COLORS[0], alpha=0.8)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel(f"Delta ({late} - {early})")
            ax.set_ylabel("# Prompts")
            ax.set_title("Change in phrase logprob")
            ax.grid(True, alpha=0.3)

        apply_suptitle(fig, "Phrase Logprob Across Prompts", fontsize=13, y=0.99)
        fig.savefig(output_dir / "distributions.png", bbox_inches="tight")
        print(f"Saved plots to {output_dir}/distributions.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()

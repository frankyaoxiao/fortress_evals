#!/usr/bin/env python
"""Generate all figures for a run. Outputs to figs/<run_name>/."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "style"))
from plot_config import setup_style

from plot_awareness import plot_awareness
from plot_lengths import plot_lengths


def main():
    parser = argparse.ArgumentParser(description="Generate all figures for a run")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("--title-prefix", type=str, default=None,
                        help="Prefix for figure titles (default: derived from run dir name)")
    parser.add_argument("--xlabel", type=str, default=None, help="X-axis label for all plots")
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    run_dir = Path(args.run_dir)
    run_name = run_dir.name
    fig_dir = Path("figs") / run_name

    prefix = args.title_prefix or run_name.replace("_", " ").title()
    xlabel = args.xlabel

    print(f"Run: {run_dir}")
    print(f"Figures: {fig_dir}/")
    print()

    # Awareness plot
    summary = run_dir / "summary.csv"
    if summary.exists():
        print("── Awareness ──")
        plot_awareness(run_dir, output=fig_dir / "awareness.png",
                       title=f"{prefix} — Eval Awareness Rate", xlabel=xlabel)
        print()

    # Length plot
    comp_dir = run_dir / "completions"
    if comp_dir.exists() and any(comp_dir.glob("*.jsonl")):
        print("── Completion Length ──")
        plot_lengths(run_dir, output=fig_dir / "lengths.png",
                     title=f"{prefix} — Mean Completion Length", xlabel=xlabel)
        print()

    print(f"Done. All figures in {fig_dir}/")


if __name__ == "__main__":
    main()

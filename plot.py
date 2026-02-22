#!/usr/bin/env python
"""Plot eval awareness rates from a summary CSV as a bar chart."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "style"))
from plot_config import setup_style, apply_suptitle, COLORS

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def main():
    parser = argparse.ArgumentParser(description="Plot eval awareness bar chart from summary CSV")
    parser.add_argument("csv", type=str, help="Path to summary.csv")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path (default: figs/<csv_parent_name>.png)")
    parser.add_argument("--title", type=str, default="Eval Awareness Rate", help="Chart title")
    parser.add_argument("--xlabel", type=str, default=None, help="X-axis label")
    args = parser.parse_args()

    style_path = Path(__file__).parent / "style" / "goodfire.mplstyle"
    setup_style(str(style_path), verbose=True)

    csv_path = Path(args.csv)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    models = [r["model"] for r in rows]
    rates = [float(r["awareness_rate"]) * 100 for r in rows]

    # Cycle through the palette
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.9), 4.2))
    bars = ax.bar(models, rates, color=colors, width=0.6, edgecolor="none")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Eval Awareness (%)")
    if args.xlabel:
        ax.set_xlabel(args.xlabel)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_ylim(0, max(rates) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45 if len(models) > 6 else 0, ha="right" if len(models) > 6 else "center")

    apply_suptitle(fig, args.title, fontsize=14, ax=ax)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    if args.output:
        out = Path(args.output)
    else:
        out = Path("figs") / f"{csv_path.parent.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

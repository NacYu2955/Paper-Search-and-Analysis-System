import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
]


def load_summary(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[row["model_name"]] = row
    return rows


def plot(summary_path: Path, output_path: Path) -> None:
    rows = load_summary(summary_path)
    base = np.array([float(rows["base_qwen"][key]) for _, key in METRICS])
    fine = np.array([float(rows["fine_tuned_selector"][key]) for _, key in METRICS])

    x = np.arange(len(METRICS))
    width = 0.36

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#5f6368",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.9, 4.57), dpi=140)
    base_color = "#2F69AC"
    fine_color = "#2E8B57"

    bars_base = ax.bar(x - width / 2, base, width, label="Base Qwen", color=base_color)
    bars_fine = ax.bar(
        x + width / 2,
        fine,
        width,
        label="Fine-tuned selector",
        color=fine_color,
    )

    for bars, values in ((bars_base, base), (bars_fine, fine)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.013,
                f"{value * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    for index, delta in enumerate(fine - base):
        color = "#2E8B57" if delta >= 0 else "#D04A1F"
        ax.text(
            x[index],
            max(base[index], fine[index]) + 0.085,
            f"{delta * 100:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
        )

    ax.set_title("Metric Comparison: Base vs Fine-tuned Selector", pad=6)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in METRICS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        handles=[bars_base, bars_fine],
        labels=["Base Qwen", "Fine-tuned selector"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.20)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=Path("output") / "selector_eval" / "selector_truefalse_summary.csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("output") / "selector_eval" / "selector_metric_comparison.png",
        type=Path,
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(args.summary, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

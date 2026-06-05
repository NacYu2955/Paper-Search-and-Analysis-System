import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_CSV = os.path.join(ROOT, "output", "eval", "topk_metrics_summary.csv")
OUT_PNG = os.path.join(ROOT, "output", "eval", "topk_metrics_comparison.png")

METHOD_ORDER = ["Dense", "Sparse", "Hybrid", "Hybrid + rerank"]
COLORS = {
    "Dense": "#2f6db3",
    "Sparse": "#d9822b",
    "Hybrid": "#2f8f5b",
    "Hybrid + rerank": "#8b4fb8",
}
MARKERS = {
    "Dense": "o",
    "Sparse": "s",
    "Hybrid": "^",
    "Hybrid + rerank": "D",
}


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["k"] = int(row["k"])
        for key in ("hit", "recall", "all_gt", "mrr", "ndcg", "mean_latency_seconds"):
            row[key] = float(row[key])
    return rows


def rows_by_method(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda item: item["k"])
    return grouped


def draw_panel(ax, rows, methods, metric, title, k_values, y_limits):
    grouped = rows_by_method(rows)
    for method in methods:
        series = [row for row in grouped.get(method, []) if row["k"] in k_values]
        if not series:
            continue
        xs = [row["k"] for row in series]
        ys = [row[metric] for row in series]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[method],
            color=COLORS[method],
            linewidth=2.4,
            markersize=6,
            label=method,
        )
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=COLORS[method],
            )

    ax.axvline(5, color="#b42318", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.text(
        4.92,
        y_limits[1] - 0.01,
        "Top 5",
        color="#b42318",
        fontsize=9,
        ha="right",
        va="top",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Top K")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(k_values)
    ax.set_xlim(min(k_values) - 0.2, max(k_values) + 0.2)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    rows = load_rows(SUMMARY_CSV)
    final_rows = [row for row in rows if row["stage"] == "final"]
    candidate_rows = [
        row
        for row in rows
        if row["stage"] == "candidate" and row["method"] != "Hybrid + rerank"
    ]

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.98, hspace=0.42, wspace=0.20)
    fig.suptitle(
        "Top K Retrieval Metrics Comparison",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "50 queries, 446-paper corpus. Dashed red line marks the recommended final output K=5.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    draw_panel(
        axes[0, 0],
        final_rows,
        METHOD_ORDER,
        "recall",
        "Final Output Recall@K",
        [1, 3, 5],
        (0.80, 1.01),
    )
    draw_panel(
        axes[0, 1],
        final_rows,
        METHOD_ORDER,
        "mrr",
        "Final Output MRR@K",
        [1, 3, 5],
        (0.80, 1.01),
    )
    draw_panel(
        axes[1, 0],
        candidate_rows,
        ["Dense", "Sparse", "Hybrid"],
        "recall",
        "Candidate Pool Recall@K",
        [1, 3, 5, 10, 20, 50],
        (0.85, 1.01),
    )
    draw_panel(
        axes[1, 1],
        candidate_rows,
        ["Dense", "Sparse", "Hybrid"],
        "mrr",
        "Candidate Pool MRR@K",
        [1, 3, 5, 10, 20, 50],
        (0.90, 0.965),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.text(
        0.5,
        0.11,
        "Check: final Hybrid + rerank reaches Recall=1.00 at Top 5; candidate recall saturates from Top 10.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

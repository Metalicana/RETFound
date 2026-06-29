from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


FOUNDATION_MODELS = [
    "retfound_oct",
    "visionfm_oct",
    "urfound_oct",
    "mirage_slo",
    "flair_slo",
    "ret_clip_slo",
    "retizero_slo",
    "urfound_slo",
    "visionfm_slo",
]

MODEL_LABELS = {
    "retfound_oct": "RETFound-OCT",
    "visionfm_oct": "VisionFM-OCT",
    "urfound_oct": "URFound-OCT",
    "mirage_slo": "MIRAGE-SLO",
    "flair_slo": "FLAIR-SLO",
    "ret_clip_slo": "RET-CLIP-SLO",
    "retizero_slo": "RetiZero-SLO",
    "urfound_slo": "URFound-SLO",
    "visionfm_slo": "VisionFM-SLO",
}

TASKS = ["amd", "dr", "glaucoma"]
TASK_LABELS = {"amd": "AMD", "dr": "DR", "glaucoma": "Glaucoma"}

CORE_COLUMNS = [
    ("race", "asian", "Asian"),
    ("race", "black", "Black"),
    ("race", "white", "White"),
    ("sex_gender", "female", "Female"),
    ("sex_gender", "male", "Male"),
    ("age_group", "younger", "Young"),
    ("age_group", "middle-aged", "Middle"),
    ("age_group", "older", "Old"),
]

INTERSECTION_SPECS = {
    "race_x_age_group": [
        ("race_x_age_group", "asian x younger", "Asian\nyoung"),
        ("race_x_age_group", "asian x middle-aged", "Asian\nmiddle"),
        ("race_x_age_group", "asian x older", "Asian\nold"),
        ("race_x_age_group", "black x younger", "Black\nyoung"),
        ("race_x_age_group", "black x middle-aged", "Black\nmiddle"),
        ("race_x_age_group", "black x older", "Black\nold"),
        ("race_x_age_group", "white x younger", "White\nyoung"),
        ("race_x_age_group", "white x middle-aged", "White\nmiddle"),
        ("race_x_age_group", "white x older", "White\nold"),
    ],
    "race_x_sex_gender": [
        ("race_x_sex_gender", "asian x female", "Asian\nfemale"),
        ("race_x_sex_gender", "asian x male", "Asian\nmale"),
        ("race_x_sex_gender", "black x female", "Black\nfemale"),
        ("race_x_sex_gender", "black x male", "Black\nmale"),
        ("race_x_sex_gender", "white x female", "White\nfemale"),
        ("race_x_sex_gender", "white x male", "White\nmale"),
    ],
    "sex_gender_x_age_group": [
        ("sex_gender_x_age_group", "female x younger", "Female\nyoung"),
        ("sex_gender_x_age_group", "female x middle-aged", "Female\nmiddle"),
        ("sex_gender_x_age_group", "female x older", "Female\nold"),
        ("sex_gender_x_age_group", "male x younger", "Male\nyoung"),
        ("sex_gender_x_age_group", "male x middle-aged", "Male\nmiddle"),
        ("sex_gender_x_age_group", "male x older", "Male\nold"),
    ],
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

F1_CMAP = LinearSegmentedColormap.from_list(
    "f1_support",
    ["#F4F5F7", "#CEDFFE", "#A3BEFA", "#5477C4", "#2E4780"],
)
F1_CMAP.set_bad("#F8F9FC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize foundation-model subgroup F1 scores.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("equi-agent/outputs/metrics/reliability_scores_by_subgroup.csv"),
        help="CSV with task/model/subgroup metrics and f1_local.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("equi-agent/outputs/figures/subgroup_f1"),
        help="Directory for PNG/SVG figures and chart-ready CSVs.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=0,
        help="Optional minimum subgroup support. Cells below this are still plotted but flagged in exports.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        model = row.get("model_name", "")
        if model not in FOUNDATION_MODELS:
            continue
        key = (row.get("task", ""), model, row.get("attribute", ""), row.get("subgroup", ""))
        indexed[key] = row
    return indexed


def chart_rows_for_columns(
    indexed: dict[tuple[str, str, str, str], dict[str, str]],
    columns: list[tuple[str, str, str]],
    min_n: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in TASKS:
        for model in FOUNDATION_MODELS:
            for attribute, subgroup, label in columns:
                row = indexed.get((task, model, attribute, subgroup))
                f1 = float_or_nan(row.get("f1_local") if row else "")
                n = int_or_zero(row.get("n") if row else "")
                n_positive = int_or_zero(row.get("n_positive") if row else "")
                n_negative = int_or_zero(row.get("n_negative") if row else "")
                f1_defined = bool(row) and n_positive > 0 and n_negative > 0
                output.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "model_name": model,
                        "model_label": MODEL_LABELS[model],
                        "attribute": attribute,
                        "subgroup": subgroup,
                        "subgroup_label": label.replace("\n", " "),
                        "f1_local": f1,
                        "f1_plot": f1 if f1_defined else float("nan"),
                        "f1_defined": f1_defined,
                        "undefined_reason": "" if f1_defined else ("no_positive_cases" if n_positive == 0 else "no_negative_cases" if n_negative == 0 else "missing"),
                        "n": n,
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "unstable": str(row.get("unstable", "")).lower() == "true" if row else True,
                        "below_min_n": n < min_n,
                    }
                )
    return output


def matrix_for_task(
    chart_rows: list[dict[str, Any]],
    task: str,
    columns: list[tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((len(FOUNDATION_MODELS), len(columns)), np.nan, dtype=float)
    support = np.zeros((len(FOUNDATION_MODELS), len(columns)), dtype=int)
    positives = np.zeros((len(FOUNDATION_MODELS), len(columns)), dtype=int)
    negatives = np.zeros((len(FOUNDATION_MODELS), len(columns)), dtype=int)
    by_key = {
        (row["task"], row["model_name"], row["attribute"], row["subgroup"]): row
        for row in chart_rows
    }
    for i, model in enumerate(FOUNDATION_MODELS):
        for j, (attribute, subgroup, _label) in enumerate(columns):
            row = by_key.get((task, model, attribute, subgroup))
            if row:
                values[i, j] = float_or_nan(row["f1_plot"])
                support[i, j] = int_or_zero(row["n"])
                positives[i, j] = int_or_zero(row["n_positive"])
                negatives[i, j] = int_or_zero(row["n_negative"])
    return values, support, positives, negatives


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.tick_params(axis="both", length=0, colors=TOKENS["muted"], labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    support: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    min_n: int,
    fontsize: float = 7.3,
) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if math.isnan(value):
                if support[i, j] > 0 and positives[i, j] == 0:
                    text = "0+"
                elif support[i, j] > 0 and negatives[i, j] == 0:
                    text = "0-"
                else:
                    text = "NA"
                color = TOKENS["muted"]
            else:
                text = f"{value:.2f}"
                color = TOKENS["panel"] if value >= 0.58 else TOKENS["ink"]
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight="semibold")
            if min_n and support[i, j] < min_n:
                ax.add_patch(
                    Rectangle(
                        (j - 0.47, i - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor="#CC6F47",
                        linewidth=1.0,
                        linestyle=(0, (2.0, 1.5)),
                    )
                )


def add_group_rules(ax: plt.Axes, columns: list[tuple[str, str, str]]) -> None:
    groups: list[str] = []
    for attribute, _subgroup, _label in columns:
        if not groups or groups[-1] != attribute:
            groups.append(attribute)
    previous = columns[0][0]
    for idx, (attribute, _subgroup, _label) in enumerate(columns[1:], start=1):
        if attribute != previous:
            ax.axvline(idx - 0.5, color=TOKENS["surface"], linewidth=2.4)
            previous = attribute


def plot_task_heatmap(
    chart_rows: list[dict[str, Any]],
    task: str,
    columns: list[tuple[str, str, str]],
    path_base: Path,
    title: str,
    subtitle: str,
    min_n: int,
    figsize: tuple[float, float],
) -> None:
    values, support, positives, negatives = matrix_for_task(chart_rows, task, columns)
    fig, ax = plt.subplots(figsize=figsize, facecolor=TOKENS["surface"])
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=F1_CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(columns)), [label for *_rest, label in columns], rotation=0)
    ax.set_yticks(range(len(FOUNDATION_MODELS)), [MODEL_LABELS[model] for model in FOUNDATION_MODELS])
    ax.set_title("")
    style_axis(ax)
    add_group_rules(ax, columns)
    annotate_heatmap(ax, values, support, positives, negatives, min_n=min_n, fontsize=7.6 if len(columns) <= 8 else 6.9)

    fig.text(0.07, 0.955, title, ha="left", va="top", fontsize=14.5, color=TOKENS["ink"], fontweight="semibold")
    fig.text(0.07, 0.915, subtitle, ha="left", va="top", fontsize=9.0, color=TOKENS["muted"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015)
    cbar.ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
    cbar.outline.set_visible(False)
    cbar.set_label("Subgroup F1", color=TOKENS["muted"], fontsize=8.5)
    fig.tight_layout(rect=[0.06, 0.05, 0.98, 0.86])
    save_figure(fig, path_base)


def plot_combined_heatmaps(
    chart_rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
    path_base: Path,
    title: str,
    subtitle: str,
    min_n: int,
) -> None:
    fig_height = 8.8 if len(columns) <= 8 else 9.4
    fig, axes = plt.subplots(3, 1, figsize=(11.6, fig_height), facecolor=TOKENS["surface"], sharex=False)
    image = None
    for ax, task in zip(axes, TASKS):
        values, support, positives, negatives = matrix_for_task(chart_rows, task, columns)
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap=F1_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(columns)), [label for *_rest, label in columns], rotation=0)
        ax.set_yticks(range(len(FOUNDATION_MODELS)), [MODEL_LABELS[model] for model in FOUNDATION_MODELS])
        ax.text(-0.07, 1.03, TASK_LABELS[task], transform=ax.transAxes, ha="left", va="bottom", fontsize=11.0, color=TOKENS["ink"], fontweight="semibold")
        style_axis(ax)
        add_group_rules(ax, columns)
        annotate_heatmap(ax, values, support, positives, negatives, min_n=min_n, fontsize=7.2 if len(columns) <= 8 else 6.3)
        ax.set_xticklabels([label for *_rest, label in columns], fontsize=8.2 if len(columns) <= 8 else 7.3)

    fig.text(0.065, 0.973, title, ha="left", va="top", fontsize=15.0, color=TOKENS["ink"], fontweight="semibold")
    fig.text(0.065, 0.942, subtitle, ha="left", va="top", fontsize=9.0, color=TOKENS["muted"])
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
        cbar.ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
        cbar.outline.set_visible(False)
        cbar.set_label("Subgroup F1", color=TOKENS["muted"], fontsize=8.5)
    if min_n and any(
        int_or_zero(row["n"]) > 0 and int_or_zero(row["n"]) < min_n
        for row in chart_rows
    ):
        fig.text(0.065, 0.025, f"Dashed orange cells have subgroup support n < {min_n}.", ha="left", va="center", fontsize=7.8, color=TOKENS["muted"])
    if any(str(row.get("f1_defined")) == "False" for row in chart_rows):
        fig.text(0.065, 0.047, "Cells labeled 0+ or 0- lack positive or negative cases, so positive-class F1 is not interpreted.", ha="left", va="center", fontsize=7.8, color=TOKENS["muted"])
    fig.subplots_adjust(left=0.18, right=0.91, top=0.89, bottom=0.07, hspace=0.36)
    save_figure(fig, path_base)


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def summarize_spread(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    undefined_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row["task"]), str(row["model_name"]))
        if not bool_from_value(row.get("f1_defined")):
            undefined_counts[key] += 1
            continue
        value = float_or_nan(row.get("f1_plot"))
        if math.isnan(value):
            undefined_counts[key] += 1
            continue
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for task in TASKS:
        for model in FOUNDATION_MODELS:
            key = (task, model)
            model_rows = grouped.get(key, [])
            values = [float_or_nan(row["f1_plot"]) for row in model_rows]
            values = [value for value in values if not math.isnan(value)]
            if values:
                ordered = sorted(model_rows, key=lambda item: (float_or_nan(item["f1_plot"]), -int_or_zero(item["n"])))
                worst = ordered[0]
                best = max(model_rows, key=lambda item: (float_or_nan(item["f1_plot"]), int_or_zero(item["n"])))
                min_f1 = float_or_nan(worst["f1_plot"])
                max_f1 = float_or_nan(best["f1_plot"])
                median_f1 = float(np.median(np.asarray(values, dtype=float)))
                spread = max_f1 - min_f1
                worst_label = str(worst["subgroup_label"])
                best_label = str(best["subgroup_label"])
                worst_n = int_or_zero(worst["n"])
                worst_n_positive = int_or_zero(worst["n_positive"])
                worst_n_negative = int_or_zero(worst["n_negative"])
            else:
                min_f1 = max_f1 = median_f1 = spread = float("nan")
                worst_label = best_label = ""
                worst_n = worst_n_positive = worst_n_negative = 0
            output.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model_name": model,
                    "model_label": MODEL_LABELS[model],
                    "min_f1": min_f1,
                    "median_f1": median_f1,
                    "max_f1": max_f1,
                    "f1_spread": spread,
                    "defined_subgroups": len(values),
                    "undefined_subgroups": undefined_counts.get(key, 0),
                    "worst_subgroup_label": worst_label,
                    "best_subgroup_label": best_label,
                    "worst_n": worst_n,
                    "worst_n_positive": worst_n_positive,
                    "worst_n_negative": worst_n_negative,
                }
            )
    return output


def plot_spread(
    spread_rows: list[dict[str, Any]],
    path_base: Path,
    title: str,
    subtitle: str,
) -> None:
    by_key = {(row["task"], row["model_name"]): row for row in spread_rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 5.7), facecolor=TOKENS["surface"], sharey=True)
    y_positions = np.arange(len(FOUNDATION_MODELS))

    for ax, task in zip(axes, TASKS):
        style_axis(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(FOUNDATION_MODELS) - 0.3, -0.7)
        ax.set_title(TASK_LABELS[task], fontsize=11.2, color=TOKENS["ink"], fontweight="semibold", pad=10)
        ax.set_xlabel("Subgroup F1", fontsize=8.8, color=TOKENS["muted"], labelpad=8)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0, 1, 6)], fontsize=8.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([MODEL_LABELS[model] for model in FOUNDATION_MODELS], fontsize=8.2)

        for y, model in enumerate(FOUNDATION_MODELS):
            row = by_key.get((task, model))
            if row is None:
                continue
            min_f1 = float_or_nan(row.get("min_f1"))
            median_f1 = float_or_nan(row.get("median_f1"))
            max_f1 = float_or_nan(row.get("max_f1"))
            if math.isnan(min_f1) or math.isnan(median_f1) or math.isnan(max_f1):
                ax.text(0.5, y, "NA", ha="center", va="center", fontsize=7.2, color=TOKENS["muted"])
                continue

            ax.hlines(y, min_f1, max_f1, color="#8FA7D6", linewidth=5.0, alpha=0.9, zorder=1)
            ax.scatter([median_f1], [y], s=38, color=TOKENS["ink"], edgecolor=TOKENS["panel"], linewidth=0.8, zorder=3)
            ax.scatter([min_f1], [y], s=30, color="#CC6F47", edgecolor=TOKENS["panel"], linewidth=0.8, zorder=4)

            worst_label = str(row.get("worst_subgroup_label", ""))
            if worst_label:
                label = f"{worst_label} {min_f1:.2f}"
                if min_f1 <= 0.18:
                    text_x = min(min_f1 + 0.035, 0.96)
                    ha = "left"
                else:
                    text_x = max(min_f1 - 0.035, 0.02)
                    ha = "right"
                ax.text(text_x, y, label, ha=ha, va="center", fontsize=6.4, color=TOKENS["muted"])

    legend_handles = [
        Line2D([0], [0], color="#8FA7D6", linewidth=5.0, label="Min-max subgroup F1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TOKENS["ink"], markeredgecolor=TOKENS["panel"], markersize=6.0, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CC6F47", markeredgecolor=TOKENS["panel"], markersize=6.0, label="Worst subgroup"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False, fontsize=8.2)
    fig.text(0.06, 0.98, title, ha="left", va="top", fontsize=15.0, color=TOKENS["ink"], fontweight="semibold")
    fig.text(0.06, 0.942, subtitle, ha="left", va="top", fontsize=9.0, color=TOKENS["muted"])
    fig.text(
        0.06,
        0.025,
        "Intervals use defined positive-class F1 across race, gender, and age strata; no-positive/no-negative cells are excluded from spreads.",
        ha="left",
        va="center",
        fontsize=7.8,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.86, bottom=0.16, wspace=0.16)
    save_figure(fig, path_base)


def summarize_worst_cells(rows: list[dict[str, Any]], limit_per_task: int = 20) -> list[dict[str, Any]]:
    worst: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not bool_from_value(row.get("f1_defined")):
            continue
        value = float_or_nan(row["f1_plot"])
        if math.isnan(value):
            continue
        by_task[str(row["task"])].append(row)
    for task, task_rows in by_task.items():
        for rank, row in enumerate(sorted(task_rows, key=lambda item: (float_or_nan(item["f1_local"]), -int_or_zero(item["n"])))[:limit_per_task], start=1):
            worst.append(
                {
                    "task": task,
                    "rank": rank,
                    "model_name": row["model_name"],
                    "model_label": row["model_label"],
                    "attribute": row["attribute"],
                    "subgroup": row["subgroup"],
                    "subgroup_label": row["subgroup_label"],
                    "f1_local": row["f1_local"],
                    "n": row["n"],
                    "n_positive": row["n_positive"],
                    "n_negative": row["n_negative"],
                    "unstable": row["unstable"],
                }
            )
    return worst


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input_csv)
    indexed = index_rows(rows)
    core_rows = chart_rows_for_columns(indexed, CORE_COLUMNS, args.min_n)
    spread_rows = summarize_spread(core_rows)
    write_csv(args.out_dir / "foundation_subgroup_f1_core.csv", core_rows)
    write_csv(args.out_dir / "foundation_subgroup_f1_core_worst_cells.csv", summarize_worst_cells(core_rows))
    write_csv(args.out_dir / "foundation_subgroup_f1_spread.csv", spread_rows)

    plot_spread(
        spread_rows,
        args.out_dir / "foundation_subgroup_f1_spread",
        "Foundation models have reliability distributions, not single reliability levels",
        "Min-max and median F1 across core race, gender, and age strata; labels mark each model's lowest defined subgroup.",
    )
    plot_combined_heatmaps(
        core_rows,
        CORE_COLUMNS,
        args.out_dir / "foundation_subgroup_f1_core_heatmaps",
        "Foundation-model F1 varies sharply by subgroup",
        "FairVision test-set subgroup F1 for foundation models; rows are models, columns are race, gender, and age strata.",
        args.min_n,
    )
    for task in TASKS:
        plot_task_heatmap(
            core_rows,
            task,
            CORE_COLUMNS,
            args.out_dir / f"foundation_subgroup_f1_{task}_core_heatmap",
            f"{TASK_LABELS[task]} subgroup F1 across foundation models",
            "Race, gender, and age-group F1 on the FairVision test split.",
            args.min_n,
            (10.8, 4.9),
        )

    for attribute, columns in INTERSECTION_SPECS.items():
        intersection_rows = chart_rows_for_columns(indexed, columns, args.min_n)
        write_csv(args.out_dir / f"foundation_subgroup_f1_{attribute}.csv", intersection_rows)
        write_csv(args.out_dir / f"foundation_subgroup_f1_{attribute}_worst_cells.csv", summarize_worst_cells(intersection_rows))
        plot_combined_heatmaps(
            intersection_rows,
            columns,
            args.out_dir / f"foundation_subgroup_f1_{attribute}_heatmaps",
            f"Foundation-model F1 across {attribute.replace('_', ' ')} strata",
            "FairVision test-set subgroup F1 for foundation models; low-support cells are most useful for reliability auditing.",
            args.min_n,
        )

    print(f"input={args.input_csv}")
    print(f"out_dir={args.out_dir}")
    for path in sorted(args.out_dir.glob("*.png")):
        print(f"figure={path}")
    for path in sorted(args.out_dir.glob("*.csv")):
        print(f"table={path}")


if __name__ == "__main__":
    main()

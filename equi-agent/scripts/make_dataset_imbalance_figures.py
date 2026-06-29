from __future__ import annotations

import argparse
import ast
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


FAIRVISION_TASKS = {
    "AMD": {
        "path": Path("OphthalmicAgent/data/AMD/data_summary_amd.csv"),
        "label_column": "amd",
        "positive": lambda value: str(value).strip().lower() != "normal",
    },
    "DR": {
        "path": Path("OphthalmicAgent/data/DR/data_summary_dr.csv"),
        "label_column": "dr",
        "positive": lambda value: str(value).strip().lower() == "vision threatening dr",
    },
    "Glaucoma": {
        "path": Path("OphthalmicAgent/data/Glaucoma/data_summary_glaucoma.csv"),
        "label_column": "glaucoma",
        "positive": lambda value: str(value).strip().lower() in {"yes", "1", "true"},
    },
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}
LABEL_COLORS = {
    "Negative": COLORS["blue"]["base"],
    "Positive": COLORS["orange"]["base"],
}
LABEL_EDGES = {
    "Negative": COLORS["blue"]["dark"],
    "Positive": COLORS["orange"]["dark"],
}
RACE_ORDER = ["white", "black", "asian", "other/missing"]
GENDER_ORDER = ["female", "male", "other/missing"]
AGE_ORDER = ["younger", "middle-aged", "older", "missing"]
ATTRIBUTE_ORDERS = {"race": RACE_ORDER, "gender": GENDER_ORDER, "age_group": AGE_ORDER}
DEMOGRAPHIC_COLORS = {
    "white": COLORS["blue"]["base"],
    "black": COLORS["orange"]["base"],
    "asian": COLORS["olive"]["base"],
    "female": "#B7A6D9",
    "male": "#8EC6B0",
    "younger": COLORS["gold"]["base"],
    "middle-aged": COLORS["olive"]["base"],
    "older": COLORS["orange"]["base"],
    "other/missing": COLORS["neutral"]["base"],
    "missing": COLORS["neutral"]["base"],
}
DEMOGRAPHIC_EDGES = {
    "white": COLORS["blue"]["dark"],
    "black": COLORS["orange"]["dark"],
    "asian": COLORS["olive"]["dark"],
    "female": "#6B5A91",
    "male": "#3F7464",
    "younger": COLORS["gold"]["dark"],
    "middle-aged": COLORS["olive"]["dark"],
    "older": COLORS["orange"]["dark"],
    "other/missing": COLORS["neutral"]["dark"],
    "missing": COLORS["neutral"]["dark"],
}
PIE_COLORS = {
    "Negative": "#A9BFF0",
    "Positive": "#E89A74",
    "white": "#B9CBF0",
    "black": "#D98A63",
    "asian": "#A6C97D",
    "female": "#B7A6D9",
    "male": "#8EC6B0",
    "younger": "#F0D968",
    "middle-aged": "#9FCB79",
    "older": "#E6956F",
    "other/missing": "#D6D9DE",
    "missing": "#D6D9DE",
}
PIE_EDGES = {
    "Negative": "#334D83",
    "Positive": "#85472C",
    "white": "#334D83",
    "black": "#85472C",
    "asian": "#436E22",
    "female": "#6B5A91",
    "male": "#3F7464",
    "younger": "#736422",
    "middle-aged": "#436E22",
    "older": "#85472C",
    "other/missing": "#5C626D",
    "missing": "#5C626D",
}
ATTRIBUTE_LABELS = {
    "disease_label": "Positive vs negative",
    "race": "Race",
    "gender": "Gender",
    "age_group": "Age group",
}
ATTRIBUTE_SHORT_LABELS = {
    "disease_label": {
        "Negative": "Neg",
        "Positive": "Pos",
    },
    "race": {
        "white": "White",
        "black": "Black",
        "asian": "Asian",
        "other/missing": "Other",
    },
    "gender": {
        "female": "Female",
        "male": "Male",
        "other/missing": "Other",
    },
    "age_group": {
        "younger": "Young",
        "middle-aged": "Middle",
        "older": "Old",
        "missing": "Missing",
    },
}
ATTRIBUTE_TINY_LABELS = {
    "disease_label": {
        "Negative": "Neg",
        "Positive": "Pos",
    },
    "race": {
        "white": "W",
        "black": "B",
        "asian": "A",
        "other/missing": "Other",
    },
    "gender": {
        "female": "F",
        "male": "M",
        "other/missing": "Other",
    },
    "age_group": {
        "younger": "Y",
        "middle-aged": "M",
        "older": "O",
        "missing": "Missing",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset imbalance figures for RetinAgent manuscript/slides.")
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/figures/dataset_imbalance"))
    parser.add_argument("--fairvision-amd", type=Path, default=FAIRVISION_TASKS["AMD"]["path"])
    parser.add_argument("--fairvision-dr", type=Path, default=FAIRVISION_TASKS["DR"]["path"])
    parser.add_argument("--fairvision-glaucoma", type=Path, default=FAIRVISION_TASKS["Glaucoma"]["path"])
    parser.add_argument("--gdp-summary", type=Path, default=Path("Datasets/GDP/data_summary.csv"))
    parser.add_argument(
        "--refuge-csv",
        type=Path,
        default=Path("Foundation_Models/FLAIR-main/local_data/dataframes/transferability/classification/25_REFUGE.csv"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"nan", "na", "none", "unknown"}:
        return "missing"
    return text


def norm_race(value: Any) -> str:
    text = norm(value)
    if "black" in text:
        return "black"
    if "white" in text or "caucasian" in text:
        return "white"
    if "asian" in text:
        return "asian"
    return "other/missing"


def norm_gender(value: Any) -> str:
    text = norm(value)
    if text in {"female", "male"}:
        return text
    return "other/missing"


def age_group(value: Any) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(age):
        return "missing"
    if age < 50:
        return "younger"
    if age < 70:
        return "middle-aged"
    return "older"


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def configure_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.975, title, ha="left", va="top", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.06, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def fairvision_sources(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    sources = dict(FAIRVISION_TASKS)
    sources["AMD"] = {**sources["AMD"], "path": args.fairvision_amd}
    sources["DR"] = {**sources["DR"], "path": args.fairvision_dr}
    sources["Glaucoma"] = {**sources["Glaucoma"], "path": args.fairvision_glaucoma}
    return sources


def summarize_fairvision(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    demo_rows: list[dict[str, Any]] = []
    raw_counts: dict[str, Any] = {}
    for task, spec in fairvision_sources(args).items():
        rows = read_rows(spec["path"])
        label_col = spec["label_column"]
        counts = Counter("Positive" if spec["positive"](row.get(label_col, "")) else "Negative" for row in rows)
        raw_counts[task] = {"label": counts, "n": len(rows), "path": str(spec["path"])}
        for label in ["Negative", "Positive"]:
            label_rows.append(
                {
                    "dataset": "FairVision",
                    "task": task,
                    "attribute": "disease_label",
                    "subgroup": label,
                    "n": counts[label],
                    "pct": pct(counts[label], len(rows)),
                    "source_file": str(spec["path"]),
                }
            )

        attr_counts = {
            "race": Counter(norm_race(row.get("race")) for row in rows),
            "gender": Counter(norm_gender(row.get("gender")) for row in rows),
            "age_group": Counter(age_group(row.get("age")) for row in rows),
        }
        raw_counts[task]["demographics"] = attr_counts
        for attribute, order in ATTRIBUTE_ORDERS.items():
            for subgroup in order:
                n = attr_counts[attribute][subgroup]
                demo_rows.append(
                    {
                        "dataset": "FairVision",
                        "task": task,
                        "attribute": attribute,
                        "subgroup": subgroup,
                        "n": n,
                        "pct": pct(n, len(rows)),
                        "source_file": str(spec["path"]),
                    }
                )
    return label_rows, demo_rows, raw_counts


def summarize_external(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    demo_rows: list[dict[str, Any]] = []
    raw_counts: dict[str, Any] = {}

    if args.gdp_summary.exists():
        rows = read_rows(args.gdp_summary)
        gdp_tasks = [
            ("GDP glaucoma detection", "glaucoma", lambda v: norm(v) == "1"),
            ("GDP progression MD", "progression.md", lambda v: norm(v) == "1"),
            ("GDP progression TD pointwise no p cut", "progression.td_pointwise_no_p_cut", lambda v: norm(v) == "1"),
        ]
        for task, column, positive_fn in gdp_tasks:
            valid = [row for row in rows if norm(row.get(column)) in {"0", "1"}]
            counts = Counter("Positive" if positive_fn(row.get(column)) else "Negative" for row in valid)
            raw_counts[task] = {"label": counts, "n": len(valid), "path": str(args.gdp_summary)}
            for label in ["Negative", "Positive"]:
                label_rows.append(
                    {
                        "dataset": "GDP",
                        "task": task,
                        "attribute": "disease_label",
                        "subgroup": label,
                        "n": counts[label],
                        "pct": pct(counts[label], len(valid)),
                        "source_file": str(args.gdp_summary),
                    }
                )

        attr_counts = {
            "race": Counter(norm_race(row.get("race")) for row in rows),
            "gender": Counter(norm_gender(row.get("gender")) for row in rows),
            "age_group": Counter(age_group(row.get("age")) for row in rows),
        }
        raw_counts["GDP demographics"] = {"demographics": attr_counts, "n": len(rows), "path": str(args.gdp_summary)}
        for attribute, order in ATTRIBUTE_ORDERS.items():
            for subgroup in order:
                n = attr_counts[attribute][subgroup]
                demo_rows.append(
                    {
                        "dataset": "GDP",
                        "task": "GDP overall",
                        "attribute": attribute,
                        "subgroup": subgroup,
                        "n": n,
                        "pct": pct(n, len(rows)),
                        "source_file": str(args.gdp_summary),
                    }
                )

    if args.refuge_csv.exists():
        rows = read_rows(args.refuge_csv)
        counts: Counter[str] = Counter()
        for row in rows:
            raw = row.get("categories", "")
            try:
                categories = [str(item).lower() for item in ast.literal_eval(raw)]
            except (SyntaxError, ValueError):
                categories = [str(raw).lower()]
            is_positive = any(category == "glaucoma" for category in categories)
            counts["Positive" if is_positive else "Negative"] += 1
        raw_counts["REFUGE glaucoma"] = {"label": counts, "n": len(rows), "path": str(args.refuge_csv)}
        for label in ["Negative", "Positive"]:
            label_rows.append(
                {
                    "dataset": "REFUGE",
                    "task": "REFUGE glaucoma",
                    "attribute": "disease_label",
                    "subgroup": label,
                    "n": counts[label],
                    "pct": pct(counts[label], len(rows)),
                    "source_file": str(args.refuge_csv),
                }
            )

    return label_rows, demo_rows, raw_counts


def plot_label_imbalance(label_rows: list[dict[str, Any]], tasks: list[str], title: str, subtitle: str, path_base: Path) -> None:
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"Negative": 0, "Positive": 0})
    totals: dict[str, int] = defaultdict(int)
    for row in label_rows:
        task = str(row["task"])
        label = str(row["subgroup"])
        by_task[task][label] += int(row["n"])
        totals[task] += int(row["n"])

    fig_height = max(3.2, 0.58 * len(tasks) + 1.8)
    fig, ax = plt.subplots(figsize=(9.4, fig_height), facecolor=TOKENS["surface"])
    y = np.arange(len(tasks))
    left = np.zeros(len(tasks))
    for label in ["Negative", "Positive"]:
        values = np.array([pct(by_task[task][label], totals[task]) for task in tasks])
        bars = ax.barh(
            y,
            values,
            left=left,
            color=LABEL_COLORS[label],
            edgecolor=LABEL_EDGES[label],
            linewidth=1.0,
            label=label,
        )
        for idx, bar in enumerate(bars):
            count = by_task[tasks[idx]][label]
            value = values[idx]
            x = left[idx] + value / 2
            if value >= 8:
                ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%\n(n={count:,})", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
            else:
                ax.text(left[idx] + value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}% (n={count:,})", ha="left", va="center", fontsize=8, color=TOKENS["ink"])
        left += values
    ax.set_yticks(y, tasks)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.set_xlabel("Share of labeled examples")
    configure_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, title, subtitle)
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.86])
    save_figure(fig, path_base)


def plot_demographic_representation(demo_rows: list[dict[str, Any]], tasks: list[str], title: str, subtitle: str, path_base: Path) -> None:
    by_key: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in demo_rows:
        task = str(row["task"])
        attr = str(row["attribute"])
        subgroup = str(row["subgroup"])
        n = int(row["n"])
        by_key[(task, attr, subgroup)] += n
        totals[(task, attr)] += n

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), facecolor=TOKENS["surface"], sharey=True)
    y = np.arange(len(tasks))
    for ax, attr, panel_title in zip(axes, ["race", "gender", "age_group"], ["Race", "Gender", "Age group"]):
        left = np.zeros(len(tasks))
        for subgroup in ATTRIBUTE_ORDERS[attr]:
            values = np.array([pct(by_key[(task, attr, subgroup)], totals[(task, attr)]) for task in tasks])
            bars = ax.barh(
                y,
                values,
                left=left,
                color=DEMOGRAPHIC_COLORS[subgroup],
                edgecolor=DEMOGRAPHIC_EDGES[subgroup],
                linewidth=1.0,
                label=subgroup,
            )
            for idx, bar in enumerate(bars):
                value = values[idx]
                if value >= 12:
                    ax.text(left[idx] + value / 2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
            left += values
        ax.set_title(panel_title, fontsize=10.5, color=TOKENS["ink"], loc="left", pad=8)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
        ax.set_xlabel("Share")
        configure_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, ncol=2, borderaxespad=0, fontsize=8)
    axes[0].set_yticks(y, tasks)
    axes[0].invert_yaxis()
    add_header(fig, title, subtitle)
    fig.tight_layout(rect=[0.04, 0.16, 0.99, 0.84], w_pad=2.0)
    save_figure(fig, path_base)


def plot_task_slide(task: str, label_rows: list[dict[str, Any]], demo_rows: list[dict[str, Any]], path_base: Path) -> None:
    task_label_rows = [row for row in label_rows if row["task"] == task]
    task_demo_rows = [row for row in demo_rows if row["task"] == task]
    if not task_label_rows or not task_demo_rows:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.7), facecolor=TOKENS["surface"])
    panels = [
        ("Disease label", "disease_label", ["Negative", "Positive"], LABEL_COLORS, LABEL_EDGES),
        ("Race", "race", RACE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Gender", "gender", GENDER_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Age group", "age_group", AGE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
    ]
    rows_by_attr = defaultdict(list)
    for row in task_label_rows + task_demo_rows:
        rows_by_attr[row["attribute"]].append(row)

    for ax, (panel_title, attr, order, colors, edges) in zip(axes, panels):
        attr_rows = rows_by_attr[attr]
        n_by_group = {str(row["subgroup"]): int(row["n"]) for row in attr_rows}
        total = sum(n_by_group.values())
        left = 0.0
        for subgroup in order:
            value = pct(n_by_group.get(subgroup, 0), total)
            ax.barh(
                [0],
                [value],
                left=[left],
                color=colors[subgroup],
                edgecolor=edges[subgroup],
                linewidth=1.0,
                label=subgroup,
            )
            if value >= 12:
                ax.text(left + value / 2, 0, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
            left += value
        ax.set_title(panel_title, fontsize=10.5, color=TOKENS["ink"], loc="left", pad=8)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
        configure_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20), frameon=False, fontsize=7.5, ncol=1, borderaxespad=0)
    total_n = sum(int(row["n"]) for row in task_label_rows)
    add_header(
        fig,
        f"{task} data imbalance",
        f"FairVision full metadata, n={total_n:,}; label prevalence and subgroup representation use the same age bins as the pipeline.",
    )
    fig.tight_layout(rect=[0.04, 0.24, 0.99, 0.80], w_pad=1.8)
    save_figure(fig, path_base)


def plot_custom_slide(
    title: str,
    subtitle: str,
    label_rows: list[dict[str, Any]],
    demo_rows: list[dict[str, Any]],
    path_base: Path,
) -> None:
    if not label_rows or not demo_rows:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.7), facecolor=TOKENS["surface"])
    panels = [
        ("Disease label", "disease_label", ["Negative", "Positive"], LABEL_COLORS, LABEL_EDGES),
        ("Race", "race", RACE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Gender", "gender", GENDER_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Age group", "age_group", AGE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
    ]
    rows_by_attr = defaultdict(list)
    for row in label_rows + demo_rows:
        rows_by_attr[row["attribute"]].append(row)

    for ax, (panel_title, attr, order, colors, edges) in zip(axes, panels):
        attr_rows = rows_by_attr[attr]
        n_by_group = {str(row["subgroup"]): int(row["n"]) for row in attr_rows}
        total = sum(n_by_group.values())
        left = 0.0
        for subgroup in order:
            value = pct(n_by_group.get(subgroup, 0), total)
            ax.barh(
                [0],
                [value],
                left=[left],
                color=colors[subgroup],
                edgecolor=edges[subgroup],
                linewidth=1.0,
                label=subgroup,
            )
            if value >= 12:
                ax.text(left + value / 2, 0, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=TOKENS["ink"])
            left += value
        ax.set_title(panel_title, fontsize=10.5, color=TOKENS["ink"], loc="left", pad=8)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
        configure_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20), frameon=False, fontsize=7.5, ncol=1, borderaxespad=0)
    add_header(fig, title, subtitle)
    fig.tight_layout(rect=[0.04, 0.24, 0.99, 0.80], w_pad=1.8)
    save_figure(fig, path_base)


def rows_for_task(rows: list[dict[str, Any]], dataset: str, task: str, attribute: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("dataset")) == dataset
        and str(row.get("task")) == task
        and str(row.get("attribute")) == attribute
    ]


def count_map(rows: list[dict[str, Any]], order: list[str]) -> tuple[dict[str, int], int]:
    counts = {key: 0 for key in order}
    for row in rows:
        subgroup = str(row["subgroup"])
        if subgroup in counts:
            counts[subgroup] += int(row["n"])
    return counts, sum(counts.values())


def draw_fingerprint_bar(
    ax: plt.Axes,
    counts: dict[str, int],
    order: list[str],
    color_map: dict[str, str],
    edge_map: dict[str, str],
) -> None:
    total = sum(counts.values())
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    for tick in [25, 50, 75]:
        ax.axvline(tick, ymin=0.27, ymax=0.73, color=TOKENS["grid"], linewidth=0.8, zorder=0)
    ax.add_patch(plt.Rectangle((0, -0.18), 100, 0.36, facecolor="#F8F9FC", edgecolor=TOKENS["axis"], linewidth=0.7, zorder=0.2))

    left = 0.0
    for subgroup in order:
        value = pct(counts.get(subgroup, 0), total)
        if value <= 0:
            continue
        ax.barh(
            [0],
            [value],
            left=[left],
            height=0.36,
            color=color_map[subgroup],
            edgecolor=edge_map[subgroup],
            linewidth=1.0,
            zorder=2,
        )
        center = left + value / 2
        if value >= 11:
            ax.text(center, 0, f"{value:.0f}%", ha="center", va="center", fontsize=8.5, color=TOKENS["ink"], zorder=3)
        elif value >= 5:
            ax.text(center, -0.31, f"{value:.0f}%", ha="center", va="center", fontsize=7.5, color=TOKENS["muted"], zorder=3)
        left += value


def add_fingerprint_legend(fig: plt.Figure) -> None:
    from matplotlib.patches import Patch

    legend_specs = [
        ("Label", [("Negative", LABEL_COLORS["Negative"], LABEL_EDGES["Negative"]), ("Positive", LABEL_COLORS["Positive"], LABEL_EDGES["Positive"])]),
        ("Race", [("White", DEMOGRAPHIC_COLORS["white"], DEMOGRAPHIC_EDGES["white"]), ("Black", DEMOGRAPHIC_COLORS["black"], DEMOGRAPHIC_EDGES["black"]), ("Asian", DEMOGRAPHIC_COLORS["asian"], DEMOGRAPHIC_EDGES["asian"])]),
        ("Gender", [("Female", DEMOGRAPHIC_COLORS["female"], DEMOGRAPHIC_EDGES["female"]), ("Male", DEMOGRAPHIC_COLORS["male"], DEMOGRAPHIC_EDGES["male"])]),
        ("Age", [("Young", DEMOGRAPHIC_COLORS["younger"], DEMOGRAPHIC_EDGES["younger"]), ("Middle", DEMOGRAPHIC_COLORS["middle-aged"], DEMOGRAPHIC_EDGES["middle-aged"]), ("Old", DEMOGRAPHIC_COLORS["older"], DEMOGRAPHIC_EDGES["older"])]),
    ]
    x_positions = [0.18, 0.39, 0.61, 0.81]
    for (title, items), x in zip(legend_specs, x_positions):
        handles = [Patch(facecolor=color, edgecolor=edge, linewidth=1.0, label=label) for label, color, edge in items]
        fig.text(x, 0.855, title, ha="left", va="center", fontsize=8.5, color=TOKENS["muted"], fontweight="semibold")
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(x, 0.84),
            frameon=False,
            ncol=len(handles),
            fontsize=8.0,
            handlelength=1.5,
            columnspacing=0.9,
        )


def plot_imbalance_fingerprint(
    fair_label_rows: list[dict[str, Any]],
    fair_demo_rows: list[dict[str, Any]],
    external_label_rows: list[dict[str, Any]],
    external_demo_rows: list[dict[str, Any]],
    path_base: Path,
) -> None:
    rows = [
        ("FairVision", "AMD", "FairVision AMD"),
        ("FairVision", "DR", "FairVision DR"),
        ("FairVision", "Glaucoma", "FairVision glaucoma"),
        ("GDP", "GDP glaucoma detection", "GDP glaucoma"),
    ]
    columns = [
        ("Disease label", "disease_label", ["Negative", "Positive"], LABEL_COLORS, LABEL_EDGES),
        ("Race", "race", RACE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Gender", "gender", GENDER_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
        ("Age group", "age_group", AGE_ORDER, DEMOGRAPHIC_COLORS, DEMOGRAPHIC_EDGES),
    ]

    fig = plt.figure(figsize=(14.4, 7.2), facecolor=TOKENS["surface"])
    grid = fig.add_gridspec(
        nrows=len(rows) + 1,
        ncols=len(columns) + 1,
        width_ratios=[1.52, 2.2, 2.2, 2.2, 2.2],
        height_ratios=[0.34, 1.0, 1.0, 1.0, 1.0],
        hspace=0.38,
        wspace=0.15,
        left=0.06,
        right=0.98,
        top=0.76,
        bottom=0.10,
    )
    add_header(
        fig,
        "Imbalance fingerprints reveal uneven support across retina benchmarks",
        "Each bar sums to 100%. Disease labels, race, gender, and age support are shown side by side; age bins use young <50, middle 50-69, old 70+.",
    )
    add_fingerprint_legend(fig)

    for col_idx, (title, *_rest) in enumerate(columns, start=1):
        ax = fig.add_subplot(grid[0, col_idx])
        ax.axis("off")
        ax.text(0, 0.10, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=TOKENS["ink"], fontweight="semibold")

    for row_idx, (dataset, task, label) in enumerate(rows):
        grid_row = row_idx + 1
        label_ax = fig.add_subplot(grid[grid_row, 0])
        label_ax.axis("off")
        label_ax.text(0.0, 0.58, label, ha="left", va="center", fontsize=11.0, color=TOKENS["ink"], fontweight="semibold")
        label_source = fair_label_rows if dataset == "FairVision" else external_label_rows
        n_total = sum(int(row["n"]) for row in rows_for_task(label_source, dataset, task, "disease_label"))
        label_ax.text(0.0, 0.22, f"n={n_total:,}", ha="left", va="center", fontsize=8.5, color=TOKENS["muted"])

        for col_idx, (title, attribute, order, colors, edges) in enumerate(columns, start=1):
            source_rows = fair_label_rows if attribute == "disease_label" and dataset == "FairVision" else None
            if attribute == "disease_label" and dataset != "FairVision":
                source_rows = external_label_rows
            if attribute != "disease_label" and dataset == "FairVision":
                source_rows = fair_demo_rows
            if attribute != "disease_label" and dataset != "FairVision":
                source_rows = external_demo_rows

            attr_rows = rows_for_task(source_rows or [], dataset, task if attribute == "disease_label" else ("GDP overall" if dataset == "GDP" else task), attribute)
            counts, total = count_map(attr_rows, order)
            ax = fig.add_subplot(grid[grid_row, col_idx])
            draw_fingerprint_bar(ax, counts, order, colors, edges)

    save_figure(fig, path_base)


def distribution_text(attribute: str, counts: dict[str, int], order: list[str]) -> str:
    total = sum(counts[item] for item in order)
    pieces = []
    for item in order:
        value = counts.get(item, 0)
        if value <= 0:
            continue
        label = ATTRIBUTE_TINY_LABELS[attribute].get(item, pretty_label(item))
        pieces.append(f"{label} {pct(value, total):.0f}")
    return " / ".join(pieces)


def draw_compact_distribution_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    attribute: str,
    counts: dict[str, int],
    order: list[str],
) -> None:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    total = sum(counts[item] for item in order)
    if total <= 0:
        ax.text(x + width / 2, y + height / 2, "No data", ha="center", va="center", fontsize=7, color=TOKENS["muted"])
        return

    dominant = max(order, key=lambda item: counts.get(item, 0))
    dominant_pct = pct(counts[dominant], total)
    dominant_label = ATTRIBUTE_SHORT_LABELS[attribute].get(dominant, pretty_label(dominant))

    ax.text(
        x,
        y + height * 0.70,
        f"{dominant_pct:.0f}% {dominant_label}",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="semibold",
        color=TOKENS["ink"],
    )

    bar_x = x
    bar_y = y + height * 0.36
    bar_w = width
    bar_h = height * 0.17
    clip = FancyBboxPatch(
        (bar_x, bar_y),
        bar_w,
        bar_h,
        boxstyle=f"round,pad=0,rounding_size={bar_h / 2}",
        facecolor="#F1F3F8",
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(clip)

    left = bar_x
    for item in order:
        value = counts.get(item, 0)
        if value <= 0:
            continue
        segment_w = bar_w * value / total
        segment = Rectangle(
            (left, bar_y),
            segment_w,
            bar_h,
            facecolor=PIE_COLORS[item],
            edgecolor="none",
            linewidth=0,
        )
        segment.set_clip_path(clip)
        ax.add_patch(segment)
        left += segment_w

    outline = FancyBboxPatch(
        (bar_x, bar_y),
        bar_w,
        bar_h,
        boxstyle=f"round,pad=0,rounding_size={bar_h / 2}",
        facecolor="none",
        edgecolor="#DADDE7",
        linewidth=0.8,
    )
    ax.add_patch(outline)

    ax.text(
        x,
        y + height * 0.15,
        distribution_text(attribute, counts, order),
        ha="left",
        va="center",
        fontsize=6.7,
        color=TOKENS["muted"],
    )


def plot_compact_imbalance_matrix(
    fair_label_rows: list[dict[str, Any]],
    fair_demo_rows: list[dict[str, Any]],
    external_label_rows: list[dict[str, Any]],
    external_demo_rows: list[dict[str, Any]],
    path_base: Path,
) -> None:
    rows = [
        ("FairVision", "AMD", "FairVision AMD"),
        ("FairVision", "DR", "FairVision DR"),
        ("FairVision", "Glaucoma", "FairVision glaucoma"),
        ("GDP", "GDP glaucoma detection", "Harvard-GDP glaucoma"),
    ]
    attributes = ["disease_label", "race", "gender", "age_group"]

    fig, ax = plt.subplots(figsize=(10.8, 4.05), facecolor=TOKENS["surface"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_margin = 0.245
    right_margin = 0.025
    top = 0.82
    bottom = 0.10
    usable_w = 1.0 - left_margin - right_margin
    col_w = usable_w / len(attributes)
    row_h = (top - bottom) / len(rows)

    ax.text(0.035, 0.91, "Benchmark", ha="left", va="center", fontsize=9.0, fontweight="semibold", color=TOKENS["muted"])
    for col_idx, attribute in enumerate(attributes):
        col_x = left_margin + col_idx * col_w
        ax.text(
            col_x,
            0.91,
            ATTRIBUTE_LABELS[attribute],
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
            color=TOKENS["ink"],
        )
        ax.plot([col_x - 0.012, col_x - 0.012], [bottom - 0.01, top + 0.04], color="#ECEEF5", linewidth=0.8)

    ax.plot([0.035, 0.97], [0.865, 0.865], color="#E2E5EE", linewidth=1.0)

    for row_idx, (dataset_name, task_name, display_name) in enumerate(rows):
        row_top = top - row_idx * row_h
        y = row_top - row_h
        center_y = y + row_h * 0.50

        label_counts, label_order = donut_counts_for(
            dataset_name,
            task_name,
            "disease_label",
            fair_label_rows,
            fair_demo_rows,
            external_label_rows,
            external_demo_rows,
        )
        total_n = sum(label_counts[item] for item in label_order)

        ax.text(0.035, center_y + 0.026, display_name, ha="left", va="center", fontsize=8.5, fontweight="semibold", color=TOKENS["ink"])
        ax.text(0.035, center_y - 0.030, f"n={total_n:,}", ha="left", va="center", fontsize=7.5, color=TOKENS["muted"])
        if row_idx > 0:
            ax.plot([0.035, 0.97], [row_top, row_top], color="#F0F2F7", linewidth=0.8)

        for col_idx, attribute in enumerate(attributes):
            col_x = left_margin + col_idx * col_w
            counts, order = donut_counts_for(
                dataset_name,
                task_name,
                attribute,
                fair_label_rows,
                fair_demo_rows,
                external_label_rows,
                external_demo_rows,
            )
            draw_compact_distribution_cell(
                ax,
                col_x,
                y + row_h * 0.13,
                col_w * 0.86,
                row_h * 0.74,
                attribute,
                counts,
                order,
            )

    ax.text(
        0.035,
        0.035,
        "Each cell shows the dominant stratum plus the full percentage split; age bins use young <50, middle 50-69, old 70+.",
        ha="left",
        va="center",
        fontsize=7.2,
        color=TOKENS["muted"],
    )
    save_figure(fig, path_base)


def slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def pretty_label(value: str) -> str:
    labels = {
        "middle-aged": "Middle",
        "younger": "Young",
        "older": "Old",
        "other/missing": "Other/missing",
    }
    return labels.get(value, value.replace("_", " ").title())


def donut_counts_for(
    dataset: str,
    task: str,
    attribute: str,
    fair_label_rows: list[dict[str, Any]],
    fair_demo_rows: list[dict[str, Any]],
    external_label_rows: list[dict[str, Any]],
    external_demo_rows: list[dict[str, Any]],
) -> tuple[dict[str, int], list[str]]:
    if attribute == "disease_label":
        source = fair_label_rows if dataset == "FairVision" else external_label_rows
        lookup_task = task
        order = ["Negative", "Positive"]
    else:
        source = fair_demo_rows if dataset == "FairVision" else external_demo_rows
        lookup_task = "GDP overall" if dataset == "GDP" else task
        order = ATTRIBUTE_ORDERS[attribute]

    attr_rows = rows_for_task(source, dataset, lookup_task, attribute)
    counts, _ = count_map(attr_rows, order)
    order = [item for item in order if counts.get(item, 0) > 0]
    return counts, order


def plot_donut(
    ax: plt.Axes,
    counts: dict[str, int],
    order: list[str],
    title: str,
    *,
    center_text: str,
    legend: bool = True,
) -> None:
    values = [counts[item] for item in order]
    total = sum(values)
    if total == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=TOKENS["muted"])
        return

    colors = [PIE_COLORS[item] for item in order]
    edges = [PIE_EDGES[item] for item in order]

    def autopct(value: float) -> str:
        return f"{value:.0f}%" if value >= 5 else ""

    wedges, _texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=autopct,
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "linewidth": 1.15, "edgecolor": TOKENS["surface"]},
        textprops={"fontsize": 9, "color": TOKENS["ink"]},
    )
    for wedge, edge in zip(wedges, edges):
        wedge.set_edgecolor(edge)
        wedge.set_linewidth(1.15)
    for text in autotexts:
        text.set_fontweight("semibold")

    ax.text(0, 0.04, center_text, ha="center", va="center", fontsize=11, fontweight="semibold", color=TOKENS["ink"])
    ax.text(0, -0.12, "examples", ha="center", va="center", fontsize=7.5, color=TOKENS["muted"])
    ax.set_title("")
    ax.set(aspect="equal")
    ax.set_facecolor(TOKENS["surface"])

    if legend:
        labels = [f"{pretty_label(item)}  {counts[item]:,} ({pct(counts[item], total):.1f}%)" for item in order]
        ax.legend(
            wedges,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            fontsize=8.0,
            ncol=1,
            handlelength=1.2,
        )


def plot_single_donut_figure(
    dataset_name: str,
    task_name: str,
    display_name: str,
    attribute: str,
    counts: dict[str, int],
    order: list[str],
    path_base: Path,
) -> None:
    total = sum(counts[item] for item in order)
    fig, ax = plt.subplots(figsize=(4.2, 4.35), facecolor=TOKENS["surface"])
    plot_donut(ax, counts, order, "", center_text=f"n={total:,}", legend=True)
    fig.tight_layout(rect=[0.04, 0.08, 0.96, 0.98])
    save_figure(fig, path_base)


def plot_donut_panel(
    dataset_name: str,
    task_name: str,
    display_name: str,
    fair_label_rows: list[dict[str, Any]],
    fair_demo_rows: list[dict[str, Any]],
    external_label_rows: list[dict[str, Any]],
    external_demo_rows: list[dict[str, Any]],
    path_base: Path,
) -> None:
    attributes = ["disease_label", "race", "gender", "age_group"]
    label_counts, label_order = donut_counts_for(
        dataset_name,
        task_name,
        "disease_label",
        fair_label_rows,
        fair_demo_rows,
        external_label_rows,
        external_demo_rows,
    )
    total = sum(label_counts[item] for item in label_order)
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.2), facecolor=TOKENS["surface"])
    for ax, attribute in zip(axes, attributes):
        counts, order = donut_counts_for(
            dataset_name,
            task_name,
            attribute,
            fair_label_rows,
            fair_demo_rows,
            external_label_rows,
            external_demo_rows,
        )
        plot_donut(ax, counts, order, "", center_text=f"n={sum(counts[item] for item in order):,}", legend=True)
    fig.tight_layout(rect=[0.02, 0.08, 0.99, 0.98], w_pad=1.2)
    save_figure(fig, path_base)


def plot_all_donut_outputs(
    fair_label_rows: list[dict[str, Any]],
    fair_demo_rows: list[dict[str, Any]],
    external_label_rows: list[dict[str, Any]],
    external_demo_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    datasets = [
        ("FairVision", "AMD", "FairVision AMD"),
        ("FairVision", "DR", "FairVision DR"),
        ("FairVision", "Glaucoma", "FairVision glaucoma"),
        ("GDP", "GDP glaucoma detection", "GDP glaucoma"),
    ]
    pie_dir = out_dir / "pies"
    for dataset_name, task_name, display_name in datasets:
        base_name = slug(display_name)
        plot_donut_panel(
            dataset_name,
            task_name,
            display_name,
            fair_label_rows,
            fair_demo_rows,
            external_label_rows,
            external_demo_rows,
            pie_dir / f"pie_panel_{base_name}",
        )
        for attribute in ["disease_label", "race", "gender", "age_group"]:
            counts, order = donut_counts_for(
                dataset_name,
                task_name,
                attribute,
                fair_label_rows,
                fair_demo_rows,
                external_label_rows,
                external_demo_rows,
            )
            plot_single_donut_figure(
                dataset_name,
                task_name,
                display_name,
                attribute,
                counts,
                order,
                pie_dir / f"pie_{base_name}_{attribute}",
            )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fair_label_rows, fair_demo_rows, _ = summarize_fairvision(args)
    external_label_rows, external_demo_rows, _ = summarize_external(args)

    write_csv(args.out_dir / "fairvision_label_counts.csv", fair_label_rows)
    write_csv(args.out_dir / "fairvision_demographic_counts.csv", fair_demo_rows)
    write_csv(args.out_dir / "external_label_counts.csv", external_label_rows)
    write_csv(args.out_dir / "external_demographic_counts.csv", external_demo_rows)

    fair_tasks = ["AMD", "DR", "Glaucoma"]
    plot_label_imbalance(
        fair_label_rows,
        fair_tasks,
        "FairVision disease labels are not equally balanced",
        "Full FairVision metadata: AMD positive is any AMD stage; DR positive is vision-threatening DR; glaucoma positive is yes.",
        args.out_dir / "fairvision_label_imbalance",
    )
    plot_demographic_representation(
        fair_demo_rows,
        fair_tasks,
        "FairVision subgroup representation is uneven across diseases",
        "Race, gender, and age-group shares across 10,000 examples per disease. Age groups: <50, 50-69, and 70+.",
        args.out_dir / "fairvision_demographic_representation",
    )
    plot_imbalance_fingerprint(
        fair_label_rows,
        fair_demo_rows,
        external_label_rows,
        external_demo_rows,
        args.out_dir / "dataset_imbalance_fingerprint",
    )
    plot_compact_imbalance_matrix(
        fair_label_rows,
        fair_demo_rows,
        external_label_rows,
        external_demo_rows,
        args.out_dir / "dataset_imbalance_compact_matrix",
    )
    plot_all_donut_outputs(
        fair_label_rows,
        fair_demo_rows,
        external_label_rows,
        external_demo_rows,
        args.out_dir,
    )
    for task in fair_tasks:
        plot_task_slide(task, fair_label_rows, fair_demo_rows, args.out_dir / f"fairvision_{task.lower()}_imbalance")

    if external_label_rows:
        external_tasks = []
        for row in external_label_rows:
            task = str(row["task"])
            if task not in external_tasks:
                external_tasks.append(task)
        plot_label_imbalance(
            external_label_rows,
            external_tasks,
            "External glaucoma datasets also show label imbalance",
            "GDP progression rows exclude NA labels; REFUGE local classification file has labels but no demographics.",
            args.out_dir / "external_label_imbalance_gdp_refuge",
        )
    if external_demo_rows:
        gdp_total = sum(int(row["n"]) for row in external_demo_rows if row["dataset"] == "GDP" and row["attribute"] == "race")
        plot_demographic_representation(
            external_demo_rows,
            ["GDP overall"],
            "GDP subgroup representation is uneven",
            f"GDP full metadata, n={gdp_total:,} rows in local summary; REFUGE local classification file has no demographic fields.",
            args.out_dir / "gdp_demographic_representation",
        )
        gdp_label = [row for row in external_label_rows if row["task"] == "GDP glaucoma detection"]
        gdp_demo = [row for row in external_demo_rows if row["dataset"] == "GDP"]
        plot_custom_slide(
            "GDP glaucoma data imbalance",
            f"GDP full metadata, n={gdp_total:,}; glaucoma label prevalence and subgroup representation use the same age bins as FairVision.",
            gdp_label,
            gdp_demo,
            args.out_dir / "gdp_glaucoma_imbalance",
        )

    print(f"wrote={args.out_dir}")
    for path in sorted(args.out_dir.rglob("*.png")):
        print(f"figure={path}")
    for path in sorted(args.out_dir.rglob("*.csv")):
        print(f"table={path}")


if __name__ == "__main__":
    main()

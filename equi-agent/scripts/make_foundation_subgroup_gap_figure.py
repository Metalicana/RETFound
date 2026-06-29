from __future__ import annotations

import argparse
import csv
import math
import os
import re
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
from matplotlib.patches import Rectangle
import numpy as np


TASK_TABLES = {
    "amd": Path("equi-agent/manuscript/tables/fairvision_foundation_demographic_amd.tex"),
    "dr": Path("equi-agent/manuscript/tables/fairvision_foundation_demographic_dr.tex"),
    "glaucoma": Path("equi-agent/manuscript/tables/fairvision_foundation_demographic_glaucoma.tex"),
}
TASK_LABELS = {"amd": "AMD", "dr": "DR", "glaucoma": "Glaucoma"}

ATTRIBUTE_ORDER = ["race", "ethnicity", "sex_gender", "age_group"]
ATTRIBUTE_LABELS = {"race": "Race", "ethnicity": "Ethnicity", "sex_gender": "Gender", "age_group": "Age"}
SUBGROUP_ORDER = {
    "race": ["asian", "black", "white"],
    "ethnicity": ["hispanic", "non-hispanic"],
    "sex_gender": ["female", "male"],
    "age_group": ["younger", "middle-aged", "older"],
}
SUBGROUP_LABELS = {
    "asian": "Asian",
    "black": "Black",
    "white": "White",
    "hispanic": "Hispanic",
    "non-hispanic": "Non-Hispanic",
    "female": "Female",
    "male": "Male",
    "younger": "Young",
    "middle-aged": "Middle",
    "older": "Old",
}
MODEL_LABELS = {
    "RETFound OCT": "RETFound-OCT",
    "MIRAGE SLO": "MIRAGE-SLO",
    "FLAIR SLO": "FLAIR-SLO",
    "RET-CLIP SLO": "RET-CLIP-SLO",
    "VisionFM SLO": "VisionFM-SLO",
    "VisionFM OCT": "VisionFM-OCT",
    "RetiZero SLO": "RetiZero-SLO",
    "UrFound SLO": "URFound-SLO",
    "UrFound OCT": "URFound-OCT",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
ATTRIBUTE_BANDS = {
    "race": "#EAF1FE",
    "ethnicity": "#F2ECFF",
    "sex_gender": "#EAF7E3",
    "age_group": "#FFF0DE",
}
ATTRIBUTE_EDGES = {
    "race": "#9FB6E8",
    "ethnicity": "#B8A5E4",
    "sex_gender": "#A9D091",
    "age_group": "#E6AF77",
}
GAP_CMAP = LinearSegmentedColormap.from_list(
    "subgroup_gap",
    ["#F7FBF7", "#DCEFC9", "#F8DA9B", "#F0A275", "#BF5C45"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact subgroup F1-gap matrix from the manuscript per-disease "
            "foundation-model demographic tables."
        )
    )
    parser.add_argument("--amd-table", type=Path, default=TASK_TABLES["amd"])
    parser.add_argument("--dr-table", type=Path, default=TASK_TABLES["dr"])
    parser.add_argument("--glaucoma-table", type=Path, default=TASK_TABLES["glaucoma"])
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/figures/subgroup_f1_gap"))
    return parser.parse_args()


def clean_latex(value: str) -> str:
    text = value.strip()
    text = text.replace("\\_", "_")
    text = text.replace("\\%", "%")
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text.strip()


def float_or_nan(value: Any) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def int_or_zero(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


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


def parse_disease_table(path: Path, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_model = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line.endswith("\\\\") or " & " not in line:
            continue
        if line.startswith("\\") or line.startswith("Model &"):
            continue
        cells = [clean_latex(cell) for cell in line[:-2].split("&")]
        if len(cells) != 15:
            continue
        model = cells[0] or current_model
        current_model = model
        attribute = cells[1]
        subgroup = cells[2]
        if attribute not in ATTRIBUTE_ORDER:
            continue
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "attribute": attribute,
                "attribute_label": ATTRIBUTE_LABELS[attribute],
                "subgroup": subgroup,
                "subgroup_label": SUBGROUP_LABELS.get(subgroup, subgroup),
                "n": int_or_zero(cells[3]),
                "n_positive": int_or_zero(cells[4]),
                "n_negative": int_or_zero(cells[5]),
                "accuracy": float_or_nan(cells[6]),
                "f1": float_or_nan(cells[7]),
                "balanced_accuracy": float_or_nan(cells[8]),
                "sensitivity": float_or_nan(cells[9]),
                "specificity": float_or_nan(cells[10]),
                "fpr": float_or_nan(cells[11]),
                "fnr": float_or_nan(cells[12]),
                "ece": float_or_nan(cells[13]),
                "unstable": cells[14],
            }
        )
    return rows


def summarize_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["attribute"], row["subgroup"])].append(row)

    summary: list[dict[str, Any]] = []
    for task in TASK_LABELS:
        for attribute in ATTRIBUTE_ORDER:
            for subgroup in SUBGROUP_ORDER[attribute]:
                model_rows = grouped.get((task, attribute, subgroup), [])
                if not model_rows:
                    continue
                support = model_rows[0]
                values = [
                    float_or_nan(row["f1"])
                    for row in model_rows
                    if int_or_zero(row["n_positive"]) > 0 and int_or_zero(row["n_negative"]) > 0
                ]
                values = [value for value in values if not math.isnan(value)]
                if values:
                    median_f1 = float(np.median(np.asarray(values, dtype=float)))
                    min_f1 = min(values)
                    max_f1 = max(values)
                    best_row = max(
                        (
                            row
                            for row in model_rows
                            if int_or_zero(row["n_positive"]) > 0
                            and int_or_zero(row["n_negative"]) > 0
                            and not math.isnan(float_or_nan(row["f1"]))
                        ),
                        key=lambda item: float_or_nan(item["f1"]),
                    )
                    best_model = best_row["model_label"]
                    best_model_f1 = float_or_nan(best_row["f1"])
                    defined = True
                    undefined_reason = ""
                else:
                    median_f1 = min_f1 = max_f1 = best_model_f1 = float("nan")
                    best_model = ""
                    defined = False
                    undefined_reason = "no_positive_cases" if int_or_zero(support["n_positive"]) == 0 else "no_negative_cases"
                summary.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "attribute": attribute,
                        "attribute_label": ATTRIBUTE_LABELS[attribute],
                        "subgroup": subgroup,
                        "subgroup_label": SUBGROUP_LABELS[subgroup],
                        "n": support["n"],
                        "n_positive": support["n_positive"],
                        "n_negative": support["n_negative"],
                        "median_f1": median_f1,
                        "min_model_f1": min_f1,
                        "max_model_f1": max_f1,
                        "best_model": best_model,
                        "best_model_f1": best_model_f1,
                        "defined": defined,
                        "undefined_reason": undefined_reason,
                    }
                )

    best_by_task_attr: dict[tuple[str, str], float] = {}
    for row in summary:
        if not row["defined"]:
            continue
        key = (row["task"], row["attribute"])
        best_by_task_attr[key] = max(best_by_task_attr.get(key, -float("inf")), float(row["median_f1"]))

    for row in summary:
        key = (row["task"], row["attribute"])
        reference = best_by_task_attr.get(key, float("nan"))
        row["best_attribute_median_f1"] = reference
        row["median_f1_gap"] = reference - float(row["median_f1"]) if row["defined"] and not math.isnan(reference) else float("nan")
    return summary


ATTRIBUTE_STACKS = [
    ["race", "ethnicity"],
    ["sex_gender", "age_group"],
]


def figure_columns(attributes: list[str] | None = None) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for attribute in attributes or ATTRIBUTE_ORDER:
        for subgroup in SUBGROUP_ORDER[attribute]:
            columns.append((attribute, subgroup))
    return columns


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def plot_gap_matrix(summary: list[dict[str, Any]], path_base: Path) -> None:
    by_key = {(row["task"], row["attribute"], row["subgroup"]): row for row in summary}
    max_gap = max(
        [float_or_nan(row["median_f1_gap"]) for row in summary if row["defined"] and not math.isnan(float_or_nan(row["median_f1_gap"]))],
        default=0.35,
    )
    gap_vmax = max(0.35, math.ceil(max_gap * 10.0) / 10.0)
    fig, axes = plt.subplots(2, 1, figsize=(8.25, 5.2), facecolor=TOKENS["surface"])

    for ax, attributes in zip(axes, ATTRIBUTE_STACKS):
        columns = figure_columns(attributes)
        ax.set_facecolor(TOKENS["panel"])
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(len(TASK_LABELS) - 0.48, -0.98)
        ax.set_xticks(range(len(columns)), [SUBGROUP_LABELS[subgroup] for _attr, subgroup in columns], fontsize=8.3)
        ax.set_yticks(range(len(TASK_LABELS)), [TASK_LABELS[task] for task in TASK_LABELS], fontsize=9.4)
        ax.tick_params(axis="both", length=0, colors=TOKENS["muted"])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for attribute in attributes:
            indices = [idx for idx, (attr, _subgroup) in enumerate(columns) if attr == attribute]
            left = min(indices) - 0.5
            width = max(indices) - min(indices) + 1.0
            ax.axvspan(
                left,
                left + width,
                facecolor=ATTRIBUTE_BANDS[attribute],
                alpha=0.95,
                zorder=0,
            )
            ax.add_patch(
                Rectangle(
                    (left + 0.03, -0.92),
                    width - 0.06,
                    0.32,
                    facecolor=ATTRIBUTE_BANDS[attribute],
                    edgecolor=ATTRIBUTE_EDGES[attribute],
                    linewidth=1.2,
                    clip_on=False,
                    zorder=2,
                )
            )

        for idx, (attribute, _subgroup) in enumerate(columns):
            if idx and attribute != columns[idx - 1][0]:
                ax.axvline(idx - 0.5, color=TOKENS["surface"], linewidth=10.0, zorder=1)
                ax.axvline(idx - 0.5, color=TOKENS["axis"], linewidth=1.2, zorder=2)

        for task_idx, task in enumerate(TASK_LABELS):
            for col_idx, (attribute, subgroup) in enumerate(columns):
                row = by_key[(task, attribute, subgroup)]
                defined = bool(row["defined"])
                if defined:
                    gap = float_or_nan(row["median_f1_gap"])
                    color = GAP_CMAP(min(max(gap / gap_vmax, 0.0), 1.0))
                    edge = "#FFFFFF"
                    main = f"{float(row['median_f1']):.2f}"
                    sub = f"Delta {gap:.2f}" if gap > 0.004 else "best"
                    text_color = TOKENS["ink"]
                else:
                    color = "#EEF0F4"
                    edge = "#FFFFFF"
                    main = "NA"
                    sub = "0+"
                    text_color = TOKENS["muted"]
                ax.add_patch(
                    Rectangle(
                        (col_idx - 0.48, task_idx - 0.42),
                        0.96,
                        0.78,
                        facecolor=color,
                        edgecolor=edge,
                        linewidth=1.2,
                        alpha=0.84,
                        zorder=2,
                    )
                )
                ax.text(col_idx, task_idx - 0.07, main, ha="center", va="center", fontsize=8.2, color=text_color, fontweight="semibold", zorder=3)
                ax.text(col_idx, task_idx + 0.15, sub, ha="center", va="center", fontsize=6.4, color=TOKENS["muted"], zorder=3)

        for attribute in attributes:
            indices = [idx for idx, (attr, _subgroup) in enumerate(columns) if attr == attribute]
            center = (min(indices) + max(indices)) / 2
            ax.text(center, -0.76, ATTRIBUTE_LABELS[attribute], ha="center", va="center", fontsize=9.0, color=TOKENS["ink"], fontweight="semibold", zorder=3)

    sm = plt.cm.ScalarMappable(cmap=GAP_CMAP, norm=plt.Normalize(vmin=0.0, vmax=gap_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.8, colors=TOKENS["muted"], length=0)
    cbar.set_label("F1 gap from best subgroup", color=TOKENS["muted"], fontsize=8.4)

    fig.text(0.5, 0.985, "Subgroup-level F1 gaps are disease-specific", ha="center", va="top", fontsize=13.4, color=TOKENS["ink"], fontweight="semibold")
    fig.text(
        0.5,
        0.035,
        "Grey NA/0+ cells have no positive cases, so positive-class F1 is not interpreted as model failure.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.14, right=0.84, top=0.86, bottom=0.12, hspace=0.32)
    save_figure(fig, path_base)


def main() -> None:
    args = parse_args()
    tables = {
        "amd": args.amd_table,
        "dr": args.dr_table,
        "glaucoma": args.glaucoma_table,
    }
    rows: list[dict[str, Any]] = []
    for task, path in tables.items():
        rows.extend(parse_disease_table(path, task))
    summary = summarize_gaps(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "foundation_subgroup_f1_gap_source_rows.csv", rows)
    write_csv(args.out_dir / "foundation_subgroup_f1_gap_summary.csv", summary)
    plot_gap_matrix(summary, args.out_dir / "foundation_subgroup_f1_gap_matrix")

    print(f"out_dir={args.out_dir}")
    print(f"source_rows={args.out_dir / 'foundation_subgroup_f1_gap_source_rows.csv'}")
    print(f"summary={args.out_dir / 'foundation_subgroup_f1_gap_summary.csv'}")
    print(f"figure={args.out_dir / 'foundation_subgroup_f1_gap_matrix.png'}")
    print(f"figure={args.out_dir / 'foundation_subgroup_f1_gap_matrix.svg'}")


if __name__ == "__main__":
    main()

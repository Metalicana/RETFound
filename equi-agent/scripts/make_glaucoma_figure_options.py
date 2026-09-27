"""Render alternative glaucoma manuscript figures from the supplied table values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "retfound-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "retfound-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "equi-agent/manuscript/figures/glaucoma_options"
DATASETS = ("FairVision30K", "Harvard GDP", "Drishti-GS", "REFUGE2")
METRICS = ("f1", "worst_group_f1", "sensitivity", "specificity", "balanced_accuracy")
INK, GRID = "#242B30", "#E7EBED"
COLORS = {"Foundation models": "#527B9A", "LLM baselines": "#C59A56", "Ours": "#008275"}
WEAK, LOSS = "#C7CDD1", "#B14D4D"
WIDTH = 183 / 25.4
MODALITIES = {
    "FairVision30K": "OCT / SLO",
    "Harvard GDP": "OCT; Ours: OCT + RNFLT",
    "Drishti-GS": "CFP",
    "REFUGE2": "CFP",
}
SOURCE_REVISION_NOTE = (
    "Drishti-GS excludes URFound at the author's request because of the identified "
    "pretraining-overlap concern; checkpoint-specific overlap has not been established. "
    "RetiZero's Drishti-GS F1 is support-weighted (0.8187), replacing positive-class F1 "
    "(0.8732) from the same predictions. All other supplied values are unchanged."
)
COMMON_CAPTION = (
    "Point estimates follow the supplied glaucoma tables with the documented Drishti-GS revisions. No confidence "
    "intervals or significance claims are inferred from aggregate values. F1 averaging "
    "definitions, evaluation cohorts and adaptation protocols must be reconciled before "
    "publication; F1 is shown as reported, without averaging across datasets. Model "
    "modalities follow the source tables and are not identical for every comparator. "
    + SOURCE_REVISION_NOTE
)
OPTIONS = [
    ("01_f1_bars", "F1 benchmark", "Included methods, ranked within each cohort",
     "Glaucoma F1 comparison. Horizontal bars show every included method, sorted by "
     "reported F1 within each dataset, on a common zero-based scale. Color identifies "
     "foundation models, LLM baselines and Ours. Rankings are descriptive point estimates."),
    ("02_balanced_accuracy_bars", "Balanced accuracy", "Sensitivity and specificity in one score",
     "Reported balanced accuracy across glaucoma benchmarks. Method order is matched to "
     "the F1 figure. Hatched bars with an asterisk identify Harvard GDP entries whose "
     "reported balanced accuracy is inconsistent with (sensitivity + specificity)/2: "
     "RETFound, 0.7006 reported versus 0.8096 implied, and GPT-5.1, 0.6275 versus 0.5661. "
     "These entries are retained exactly as supplied and require verification."),
    ("03_subgroup_dumbbell", "Overall and worst-group F1", "Demographic performance on the two Harvard cohorts",
     "Overall and worst-group F1 on FairVision30K and Harvard GDP. Filled circles show "
     "overall F1 and open circles show worst-group F1. Each connecting line joins the "
     "two reported scores for the same method; it is not an uncertainty interval. "
     "Subgroup F1 was not supplied for Drishti-GS or REFUGE2."),
    ("04_sensitivity_specificity", "Sensitivity and specificity", "Operating-point trade-offs for every method",
     "Glaucoma sensitivity-specificity operating points. Each marker is one reported "
     "method at its evaluation threshold; the panels are not ROC curves. Upper-right "
     "points have both higher sensitivity and higher specificity. Shapes and colors "
     "identify foundation models, LLM baselines and Ours. Both axes use a common 0-1 scale."),
    ("05_strongest_comparator", "Difference from the strongest comparator", "Compact summary, including the Drishti reversal",
     "Difference between Ours and the highest-scoring included non-agent method in each "
     "dataset, selected separately for F1 and worst-group F1. The comparator name is shown "
     "beside each dataset. Positive differences favor Ours; negative differences favor "
     "the comparator. This is a descriptive comparison against the observed table maximum, "
     "not a deployable selection rule. Differences are percentage points, without "
     "inferred uncertainty. Missing subgroup values are not treated as zero."),
]


def group(model):
    if model == "Ours":
        return "Ours"
    return "LLM baselines" if model.startswith(("GPT", "Claude")) else "Foundation models"


def color(row):
    return COLORS[group(row["model"])]


def display(model, short=False):
    if model == "Claude Haiku 4.5":
        return "Haiku 4.5" if short else model
    return model.replace("GPT ", "GPT-")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        key = (row["dataset"], row["model"])
        if key in seen:
            raise ValueError(f"Duplicate model/dataset: {key}")
        seen.add(key)
        for metric in METRICS:
            row[metric] = float(row[metric]) if row[metric] else None
            value = row[metric]
            if value is None and metric != "worst_group_f1":
                raise ValueError(f"Missing metric {metric}: {key}")
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid metric {metric}: {key}")
        implied = (row["sensitivity"] + row["specificity"]) / 2
        row["implied_balanced_accuracy"] = implied
        row["ba_mismatch"] = abs(implied - row["balanced_accuracy"]) > 0.00015
    for dataset in DATASETS:
        if sum(r["dataset"] == dataset and r["model"] == "Ours" for r in rows) != 1:
            raise ValueError(f"Expected one Ours row for {dataset}")
    return rows


def ranked(rows, dataset):
    return sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: -r["f1"])


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#A5AEB4", "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": 0.45,
    })


def title(ax, letter, name, subtitle=None):
    ax.set_title(f"{letter}   {name}", loc="left", fontweight="bold", pad=23 if subtitle else 11)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, color="#657077", fontsize=6.5)


def score_axis(ax, count, label):
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(label, labelpad=5)
    ax.set_ylim(count - 0.45, -0.65)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.45)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2, width=0.5)


def legend(fig, marker=False, y=0.98):
    if marker:
        handles = [Line2D([], [], marker=m, color=c, linestyle="none", markersize=s,
                          label=g) for (g, c), m, s in zip(COLORS.items(), ("o", "^", "*"), (4, 5, 8))]
    else:
        handles = [Patch(facecolor=c, label=g) for g, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, y),
               ncol=3, frameon=False, handlelength=1.3, columnspacing=2.0, fontsize=7)


def model_labels(ax, entries):
    ax.set_yticks(range(len(entries)), [display(r["model"]) for r in entries])
    for label, row in zip(ax.get_yticklabels(), entries):
        if row["model"] == "Ours":
            label.set_color(COLORS["Ours"])
            label.set_fontweight("bold")


def bars(rows, metric):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.85))
    fig.subplots_adjust(left=0.178, right=0.978, bottom=0.09, top=0.88, wspace=0.80, hspace=0.53)
    legend(fig)
    for letter, dataset, ax in zip("abcd", DATASETS, axes.flat):
        entries = ranked(rows, dataset)
        for y, row in enumerate(entries):
            flag = metric == "balanced_accuracy" and row["ba_mismatch"]
            if row["model"] == "Ours":
                ax.axhspan(y - 0.44, y + 0.44, color=COLORS["Ours"], alpha=0.065, lw=0)
            ax.barh(y, row[metric], height=0.59, color="white" if flag else color(row),
                    edgecolor=color(row), linewidth=0.7 if flag else 0, hatch="////" if flag else None)
            if row["model"] == "Ours" or flag:
                value = row[metric]
                inside = value > 0.86
                ax.text(value - 0.02 if inside else value + 0.015, y,
                        f"{value:.3f}" + ("*" if flag else ""), va="center",
                        ha="right" if inside else "left", fontsize=6.5,
                        color="white" if inside and not flag else color(row), fontweight="bold")
        score_axis(ax, len(entries), "F1 score" if metric == "f1" else "Balanced accuracy")
        model_labels(ax, entries)
        title(ax, letter, dataset, MODALITIES[dataset])
    note = ("* Harvard GDP: two reported balanced accuracies conflict with sensitivity/specificity; see data audit."
            if metric == "balanced_accuracy" else "Reported point estimates. F1 definitions and evaluation cohorts require reconciliation before publication.")
    fig.text(0.178, 0.018, note, fontsize=5.8, color="#657077")
    return fig


def subgroup(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.9))
    fig.subplots_adjust(left=0.178, right=0.978, bottom=0.14, top=0.80, wspace=0.80)
    handles = [Line2D([], [], marker="o", color=INK, linestyle="none", markersize=4, label="Overall F1"),
               Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none",
                      markersize=4, label="Worst-group F1")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=2, frameon=False)
    for letter, dataset, ax in zip("ab", DATASETS[:2], axes):
        entries = ranked(rows, dataset)
        for y, row in enumerate(entries):
            ours = row["model"] == "Ours"
            c = COLORS["Ours"] if ours else "#667A89"
            if ours:
                ax.axhspan(y - 0.44, y + 0.44, color=c, alpha=0.065, lw=0)
            ax.plot([row["worst_group_f1"], row["f1"]], [y, y], color=c if ours else WEAK,
                    linewidth=1.8 if ours else 1.1, zorder=2)
            ax.scatter(row["f1"], y, s=27 if ours else 17, color=c, zorder=3)
            ax.scatter(row["worst_group_f1"], y, s=27 if ours else 17, facecolor="white",
                       edgecolor=c, linewidth=1.1, zorder=3)
        score_axis(ax, len(entries), "F1 score")
        model_labels(ax, entries)
        title(ax, letter, dataset)
    fig.text(0.178, 0.022, "Lines connect overall and worst-group F1 for the same method; they are not confidence intervals.",
             fontsize=5.8, color="#657077")
    return fig


# Label positions in axes coordinates are fixed for the supplied table, so crowded
# operating points remain readable without implying movement in the measurements.
POINT_LABELS = {
    "FairVision30K": {
        "RETFound": (.79, .81), "VisionFM": (.98, .55), "URFound": (.61, .77),
        "FLAIR": (.70, .43), "MIRAGE": (.48, .56), "RET-CLIP": (.39, .68),
        "RetiZero": (.40, .87), "GPT 5.1": (.84, .045), "GPT 5.6-luna": (.39, .19),
        "Claude Haiku 4.5": (.48, .30), "Ours": (.97, .93),
    },
    "Harvard GDP": {
        "RETFound": (.99, .87), "VisionFM": (.57, .90), "URFound": (.38, .57),
        "GPT 5.1": (.74, .24), "GPT 5.6-luna": (.99, .46),
        "Claude Haiku 4.5": (.21, .66), "Ours": (.98, .59),
    },
    "Drishti-GS": {
        "RETFound": (.70, .38), "MIRAGE": (.02, .74), "RET-CLIP": (.98, .55),
        "RetiZero": (.46, .97), "GPT 5.1": (.43, .77),
        "GPT 5.6-luna": (.43, .60), "Claude Haiku 4.5": (.96, .29),
        "Ours": (.70, .87),
    },
    "REFUGE2": {
        "RETFound": (.94, .45), "RET-CLIP": (.50, .98), "RetiZero": (.27, .84),
        "URFound": (.02, .89), "GPT 5.1": (.95, .025), "GPT 5.6-luna": (.95, .23),
        "Claude Haiku 4.5": (.63, .13), "Ours": (.95, .65),
    },
}


def operating_points(rows):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.25))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.105, top=0.88, wspace=0.29, hspace=0.52)
    legend(fig, marker=True)
    for letter, dataset, ax in zip("abcd", DATASETS, axes.flat):
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.035, 1.075)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Specificity", labelpad=4)
        ax.set_ylabel("Sensitivity", labelpad=4)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        title(ax, letter, dataset, MODALITIES[dataset])
        for row in ranked(rows, dataset):
            model, g = row["model"], group(row["model"])
            marker = {"Foundation models": "o", "LLM baselines": "^", "Ours": "*"}[g]
            ax.scatter(row["specificity"], row["sensitivity"], s=78 if model == "Ours" else 22,
                       marker=marker, c=color(row), edgecolors="white", linewidths=0.5, zorder=4)
            pos = POINT_LABELS[dataset][model]
            ax.annotate(display(model, short=True), (row["specificity"], row["sensitivity"]),
                        xytext=pos, textcoords="axes fraction", fontsize=6.4,
                        ha="right" if pos[0] > .75 else "left", va="center",
                        fontweight="bold" if model == "Ours" else "normal",
                        color=color(row) if model == "Ours" else INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.65, "alpha": .94},
                        arrowprops={"arrowstyle": "-", "color": "#A8B0B5", "lw": .5}, zorder=5)
    fig.text(0.09, 0.023, "One reported operating point per method; these panels are not ROC curves. Higher is better on both axes.",
             fontsize=5.8, color="#657077")
    return fig


def best_comparator(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.15))
    fig.subplots_adjust(left=.24, right=.97, bottom=.20, top=.79, wspace=.90)
    for letter, metric, ax in zip("ab", ("f1", "worst_group_f1"), axes):
        labels = []
        for y, dataset in enumerate(DATASETS):
            entries = ranked(rows, dataset)
            ours = next(r for r in entries if r["model"] == "Ours")
            candidates = [r for r in entries if r["model"] != "Ours" and r[metric] is not None]
            if ours[metric] is None or not candidates:
                labels.append(dataset)
                ax.text(0, y, "Not reported", color="#7A8287", fontsize=6.6, ha="center", va="center")
                continue
            best = max(candidates, key=lambda r: r[metric])
            delta = (ours[metric] - best[metric]) * 100
            labels.append(f"{dataset}\nvs {display(best['model'])}")
            c = COLORS["Ours"] if delta >= 0 else LOSS
            ax.plot([0, delta], [y, y], color=c, linewidth=2.0, solid_capstyle="round")
            ax.scatter(delta, y, s=28, color=c, zorder=3)
            ax.annotate(f"{delta:+.1f}", (delta, y), xytext=(6, 9 if delta < 0 else 0),
                        textcoords="offset points", ha="left",
                        va="center", color=c, fontweight="bold", fontsize=7)
        ax.axvline(0, color="#919BA1", linewidth=.8)
        ax.set_xlim(-16, 16)
        ax.set_xticks([-15, -10, -5, 0, 5, 10, 15], ["-15", "-10", "-5", "0", "+5", "+10", "+15"])
        ax.set_yticks(range(4), labels, fontsize=6.6)
        ax.set_ylim(3.65, -.55)
        ax.tick_params(axis="y", length=0)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.xaxis.grid(color=GRID, linewidth=.4)
        ax.set_axisbelow(True)
        ax.set_xlabel("Ours minus comparator (percentage points)", fontsize=6.4, labelpad=6)
        title(ax, letter, "F1 score" if metric == "f1" else "Worst-group F1")
    fig.text(.24, .035, "Comparator = highest included non-agent score for that metric. Point differences, without inferred uncertainty.",
             fontsize=5.8, color="#657077")
    return fig


def write_deliverables(out, figures, rows, source):
    for (stem, _, _, _), fig in zip(OPTIONS, figures):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(out / f"{stem}.{extension}", dpi=300)
        plt.close(fig)
    mismatches = [{"dataset": r["dataset"], "model": r["model"],
                   "reported": r["balanced_accuracy"], "implied": r["implied_balanced_accuracy"]}
                  for r in rows if r["ba_mismatch"]]
    (out / "data_audit.json").write_text(json.dumps({
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(rows), "counts": {d: sum(r["dataset"] == d for r in rows) for d in DATASETS},
        "balanced_accuracy_mismatches": mismatches,
        "source_revisions": SOURCE_REVISION_NOTE,
        "values_corrected_during_render": False, "intervals_estimated": False,
        "f1_averaging": "Not specified consistently in supplied tables; reconcile source predictions.",
        "sample_sizes": "Not inferred from dataset names; no n labels added.",
    }, indent=2) + "\n")
    captions = "\n\n".join(f"## {name}\n\n{caption} {COMMON_CAPTION}" for _, name, _, caption in OPTIONS)
    (out / "captions.md").write_text(captions + "\n")
    tex = [r"\documentclass{article}", r"\usepackage[a4paper,margin=12mm]{geometry}",
           r"\usepackage[T1]{fontenc}", r"\usepackage{graphicx}", r"\usepackage{caption}",
           r"\captionsetup{font=small}", r"\begin{document}"]
    for stem, name, _, caption in OPTIONS:
        tex += [r"\begin{figure}[p]", r"\centering", rf"\includegraphics[width=\linewidth]{{{stem}.pdf}}",
                rf"\caption{{{caption} {COMMON_CAPTION}}}", rf"\label{{fig:{stem}}}",
                r"\end{figure}", r"\clearpage"]
    tex.append(r"\end{document}")
    (out / "overleaf_figures.tex").write_text("\n".join(tex) + "\n")
    blocks = []
    for stem, name, subtitle, _ in OPTIONS:
        links = " | ".join(f'<a href="{stem}.{ext}">{ext.upper()}</a>' for ext in ("pdf", "svg", "png"))
        blocks.append(f'<section id="{stem}"><h2>{stem[:2]} / {html.escape(name)}</h2>'
                      f'<p>{html.escape(subtitle)} &nbsp; {links}</p>'
                      f'<a href="{stem}.pdf"><img src="{stem}.svg" alt="{html.escape(name)}"></a></section>')
    (out / "index.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glaucoma figure options</title>
<style>
* { box-sizing: border-box; } body { margin: 0; font: 15px/1.5 Arial, sans-serif; color: #242b30; }
header, main { max-width: 1060px; margin: auto; padding: 28px; } header { padding-bottom: 0; }
h1 { font-size: 26px; margin: 0 0 12px; } h2 { font-size: 20px; margin: 0; }
p { margin: 8px 0 16px; } a { color: #006d65; } section { border-top: 1px solid #dfe5e8; padding: 24px 0 38px; }
img { display: block; width: 100%; max-width: 850px; height: auto; margin: 22px auto 0; }
.audit { border-left: 3px solid #b14d4d; padding: 4px 16px; background: #fff8f7; }
nav { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 18px 0; }
@media (max-width: 600px) { header, main { padding: 18px 12px; } h1 { font-size: 22px; } }
</style><header><h1>Glaucoma / figure alternatives</h1>
<p>Four datasets with the requested Drishti-GS revisions. Editable SVG, vector PDF and 300 dpi PNG.</p>
<p>Drishti-GS: URFound excluded for pretraining-overlap concerns; RetiZero F1 updated to support-weighted 0.8187.</p>
<p class="audit">Two Harvard GDP balanced accuracies need verification. Hatched bars retain the supplied values.
See <a href="data_audit.json">data audit</a> and <a href="captions.md">captions</a>.</p>
<nav>""" + "".join(f'<a href="#{s}">{s[:2]} {html.escape(n)}</a>' for s, n, _, _ in OPTIONS) +
        '</nav><p><a href="glaucoma_overleaf_bundle.zip">Overleaf bundle</a></p></header><main>' +
        "\n".join(blocks) + "</main></html>\n")
    (out / "README.md").write_text(
        "# Glaucoma figure alternatives\n\n"
        "Source: the four tables supplied in the conversation, including the completed REFUGE2 rows. "
        "The existing manuscript and TikZ overview are preserved.\n\n"
        + SOURCE_REVISION_NOTE + "\n\n"
        "Open index.html to compare figures. 01 is the broad benchmark view; 03 emphasizes subgroup "
        "performance; 04 shows the sensitivity-specificity trade-off. 05 explicitly shows the Drishti "
        "loss relative to the strongest included comparator, now RetiZero. 02 needs the marked GDP values reconciled.\n\n"
        "All figures are 183 mm wide with vector PDF/SVG exports and 300 dpi PNG previews. "
        "Upload glaucoma_overleaf_bundle.zip to Overleaf and compile overleaf_figures.tex, or include "
        "individual PDFs using graphicx. No TikZ dependencies are needed.\n\n"
        "The PDFs were generated and visually checked locally. The TeX wrapper was not compiled "
        "locally because a LaTeX installation is unavailable.\n\n"
        "Reproduce from repository root:\n\n"
        "```bash\npython equi-agent/scripts/make_glaucoma_figure_options.py\n```\n\n"
        "Data are in glaucoma_table_values.csv; no values are silently corrected. The figures do not "
        "assert identical modalities, training protocols, F1 averaging, sample sizes or statistical "
        "significance. Confirm those against the case-level results before publication. In particular, "
        "the REFUGE2 values appear to use a different F1 convention from some other rows; metric "
        "definitions cannot be recovered definitively from these aggregate tables alone.\n\n"
        "No confidence intervals can be recovered from the supplied point estimates alone. "
        "The strongest-comparator figure uses the observed maximum separately for each metric; "
        "this is descriptive and must not be presented as model selection evaluated independently.\n"
    )
    with zipfile.ZipFile(out / "glaucoma_overleaf_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["overleaf_figures.tex", "captions.md", "data_audit.json", "README.md", "glaucoma_table_values.csv"]:
            bundle.write(source if name == "glaucoma_table_values.csv" else out / name, arcname=name)
        for stem, _, _, _ in OPTIONS:
            bundle.write(out / f"{stem}.pdf", arcname=f"{stem}.pdf")
    # A single preview keeps all alternatives easy to compare in the chat.
    sheet = Image.new("RGB", (1800, 2400), "#f3f5f6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26) if Path("/System/Library/Fonts/Helvetica.ttc").exists() else ImageFont.load_default(size=26)
    for i, (stem, name, _, _) in enumerate(OPTIONS):
        x, y = (i % 2) * 900, (i // 2) * 800
        draw.text((x + 30, y + 20), f"{i + 1:02}  {name}", fill=INK, font=font)
        with Image.open(out / f"{stem}.png") as picture:
            picture.thumbnail((850, 730), Image.Resampling.LANCZOS)
            sheet.paste(picture, (x + (900 - picture.width) // 2, y + 58 + (730 - picture.height) // 2))
    sheet.save(out / "contact_sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT / "glaucoma_table_values.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.source)
    style()
    figures = [bars(rows, "f1"), bars(rows, "balanced_accuracy"), subgroup(rows),
               operating_points(rows), best_comparator(rows)]
    write_deliverables(args.out_dir, figures, rows, args.source)
    print(f"Rendered {len(figures)} alternatives from {len(rows)} rows: {args.out_dir / 'index.html'}")
    print(f"Balanced-accuracy mismatches retained and marked: {sum(r['ba_mismatch'] for r in rows)}")


if __name__ == "__main__":
    main()

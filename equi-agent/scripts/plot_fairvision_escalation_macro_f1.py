"""Plot a saved-score sweep for historical deterministic FairVision fusion.

Read-only with respect to experiment inputs; no inference or threshold fitting.
This is deliberately not a plot of the live OphthalmicAgent experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "equi-agent/outputs/fairvision_reliability_selective_arbitration"
OUTPUT = ROOT / "equi-agent/docs/paper_figures/escalation_coverage"
STEM = "historical_fusion_escalation_macro_f1"
TASKS = ("glaucoma", "amd", "dr")
TITLES = ("Glaucoma", "AMD", "Diabetic retinopathy")
COLORS = ("#008577", "#3468AD", "#B96921")
DISPLAY_MIN_CLASS = 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary(value):
    number = float(value)
    require(number in (0, 1), f"Invalid binary label: {value!r}")
    return int(number)


def load_cases(source, expected_per_task=3000):
    summary = json.loads((source / "selective_arbitration_summary.json").read_text())
    require(set(summary["tasks"]) == set(TASKS), "Unexpected task set")
    require(summary["test_common_cases"] == expected_per_task * len(TASKS), "Unexpected cohort size")
    require(summary["selective_escalation"]["risk_coverage_rank_column"] == "risk_score",
            "Source does not identify risk_score as its ranking score")
    cases = defaultdict(list)
    seen = set()
    with (source / "selective_arbitration_predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            require(task in TASKS and row["split"] == "test", "Unexpected task or split")
            require(row["dataset"] == "harvard_fairvision"
                    and row["model_name"] == "reliability_shrunk_weighted",
                    "These figures require the historical deterministic fusion predictions")
            require(bool(row["image_id"]), "Missing case identifier")
            key = (task, row["image_id"])
            require(key not in seen, f"Duplicate case: {key}")
            seen.add(key)
            truth, prediction = binary(row["y_true"]), binary(row["y_pred"])
            score = float(row["risk_score"])
            require(math.isfinite(score) and 0 <= score <= 1, f"Invalid risk score: {key}")
            probability = float(row["y_prob"])
            disagreement = float(row["disagreement_rate"])
            reliability = float(row["weighted_reliability"])
            require(all(math.isfinite(x) for x in (probability, disagreement, reliability)),
                    f"Nonfinite source evidence: {key}")
            require(0 <= probability <= 1 and 0 <= disagreement <= 1, f"Invalid source evidence: {key}")
            require(prediction == int(probability >= 0.5), f"Unexpected prediction rule: {key}")
            expected_score = max(1 - min(1, 2 * abs(probability - .5)), disagreement,
                max(0, summary["selective_escalation"]["low_reliability_threshold"] - reliability))
            require(math.isclose(score, expected_score, abs_tol=1e-12), f"Risk formula mismatch: {key}")
            cases[task].append(dict(truth=truth, prediction=prediction, risk_score=score))
    for task in TASKS:
        require(len(cases[task]) == expected_per_task, f"Incomplete cohort for {task}")
        require({c["truth"] for c in cases[task]} == {0, 1}, f"Missing reference class for {task}")
    return cases, summary


def point(task, threshold, counts, total):
    tn, fp, fn, tp = counts
    accepted = sum(counts)
    positive, negative = tp + fn, tn + fp
    pos_denom, neg_denom = 2 * tp + fp + fn, 2 * tn + fp + fn
    positive_f1 = 2 * tp / pos_denom if pos_denom else 0.
    negative_f1 = 2 * tn / neg_denom if neg_denom else 0.
    both_classes = positive > 0 and negative > 0
    return dict(task=task, threshold=threshold, total=total, accepted=accepted,
        escalated=total - accepted, escalation_percent=100 * (total - accepted) / total,
        coverage=accepted / total, accepted_positive=positive, accepted_negative=negative,
        tn=tn, fp=fp, fn=fn, tp=tp,
        macro_f1=(positive_f1 + negative_f1) / 2 if accepted else None,
        positive_f1=positive_f1 if accepted else None, both_classes=both_classes,
        display_support=("empty" if not accepted else "one_class" if not both_classes else
                         "sparse" if min(positive, negative) < DISPLAY_MIN_CLASS else "supported"))


def sweep(cases, task):
    require(bool(cases), "Cannot sweep an empty cohort")
    ranked = sorted(cases, key=lambda row: row["risk_score"])
    counts = [0, 0, 0, 0]
    # Equal saved scores enter together; a score threshold cannot split a tie.
    points = [point(task, math.nextafter(ranked[0]["risk_score"], -math.inf), counts, len(cases))]
    for score, group in groupby(ranked, key=lambda row: row["risk_score"]):
        for row in group:
            counts[2 * row["truth"] + row["prediction"]] += 1
        points.append(point(task, score, counts, len(cases)))
    if points[-1]["threshold"] < 1:
        points.append(point(task, 1., counts, len(cases)))
    return points


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_record(path):
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def draw(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .6})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.05), sharex=True, sharey=True)
    fig.subplots_adjust(left=.082, right=.982, bottom=.265, top=.745, wspace=.16)
    fig.text(.082, .94, "Accepted-case performance as more cases are referred", fontsize=11, weight="bold")
    fig.text(.082, .874, "Historical deterministic fusion | 3,000 test cases per task | Not live RetinAgent",
             fontsize=8, color="#50545A")
    for i, (ax, task, title, color) in enumerate(zip(axes, TASKS, TITLES, COLORS)):
        points = sorted(curves[task], key=lambda row: row["escalation_percent"])
        supported = [row for row in points if row["display_support"] == "supported"]
        sparse = [row for row in points if row["display_support"] == "sparse"]
        # Counts only decrease as escalation rises; connect the single support boundary.
        if supported and sparse:
            sparse.insert(0, supported[-1])
        for section, linestyle in ((supported, "solid"), (sparse, "dashed")):
            ax.plot([r["escalation_percent"] for r in section], [r["macro_f1"] for r in section],
                    color=color, linewidth=1.4, linestyle=linestyle)
        baseline = points[0]
        ax.scatter([0], [baseline["macro_f1"]], color=color, s=16, zorder=3, clip_on=False)
        ax.annotate(f"{baseline['macro_f1']:.3f}", (0, baseline["macro_f1"]),
                    xytext=(5, -13), textcoords="offset points", color=color, fontsize=8)
        ax.set_title(f"{'abc'[i]}   {title}", fontsize=9, pad=11, loc="left")
        ax.set(xlim=(0, 100), ylim=(.45, 1.025), xticks=[0, 25, 50, 75, 100],
               yticks=[.5, .6, .7, .8, .9, 1.])
        ax.grid(axis="y", linewidth=.5, color="#E1E4E6")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9CA2A8")
        ax.tick_params(length=2.5, color="#9CA2A8")
    axes[0].set_ylabel("Accepted-case macro-F1", labelpad=8)
    fig.supxlabel("Cases escalated to clinician review (%)", x=.532, y=.136, fontsize=9)
    fig.legend([Line2D([0], [0], color="#50545A", lw=1.4),
                Line2D([0], [0], color="#50545A", lw=1.4, ls="--")],
               ["At least 20 accepted cases per class", "Fewer than 20 in either class"],
               loc="lower center", bbox_to_anchor=(.54, .035), frameon=False, ncol=2,
               fontsize=7.5, columnspacing=1.8, handlelength=2.3)
    fig.text(.082, .005, "Single-class and empty subsets are not plotted. All thresholds and counts are exported.",
             fontsize=7, color="#50545A")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{STEM}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out_dir.resolve()
    require(not output.is_relative_to(source), "Do not write figure outputs inside the source run")
    cases, summary = load_cases(source)
    curves = {task: sweep(cases[task], task) for task in TASKS}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "threshold_curve.csv", [row for task in TASKS for row in curves[task]])
    checkpoints = []
    for task in TASKS:
        for threshold in (1., .8, .6, .4, .2, 0.):
            counts = [0, 0, 0, 0]
            for row in cases[task]:
                if row["risk_score"] <= threshold:
                    counts[2 * row["truth"] + row["prediction"]] += 1
            checkpoints.append(point(task, threshold, counts, len(cases[task])))
    write_csv(output / "threshold_checkpoints.csv", checkpoints)
    provenance = dict(experiment="historical deterministic reliability-shrunk weighted fusion",
        live_retinagent=False, inputs=[source_record(source / name) for name in
            ("selective_arbitration_predictions.csv", "selective_arbitration_summary.json")],
        generator=source_record(Path(__file__).resolve()), score_column="risk_score",
        acceptance_rule="risk_score <= threshold; final classification labels remain fixed",
        sweep="Every distinct saved risk score, with ties kept together; no threshold optimization",
        metric="Fixed-label [0,1] macro-F1; zero_division=0; empty accepted set undefined",
        plotted_support="Both reference classes present; dashed if either has fewer than 20 cases",
        sparse_marker_is_display_only=True, original_reliability=summary["reliability_score"],
        source_models=summary["models_requested"],
        original_escalation_policy=summary["selective_escalation"],
        full_coverage={task: curves[task][-1] for task in TASKS},
        limitations=["Not the live 250-case OphthalmicAgent runs.",
            "Post-hoc score sweep, not a sweep of the original joint conformal/escalation policy.",
            "Recomputed macro-F1, not the positive-class F1 in the old risk_coverage_curve.csv.",
            "Descriptive point estimates without uncertainty intervals or run-to-run variability.",
            "The accepted cohort's class composition changes with threshold.",
            "No clinician outcomes are available for escalated cases."])
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    draw(curves, output)
    for task in TASKS:
        full = curves[task][-1]
        print(f"{task}: N={full['total']}, full-coverage macro-F1={full['macro_f1']:.4f}")
    print(f"Historical figure (not live RetinAgent): {output / (STEM + '.pdf')}")


if __name__ == "__main__":
    main()

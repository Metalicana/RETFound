"""Audit downloaded CECSL outputs and draw explicitly historical result figures.

No inference, fitting, threshold selection, network access, or patient-level export.
The caller supplies a separately extracted archive; original results are read-only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from build_paper_drawio import Page, write_document, BLUE, TEAL, AMBER, GRAY, INK


BOUNDARIES = "equi-agent/outputs/metrics/foundation_reliability_boundaries_glaucoma_f1_bonus"
SELECTIVE = "equi-agent/outputs/fairvision_reliability_selective_arbitration"
LIVE = "equi-agent/outputs/experiments/fairvision_live_glaucoma_250_score_cf_cdr_v1"
EXTERNAL = (
    ("PAPILA", "CFP", 81,
     "equi-agent/outputs/benchmarks/papila_glaucoma_foundations_oof_v2/retfound/predictions_test.csv",
     "OphthalmicAgent/outputs/papila/agentic_retfound_oof_v2/predictions.csv"),
    ("Drishti-GS", "CFP", 51,
     "equi-agent/outputs/benchmarks/drishti_glaucoma_foundations_v1/retfound/predictions_test.csv",
     "OphthalmicAgent/outputs/drishti/agentic_retfound_cfp_v1/predictions.csv"),
    ("GAMMA", "OCT + CFP", 20,
     "OphthalmicAgent/outputs/gamma/retfound_oct_fairvision_init_v1/predictions_test.csv",
     "OphthalmicAgent/outputs/gamma/agentic_retfound_oct_v1/predictions.csv"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binary(value):
    number = float(value)
    require(number in (0, 1), f"Expected binary value; got {value!r}")
    return int(number)


def flag(value):
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    return bool(binary(value))


def metrics(pairs):
    counts = Counter((binary(y), binary(p)) for y, p in pairs)
    tn, fp, fn, tp = (counts[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1)))
    n = tn + fp + fn + tp
    require(n > 0, "Cannot score an empty cohort")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(n=n, tn=tn, fp=fp, fn=fn, tp=tp, f1=f1,
                weighted_f1=((tp + fn) * f1 + (tn + fp) * f0) / n,
                accuracy=(tp + tn) / n, sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.sources = {}

    def path(self, name):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "Input path escapes archive root")
        self.sources[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  bytes=path.stat().st_size)
        return path

    def csv(self, name):
        with self.path(name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def json(self, name):
        return json.loads(self.path(name).read_text())


def unique_rows(rows, key):
    result = {}
    for row in rows:
        require(bool(row[key]), f"Empty {key}")
        require(row[key] not in result, f"Duplicate {key} in a locked cohort")
        result[row[key]] = row
    return result


def audit_boundaries(inputs):
    search = unique_rows(inputs.csv(f"{BOUNDARIES}/formula_search_oof.csv"), "formula")
    cohort = None
    for formula, summary in search.items():
        rows = unique_rows(inputs.csv(f"{BOUNDARIES}/case_winners_{formula}.csv"), "case_id")
        require(len(rows) == int(summary["case_n"]) == 1000, "Boundary cohort size changed")
        require(all(row["split"] == "val" for row in rows.values()), "Not validation-only")
        labels = {key: binary(row["y_true"]) for key, row in rows.items()}
        require(cohort is None or labels == cohort, "Boundary candidates use different cases/labels")
        cohort = labels
        measured = metrics((row["y_true"], row["selected_vote"]) for row in rows.values())
        for source, computed in (("selector_f1", "f1"), ("selector_accuracy", "accuracy"),
                                 ("selector_balanced_accuracy", "balanced_accuracy")):
            require(math.isclose(float(summary[source]), measured[computed], abs_tol=1e-10),
                    f"Cannot reproduce {formula}: {source}")
        search[formula] = measured
    inputs.path(f"{BOUNDARIES}/methodology_report.md")
    return search


def paired_rows(baseline, attempts):
    base = unique_rows(baseline, "case_id")
    # Use the last recorded attempt, not the last correct or last valid attempt.
    agent = {}
    for row in attempts:
        require(row.get("split") == "test", "Non-test agent attempt")
        agent[row["case_id"]] = row
    require(base.keys() == agent.keys(), "Baseline/agent case sets differ")
    groups = defaultdict(list)
    for key, row in base.items():
        require(row["split"] == "test", "Non-test baseline row")
        require(binary(row["y_true"]) == binary(agent[key]["Ground_Truth"]), "Paired labels differ")
        groups[row.get("patient_id") or key].append(
            (binary(row["y_true"]), binary(row["y_pred"]), binary(agent[key]["Pred_GL"])))
    return list(groups.values())


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def audit_external(inputs):
    result = []
    for dataset, modality, expected, base_path, agent_path in EXTERNAL:
        base = inputs.csv(base_path)
        attempts = inputs.csv(agent_path)
        require(len(base) == expected, f"{dataset}: locked cohort size changed")
        groups = paired_rows(base, attempts)
        rows = [row for group in groups for row in group]
        baseline = metrics((y, b) for y, b, a in rows)
        agent = metrics((y, a) for y, b, a in rows)
        rng = random.Random(2026)
        deltas = []
        for _ in range(2000):
            sampled = [row for group in rng.choices(groups, k=len(groups)) for row in group]
            b = metrics((y, b) for y, b, a in sampled)
            a = metrics((y, a) for y, b, a in sampled)
            deltas.append(a["weighted_f1"] - b["weighted_f1"])
        result.append(dict(dataset=dataset, modality=modality, baseline=baseline, agent=agent,
                           clusters=len(groups), extra_attempts=len(attempts) - expected,
                           delta=agent["weighted_f1"] - baseline["weighted_f1"],
                           ci=[percentile(deltas, .025), percentile(deltas, .975)],
                           baseline_path=base_path, agent_path=agent_path))
    return result


def audit_selective(inputs):
    summary = inputs.json(f"{SELECTIVE}/selective_arbitration_summary.json")
    curve = inputs.csv(f"{SELECTIVE}/risk_coverage_curve.csv")
    predictions = inputs.csv(f"{SELECTIVE}/selective_arbitration_predictions.csv")
    result = {}
    for task in ("amd", "dr", "glaucoma"):
        rows = [row for row in predictions if row["task"] == task]
        unique_rows(rows, "image_id")
        require(len(rows) == 3000, "Expected 3000 held-out cases per task")
        require(all(row["split"] == "test" for row in rows), "Non-test selective predictions")
        full = metrics((r["y_true"], r["y_pred"]) for r in rows)
        points = []
        for row in curve:
            if row["task"] != task:
                continue
            tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
            n = tn + fp + fn + tp
            coverage = float(row["coverage"])
            require(n == int(row["n"]) and math.isclose(n / len(rows), coverage),
                    "Curve count/coverage mismatch")
            require(tp + fn == int(row["n_positive"]) and tn + fp == int(row["n_negative"]),
                    "Curve class support mismatch")
            risk = (fp + fn) / n
            require(math.isclose(1 - risk, float(row["accuracy"])), "Curve error count mismatch")
            if coverage == 1:
                require(all(int(row[key]) == full[key] for key in ("tn", "fp", "fn", "tp")),
                        "Full-coverage curve and prediction counts differ")
            points.append(dict(coverage=coverage, risk=risk, positive_fraction=(tp + fn) / n, n=n))
        require(any(p["coverage"] == 1 for p in points), "Missing full-coverage point")
        result[task] = dict(points=sorted(points, key=lambda p: p["coverage"]), full=full,
                            accepted_n=sum(flag(row["accepted"]) for row in rows))
    return dict(tasks=result, score=summary["reliability_score"],
                ranking="risk_score, increasing; saved coverage grid, not a new selected policy")


def audit_live(inputs):
    summaries = []
    for path in sorted((inputs.root / "equi-agent/outputs/experiments").glob("fairvision*250*/*summary.json")):
        summary = inputs.json(str(path.relative_to(inputs.root)))
        if "dry_run" in summary:
            summaries.append(dict(directory=path.parent.name,
                                  **{k: summary.get(k) for k in ("dry_run", "cases", "errors")}))
    rows = inputs.csv(f"{LIVE}/equi_agent_live_predictions.csv")
    unique_rows(rows, "image_id")
    require(len(rows) == 250, "Live candidate cohort changed")
    score = metrics((row["y_true"], row["y_pred"]) for row in rows)
    other = inputs.csv("OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv")
    invalid_truth = sum(row["Ground_Truth"] not in ("0", "1") for row in other)
    config = inputs.json(f"{LIVE}/equi_agent_live_resolved_config.json")
    return dict(runs=summaries, live_candidate=score,
                live_candidate_case_list=config["arguments"].get("case_list_csv"),
                ophth_glaucoma_rows=len(other), ophth_glaucoma_invalid_truth=invalid_truth)


def dot(page, x, y, color, diameter=12):
    page.cell("", x - diameter / 2, y - diameter / 2, diameter, diameter,
              f"ellipse;fillColor={color};strokeColor=#ffffff;strokeWidth=1;")


def axis(page, x, y, w, h, xlim, ylim, xticks, yticks, xlabel, ylabel):
    def point(a, b):
        return (x + (a - xlim[0]) / (xlim[1] - xlim[0]) * w,
                y + h - (b - ylim[0]) / (ylim[1] - ylim[0]) * h)
    for value in yticks:
        px, py = point(xlim[0], value)
        page.line([(px, py), (px + w, py)], "#E5E9EC", width=1)
        page.text(f"{value:g}", px - 60, py - 13, 48, 26, 17, GRAY, align="right")
    page.line([(x, y), (x, y + h), (x + w, y + h)], INK, width=1.4)
    for value in xticks:
        px, py = point(value, ylim[0])
        page.line([(px, py), (px, py + 5)], INK, width=1)
        page.text(f"{value:g}", px - 26, py + 10, 52, 26, 17, GRAY, align="center")
    page.text(xlabel, x, y + h + 47, w, 32, 19, align="center")
    page.text(ylabel, x, y - 38, w, 30, 19, GRAY)
    return point


def boundary_figure(search):
    page = Page("01 Reliability development")
    page.header("S1", "What did the earlier reliability experiments show?",
                "Recovered development analysis: 1,000 validation cases, nine models; positive-class F1.")
    page.text("a  Selection rules", 60, 231, 620, 40, 24, bold=True)
    labels = [("global_accuracy", "Global accuracy"), ("demographic_accuracy", "Demographics"),
              ("directional_demographic_accuracy", "Demographics + vote"),
              ("probability_bin_accuracy", "Probability bins"),
              ("hierarchical_demo_bin_accuracy", "Hierarchical demo + bins"),
              ("logistic_correctness", "Logistic correctness"), ("tree_correctness", "Tree correctness")]
    start, width = 350, 345
    for tick in (.65, .70, .75, .80):
        x = start + (tick - .65) / .15 * width
        page.line([(x, 305), (x, 670)], "#E5E9EC", width=1)
        page.text(f"{tick:.2f}", x - 30, 688, 60, 25, 17, GRAY, align="center")
    for i, (key, label) in enumerate(labels):
        y = 327 + 53 * i
        page.text(label, 60, y - 20, 273, 40, 20)
        x = start + (search[key]["f1"] - .65) / .15 * width
        dot(page, x, y, BLUE)
    page.line([(start, 670), (start + width, 670)], INK)
    page.text("Selector F1", start, 730, width, 30, 19, align="center")
    page.text("b  Positive-vote bonus sensitivity", 830, 231, 590, 40, 24, bold=True)
    point = axis(page, 860, 375, 480, 295, (0, .05), (.68, .78),
                 (0, .01, .02, .03, .04, .05), (.68, .70, .72, .74, .76, .78),
                 "Positive-vote bonus", "Selector F1")
    for family, name, color in (("probability_bin", "Probability bins", BLUE),
                                 ("hierarchical_demo_bin", "Hierarchical demo + bins", TEAL)):
        values = [(0, search[f"{family}_accuracy"]["f1"])]
        for amount in (1, 2, 3, 5):
            values.append((amount / 100, search[f"{family}_f1_bonus_0p0{amount}"]["f1"]))
        page.line([point(x, y) for x, y in values], color, width=2.5)
        for x, y in values:
            dot(page, *point(x, y), color, diameter=10)
        legend_y = 282 if color == BLUE else 310
        page.line([(900, legend_y), (932, legend_y)], color, width=2.5)
        page.text(name, 947, legend_y - 12, 400, 25, 17, color)
    page.text("Exploratory OOF search, not an independent test: these results do not establish the paper's five fixed risk coefficients.",
              60, 796, 1350, 42, 19, AMBER)
    page.text("All 17 candidates and reproduced metrics are retained in source_data.json; panel a shows the seven unmodified rule families.",
              60, 848, 1350, 28, 16, GRAY)
    return page


def selective_figure(data):
    page = Page("02 Deterministic selection")
    page.header("S2", "Selective performance depends on which cases remain",
                "Historical deterministic arbitration, not the live LLM agent: 3,000 held-out cases per task.")
    page.text("a  Error among retained cases", 100, 236, 610, 40, 24, bold=True)
    page.text("b  Class mix among retained cases", 825, 236, 595, 40, 24, bold=True)
    max_risk = max(p["risk"] for d in data["tasks"].values() for p in d["points"])
    ymax = max(.3, math.ceil(max_risk * 10) / 10)
    xmin = min(p["coverage"] for d in data["tasks"].values() for p in d["points"])
    xticks = [i / 10 for i in range(math.ceil(xmin * 10), 11)]
    left = axis(page, 130, 350, 525, 315, (xmin, 1), (0, ymax), xticks,
                [i / 10 for i in range(round(ymax * 10) + 1)], "Coverage", "Error fraction")
    right = axis(page, 870, 350, 525, 315, (xmin, 1), (0, 1), xticks,
                 (0, .25, .5, .75, 1), "Coverage", "Fraction disease positive")
    for i, (task, color) in enumerate((("amd", BLUE), ("dr", AMBER), ("glaucoma", TEAL))):
        points = data["tasks"][task]["points"]
        for transform, field in ((left, "risk"), (right, "positive_fraction")):
            page.line([transform(p["coverage"], p[field]) for p in points], color, width=2.5)
            for p in points:
                dot(page, *transform(p["coverage"], p[field]), color, diameter=9)
        x = 470 + 230 * i
        page.line([(x, 292), (x + 34, 292)], color, width=2.5)
        page.text(task.upper() if task != "glaucoma" else "Glaucoma", x + 44, 278, 180, 30, 20, color)
    page.text("Saved risk-score ranking; no confidence intervals or competing-method curves are inferred. Curves are not clinical utility estimates.",
              60, 773, 1350, 53, 19, GRAY)
    accepted = data["tasks"]["dr"]["accepted_n"]
    page.text(f"The saved full escalation policy accepted {accepted:,}/3,000 DR cases. Its operating point is separate from this coverage sweep.",
              60, 836, 1350, 39, 18, AMBER)
    return page


def external_figure(data):
    page = Page("03 Historical paired external")
    page.header("S3", "Paired external results include gains, losses and ties",
                "Earlier recovered RETFound-fed agent runs, not a reconstruction of the current manuscript rows.")
    page.text("a  Change in support-weighted F1", 60, 239, 930, 44, 25, bold=True)
    xmin = min(-.2, math.floor(min(row["ci"][0] for row in data) * 10) / 10)
    xmax = max(.2, math.ceil(max(row["ci"][1] for row in data) * 10) / 10)
    start, width = 400, 650
    def xval(value):
        return start + (value - xmin) / (xmax - xmin) * width
    for i in range(round(xmin * 10), round(xmax * 10) + 1):
        value = i / 10
        x = xval(value)
        page.line([(x, 328), (x, 644)], GRAY if i == 0 else "#E5E9EC", width=1.5 if i == 0 else 1)
        page.text(f"{value:+.1f}" if value else "0", x - 35, 669, 70, 30, 18, GRAY, align="center")
    page.text("Baseline / agent", 1135, 294, 260, 35, 20, GRAY, align="center")
    for i, row in enumerate(data):
        y = 362 + 110 * i
        page.text(row["dataset"], 60, y - 33, 310, 36, 24, bold=True)
        page.text(f"n={row['baseline']['n']}; {row['modality']}", 60, y + 5, 300, 28, 18, GRAY)
        color = TEAL if row["delta"] > 0 else AMBER if row["delta"] < 0 else GRAY
        a, b = (xval(v) for v in row["ci"])
        page.line([(a, y), (b, y)], color, width=2)
        for x in (a, b):
            page.line([(x, y - 7), (x, y + 7)], color, width=2)
        dot(page, xval(row["delta"]), y, color, diameter=14)
        page.text(f"{row['baseline']['weighted_f1']:.3f} / {row['agent']['weighted_f1']:.3f}",
                  1135, y - 20, 260, 40, 23, align="center")
    page.line([(start, 644), (start + width, 644)], INK)
    page.text("Favors RETFound", start, 719, 275, 35, 20, AMBER)
    page.text("Favors agent", start + width - 250, 719, 250, 35, 20, TEAL, align="right")
    page.text("Points: agent minus baseline. Lines: 95% paired percentile bootstrap intervals (2,000 resamples; seed 2026).",
              60, 789, 1350, 35, 18, GRAY)
    page.text("PAPILA: patient clusters (both eyes together). Drishti-GS: patient IDs. GAMMA: case resampling; one case per patient assumed.",
              60, 831, 1350, 46, 17, GRAY)
    return page


def report(data, inputs):
    live = data["live"]
    lines = ["# Recovered CECSL Figure Inputs", "",
             "This audit reads a separate extraction of the 2026-09-27 transfer. No model, LLM, "
             "prompt, original output or manuscript was changed. All figures in this directory are "
             "historical/exploratory drafts, not replacements for verified final manuscript figures.", "",
             "## What Is Recovered", "",
             "- All 17 reliability-search candidates reproduce from their 1,000-case validation "
             "winner CSVs, including F1, accuracy and balanced accuracy.",
             "- Three tasks have deterministic risk-coverage results on 3,000 test cases each.",
             "- PAPILA, Drishti-GS and GAMMA have fully paired earlier baseline/agent outputs.", "",
             "## Main-Figure Blockers", "",
             f"- The recovered live 250-case glaucoma run has weighted F1 "
             f"**{live['live_candidate']['weighted_f1']:.4f}** (binary F1 "
             f"{live['live_candidate']['f1']:.4f}), not the previously supplied paper value **0.7704**. "
             "Do not substitute or relabel this run.",
             f"- Its configured case list is `{live['live_candidate_case_list']}`.",
             f"- `OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv` has "
             f"{live['ophth_glaucoma_rows']} rows and {live['ophth_glaucoma_invalid_truth']} nonbinary "
             "ground-truth value(s). It is excluded from all result plots, not silently relabelled.",
             "- Live AMD/DR manuscript runs and matched final GDP/REFUGE agent outputs are not "
             "established by this transfer. Existing aggregates alone do not establish their cohorts.",
             "- The three paired external runs below do not reproduce all subsequently updated "
             "manuscript rows. They are historical comparisons, not final reported estimates.", "",
             "## Candidate Run Status", "", "| Directory | Dry run | Cases | Errors |",
             "| --- | --- | ---: | ---: |"]
    for row in live["runs"]:
        lines.append(f"| {row['directory']} | {row['dry_run']} | {row['cases']} | {row['errors']} |")
    lines += ["", "## Reliability Evidence", "",
              "The saved report describes case-grouped out-of-fold evaluation on validation data. "
              "These are successive exploratory searches on the same development set; they do not "
              "provide an independent confirmation of the selected bonus, nested threshold tuning, "
              "or the paper's five fixed risk coefficients. OOF fold assignments are not recovered. "
              "The 0.974 any-model-correct oracle is not a deployable selector and is not plotted.", "",
              "| Rule | Positive-class F1 | Accuracy |", "| --- | ---: | ---: |"]
    for formula, row in data["boundaries"].items():
        lines.append(f"| {formula} | {row['f1']:.4f} | {row['accuracy']:.4f} |")
    lines += ["", "## Selective-Performance Scope", "",
              "The curve is the saved deterministic ensemble's increasing `risk_score` ranking. "
              "Its full-cohort confusion counts match the predictions. Each point's coverage, "
              "error rate and class support are checked against its saved counts; tied-score "
              "membership is not reconstructed. These are descriptive curves without uncertainty "
              "bands or matched comparator curves, not evidence of LLM-agent superiority.", "",
              "Saved reliability weights: `" + json.dumps(data["selective"]["score"]["weights"], sort_keys=True) + "`.",
              "This differs from the paper's FNR/FPR/ECE/AUROC/F1 risk score. The hard escalation "
              "policy and the continuous coverage sweep are different analyses.", "",
              "## Historical External Comparisons", "",
              "Last recorded test attempt per case, with exact baseline/agent ID and label "
              "agreement required. No invalid final prediction is dropped or replaced with an "
              "earlier valid prediction. All three cohorts here have complete valid final outputs. "
              "Retries were performed in the original runs; this audit does not establish their "
              "selection protocol. F1 conventions are explicitly separated.", "",
              "| Dataset | Cases | Extra attempts | Baseline binary F1 | Agent binary F1 | "
              "Baseline weighted F1 | Agent weighted F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["external"]:
        b, a = row["baseline"], row["agent"]
        lines.append(f"| {row['dataset']} | {b['n']} | {row['extra_attempts']} | {b['f1']:.4f} | "
                     f"{a['f1']:.4f} | {b['weighted_f1']:.4f} | {a['weighted_f1']:.4f} |")
    lines += ["", "The paired bootstrap estimates uncertainty conditional on these fixed predictions; "
              "it does not include training-seed, threshold-selection or LLM variability. The "
              "GAMMA interval is exactly zero because both methods predicted identical labels, "
              "not because the experiment establishes equivalence.", "",
              "## Files And Reproduction", "",
              "- `recovered_experiments.drawio`: three fully editable result pages.",
              "- `01_reliability_development.drawio`, `02_selective_diagnostics.drawio`, "
              "`03_paired_external.drawio`: individual figures.",
              "- `source_data.json`: only aggregate metrics, provenance paths and source hashes. "
              "No case IDs, clinical narratives or images are exported.", "",
              "```bash", "python equi-agent/scripts/build_recovered_experiment_figures.py \\",
              "  --inputs-root /path/to/separate/extracted/archive", "```", "",
              "Generated figures are overwritten on rebuild; use a differently named copy for manual edits.", "",
              f"Source files checked: {len(inputs.sources)}. See `source_data.json` for SHA-256 values.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs/paper_figures/recovered")
    args = parser.parse_args()
    inputs = Inputs(args.inputs_root)
    data = dict(boundaries=audit_boundaries(inputs), selective=audit_selective(inputs),
                external=audit_external(inputs), live=audit_live(inputs))
    data["sources"] = inputs.sources
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages = [boundary_figure(data["boundaries"]), selective_figure(data["selective"]),
             external_figure(data["external"])]
    for name, page in zip(("01_reliability_development", "02_selective_diagnostics", "03_paired_external"), pages):
        write_document(args.out_dir / f"{name}.drawio", [page])
    write_document(args.out_dir / "recovered_experiments.drawio", pages)
    (args.out_dir / "source_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "audit.md").write_text(report(data, inputs))
    print(f"Verified {len(inputs.sources)} source files; wrote {args.out_dir}")


if __name__ == "__main__":
    main()

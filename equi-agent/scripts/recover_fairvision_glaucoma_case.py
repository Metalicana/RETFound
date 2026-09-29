"""Recover an invalid exported label and a skipped RETFound row from saved evidence.

Original inputs are never edited. No model, API, or probability-metric evaluation
is run. A rounded trace score is used only when its thresholded label is certain.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from build_recovered_experiment_figures import binary, metrics, require


ROOT = Path(__file__).resolve().parents[2]
CASE = "data_07199.npz"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index(rows, field):
    result = {}
    for row in rows:
        key = Path(row[field]).name
        require(key not in result, f"Duplicate case: {key}")
        result[key] = row
    return result


def source_label(row):
    value = row["glaucoma"].strip().lower()
    require(value in {"yes", "no"}, f"Invalid source label: {value}")
    require(row["use"].strip().lower() == "test", "Source case is not test")
    return int(value == "yes")


def rounded_score_decision(percent, threshold):
    value, cut = Decimal(str(percent)), Decimal(str(threshold)) * 100
    require(value.is_finite() and cut.is_finite(), "Nonfinite score or threshold")
    require(0 <= value <= 100 and 0 <= cut <= 100, "Invalid probability range")
    require(value == value.quantize(Decimal("0.01")), "Expected a two-decimal percentage")
    lo, hi = max(Decimal(0), value - Decimal("0.005")), min(Decimal(100), value + Decimal("0.005"))
    if hi < cut:
        return 0, [float(lo / 100), float(hi / 100)]
    if lo >= cut:
        return 1, [float(lo / 100), float(hi / 100)]
    raise ValueError("Rounded score straddles the decision threshold; exact output is required")


def recover(manifest, summary, agent_rows, baseline_rows, traces, case=CASE, expected_cases=250):
    selected = [r for r in manifest if r["Task_Folder"].lower() == "glaucoma"]
    selected_by_id = index(selected, "filename")
    source = index(summary, "filename")
    agent = index(agent_rows, "Filename")
    baseline = index(baseline_rows, "Filename")
    require(len(selected) == expected_cases and set(selected_by_id) == set(agent), "Agent/manifest cohort mismatch")
    require(set(selected_by_id) - set(baseline) == {case} and not set(baseline) - set(selected_by_id),
            "Expected only the specified case to be missing from the baseline")
    require(all(r["Task_Folder"].lower() == "glaucoma" for r in agent_rows), "Agent task mismatch")
    require(all(r["Model"] == "RETFound" and r["Modality"] == "OCT"
                and r["Disease"].lower() == "glaucoma" for r in baseline_rows), "Baseline identity mismatch")
    thresholds = {Decimal(r["Decision_Threshold"]) for r in baseline_rows}
    require(len(thresholds) == 1, "Baseline threshold is not constant")
    threshold = next(iter(thresholds))
    for key in selected_by_id:
        truth = source_label(source[key])
        for column, source_column in (("Age", "age"), ("Gender", "gender"), ("Race", "race")):
            expected = source[key][source_column].strip().lower()
            actual = agent[key][column].strip().lower()
            require(float(actual) == float(expected) if column == "Age" else actual == expected,
                    f"Case metadata mismatch: {key}/{column}")
        binary(agent[key]["Pred_GL"])
        if key == case:
            require(float(selected_by_id[key]["Ground_Truth"]) == float(agent[key]["Ground_Truth"]) == -1,
                    "This recovery expects the documented -1 exported label")
        else:
            require(binary(selected_by_id[key]["Ground_Truth"]) == binary(agent[key]["Ground_Truth"])
                    == binary(baseline[key]["Ground_Truth"]) == truth, f"Other label mismatch: {key}")
            probability = Decimal(baseline[key]["Probability_Positive"])
            require(probability.is_finite() and 0 <= probability <= 1, f"Invalid baseline probability: {key}")
            require(binary(baseline[key]["Prediction"]) == int(probability >= threshold),
                    f"Baseline prediction/threshold mismatch: {key}")

    candidates, compared_cases, compared_traces = [], set(), 0
    for line, trace in enumerate(traces, 1):
        if trace.get("task", "").lower() != "glaucoma":
            continue
        key = Path(trace.get("case_id", "")).name
        score = trace.get("evidence", {}).get("retfound_glaucoma_probability_percent")
        if score is None:
            continue
        if key in baseline:
            difference = abs(Decimal(str(score)) - Decimal(baseline[key]["Probability_Positive"]) * 100)
            require(difference <= Decimal("0.005000001"), f"Trace and baseline probabilities differ: {key}")
            compared_cases.add(key)
            compared_traces += 1
        if key == case:
            candidates.append({"line": line, "percent": score, "fingerprint": trace.get("fingerprint", "")})
    require(compared_cases, "No overlapping cases to check trace/baseline consistency")
    require(candidates, "No saved RETFound evidence for the missing case")
    # Never choose among conflicting attempts according to correctness.
    require(len({Decimal(str(r["percent"])) for r in candidates}) == 1, "Conflicting saved scores for missing case")
    prediction, probability_bounds = rounded_score_decision(candidates[-1]["percent"], threshold)
    truth = source_label(source[case])

    fixed_manifest = [dict(r) for r in manifest]
    for r in fixed_manifest:
        if r["Task_Folder"].lower() == "glaucoma" and Path(r["filename"]).name == case:
            r["Ground_Truth"] = str(truth)
    fixed_agent = [dict(r) for r in agent_rows]
    for r in fixed_agent:
        if Path(r["Filename"]).name == case:
            r["Ground_Truth"] = str(truth)
            r["Is_Correct"] = str(int(binary(r["Pred_GL"]) == truth))
    new_row = {field: "" for field in baseline_rows[0]}
    meta = selected_by_id[case]
    row_index = next(i for i, r in enumerate(manifest) if r is meta)
    age = float(meta["Age"])
    new_row.update(Row_Index=str(row_index), Filename=meta["filename"], Model="RETFound", Modality="OCT",
                   Disease="GLAUCOMA", Ground_Truth=str(truth), Decision_Threshold=str(threshold),
                   Prediction=str(prediction), Is_Correct=str(int(prediction == truth)), Age=meta["Age"],
                   Age_Group="young" if age < 50 else "older" if age >= 70 else "middle",
                   Gender=meta["Gender"], Race=meta["Race"], Recovery_Source="rounded_saved_agent_evidence",
                   Probability_Percent_Rounded=str(candidates[-1]["percent"]))
    # Deliberately leave Probability_Positive empty: exact precision is unavailable.
    fixed_baseline = [{**r, "Recovery_Source": "original_export", "Probability_Percent_Rounded": ""}
                      for r in baseline_rows] + [new_row]
    fixed_baseline.sort(key=lambda r: int(r["Row_Index"]))
    before = {
        "RETFound": metrics((r["Ground_Truth"], r["Prediction"]) for r in baseline_rows),
        "Ours": metrics((r["Ground_Truth"], r["Pred_GL"]) for r in agent_rows if Path(r["Filename"]).name != case),
    }
    after = {
        "RETFound": metrics((r["Ground_Truth"], r["Prediction"]) for r in fixed_baseline),
        "Ours": metrics((r["Ground_Truth"], r["Pred_GL"]) for r in fixed_agent),
    }
    report = {"case_id": case, "ground_truth": truth, "agent_prediction": binary(agent[case]["Pred_GL"]),
              "retfound_prediction": prediction, "retfound_threshold": float(threshold),
              "retfound_probability_percent_rounded": candidates[-1]["percent"],
              "retfound_probability_bounds": probability_bounds, "exact_probability_recovered": False,
              "matching_trace_cases": len(compared_cases), "matching_trace_records": compared_traces,
              "case_traces": candidates, "before": before, "after": after,
              "npz_verified": False,
              "limitations": ["Reference label comes from the dataset summary, not model agreement.",
                              "Trace scores match the baseline on overlapping cases; this is not checkpoint-hash verification.",
                              "Do not compute full-cohort AUROC or calibration by silently dropping the missing exact probability.",
                              "Worst-group F1 and other model rows are not recalculated here."]}
    return fixed_manifest, fixed_agent, fixed_baseline, report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "OphthalmicAgent/data/fairvision_250each.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "OphthalmicAgent/data/Glaucoma/data_summary_glaucoma.csv")
    parser.add_argument("--agent", type=Path, default=ROOT / "OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "OphthalmicAgent/_extras/CSVs/raw_model_predictions/retfound_glaucoma_predictions.csv")
    parser.add_argument("--traces", type=Path, default=ROOT / "OphthalmicAgent/outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl")
    parser.add_argument("--case-npz", type=Path, help="Optional independent verification of the original case's glaucoma label")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {key: getattr(args, key).resolve() for key in ("manifest", "summary", "agent", "baseline", "traces")}
    traces = [json.loads(line) for line in paths["traces"].read_text().splitlines() if line.strip()]
    manifest, agent, baseline, report = recover(*(read_csv(paths[key]) for key in ("manifest", "summary", "agent", "baseline")), traces)
    if args.case_npz:
        import numpy as np
        require(args.case_npz.name == CASE, "Wrong case NPZ filename")
        with np.load(args.case_npz, allow_pickle=False) as data:
            require(binary(np.asarray(data["glaucoma"]).item()) == report["ground_truth"], "NPZ label conflicts with dataset summary")
        paths["case_npz"] = args.case_npz.resolve()
        report["npz_verified"] = True
    report["sources"] = {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for key, path in paths.items()}
    outputs = {"manifest_recovered.csv": manifest, "agent_predictions_recovered.csv": agent,
               "retfound_predictions_recovered.csv": baseline}
    directory = args.out_dir.resolve()
    for name in (*outputs, "recovery_report.json", "recovery_report.md"):
        require((directory / name).resolve() not in paths.values(), "Refusing to overwrite an original input")
        require(not (directory / name).is_symlink(), "Refusing a symlink output")
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "recovery_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = ["# FairVision Glaucoma Case Recovery", "",
            f"Case: `{CASE}`. Dataset summary label: {report['ground_truth']}. Existing agent decision: {report['agent_prediction']}.",
            f"Saved RETFound score: {report['retfound_probability_percent_rounded']}%; threshold: {report['retfound_threshold'] * 100:g}%; recovered decision: {report['retfound_prediction']}.",
            f"Trace/baseline checks: {report['matching_trace_records']} matching records across {report['matching_trace_cases']} cases.",
            "Original files, predictions, prompts and figures are unchanged. Corrected copies are in this directory.",
            f"Original NPZ independently checked: {report['npz_verified']}.", "",
            "| Export | Model | N | Weighted F1 | Sensitivity | Specificity | Balanced accuracy |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for version in ("before", "after"):
        for model, values in report[version].items():
            text.append(f"| {version} | {model} | {values['n']} | " + " | ".join(
                f"{values[k]:.4f}" for k in ("weighted_f1", "sensitivity", "specificity", "balanced_accuracy")) + " |")
    text += ["", *[f"- {note}" for note in report["limitations"]], "",
             "The skipped RETFound row's exact probability remains blank; the rounded trace score is stored in a separate column."]
    (directory / "recovery_report.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"\nwrote={directory}")


if __name__ == "__main__":
    main()

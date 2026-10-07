"""Read-only review of saved FairVision glaucoma disagreements. No API calls.

Keep every trace version: historical prediction rows do not link their final
orchestrator response to a counterfactual fingerprint. A counterfactual rationale
is not a verified explanation of the exported final decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path

import audit_fairvision_glaucoma_uncertainty as uncertainty
from CounterfactualAgent.counterfactual_agent import SCENARIOS

runner = uncertainty.runner
ROOT = runner.ROOT


def load_traces(path, locked, baseline):
    result = defaultdict(list)
    with path.open() as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            record = json.loads(text)
            key = record["case_id"]
            if key not in locked:
                continue
            runner.require(record["task"] == "glaucoma", f"Wrong trace task at line {line}")
            payload = {k: record[k] for k in ("case_id", "deployment", "evidence", "prompt_version")}
            runner.require(runner.digest(payload) == record["fingerprint"], f"Trace fingerprint mismatch at line {line}")
            scenarios = uncertainty.indexed(record["scenarios"], "name")
            runner.require(set(scenarios) == set(SCENARIOS), f"Incomplete scenarios at line {line}")
            runner.require(all(type(s["diagnosis"]) is int and s["diagnosis"] in (-1, 0, 1)
                               for s in scenarios.values()), f"Invalid scenario label at line {line}")
            runner.require(scenarios["full_evidence"]["diagnosis"] == record["full_evidence_diagnosis"],
                           f"Full-evidence label mismatch at line {line}")
            score = Decimal(str(record["evidence"]["retfound_glaucoma_probability_percent"]))
            runner.require(score.is_finite() and 0 <= score <= 100, f"Invalid trace probability at line {line}")
            original = baseline[key]
            if original["Probability_Positive"]:
                expected = Decimal(original["Probability_Positive"]) * 100
                runner.require(abs(score - expected) <= Decimal("0.005000001"),
                               f"Trace/baseline score mismatch for {key} at line {line}")
            else:
                runner.require(score == Decimal(original["Probability_Percent_Rounded"]),
                               f"Recovered trace score mismatch for {key} at line {line}")
            result[key].append(dict(source_line=line, record=record))
    return result


def build_review(cases, predictions, traces, baseline, include_shared_errors=False):
    rows, packets = [], []
    for i, case in enumerate(cases):
        a, b, truth = predictions["RetinAgent"][i], predictions["RETFound"][i], case["truth"]
        if a == b and (not include_shared_errors or a == truth):
            continue
        key = case["case_id"]
        matches = traces.get(key, [])
        labels = sorted({t["record"]["full_evidence_diagnosis"] for t in matches})
        scenarios = [uncertainty.indexed(t["record"]["scenarios"], "name") for t in matches]
        row = dict(case_id=key, truth=truth, retfound_prediction=b, agent_prediction=a,
                   outcome=("shared_error" if a == b else "regression" if b == truth else "correction"),
                   error_type=("false_negative" if truth else "false_positive") if a != truth else "",
                   retfound_probability=baseline[key]["Probability_Positive"],
                   retfound_threshold=baseline[key]["Decision_Threshold"],
                   trace_records=len(matches), trace_versions=len({t["record"]["fingerprint"] for t in matches}),
                   source_lines=",".join(str(t["source_line"]) for t in matches),
                   full_evidence_labels=",".join(str(label) for label in labels),
                   all_saved_full_labels_differ_from_final=bool(matches) and a not in labels,
                   final_response_link="unavailable in prediction CSV",
                   cdr_values="; ".join(sorted({str(t["record"]["evidence"]["vertical_cup_to_disc_ratio"]) for t in matches})))
        for name in SCENARIOS[1:]:
            row[f"any_trace_flip_{name}"] = (any(s[name]["diagnosis"] != s["full_evidence"]["diagnosis"]
                                                for s in scenarios) if matches else None)
        rows.append(row)
        packets.append(dict(comparison=row, saved_traces=matches))
    return rows, packets


def quote(text):
    return "\n".join("> " + line for line in str(text).splitlines()) + "\n"


def case_report(packet, source):
    c = packet["comparison"]
    lines = [f"# {Path(c['case_id']).stem} {c['outcome']}", "",
        f"Dataset label: **{c['truth']}**. RETFound: **{c['retfound_prediction']}**. Exported agent: **{c['agent_prediction']}**.",
        f"RETFound probability: {float(c['retfound_probability'])*100:.2f}%; decision threshold: {float(c['retfound_threshold'])*100:g}%.", "",
        "These are saved counterfactual-agent outputs, not the final orchestrator response. "
        "No trace version is selected as the one used by the final decision. Labels: 1 positive, 0 negative, -1 inconclusive.", ""]
    if not packet["saved_traces"]:
        lines.append("No trace available in the selected source file.")
    for saved in packet["saved_traces"]:
        r, line = saved["record"], saved["source_line"]
        e = r["evidence"]
        lines += [f"## Saved Trace Line {line}", "", f"Source: [{source.name}:{line}]({source.resolve()}:{line}).",
                  f"Fingerprint: `{r['fingerprint']}`. Deployment: `{r['deployment']}`. Prompt: `{r['prompt_version']}`.",
                  f"CDR: {e['vertical_cup_to_disc_ratio']}; reliability trust: {e['demographic_reliability_trust_score']}.", "",
                  "| Scenario | Label |", "|---|---:|"]
        lines += [f"| {s['name']} | {s['diagnosis']} |" for s in r["scenarios"]]
        for title, field in (("Patient Narrative", "patient_narrative"), ("OCT Report", "oct_specialist_report"),
                             ("SLO Report", "slo_specialist_report")):
            lines += ["", f"### {title}", "", quote(e[field])]
        for s in r["scenarios"]:
            lines += ["", f"### {s['name']}", "", quote(s["reasoning"])]
        lines += ["", "### Counterfactual Interpretation", "", quote(r["interpretation"])]
    return "\n".join(lines) + "\n"


def summary_report(rows, include_shared_errors=False):
    regressions = [r for r in rows if r["outcome"] == "regression"]
    corrections = [r for r in rows if r["outcome"] == "correction"]
    shared = [r for r in rows if r["outcome"] == "shared_error"]
    errors = [r for r in rows if r["error_type"]]
    counts = Counter(r["error_type"] for r in regressions)
    title = "Error And Disagreement" if include_shared_errors else "Disagreement"
    lines = [f"# FairVision Glaucoma {title} Trace Review", "",
        f"RETFound was correct and RetinAgent wrong on {len(regressions)} cases: "
        f"{counts['false_positive']} false alarms and {counts['false_negative']} missed positives. "
        f"RetinAgent corrected {len(corrections)} other RETFound errors.", "",
        "All available versions are retained. The prediction CSV contains final labels but no raw final response "
        "or counterfactual fingerprint. Counterfactual and final labels may legitimately differ; a mismatch "
        "does not establish a parsing error or an incorrect override by the orchestrator.", "",
        ]
    if include_shared_errors:
        lines += [f"All {len(errors)} RetinAgent errors are included: "
            f"{sum(r['error_type'] == 'false_negative' for r in errors)} false negatives and "
            f"{sum(r['error_type'] == 'false_positive' for r in errors)} false positives. "
            f"{len(shared)} errors are shared with RETFound.", "",
            f"Traces are available for {sum(r['trace_records'] > 0 for r in errors)}/{len(errors)} error cases. "
            "Missing traces remain missing, not inferred from another cache.", ""]
    groups = [("Regression", regressions), ("Correction", corrections)]
    if include_shared_errors:
        groups += [("Shared error", shared), ("All agent errors", errors)]
    lines += ["## Trace Coverage", "",
        "| Outcome | Cases | Cases with traces | Multiple versions | All saved full-evidence labels differ from final |",
        "|---|---:|---:|---:|---:|"]
    for outcome, selected in groups:
        lines.append(f"| {outcome} | {len(selected)} | {sum(r['trace_records'] > 0 for r in selected)} | "
                     f"{sum(r['trace_versions'] > 1 for r in selected)} | "
                     f"{sum(r['all_saved_full_labels_differ_from_final'] for r in selected)} |")
    lines += ["", "## Saved Scenario Sensitivity", "",
        "Counts are cases with at least one saved trace in which the named scenario differs from that trace's "
        "full-evidence label, including switches to/from -1. These are model-reported hypothetical scenarios "
        "from one request containing all evidence, not independent reruns with evidence actually removed.", "",
        "Denominators below include only cases with a trace. A case counts once if any saved version changes; "
        "multiple versions give more opportunities for a change. These are not causal attribution counts.", "",
        "| Source described as removed | " + " | ".join(name for name, _ in groups) + " |",
        "|---|" + "---:|" * len(groups)]
    for scenario in SCENARIOS[1:]:
        values = [f"{sum(bool(r[f'any_trace_flip_{scenario}']) for r in selected)}/"
                  f"{sum(r['trace_records'] > 0 for r in selected)}" for _, selected in groups]
        lines.append(f"| {scenario} | " + " | ".join(values) + " |")
    sections = [("Regressions", regressions), ("Corrections For Comparison", corrections)]
    if include_shared_errors:
        sections.append(("Errors Shared With RETFound", shared))
    for title, selected in sections:
        lines += ["", f"## {title}", "", "| Case | Truth | RETFound % | Final agent | CDR values | Saved full labels | Trace lines |",
                  "|---|---:|---:|---:|---|---|---|"]
        for r in selected:
            case = Path(r["case_id"]).stem
            lines.append(f"| [{case}](cases/{case}.md) | {r['truth']} | {float(r['retfound_probability'])*100:.2f} | "
                         f"{r['agent_prediction']} | {r['cdr_values']} | {r['full_evidence_labels']} | {r['source_lines']} |")
    lines += ["", "## Limits And Next Evidence", "",
        "- No new inference, relabelling or selection of the best trace attempt was performed.",
        "- The run-specific trace file is used; unrelated caches are not mixed into the analysis.",
        "- Probability agreement and trace fingerprint validation establish internal consistency, not checkpoint identity or a final-response link.",
        "- Recover the original stdout/log or final-response records, including parsing messages, before assigning final-stage causes.",
        "- Inspect the original model input images and disc/cup masks before calling visual descriptions or CDR measurements incorrect.",
        "- Compare candidate fixes against corrections as well as regressions on development data. Inspecting these test failures makes changes exploratory; it does not create new held-out evidence.",
        "- Raw records, line numbers and fingerprints are preserved in case_packets.json. Predictions and running experiment code are unchanged."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery")
    parser.add_argument("--traces", type=Path, default=ROOT / "OphthalmicAgent/outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_trace_review_v1")
    parser.add_argument("--include-shared-errors", action="store_true",
                        help="Include every agent error, including cases RETFound also got wrong")
    args = parser.parse_args()
    cases, predictions, _, sources, _ = uncertainty.load_paper(args.paper_root, 250)
    baseline = uncertainty.indexed(runner.read_csv(args.paper_root / "retfound_predictions_recovered.csv"), "Filename")
    traces = load_traces(args.traces, {c["case_id"] for c in cases}, baseline)
    rows, packets = build_review(cases, predictions, traces, baseline, args.include_shared_errors)
    artifacts = {"report.md": summary_report(rows, args.include_shared_errors),
                 "case_packets.json": json.dumps(packets, indent=2) + "\n"}
    for packet in packets:
        key = Path(packet["comparison"]["case_id"]).stem
        path = f"cases/{key}.md"
        runner.require(path not in artifacts, f"Duplicate case filename: {key}")
        artifacts[path] = case_report(packet, args.traces)
    sources.append(args.traces)
    source_paths = {p.resolve() for p in sources}
    csv_files = {"comparisons.csv": rows}
    if args.include_shared_errors:
        csv_files["errors.csv"] = [r for r in rows if r["error_type"]]
        csv_files["saved_full_evidence_reasoning.csv"] = [
            dict(case_id=p["comparison"]["case_id"], outcome=p["comparison"]["outcome"],
                 truth=p["comparison"]["truth"], final_prediction=p["comparison"]["agent_prediction"],
                 source=str(args.traces), source_line=t["source_line"], fingerprint=t["record"]["fingerprint"],
                 trace_label=t["record"]["full_evidence_diagnosis"],
                 reasoning=next(s["reasoning"] for s in t["record"]["scenarios"] if s["name"] == "full_evidence"))
            for p in packets for t in p["saved_traces"]]
    for name in (*artifacts, *csv_files, "provenance.json"):
        target = args.out_dir / name
        runner.require(not target.is_symlink() and target.resolve() not in source_paths, "Unsafe output path")
    for name, text in artifacts.items():
        target = args.out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for name, content in csv_files.items():
        runner.audit.write_csv(args.out_dir / name, content)
    runner.write_json(args.out_dir / "provenance.json", dict(api_calls=0, sources={str(p.resolve()): runner.sha(p) for p in sources},
        code={str(p.relative_to(ROOT)): runner.sha(p) for p in (Path(__file__), Path(uncertainty.__file__), *runner.code_paths())},
        trace_selection="all records from named run-specific file; no final fingerprint link available",
        include_shared_errors=args.include_shared_errors,
        agent_error_cases=sum(bool(r["error_type"]) for r in rows),
        shared_error_cases=sum(r["outcome"] == "shared_error" for r in rows),
        regression_cases=sum(r["outcome"] == "regression" for r in rows), correction_cases=sum(r["outcome"] == "correction" for r in rows)))
    print(artifacts["report.md"])
    print(f"wrote={args.out_dir.resolve()}; no API calls")


if __name__ == "__main__":
    main()

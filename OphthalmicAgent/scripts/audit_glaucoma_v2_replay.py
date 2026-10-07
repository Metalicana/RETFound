"""Offline review of downloaded V2 exports; never creates a ledger or API client."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import run_glaucoma_v2_errors as errors

replay = errors.replay
base = replay.base
DEFAULT_OUT = base.ROOT / "equi-agent/outputs/audits/glaucoma_v2_replay_review_20261007"


def cell(value):
    return "" if value is None else str(value)


def validate_exports(bundle, exported_errors, receipts):
    """Reconstruct results from raw responses before trusting exported diagnoses."""
    cases = {c["case_id"]: c for c in bundle["cases"]}
    attempts = {}
    for receipt in receipts:
        key = receipt["case_id"], receipt["arm"]
        case_id, arm = key
        base.require(key not in attempts, f"Duplicate receipt: {key}")
        base.require(case_id in cases and arm in replay.ARMS, f"Unknown case/arm: {key}")
        request = cases[case_id]["requests"][arm]
        base.require(receipt["request_json"] == request and receipt["request_hash"] == base.digest(request),
                     f"Frozen request mismatch: {key}")
        base.require(receipt["status"] in ("reserved", "received", "failed", "invalid", "valid"),
                     f"Unknown receipt status: {key}")
        if receipt["status"] == "valid":
            parsed = replay.parse_response(arm, receipt["raw"])
            base.require(parsed == receipt["parsed_json"], f"Saved parse mismatch: {key}")
            response = receipt["response_json"]
            choices = response.get("choices", [])
            base.require(bool(choices) and choices[0]["finish_reason"] == "stop" and
                         choices[0]["message"]["content"] == receipt["raw"] and
                         not choices[0]["message"].get("refusal"), f"Invalid response envelope: {key}")
            base.require(isinstance(response.get("model"), str) and bool(response["model"]),
                         f"Missing returned model: {key}")
        else:
            base.require(receipt["parsed_json"] is None, f"Nonvalid attempt has a parsed result: {key}")
        attempts[key] = receipt
    base.require(len(attempts) <= bundle["api_budget"], "Export exceeds frozen call budget")

    rows = []
    for evaluation in bundle["evaluation"]:
        row = dict(evaluation)
        case = cases.get(row["case_id"])
        if case:
            row.update(source_line=case["source_line"], evidence_sha256=case["evidence_sha256"],
                       cdr_status=replay.measurement_checks(case["evidence"])["cdr_status"],
                       saved_full_evidence_label=case["trace"]["full_evidence_diagnosis"])
        else:
            row.update(source_line="", evidence_sha256="", cdr_status="missing_evidence",
                       saved_full_evidence_label="")
        for arm in replay.ARMS:
            receipt = attempts.get((row["case_id"], arm))
            status = receipt["status"] if receipt else ("not_attempted" if case else "missing_evidence")
            parsed = receipt["parsed_json"] if status == "valid" else {}
            row[arm + "_status"] = status
            for target, source in (("prediction", "diagnosis"), ("reasoning", "reasoning"),
                                   ("escalation_required", "escalation_required"),
                                   ("escalation_reason", "escalation_reason")):
                row[arm + "_" + target] = parsed.get(source, "")
        rows.append(row)
    expected = {r["case_id"]: r for r in rows if r["group"] == "historical_error"}
    actual = {r["case_id"]: r for r in exported_errors}
    base.require(len(actual) == len(exported_errors), "Duplicate exported error case")
    base.require(set(actual) == set(expected), "Export must retain all historical errors, including missing evidence")
    for case_id, row in expected.items():
        base.require({k: cell(v) for k, v in row.items()} == actual[case_id],
                     f"CSV/receipt/bundle mismatch: {case_id}")
    return rows, attempts


def summarize(bundle, rows, receipts):
    selected, result = errors.analyse(rows)
    valid = [r for r in selected if r["v2_status"] == "valid"]
    pairs = [r for r in selected if all(r[a + "_status"] == "valid" for a in replay.ARMS)]
    result["diagnosis_disagreements"] = sum(r[replay.ARMS[0] + "_prediction"] != r["v2_prediction"] for r in pairs)
    result["repaired_case_ids"] = [r["case_id"] for r in valid if r["v2_prediction"] == r["truth"]]
    result["missing_evidence_case_ids"] = [r["case_id"] for r in selected if not r["eligible"]]
    result["trace_agreement"] = {}
    for arm in replay.ARMS:
        assessed = [r for r in selected if r[arm + "_status"] == "valid"]
        binary_trace = [r for r in assessed if r["saved_full_evidence_label"] in (0, 1)]
        result["trace_agreement"][arm] = dict(valid=len(assessed), binary_trace=len(binary_trace),
            matches=sum(r[arm + "_prediction"] == r["saved_full_evidence_label"] for r in binary_trace))
    result["v2_review_flags"] = {
        outcome: dict(valid=len(group), flagged=sum(r["v2_escalation_required"] is True for r in group))
        for outcome, group in (
            ("repaired", [r for r in valid if r["v2_prediction"] == r["truth"]]),
            ("still_wrong", [r for r in valid if r["v2_prediction"] != r["truth"]]),
            ("all_valid_errors", valid))}
    result["anchor_comparison"] = {}
    for correct, name in ((True, "retfound_correct_agent_wrong"), (False, "shared_historical_error")):
        group = [r for r in selected if (r["retfound_prediction"] == r["truth"]) == correct]
        assessed = [r for r in group if r["v2_status"] == "valid"]
        repaired = sum(r["v2_prediction"] == r["truth"] for r in assessed)
        result["anchor_comparison"][name] = dict(expected=len(group), valid=len(assessed), repaired=repaired,
            still_wrong=len(assessed) - repaired, unassessed=len(group) - len(assessed))
    controls = [r for r in rows if r["group"] == "historical_correction_control"]
    result["correction_controls"] = {}
    for arm in replay.ARMS:
        assessed = [r for r in controls if r[arm + "_status"] == "valid"]
        result["correction_controls"][arm] = dict(expected=len(controls), valid=len(assessed),
            correct=sum(r[arm + "_prediction"] == r["truth"] for r in assessed))
    usage = Counter()
    for receipt in receipts:
        response = receipt.get("response_json") or {}
        usage.update({k: v for k, v in (response.get("usage") or {}).items()
                      if k in ("prompt_tokens", "completion_tokens", "total_tokens") and type(v) is int})
    result["receipts"] = dict(count=len(receipts), statuses=dict(Counter(r["status"] for r in receipts)),
        returned_models=dict(Counter(r["response_json"]["model"] for r in receipts if r["status"] == "valid")),
        first_started_utc=min((r["started_utc"] for r in receipts if r["started_utc"]), default=None),
        last_finished_utc=max((r["finished_utc"] for r in receipts if r["finished_utc"]), default=None),
        reported_token_usage=dict(usage))
    result["bundle_fingerprint"] = bundle["fingerprint"]
    return result


def case_text(row, case, attempts):
    lines = [row["case_id"], "", "EVALUATION LABELS (not sent to the API)",
             json.dumps({k: row[k] for k in ("truth", "historical_prediction", "retfound_prediction", "group")}, indent=2)]
    if case:
        lines += ["", f"Frozen source trace line: {case['source_line']}", "SAVED EVIDENCE (not revalidated imaging)",
                  json.dumps(case["evidence"], indent=2, ensure_ascii=True), "", "SAVED COUNTERFACTUAL TRACE",
                  json.dumps(case["trace"], indent=2, ensure_ascii=True)]
    else:
        lines += ["", "No eligible saved evidence; no prediction or rationale imputed."]
    for arm in replay.ARMS:
        receipt = attempts.get((row["case_id"], arm))
        lines += ["", arm.upper() + ": " + row[arm + "_status"]]
        if receipt:
            lines += [f"Request SHA256: {receipt['request_hash']}", "RAW SAVED RESPONSE", receipt["raw"] or "(no response)"]
    return "\n".join(lines) + "\n"


def report_text(summary, rows):
    lines = ["V2 GLAUCOMA REPLAY: OFFLINE EXPORT AUDIT", "",
             f"Receipts checked: {summary['receipts']['count']}; statuses: {summary['receipts']['statuses']}",
             f"Returned model strings: {summary['receipts']['returned_models']}",
             f"Historical errors: {summary['historical_errors']}; eligible: {summary['eligible_errors']}; "
             f"paired valid: {summary['paired_valid']}.", ""]
    for arm, data in summary["arms"].items():
        lines.append(f"{arm}: {data['repaired']} repaired, {data['still_wrong']} still wrong, {data['unassessed']} unassessed.")
    lines += [f"Paired outcomes: {summary['paired_outcomes']}",
              f"Arm diagnosis disagreements: {summary['diagnosis_disagreements']}", "", "REPAIRED CASES",
              "Case           Truth Historical RETFound Saved trace Control V2"]
    for row in rows:
        if row["case_id"] in summary["repaired_case_ids"]:
            labels = [row[k] for k in ("truth", "historical_prediction", "retfound_prediction",
                      "saved_full_evidence_label", "legacy_prompt_control_prediction", "v2_prediction")]
            lines.append(Path(row["case_id"]).stem + "  " + "  ".join(str(v) for v in labels))
    lines += ["", "FALSE NEGATIVE / FALSE POSITIVE BREAKDOWN (V2)",
              json.dumps(summary["arms"]["v2"]["error_types"], indent=2), "", "ANCHOR COMPARISON (V2)",
              json.dumps(summary["anchor_comparison"], indent=2), "", "AGREEMENT WITH SAVED FULL-EVIDENCE TRACE",
              json.dumps(summary["trace_agreement"], indent=2), "", "V2 EXPLICIT REVIEW FLAGS",
              json.dumps(summary["v2_review_flags"], indent=2), "", "HISTORICAL CORRECTION CONTROLS",
              json.dumps(summary["correction_controls"], indent=2), "", "INTERPRETATION LIMITS",
              summary["interpretation"],
              "Trace agreement is not proof of causal anchoring; the same underlying evidence is shared.",
              "Legacy control has no matched structured review flag; do not infer absence of review.",
              "Replay evidence is a saved first trace, not a verified historical final-stage request.",
              "No original images or segmentation masks were inspected. Rationales are model statements, not verified anatomy.",
              "No full-cohort F1 is recomputed. No new predictions replace historical results.",
              "The receipt export was checked internally against the frozen bundle; it is not a signed provider audit.",
              "", "See cases.csv and cases/*.txt for every selected error and attempted correction control."]
    return "\n".join(lines) + "\n"


def audit(bundle_path, run_root, out):
    sources = [bundle_path, run_root / "failed_case_results.csv", run_root / "api_receipts.jsonl", Path(__file__)]
    out = out.resolve()
    base.require(not out.is_relative_to(run_root.resolve()) and not run_root.resolve().is_relative_to(out) and
                 not out.is_relative_to(bundle_path.parent.resolve()) and
                 not bundle_path.resolve().is_relative_to(out), "Audit output must be separate from frozen inputs and run")
    hashes = {str(p.resolve()): base.sha(p) for p in sources}
    bundle = replay.load_bundle(bundle_path)
    receipts = [json.loads(line) for line in sources[2].read_text().splitlines() if line.strip()]
    rows, attempts = validate_exports(bundle, base.read_csv(sources[1]), receipts)
    summary = summarize(bundle, rows, receipts)
    cases = {c["case_id"]: c for c in bundle["cases"]}
    selected = [r for r in rows if r["group"] == "historical_error" or any((r["case_id"], a) in attempts for a in replay.ARMS)]
    (out / "cases").mkdir(parents=True, exist_ok=True)
    for row in selected:
        case = cases.get(row["case_id"])
        row["retfound_probability_percent"] = case["evidence"]["retfound_glaucoma_probability_percent"] if case else ""
        row["saved_cdr"] = case["evidence"]["vertical_cup_to_disc_ratio"] if case else ""
        receipt = attempts.get((row["case_id"], "v2"))
        row["v2_overview"] = receipt["parsed_json"]["overview"] if row["v2_status"] == "valid" else ""
        (out / "cases" / (Path(row["case_id"]).stem + ".txt")).write_text(case_text(row, case, attempts))
    base.audit.write_csv(out / "cases.csv", selected)
    index_fields = ("case_id", "group", "truth", "historical_prediction", "retfound_prediction",
                    "retfound_probability_percent", "saved_cdr", "saved_full_evidence_label",
                    "legacy_prompt_control_status", "legacy_prompt_control_prediction", "v2_status",
                    "v2_prediction", "v2_escalation_required", "v2_overview")
    base.audit.write_csv(out / "case_index.csv", [{k: r[k] for k in index_fields} for r in selected])
    base.write_json(out / "summary.json", summary)
    (out / "report.txt").write_text(report_text(summary, rows))
    base.require(hashes == {str(p.resolve()): base.sha(p) for p in sources}, "Input changed during audit")
    base.write_json(out / "provenance.json", dict(audited_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes, runtime_code_sha256=replay.code_hashes(),
        analysis_helper_sha256=base.sha(Path(errors.__file__)), bundle_fingerprint=bundle["fingerprint"],
        paid_calls=0, original_images_inspected=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=replay.DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=replay.DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.bundle, args.run_root, args.out_dir)
    print(f"Verified {result['receipts']['count']} exported attempts. Offline report: {args.out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()

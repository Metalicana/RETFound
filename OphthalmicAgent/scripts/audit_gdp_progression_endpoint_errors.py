"""Audit saved staged GDP predictions and cross-endpoint errors without API calls.

Reads frozen case evidence, evaluation labels and final case traces. Does not
load images, train models, change predictions or import an API client.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


TARGETS = ("md", "vfi", "td_pointwise", "md_fast", "md_fast_no_p_cut", "td_pointwise_no_p_cut")
PRIMARY = "td_pointwise_no_p_cut"
SCENARIOS = ("full_evidence", "without_helper_predictions", "without_structural_interpretation",
             "without_functional_interpretation", "without_demographic_reliability")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def binary(value):
    require(not isinstance(value, bool), f"Boolean is not a prediction label: {value!r}")
    number = float(value)
    require(number in (0, 1), f"Invalid binary label: {value!r}")
    return int(number)


def probability(value):
    number = float(value)
    require(math.isfinite(number) and 0 <= number <= 1, f"Invalid probability: {value!r}")
    return number


def nonempty(value):
    require(isinstance(value, str) and bool(value.strip()), "Missing saved explanation")
    return value


def index(rows, field, expected, *, filenames=False):
    result = {}
    for row in rows:
        key = Path(row[field]).stem if filenames else row[field]
        require(key not in result, f"Duplicate case: {key}")
        result[key] = row
    require(set(result) == set(expected), f"Case alignment mismatch for {field}")
    return result


def load_run(root, expected_cases=200):
    config = read_json(root / "resolved_config.json")
    fingerprint = config["fingerprint"]
    require(fingerprint == digest({k: v for k, v in config.items() if k != "fingerprint"}),
            "Saved configuration fingerprint is invalid")
    require(set(config["targets"]) == set(TARGETS), "Expected all six progression endpoints")
    ids = config["case_ids"]
    require(len(ids) == len(set(ids)) == expected_cases, f"Expected {expected_cases} unique frozen cases")
    require(all(Path(c).name == c and c not in (".", "..") for c in ids), "Invalid frozen case ID")
    summary = read_json(root / "summary.json")
    require(summary.get("run_fingerprint") == fingerprint, "Summary/config fingerprint mismatch")
    require(summary.get("dry_run") is False and summary.get("complete_live_cohort") is True
            and summary.get("completed_cases") == expected_cases
            and summary.get("expected_cases") == expected_cases and not summary.get("missing_cases"),
            "Audit requires a complete live cohort, not a partial smoke or dry run")
    prepared = read_json(root / "prepared_cases.json")
    require(digest(prepared) == config["evidence_sha256"], "Saved evidence hash mismatch")
    prepared = index(prepared, "case_id", ids)
    answers = read_json(root / "evaluation_labels.json")
    require(set(answers) == set(TARGETS), "Evaluation labels must cover exactly six endpoints")
    labels = {}
    for target, rows in answers.items():
        rows = index(rows, "image_id", ids, filenames=True)
        require(all(r["split"] == "test" and r["task"] == "progression_forecasting" for r in rows.values()),
                f"Non-test/non-progression labels: {target}")
        labels[target] = {c: binary(r["y_true"]) for c, r in rows.items()}
    require({p.stem for p in (root / "cases").glob("*.json")} == set(ids), "Incomplete or extra final case files")
    records = []
    for case_id in ids:
        case = prepared[case_id]
        saved = read_json(root / "cases" / f"{case_id}.json")
        require(saved.get("case_id") == case_id and saved.get("run_fingerprint") == fingerprint,
                f"Final trace provenance mismatch: {case_id}")
        require(set(saved["predictions"]) == set(case["helper_predictions"]) == set(case["reliability"]) == set(TARGETS),
                f"Missing or extra endpoints: {case_id}")
        counterfactual = saved["counterfactual"]
        scenarios = counterfactual["scenarios"]
        require(set(scenarios) == set(SCENARIOS), f"Invalid counterfactual scenarios: {case_id}")
        for name, scenario in scenarios.items():
            require(set(scenario["predictions"]) == set(TARGETS), f"Invalid scenario endpoints: {case_id}/{name}")
            nonempty(scenario["reasoning"])
        nonempty(counterfactual["interpretation"])
        for target in TARGETS:
            helper = case["helper_predictions"][target]
            final = saved["predictions"][target]
            reliability = case["reliability"][target]
            require(reliability.get("source") == "development_oof" and reliability.get("development_cases") == 300,
                    f"Expected development-OOF reliability: {case_id}/{target}")
            y, h, a = labels[target][case_id], binary(helper["prediction"]), binary(final["prediction"])
            p, threshold = probability(helper["probability"]), probability(helper["source_threshold"])
            require(h == int(p >= threshold), f"Helper threshold/label mismatch: {case_id}/{target}")
            require(type(final["review_required"]) is bool, f"Invalid review flag: {case_id}/{target}")
            transition = ("unchanged_correct" if a == y else "unchanged_wrong") if h == a else (
                "corrected_false_negative" if y == a == 1 else
                "corrected_false_positive" if y == a == 0 else
                "introduced_false_positive" if a == 1 else "introduced_false_negative")
            row = {"case_id": case_id, "target": target, "y_true": y,
                   "helper_probability": p, "helper_threshold": threshold, "helper_prediction": h,
                   "agent_prediction": a, "transition": transition, "review_required": final["review_required"],
                   "trust_score": probability(reliability["trust_score"]),
                   **{f"oof_{key}": reliability["global"][key] for key in ("auroc", "f1", "ece", "fpr", "fnr")},
                   **{f"cf_{s}": binary(scenarios[s]["predictions"][target]) for s in SCENARIOS},
                   "agent_reasoning": nonempty(final["reasoning"]),
                   "cf_full_reasoning": scenarios["full_evidence"]["reasoning"],
                   "cf_without_helper_reasoning": scenarios["without_helper_predictions"]["reasoning"],
                   "cf_interpretation": counterfactual["interpretation"]}
            records.append(row)
    paths = [root / name for name in ("resolved_config.json", "summary.json", "prepared_cases.json", "evaluation_labels.json")]
    paths.extend(root / "cases" / f"{case_id}.json" for case_id in ids)
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return config, records, sources


def ratio(a, b):
    return a / b if b else None


def metrics(rows, column):
    tp = sum(r["y_true"] == r[column] == 1 for r in rows)
    tn = sum(r["y_true"] == r[column] == 0 for r in rows)
    fp = sum(r["y_true"] == 0 and r[column] == 1 for r in rows)
    fn = sum(r["y_true"] == 1 and r[column] == 0 for r in rows)
    positive = ratio(2 * tp, 2 * tp + fp + fn) or 0.0
    negative = ratio(2 * tn, 2 * tn + fp + fn) or 0.0
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "positive_f1": positive,
            "macro_f1": (positive + negative) / 2,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if None not in (sensitivity, specificity) else None}


def analyze(records):
    endpoint_rows, new_fp, by_case = [], {}, {}
    for row in records:
        by_case.setdefault(row["case_id"], {})[row["target"]] = row
    for target in TARGETS:
        rows = [r for r in records if r["target"] == target]
        added = [r for r in rows if r["transition"] == "introduced_false_positive"]
        new_fp[target] = {r["case_id"] for r in added}
        endpoint_rows.append({"target": target, "n": len(rows), "positives": sum(r["y_true"] for r in rows),
            **{f"helper_{k}": v for k, v in metrics(rows, "helper_prediction").items()},
            **{f"agent_{k}": v for k, v in metrics(rows, "agent_prediction").items()},
            "corrections": sum(r["transition"].startswith("corrected_") for r in rows),
            "regressions": sum(r["transition"].startswith("introduced_") for r in rows),
            "new_false_positives": len(added),
            "new_fp_already_positive_in_cf_full": sum(r["cf_full_evidence"] == 1 for r in added),
            "new_fp_flipped_only_at_final_stage": sum(r["cf_full_evidence"] == 0 for r in added),
            "new_fp_primary_helper_positive": sum(by_case[r["case_id"]][PRIMARY]["helper_prediction"] for r in added),
            "new_fp_primary_agent_positive": sum(by_case[r["case_id"]][PRIMARY]["agent_prediction"] for r in added),
            "new_fp_primary_truth_positive": sum(by_case[r["case_id"]][PRIMARY]["y_true"] for r in added)})
    overlaps = []
    for a, b in itertools.combinations(TARGETS, 2):
        intersection, union = new_fp[a] & new_fp[b], new_fp[a] | new_fp[b]
        overlaps.append({"target_a": a, "target_b": b, "new_fp_a": len(new_fp[a]), "new_fp_b": len(new_fp[b]),
                         "shared_new_fp": len(intersection), "jaccard": ratio(len(intersection), len(union)),
                         "shared_case_ids": ";".join(sorted(intersection))})
    patterns = []
    for case_id, rows in sorted(by_case.items()):
        affected = [t for t in TARGETS if case_id in new_fp[t]]
        patterns.append({"case_id": case_id, "new_fp_endpoints": len(affected), "new_fp_targets": ";".join(affected),
            **{f"{t}_{label}": rows[t][column] for t in TARGETS for label, column in (
                ("truth", "y_true"), ("helper", "helper_prediction"), ("agent", "agent_prediction"))},
            "primary_helper_probability": rows[PRIMARY]["helper_probability"]})
    patterns.sort(key=lambda r: (-r["new_fp_endpoints"], r["case_id"]))
    return endpoint_rows, overlaps, patterns, by_case


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def quote(value):
    return "\n".join("> " + line for line in value.splitlines())


def report(config, endpoint_rows, overlaps, patterns, by_case, max_trace_cases):
    lines = ["# GDP Progression Endpoint Error Audit", "",
             f"Prompt version: `{config['prompt_version']}`. Cohort: {len(by_case)} cases per endpoint.",
             "Read-only analysis of frozen evidence and saved final traces. No API calls or prediction changes.",
             "A new false positive means truth=0, helper=0, agent=1; existing helper false positives are counted separately in the totals.", "",
             "## Endpoint Summary", "",
             "| Endpoint | Positives | Helper TP/FP | Agent TP/FP | Positive F1 helper -> agent | Macro-F1 helper -> agent | Corrections | Regressions |",
             "|---|---:|---:|---:|---|---|---:|---:|"]
    for r in endpoint_rows:
        lines.append(f"| {r['target']} | {r['positives']} | {r['helper_tp']}/{r['helper_fp']} | {r['agent_tp']}/{r['agent_fp']} | "
                     f"{fmt(r['helper_positive_f1'])} -> {fmt(r['agent_positive_f1'])} | "
                     f"{fmt(r['helper_macro_f1'])} -> {fmt(r['agent_macro_f1'])} | {r['corrections']} | {r['regressions']} |")
    lines += ["", "## Where New False Positives Appear", "",
              "CF full is the counterfactual stage's full-evidence forecast. These are sequential LLM outputs, not independent experiments.", "",
              "| Endpoint | New FP | Already positive in CF full | CF full negative, final positive | Primary helper positive | Primary agent positive | Primary truth positive |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in endpoint_rows:
        cells = [r[k] for k in ("target", "new_false_positives", "new_fp_already_positive_in_cf_full",
                 "new_fp_flipped_only_at_final_stage", "new_fp_primary_helper_positive", "new_fp_primary_agent_positive", "new_fp_primary_truth_positive")]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += ["", f"Primary means `{PRIMARY}`. The last three columns count primary-endpoint positives among each row's new false positives.",
              "", "## Shared New False Positives", "",
              "| Endpoint A | Endpoint B | New FP A/B | Shared cases | Jaccard |",
              "|---|---|---:|---:|---:|"]
    for r in overlaps:
        lines.append(f"| {r['target_a']} | {r['target_b']} | {r['new_fp_a']}/{r['new_fp_b']} | {r['shared_new_fp']} | {fmt(r['jaccard'])} |")
    affected = [r for r in patterns if r["new_fp_endpoints"]]
    lines += ["", f"Patients with any new false positive: {len(affected)}; with new false positives on at least two endpoints: "
              f"{sum(r['new_fp_endpoints'] >= 2 for r in affected)}.", "",
              "Overlap can identify repeated errors, but does not establish that one endpoint's evidence caused another endpoint's prediction. "
              "Endpoint outcomes are correlated. Interpret the saved explanations alongside the numeric evidence."]
    overview = "\n".join(lines) + "\n"
    lines += ["", "## Case Review", "",
              f"Showing up to {max_trace_cases} patients, ordered by number of new false-positive endpoints, then case ID. "
              "Every case-endpoint explanation and evidence score is in case_endpoint_audit.csv."]
    for pattern in affected[:max_trace_cases]:
        case_id = pattern["case_id"]
        rows = by_case[case_id]
        lines += ["", f"### {case_id}", "", f"New false-positive endpoints: {pattern['new_fp_targets']}.", "",
                  "| Endpoint | Truth | Helper p / label | Agent | CF full | CF without helper | OOF F1 | OOF AUROC | OOF ECE | Trust |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for target in TARGETS:
            r = rows[target]
            lines.append(f"| {target} | {r['y_true']} | {r['helper_probability']:.6g} / {r['helper_prediction']} | "
                         f"{r['agent_prediction']} | {r['cf_full_evidence']} | {r['cf_without_helper_predictions']} | "
                         f"{fmt(r['oof_f1'])} | {fmt(r['oof_auroc'])} | {fmt(r['oof_ece'])} | {fmt(r['trust_score'])} |")
        for target in TARGETS:
            r = rows[target]
            if r["helper_prediction"] != r["agent_prediction"] or target == PRIMARY:
                lines += ["", f"**{target}: {r['transition']}**", "", quote(r["agent_reasoning"])]
        lines += ["", "**Counterfactual full-evidence explanation (shared across endpoints)**", "",
                  quote(rows[PRIMARY]["cf_full_reasoning"]), "",
                  "**Counterfactual without-helper explanation (shared across endpoints)**", "",
                  quote(rows[PRIMARY]["cf_without_helper_reasoning"])]
    return overview, "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-trace-cases", type=int, default=5)
    args = parser.parse_args()
    require(args.max_trace_cases >= 0, "--max-trace-cases must be nonnegative")
    root = args.run_dir.expanduser().resolve()
    out = (args.out_dir or root / "audit_endpoint_errors").expanduser().resolve()
    require(out != root and all(not out.is_relative_to(root / name) for name in ("cases", "stage_cache", "assets")),
            "Use a separate audit output directory, not the run root or input/cache directories")
    config, records, sources = load_run(root)
    endpoint_rows, overlaps, patterns, by_case = analyze(records)
    overview, markdown = report(config, endpoint_rows, overlaps, patterns, by_case, args.max_trace_cases)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("endpoint_summary.csv", endpoint_rows), ("case_endpoint_audit.csv", records),
                       ("false_positive_overlap.csv", overlaps), ("case_patterns.csv", patterns)):
        write_csv(out / name, rows)
    (out / "audit_report.md").write_text(markdown)
    (out / "summary.json").write_text(json.dumps({
        "run_dir": str(root), "run_fingerprint": config["fingerprint"], "prompt_version": config["prompt_version"],
        "case_count": len(by_case), "endpoint_rows": endpoint_rows, "api_calls": 0,
        "prediction_changes": 0, "input_sha256": sources,
        "limitations": ["Analysis of saved helper evidence and final case traces, not a rerun or new performance estimate.",
                        "External native checkpoints, source label manifests and image files are not revalidated here.",
                        "Cross-endpoint overlap is descriptive and does not prove causal evidence transfer."]
    }, indent=2, allow_nan=False) + "\n")
    print(overview)
    print(f"Full explanations: {out / 'audit_report.md'}")
    print(f"All case-endpoint rows: {out / 'case_endpoint_audit.csv'}")


if __name__ == "__main__":
    main()

"""Continue the frozen V2 pilot on historical errors only. No new prompts or evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sqlite3

import run_glaucoma_v2_replay as replay

base = replay.base
SCOPE = "historical_errors_only_20261007"


def error_cases(bundle):
    evaluation = {r["case_id"]: r for r in bundle["evaluation"]}
    base.require(len(evaluation) == len(bundle["evaluation"]), "Duplicate evaluation case")
    for row in evaluation.values():
        is_error = row["historical_prediction"] != row["truth"]
        base.require(is_error == (row["group"] == "historical_error"), "Historical error designation mismatch")
    selected = [c for c in bundle["cases"] if evaluation[c["case_id"]]["group"] == "historical_error"]
    base.require({c["case_id"] for c in selected} ==
                 {k for k, r in evaluation.items() if r["group"] == "historical_error" and r["eligible"]},
                 "Eligible historical error IDs do not match frozen inputs")
    return selected


def attempted_keys(bundle, root):
    path = root / "ledger.sqlite3"
    if not path.exists():
        return set()
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        metadata = dict(db.execute("SELECT name, value FROM metadata"))
        base.require(metadata.get("bundle") == bundle["fingerprint"] and
                     metadata.get("budget") == str(bundle["api_budget"]), "Ledger belongs to another run")
        keys = set(db.execute("SELECT case_id, arm FROM attempts"))
        allowed = {(c["case_id"], arm) for c in bundle["cases"] for arm in replay.ARMS}
        base.require(keys <= allowed, "Unknown case/arm in ledger")
        return keys
    finally:
        db.close()


def plan(bundle, root):
    cases = error_cases(bundle)
    targets = {(c["case_id"], arm) for c in cases for arm in replay.ARMS}
    old = attempted_keys(bundle, root)
    errors = [r for r in bundle["evaluation"] if r["group"] == "historical_error"]
    return dict(scope=SCOPE, historical_errors=len(errors), eligible_errors=len(cases),
                missing_evidence=[r["case_id"] for r in errors if not r["eligible"]],
                total_error_requests=len(targets), error_requests_already_reserved=len(targets & old),
                remaining_error_requests=len(targets - old),
                prior_nonerror_requests_retained=len(old - targets),
                original_shared_budget=bundle["api_budget"],
                budget_remaining=bundle["api_budget"] - len(old))


def execute(bundle, root, client_factory):
    selected = error_cases(bundle)
    receipt = dict(scope=SCOPE, bundle_fingerprint=bundle["fingerprint"],
                   selection="all historical errors; no pilot outcomes used to choose cases",
                   continuation_code_sha256=base.sha(Path(__file__)),
                   case_ids=[c["case_id"] for c in selected])
    replay.ensure_run_root(root, bundle)
    with replay.run_lock(root):
        path = root / "errors_only_selection.json"
        if path.exists():
            base.require(json.loads(path.read_text()) == receipt, "Failure-only continuation changed")
        else:
            base.write_json(path, receipt)
    # This is an execution subset, like max_cases, not a new bundle. Keep the
    # original identity, requests and ledger so pilot reservations are reused.
    replay.execute({**bundle, "cases": selected}, root, client_factory)


def analyse(rows):
    errors = [r for r in rows if r["group"] == "historical_error"]
    base.require(len({r["case_id"] for r in errors}) == len(errors), "Duplicate error row")
    report = dict(scope=SCOPE, historical_errors=len(errors),
                  eligible_errors=sum(str(r["eligible"]).lower() == "true" for r in errors), arms={})
    for arm in replay.ARMS:
        valid = [r for r in errors if r[arm + "_status"] == "valid"]
        repairs = sum(base.binary(r[arm + "_prediction"]) == base.binary(r["truth"]) for r in valid)
        report["arms"][arm] = dict(valid=len(valid), repaired=repairs, still_wrong=len(valid) - repairs,
            unassessed=len(errors) - len(valid), repair_rate_among_valid=repairs / len(valid) if valid else None,
            statuses=dict(Counter(r[arm + "_status"] for r in errors)), error_types={})
        for truth, name in ((1, "historical_false_negatives"), (0, "historical_false_positives")):
            group = [r for r in errors if base.binary(r["truth"]) == truth]
            assessed = [r for r in group if r[arm + "_status"] == "valid"]
            correct = sum(base.binary(r[arm + "_prediction"]) == truth for r in assessed)
            report["arms"][arm]["error_types"][name] = dict(expected=len(group), valid=len(assessed),
                repaired=correct, still_wrong=len(assessed) - correct, unassessed=len(group) - len(assessed))
    pairs = [r for r in errors if all(r[arm + "_status"] == "valid" for arm in replay.ARMS)]
    outcomes = Counter()
    for row in pairs:
        a, b = (base.binary(row[arm + "_prediction"]) == base.binary(row["truth"]) for arm in replay.ARMS)
        outcomes["both_repaired" if a and b else "v2_only_repaired" if b else "legacy_only_repaired" if a else "neither_repaired"] += 1
    report["paired_valid"] = len(pairs)
    report["paired_outcomes"] = {k: outcomes[k] for k in
        ("both_repaired", "v2_only_repaired", "legacy_only_repaired", "neither_repaired")}
    report["eligible_error_pairs_complete"] = len(pairs) == report["eligible_errors"]
    report["all_historical_errors_assessed_in_both_arms"] = len(pairs) == len(errors)
    report["interpretation"] = ("Descriptive repair rates on previously failed cases only; not full-cohort accuracy/F1 "
                                "or a significance claim. Correct-case regressions are not assessed by this continuation.")
    return errors, report


def collect(bundle, root):
    # Preserve and validate every pilot result, including any correct-case controls.
    with redirect_stdout(StringIO()):
        replay.collect(bundle, root)
    rows, report = analyse(base.read_csv(root / "case_results.csv"))
    report["plan"] = plan(bundle, root)
    report["bundle_fingerprint"] = bundle["fingerprint"]
    base.audit.write_csv(root / "failed_case_results.csv", rows)
    base.write_json(root / "failed_case_summary.json", report)
    lines = ["V2 FAILURE-ONLY REPLAY", "", f"Historical errors: {report['historical_errors']}; "
             f"eligible saved packets: {report['eligible_errors']}; paired valid: {report['paired_valid']}.", ""]
    for arm, value in report["arms"].items():
        rate = value["repair_rate_among_valid"]
        fraction = f"{100 * rate:.1f}%" if rate is not None else "not available"
        lines += [f"{arm}: {value['repaired']} repaired, {value['still_wrong']} still wrong, "
                  f"{value['unassessed']} unassessed. Repair rate among {value['valid']} valid responses: {fraction}."]
    lines += ["", "Paired comparison: " + json.dumps(report["paired_outcomes"], sort_keys=True),
              f"Eligible error pairs complete: {report['eligible_error_pairs_complete']}",
              f"All historical errors assessed: {report['all_historical_errors_assessed_in_both_arms']}",
              "", report["interpretation"],
              "Missing, invalid and interrupted responses remain unassessed, not negative or wrong predictions.",
              "Existing correct-case pilot results are retained in case_results.csv; no new correct-case calls are scheduled."]
    (root / "failed_case_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run", "collect"), default="preflight")
    parser.add_argument("--bundle", type=Path, default=replay.DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=replay.DEFAULT_RUN)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    bundle = replay.load_bundle(args.bundle)
    if args.stage == "preflight":
        print(json.dumps(plan(bundle, args.run_root), indent=2))
        print("No API calls or output changes. Resume the same run root; do not rebuild the bundle.")
    elif args.stage == "run":
        base.require(args.allow_api, "Paid run requires --allow-api")
        base.require(replay.os.environ.get("AZURE_OPENAI_ENDPOINT") and replay.os.environ.get("AZURE_OPENAI_API_KEY"),
                     "Missing Azure credentials")
        try:
            execute(bundle, args.run_root, replay.azure_factory(bundle))
        finally:
            if (args.run_root / "run_identity.json").exists():
                collect(bundle, args.run_root)
    else:
        collect(bundle, args.run_root)


if __name__ == "__main__":
    main()

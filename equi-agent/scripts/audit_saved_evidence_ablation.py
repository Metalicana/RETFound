"""Saved trace sensitivity and Drishti threshold provenance; stdlib, no inference.

Cache versions are not repeated patients. Preserve them without selecting a
version using reference labels or agreement with a final pipeline prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("glaucoma", "amd", "dr")
REMOVED = (
    "without_demographic_reliability", "without_retfound_probability",
    "without_cdr_tool", "without_visual_interpretation",
)
NAMES = ("full_evidence", *REMOVED)
MANIFEST = ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery/manifest_recovered.csv"
SOURCES = {
    "shared_cache": ROOT / "OphthalmicAgent/outputs/counterfactual/glaucoma_counterfactual_traces.jsonl",
    "glaucoma_run_cache": ROOT / "OphthalmicAgent/outputs/glaucoma_counterfactual_250/counterfactual_traces.jsonl",
}
THRESHOLD_PATH = "OphthalmicAgent/Orchestrator/drishti.py"
THRESHOLD_COMMIT = "e92c5367d5b4c6d372c5bceeb8515d43628ffb79"
DEFAULT_OUT = ROOT / "equi-agent/outputs/audits/evidence_ablation_followup_20261006"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(record):
    payload = {k: record[k] for k in ("case_id", "deployment", "evidence", "prompt_version")}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def case_key(path):
    path = Path(path)
    tasks = [p.lower() for p in path.parts if p.lower() in TASKS]
    require(len(tasks) == 1, f"Ambiguous input task: {path}")
    return tasks[0], path.stem


def labels(scenarios, allow_extra_strings=False):
    require(isinstance(scenarios, list), "Scenarios must be a list")
    result, ignored = {}, 0
    for row in scenarios:
        if allow_extra_strings and isinstance(row, str):
            ignored += 1
            continue
        require(isinstance(row, dict), "Invalid scenario entry")
        name, value = row.get("name"), row.get("diagnosis")
        require(name in NAMES and name not in result, "Unknown/duplicate scenario")
        require(type(value) is int and value in (-1, 0, 1), "Invalid scenario diagnosis")
        result[name] = value
    require(set(result) == set(NAMES), "Missing scenario")
    return result, ignored


def load_locked(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    locked = {task: set() for task in TASKS}
    for row in rows:
        task, key = case_key(row["filename"])
        require(task == row["Task_Folder"].lower(), "Manifest task conflict")
        require(key not in locked[task], "Duplicate manifest case")
        locked[task].add(key)
    require(all(len(keys) == 250 for keys in locked.values()), "Expected 250 locked cases per task")
    return locked


def load_traces(path, source, locked):
    records, by_fingerprint = [], {}
    with path.open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            record = json.loads(text)
            task, key = case_key(record["case_id"])
            require(key in locked[task], f"Trace not in locked cohort: {task}/{key}")
            require(fingerprint(record) == record["fingerprint"], f"Fingerprint mismatch: {path}:{line}")
            values, _ = labels(record["scenarios"])
            raw = json.loads(record["raw_response"])
            raw_values, ignored = labels(raw["scenarios"], allow_extra_strings=True)
            require(values == raw_values, f"Raw/normalized diagnoses differ: {path}:{line}")
            full = values["full_evidence"]
            changed = {name for name in REMOVED if values[name] != full}
            require(record["full_evidence_diagnosis"] == full and
                    set(record["label_flip_scenarios"]) == changed and
                    record["label_flip_count"] == len(changed) and
                    record["evidence_sensitive"] == bool(changed), f"Saved summary mismatch: {path}:{line}")
            digest = record["fingerprint"]
            signature = (task, key, values)
            require(digest not in by_fingerprint or by_fingerprint[digest] == signature,
                    f"Conflicting responses for one fingerprint: {path}:{line}")
            duplicate = digest in by_fingerprint
            by_fingerprint[digest] = signature
            records.append(dict(source=source, source_line=line, task=task, case_id=key,
                recorded_task=record["task"], prompt_version=record["prompt_version"],
                deployment=record["deployment"], fingerprint=digest, duplicate_record=duplicate,
                raw_extra_string_entries=ignored,
                raw_mentions_glaucoma="glaucoma" in record["raw_response"].lower(), **values))
    return records


def summarize(records, locked):
    groups = defaultdict(list)
    for row in records:
        if not row["duplicate_record"]:
            groups[row["source"], row["task"], row["case_id"]].append(row)
    cases = []
    for (source, task, key), versions in sorted(groups.items()):
        for scenario in REMOVED:
            pairs = [(r["full_evidence"], r[scenario]) for r in versions]
            changes = [a != b for a, b in pairs]
            binary = [a != b and a in (0, 1) and b in (0, 1) for a, b in pairs]
            cases.append(dict(source=source, task=task, case_id=key, scenario=scenario,
                versions=len(versions), source_lines=";".join(str(r["source_line"]) for r in versions),
                fingerprints=";".join(r["fingerprint"] for r in versions),
                first_change=changes[0], last_change=changes[-1],
                any_change=any(changes), all_change=all(changes),
                version_disagreement=any(changes) != all(changes),
                any_binary_flip=any(binary), all_binary_flip=all(binary),
                any_inconclusive_transition=any(a != b and -1 in (a, b) for a, b in pairs)))
    summaries, inventory, transitions = [], [], []
    for source, task in sorted({(r["source"], r["task"]) for r in records}):
        raw = [r for r in records if (r["source"], r["task"]) == (source, task)]
        unique = [r for r in raw if not r["duplicate_record"]]
        counts = Counter(r["case_id"] for r in unique)
        inventory.append(dict(source=source, task=task, records=len(raw), unique_versions=len(unique),
            cases=len(counts), expected_cases=len(locked[task]), missing_cases=len(locked[task])-len(counts),
            cases_with_multiple_versions=sum(n > 1 for n in counts.values()),
            version_count_distribution=json.dumps(dict(sorted(Counter(counts.values()).items()))),
            recorded_tasks=";".join(sorted({r["recorded_task"] for r in raw})),
            prompt_versions=";".join(sorted({r["prompt_version"] for r in raw})),
            records_with_raw_extra_strings=sum(r["raw_extra_string_entries"] > 0 for r in raw),
            records_mentioning_glaucoma=sum(r["raw_mentions_glaucoma"] for r in raw)))
        for scenario in REMOVED:
            selected = [c for c in cases if (c["source"], c["task"], c["scenario"]) == (source, task, scenario)]
            summary = dict(source=source, task=task, scenario=scenario, cases=len(selected))
            for field in ("first_change", "last_change", "any_change", "all_change", "version_disagreement",
                          "any_binary_flip", "all_binary_flip", "any_inconclusive_transition"):
                total = sum(c[field] for c in selected)
                summary[field + "_n"] = total
                summary[field + "_percent"] = total / len(selected) * 100
            summaries.append(summary)
            pairs = Counter((r["full_evidence"], r[scenario]) for r in unique)
            transitions.extend(dict(source=source, task=task, scenario=scenario,
                full_evidence=a, removed_source=b, trace_versions=n) for (a, b), n in sorted(pairs.items()))
    return cases, summaries, inventory, transitions


def threshold_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True)
    history = git("log", "--format=%H", "--reverse", "-S", "0.48", "--", THRESHOLD_PATH).splitlines()
    require(history and history[0] == THRESHOLD_COMMIT, "Unexpected threshold history; inspect before reporting")
    before = git("show", f"{THRESHOLD_COMMIT}^:{THRESHOLD_PATH}")
    after = git("show", f"{THRESHOLD_COMMIT}:{THRESHOLD_PATH}")
    require("0.48" not in before and "greater than 0.48" in after, "Threshold introduction not verified")
    diff = git("show", "--format=fuller", "--no-ext-diff", THRESHOLD_COMMIT, "--", THRESHOLD_PATH)
    return dict(commit=THRESHOLD_COMMIT,
        author_date=git("show", "-s", "--format=%aI", THRESHOLD_COMMIT).strip(),
        commit_message=git("show", "-s", "--format=%s", THRESHOLD_COMMIT).strip(),
        source=THRESHOLD_PATH, parent_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
        introduced_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
        interpretation="Prompt suspicion heuristic; not an executable forced-positive decision rule.",
        numerical_justification="NOT ESTABLISHED: no inspected citation, validation sweep or selection receipt.",
        historical_test_tuning="NOT ESTABLISHED by commit date or prompt text."), diff


def ablation_availability(root):
    variants = ("retinagent_full", "agents_without_reliability")
    counts = {v: len(list((root / "agent/glaucoma" / v).glob("*.json"))) for v in variants}
    receipt = (root / "live_receipt.json").is_file()
    root = root.resolve()
    argument = str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root)
    return dict(run_root=str(root), decision_file_counts=counts, live_receipt_present=receipt,
        status="ready_for_provenance_validation" if receipt and all(n == 250 for n in counts.values())
               else "requested_paired_predictions_not_available_in_this_local_bundle",
        requested_full_f1=.771, requested_no_reliability_f1=.702,
        note="Requested aggregate values are not independently verified ablation scores. "
             "Do not substitute historical full-agent predictions or hypothetical trace scenario labels.",
        command='python OphthalmicAgent/scripts/audit_fairvision_glaucoma_uncertainty.py '
                f'--source ablation --run-root {shlex.quote(argument)} '
                '--resamples 10000 --seed 20261006 --confidence 0.95 '
                '--out-dir "equi-agent/outputs/audits/glaucoma_ablation_paired_20261006"')


def report(summaries, inventory, threshold, ablation):
    unavailable = ablation["status"] != "ready_for_provenance_validation"
    availability_note = ("The local preparation bundle lacks completed paired-arm decisions. This does not\n"
        "establish whether the cluster run completed." if unavailable else
        "Paired decision files are present. Run the command above to validate and analyse them.")
    run_inventory = next(r for r in inventory if r["source"] == "glaucoma_run_cache")
    extras = sum(r["records_with_raw_extra_strings"] for r in inventory if r["source"] == "shared_cache")
    shared = [r for r in inventory if r["source"] == "shared_cache"]
    require(all(r["recorded_tasks"] == "glaucoma" and
                r["prompt_versions"] == "glaucoma_evidence_ablation_v1" and
                r["records_mentioning_glaucoma"] == r["records"] for r in shared),
            "Shared prompt identity changed; revise historical interpretation before reporting")
    lines = ["SAVED EVIDENCE-ABLATION FOLLOW-UP", "No API calls, inference or training.", "",
        "1. REQUESTED GLAUCOMA PAIRED TEST", json.dumps(ablation, indent=2), "",
        availability_note, "CI and McNemar p cannot be recovered",
        "from two rounded F1 values. The command above reads saved decisions only, checks",
        "run fingerprints, complete case pairing and identical upstream evidence, and computes",
        "10,000 class-stratified paired percentile bootstrap replicates plus exact McNemar.",
        "Use the full arm from that run, not the historical 0.771 score. A discrepancy between",
        "the paired full score and 0.771 must be reported, not overwritten. McNemar tests",
        "paired error rates, not F1; the bootstrap difference CI addresses macro-F1.", "",
        "2. DRISHTI CDR > 0.48", json.dumps(threshold, indent=2),
        "The added instruction asks for suspicion and corroborating signs. It does not force",
        "a positive diagnosis. The same commit changed the deployment default and temperature",
        "handling. The evaluator's validation-selected CFP probability threshold is separate",
        "from this CDR heuristic. A development-only rationale for the exact 0.48 is not saved",
        "in the inspected code/history; do not invent one or infer test tuning from the date.", "",
        "3. PER-CASE SAVED-SCENARIO CHANGES", "",
        "Primary descriptive definition: a case changes if AT LEAST ONE distinct saved",
        "evidence version has a different scenario label from its own full_evidence label.",
        "The denominator is unique cases WITH traces, not records. -1 means inconclusive",
        "in this prompt. The binary-only column counts 0<->1; it excludes transitions to/from -1.",
        "No one canonical trace version is identified. First/last file-order rates and the",
        "all-version counts are exported, without claiming first/last are dated runs.", "",
        "Source / input cohort / removed source: changed cases (%); binary-only cases (%)"]
    for r in summaries:
        lines.append(f"{r['source']} / {r['task']} / {r['scenario']}: "
            f"{r['any_change_n']}/{r['cases']} ({r['any_change_percent']:.1f}%); "
            f"{r['any_binary_flip_n']}/{r['cases']} ({r['any_binary_flip_percent']:.1f}%)")
    lines += ["", "COVERAGE AND VERSION INVENTORY", *[json.dumps(r, sort_keys=True) for r in inventory], "",
        "IMPORTANT INTERPRETATION", "All three input cohorts in shared_cache have task=glaucoma and",
        "prompt_version=glaucoma_evidence_ablation_v1. All raw responses mention glaucoma.",
        "Current AMD/DR evaluators call the same shared CounterfactualAgent. Those cohorts'",
        "rates therefore are NOT validated AMD/DR-specific diagnosis-ablation results.",
        f"The separate glaucoma_run_cache covers {run_inventory['cases']} of "
        f"{run_inventory['expected_cases']} cases. Its {run_inventory['missing_cases']} missing cases",
        "are not counted as unchanged. Caches are never merged into one purported run.",
        "These are hypothetical scenarios jointly returned in one request containing all",
        "evidence, not independent pipeline runs with inputs actually withheld. The saved",
        "without_demographic_reliability instruction removes patient narrative AND trust score.",
        "without_visual_interpretation removes both OCT and SLO specialist reports.",
        "They do not estimate the effect of removing reliability from the whole pipeline,",
        "nor final-orchestrator diagnosis or escalation changes. Historical final CSVs do",
        "not identify their corresponding counterfactual fingerprint.",
        "More versions give a case more chances to have any change. These percentages",
        "are cache descriptions, not comparable per-run causal effects. All-version and",
        "any-version percentages are not confidence interval bounds.",
        f"{extras} shared-cache raw responses include extra string entries after the five valid",
        "scenario objects. These are counted in inventory; all five raw diagnoses agree",
        "with their normalized stored diagnoses. Fingerprints and saved flip fields verify.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-root", type=Path, default=ROOT / "OphthalmicAgent/outputs/fairvision_ablation_v1")
    args = parser.parse_args()
    locked = load_locked(MANIFEST)
    records = [r for source, path in SOURCES.items() for r in load_traces(path, source, locked)]
    cases, summaries, inventory, transitions = summarize(records, locked)
    threshold, diff = threshold_provenance()
    availability = ablation_availability(args.run_root)
    inputs = [MANIFEST, *SOURCES.values(), ROOT / THRESHOLD_PATH,
        ROOT / "OphthalmicAgent/evaluate_drishti_agentic_cfp.py",
        ROOT / "OphthalmicAgent/CounterfactualAgent/counterfactual_agent.py", Path(__file__)]
    inputs += [ROOT / f"OphthalmicAgent/evaluate_fairvision_{task}_agentic.py" for task in TASKS]
    inputs += [p for p in (args.run_root / "config.json", args.run_root / "table4.md") if p.is_file()]
    source_paths = {p.resolve() for p in inputs}
    artifacts = {
        "report.txt": report(summaries, inventory, threshold, availability),
        "drishti_threshold.diff": diff,
        "drishti_threshold_provenance.json": json.dumps(threshold, indent=2) + "\n",
        "ablation_availability.json": json.dumps(availability, indent=2) + "\n",
        "provenance.json": json.dumps(dict(api_calls=0, independent_task_runs_verified=False,
            aggregation="at least one distinct saved evidence version per case, separately by cache",
            sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in inputs}),
            indent=2) + "\n",
    }
    tables = dict(trace_records=records, case_scenarios=cases, scenario_summary=summaries,
                  cache_inventory=inventory, trace_transitions=transitions)
    targets = [args.out_dir / name for name in (*artifacts, *(name + ".csv" for name in tables))]
    for target in targets:
        require(not target.is_symlink() and target.resolve() not in source_paths, "Unsafe output path")
        require(not target.exists(), f"Output already exists; use a fresh --out-dir: {target}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in artifacts.items():
        (args.out_dir / name).write_text(contents, encoding="utf-8")
    for name, rows in tables.items():
        with (args.out_dir / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(artifacts["report.txt"])
    print(f"Wrote {args.out_dir}; zero API calls.")


if __name__ == "__main__":
    main()

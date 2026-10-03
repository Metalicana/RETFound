"""Find AMD/GDP case-level result candidates and audit counts, labels and F1 definitions.

Read-only, standard library, no API calls. Does not choose a run by its score,
repair labels, discard review cases, or fill missing predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".git", "__pycache__", "checkpoints", "weights", "feature_cache", "raw", "audits"}
TASKS = {"amd": "amd", "dr": "dr", "glaucoma": "glaucoma", "glaucoma_detection": "glaucoma"}


def first(row, keys):
    return next((str(row[k]).strip() for k in keys if row.get(k) not in (None, "")), "")


def binary(value):
    try:
        number = float(value)
        return int(number) if number in (0, 1) else None
    except (TypeError, ValueError):
        return None


def identifier(row):
    value = first(row, ("image_id", "Filename", "filename", "BScan_Path", "Patient_ID", "patient_id", "case_id"))
    name = Path(value).stem
    if name.isdigit():
        return "data_" + name.zfill(4)
    return name


def dataset_task(row, path):
    raw_task = first(row, ("task", "Task_Folder", "Disease")).lower()
    if "progression" in raw_task or "progression" in str(path).lower():
        return None
    dataset = first(row, ("dataset",)).lower()
    if "gdp" in dataset or "gdp" in str(path).lower():
        return "gdp", "glaucoma"
    if dataset and "fairvision" not in dataset:
        return None
    task = TASKS.get(raw_task)
    return ("fairvision", task) if task else None


def metrics(rows):
    counts = Counter((r["truth"], r["prediction"]) for r in rows)
    tn, fp, fn, tp = (counts[k] for k in ((0, 0), (0, 1), (1, 0), (1, 1)))
    p, n = tp + fn, tn + fp
    pos_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    neg_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    se, sp = tp / p if p else None, tn / n if n else None
    return dict(n=len(rows), positive=p, negative=n, tp=tp, tn=tn, fp=fp, fn=fn,
                f1_positive=pos_f1, f1_macro=(pos_f1 + neg_f1) / 2,
                f1_weighted=(p * pos_f1 + n * neg_f1) / len(rows) if rows else None,
                sensitivity=se, specificity=sp, balanced_accuracy=(se + sp) / 2 if p and n else None)


def demographics(row):
    age = first(row, ("Age", "age"))
    try:
        age = float(age)
        age = "unknown" if not math.isfinite(age) or age < 0 else "<50" if age < 50 else "50-69" if age < 70 else ">=70"
    except ValueError:
        age = "unknown"
    return dict(age_group=age, race=first(row, ("Race", "race")).lower(),
                sex=first(row, ("Gender", "sex_gender", "gender")).lower(),
                ethnicity=first(row, ("Ethnicity", "ethnicity")).lower())


def load_manifest(path):
    expected = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "test").lower() != "test":
                continue
            key = dataset_task(row, path)
            if key is None:
                continue
            case = identifier(row)
            if not case or case in expected[key]:
                raise ValueError(f"Missing/duplicate manifest ID: {path}: {key}/{case}")
            expected[key][case] = dict(truth=binary(first(row, ("Ground_Truth", "y_true", "label"))),
                                       **demographics(row))
    return expected


def summarize(group, expected):
    ids = Counter(r["id"] for r in group)
    duplicate_rows = sum(count - 1 for count in ids.values())
    valid = [r for r in group if r["truth"] is not None and r["prediction"] is not None and r["id"]]
    observed = set(ids) - {""}
    conflict = sum(r["id"] in expected and expected[r["id"]]["truth"] != r["truth"] for r in group) if expected else None
    complete = bool(expected) and observed == set(expected) and not duplicate_rows and len(valid) == len(group) and conflict == 0
    summary = dict(rows=len(group), unique_ids=len(observed), duplicate_rows=duplicate_rows,
                   invalid_truth=sum(r["truth"] is None for r in group),
                   invalid_prediction=sum(r["prediction"] is None for r in group),
                   missing_id_rows=ids[""], valid_rows=len(valid), expected_cases=len(expected) if expected else None,
                   missing_cases=len(set(expected) - observed) if expected else None,
                   extra_cases=len(observed - set(expected)) if expected else None,
                   label_conflicts=conflict, complete_locked_cohort=complete,
                   cohort_sha256=hashlib.sha256(json.dumps(sorted(observed)).encode()).hexdigest(),
                   labelled_cohort_sha256=hashlib.sha256(json.dumps(sorted((r["id"], r["truth"]) for r in valid)).encode()).hexdigest(),
                   status="complete_locked_cohort" if complete else "candidate_only_not_a_paired_main_table")
    subgroup_rows = []
    if valid and not duplicate_rows:
        summary.update(metrics(valid))
        for attribute in ("race", "sex", "age_group", "ethnicity"):
            groups = defaultdict(list)
            for r in valid:
                value = expected.get(r["id"], r).get(attribute, "")
                if value not in ("", "unknown", "nan", "none", "-1"):
                    groups[value].append(r)
            for value, rows in sorted(groups.items()):
                subgroup_rows.append(dict(attribute=attribute, subgroup=value, **metrics(rows)))
        # Both definitions, and both attribute sets; the report must not silently choose one.
        for kind in ("macro", "weighted"):
            for name, allowed in (("race_sex_age", {"race", "sex", "age_group"}),
                                  ("including_ethnicity", {"race", "sex", "age_group", "ethnicity"})):
                eligible = [r[f"f1_{kind}"] for r in subgroup_rows if r["attribute"] in allowed]
                summary[f"worst_{name}_f1_{kind}"] = min(eligible, default=None)
    return summary, subgroup_rows


def audit_file(path, expected):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not fields.intersection({"y_true", "Ground_Truth"}) or not fields.intersection({"y_pred", "Pred_AMD", "Pred_DR", "Pred_GL", "Prediction"}):
            return [], []
        for row in reader:
            key = dataset_task(row, path)
            if key is None:
                continue
            dataset, task = key
            prediction = first(row, ("y_pred", {"amd": "Pred_AMD", "dr": "Pred_DR", "glaucoma": "Pred_GL"}[task], "Prediction"))
            model = first(row, ("model_name", "Model")) or "unspecified"
            split = first(row, ("split",)) or "unspecified"
            groups[dataset, task, model, split].append(dict(id=identifier(row),
                truth=binary(first(row, ("y_true", "Ground_Truth"))), prediction=binary(prediction), **demographics(row)))
    reports, subgroups = [], []
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if groups else ""
    for (dataset, task, model, split), rows in sorted(groups.items()):
        meta = dict(path=str(path), sha256=sha, dataset=dataset, task=task, model=model, split=split)
        summary, subs = summarize(rows, expected.get((dataset, task), {}))
        if split not in ("test", "unspecified"):
            summary["complete_locked_cohort"] = False
            summary["status"] = "non_test_split"
        reports.append({**meta, **summary})
        subgroups.extend({**meta, **r} for r in subs)
    return reports, subgroups


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_progression(root, run):
    script = root / "OphthalmicAgent/scripts/audit_gdp_progression_endpoint_errors.py"
    spec = importlib.util.spec_from_file_location("endpoint_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, records, sources = module.load_run(run)
    labels = json.loads((run / "evaluation_labels.json").read_text())
    output, subgroups = [], []
    for target in module.TARGETS:
        metadata = {Path(r["image_id"]).stem: r for r in labels[target]}
        for name, field in (("Native helper", "helper_prediction"), ("OphthalmicAgent", "agent_prediction")):
            rows = [dict(truth=r["y_true"], prediction=r[field], metadata=metadata[r["case_id"]])
                    for r in records if r["target"] == target]
            eligible = []
            for attribute in ("race", "sex_gender", "age_group"):
                groups = defaultdict(list)
                for row in rows:
                    value = row["metadata"].get(attribute, "").strip().lower()
                    if value not in ("", "unknown", "missing", "nan", "none", "-1"):
                        groups[value].append(row)
                measured = [(value, metrics(group)) for value, group in sorted(groups.items())]
                sufficient = [m for _, m in measured if m["positive"] >= 20 and m["negative"] >= 20]
                for value, m in measured:
                    used = len(sufficient) >= 2 and m["positive"] >= 20 and m["negative"] >= 20
                    subgroups.append(dict(target=target, method=name, attribute=attribute, subgroup=value,
                                          eligible_for_worst_group=used, **m))
                if len(sufficient) >= 2:
                    eligible.extend(sufficient)
            output.append(dict(target=target, method=name, **metrics(rows),
                               worst_group_positive_f1=min((m["f1_positive"] for m in eligible), default=None),
                               worst_group_macro_f1=min((m["f1_macro"] for m in eligible), default=None)))
    meta = {k: config[k] for k in ("fingerprint", "prompt_version", "include_oct", "input_protocol", "label_policy", "targets", "calls_per_case")}
    meta.update(verified_complete_live_cohort=True, cases=200, source_hashes=sources,
                note="Saved provenance does not establish that prompt development was blinded to test outcomes.")
    return output, subgroups, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/retinagent_manuscript_audit"))
    parser.add_argument("--progression-run", type=Path, help="Optional complete staged run to verify and rescore without APIs")
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.out_dir.resolve()
    csv.field_size_limit(10_000_000)
    manifests, expected = [], {}
    for rel in ("OphthalmicAgent/data/fairvision_250each.csv", "equi-agent/outputs/manifests/gdp_glaucoma_detection.csv"):
        path = root / rel
        if path.is_file():
            expected.update(load_manifest(path))
            manifests.append(dict(path=rel, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    reports, subgroups, errors = [], [], []
    for base in (root / "OphthalmicAgent", root / "equi-agent/outputs"):
        for folder, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(folder) / d).is_symlink()
                             and not (Path(folder) / d).resolve().is_relative_to(out))
            for name in sorted(files):
                path = Path(folder) / name
                if path.suffix.lower() != ".csv" or path.is_symlink():
                    continue
                try:
                    rows, subs = audit_file(path, expected)
                    reports.extend(rows)
                    subgroups.extend(subs)
                except (OSError, ValueError, csv.Error, KeyError, TypeError) as error:
                    errors.append(dict(path=str(path), error=str(error)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "candidates.csv", reports)
    write_csv(out / "subgroups.csv", subgroups)
    names = ["candidates.csv", "subgroups.csv", "audit.json"]
    if args.progression_run:
        results, groups, meta = audit_progression(root, args.progression_run.resolve())
        write_csv(out / "progression_results.csv", results)
        write_csv(out / "progression_subgroups.csv", groups)
        (out / "progression_provenance.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        names.extend(["progression_results.csv", "progression_subgroups.csv", "progression_provenance.json"])
    (out / "audit.json").write_text(json.dumps(dict(manifests=manifests, candidates=reports, errors=errors,
        includes_patient_rows=False, api_calls=0,
        notes=["Candidates are not automatically promoted to canonical runs.",
               "Metrics with incomplete cohorts are descriptive only; no missing or invalid values are imputed.",
               "Duplicate attempts suppress metrics; choose an explicit retry policy before rescoring.",
               "Subgroup tables include both F1 definitions and support; no post-hoc support cutoff is chosen.",
               "Recorded settings, input equality, thresholds and live-vs-dry-run provenance still need verification."]), indent=2, allow_nan=False) + "\n")
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(out / name, name)
    for task in ("amd", "glaucoma"):
        matched = [r for r in reports if r["task"] == task and r["complete_locked_cohort"]]
        print(f"{task}: {len(matched)} candidate model/split groups match locked cases and labels")
    print(f"candidate_groups={len(reports)} read_errors={len(errors)} api_calls=0\narchive={out.with_suffix('.zip')}")


if __name__ == "__main__":
    main()

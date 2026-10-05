"""Read-only source audit and offline analyses; never imports an API client.

Historical missing predictions remain missing. Partial-cohort and ambiguous-run
analyses are explicitly labelled and must not replace complete canonical results.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import audit_fairvision_glaucoma_uncertainty as uncertainty
import audit_manuscript_result_gaps as metrics
import audit_reliability_weight_sensitivity as sensitivity

TASKS = ("glaucoma", "amd", "dr")
PRED = dict(glaucoma="Pred_GL", amd="Pred_AMD", dr="Pred_DR")
RECOVERY = ROOT / "equi-agent/outputs/audits/fairvision_glaucoma_case_recovery"
RESULTS = ROOT / "OphthalmicAgent/results"
RAW = ROOT / "OphthalmicAgent/_extras/CSVs/raw_model_predictions"
SOURCES = {}


def record(path):
    path = Path(path).resolve()
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return name


def read(path):
    record(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    record(path)
    return json.loads(Path(path).read_text())


def write(out, name, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (out / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def case_id(row):
    return Path(row.get("Filename") or row.get("filename") or row.get("image_id")
                or row.get("case_id") or row.get("Patient_ID")).stem


def index(rows):
    result = {}
    for row in rows:
        key = case_id(row)
        if key in result:
            raise ValueError(f"Duplicate case {key}")
        result[key] = row
    return result


def binary(value):
    return metrics.binary(value)


def stat(rows):
    return metrics.metrics(rows)


def subgroup_rows(rows, task, method):
    result = []
    for attribute in uncertainty.ATTRIBUTES:
        for group in sorted({r[attribute] for r in rows} - uncertainty.UNKNOWN):
            members = [r for r in rows if r[attribute] == group]
            result.append(dict(task=task, method=method, attribute=attribute, subgroup=group, **stat(members)))
    return result


def load_methods(locked, out):
    methods, summaries, groups, omissions, aligned = {}, [], [], [], []
    for task in TASKS:
        specs = [
            ("RetinAgent", RECOVERY / "agent_predictions_recovered.csv" if task == "glaucoma"
             else RESULTS / f"fairvision/fairvision_{task}_agentic.csv", PRED[task]),
            ("RETFound", RECOVERY / "retfound_predictions_recovered.csv" if task == "glaucoma"
             else RAW / f"retfound_{task}_predictions.csv", "Prediction"),
            ("MIRAGE", RAW / f"mirage_{task}_predictions.csv", "Prediction"),
        ]
        for model in ("visionfm_oct", "urfound_oct", "visionfm_slo", "urfound_slo", "flair_slo",
                      "ret_clip_slo", "retizero_slo"):
            if task == "glaucoma" and model in {"visionfm_slo", "urfound_slo"}:
                continue
            specs.append((model, ROOT / f"equi-agent/outputs/predictions/fairvision_{task}_{model}_test_thresholded.csv", "y_pred"))
        for model, slug in (("GPT-5.1", "gpt51"), ("GPT-5.6-luna", "gpt56"),
                            ("Claude Haiku 4.5", "claude" if task == "glaucoma" else "claude45")):
            specs.append((model, RESULTS / f"fairvision/fairvision_{task}_{slug}_predictions.csv", PRED[task]))
        methods[task] = {}
        for method, path, field in specs:
            if not path.exists():
                omissions.append(dict(task=task, method=method, reason="NOT FOUND", source=str(path)))
                continue
            source = record(path)
            raw = index(read(path))
            rows = []
            recovered_truth = 0
            for key, meta in locked[task].items():
                if key not in raw:
                    omissions.append(dict(task=task, method=method, case_id=key, reason="missing row", source=source))
                    continue
                r = raw[key]
                y = binary(r.get("Ground_Truth", r.get("y_true")))
                p = binary(r.get(field, r.get(PRED[task]) if field == "Prediction" else None))
                if y is not None and y != meta["truth"]:
                    raise ValueError(f"Label mismatch: {source}/{key}")
                if y is None:
                    if not (task == "glaucoma" and key == "data_07199"):
                        raise ValueError(f"Unexpected missing truth: {source}/{key}")
                    recovered_truth += 1
                if p is None:
                    omissions.append(dict(task=task, method=method, case_id=key,
                                          truth=meta["truth"], reason="invalid/missing prediction", source=source))
                    continue
                rows.append({**meta, "prediction": p})
            methods[task][method] = rows
            group_stats = subgroup_rows(rows, task, method)
            groups.extend(group_stats)
            summaries.append(dict(task=task, method=method, source=source, expected_n=250,
                status="complete" if len(rows) == 250 else "PARTIAL; not a full-cohort result",
                recovered_truth_rows=recovered_truth, **stat(rows),
                worst_group_macro_f1=min((r["f1_macro"] for r in group_stats), default=None),
                worst_group_weighted_f1=min((r["f1_weighted"] for r in group_stats), default=None)))
            aligned.extend(dict(task=task, method=method, source=source, **r) for r in rows)
    write(out, "fairvision_metrics.csv", summaries)
    write(out, "fairvision_subgroups.csv", groups)
    write(out, "fairvision_missing.csv", omissions)
    write(out, "fairvision_aligned_predictions.csv", aligned)
    return methods


def paired_tests(methods, out):
    intervals, tests = [], []
    for task in TASKS:
        comparisons = (["visionfm_oct", "urfound_oct", "GPT-5.1"] if task == "amd" else
                       ["RETFound", "GPT-5.1"] if task == "dr" else ["GPT-5.1"])
        agent = {r["case_id"]: r for r in methods[task]["RetinAgent"]}
        for comparator in comparisons:
            baseline = {r["case_id"]: r for r in methods[task][comparator]}
            keys = sorted(agent.keys() & baseline.keys())
            scope = "complete locked 250" if len(keys) == 250 else "complete-response intersection only; NOT full cohort"
            cases = [agent[key] for key in keys]
            pred = {"RetinAgent": [agent[k]["prediction"] for k in keys],
                    comparator: [baseline[k]["prediction"] for k in keys]}
            ci, paired, _, _, _ = uncertainty.analyse(cases, pred, "RetinAgent", resamples=10000, seed=20261005)
            intervals.extend(dict(task=task, analysis_comparator=comparator, scope=scope, **r) for r in ci)
            tests.extend(dict(task=task, scope=scope, n=len(keys), **r) for r in paired)
    # One declared family for the available FairVision comparisons. Partial AMD
    # analyses remain diagnostic; this adjustment cannot make them full-cohort tests.
    for row, adjusted in zip(tests, uncertainty.holm([r["p_value"] for r in tests])):
        row["p_holm_available_family"] = adjusted
    write(out, "paired_intervals.csv", intervals)
    write(out, "paired_tests.csv", tests)


def gdp_candidate(out):
    path = RESULTS / "gdp/gdp_test_claude_baseline_predictions.csv"
    raw = index(read(path))
    scope = "UNVERIFIED ATTRIBUTION: agent-format file named Claude baseline; not a canonical RetinAgent result"
    cases = []
    for key, r in sorted(raw.items()):
        y, p = binary(r["Ground_Truth"]), binary(r["Pred_GL"])
        assert y is not None and p is not None
        cases.append(dict(case_id=key, truth=y, prediction=p, **uncertainty.runner.demographic(r)))
    assert len(cases) == 400
    predictions = {"Unverified agent-format candidate": [r["prediction"] for r in cases]}
    sources = [dict(method="Unverified agent-format candidate", source=record(path), scope=scope, **stat(cases))]
    for model, name in (("retfound", "RETFound"), ("visionfm", "VisionFM")):
        source = ROOT / f"equi-agent/outputs/predictions/gdp_glaucoma_detection_{model}_oct.csv"
        baseline = index(read(source))
        assert set(baseline) == set(raw)
        for r in cases:
            assert binary(baseline[r["case_id"]]["y_true"]) == r["truth"]
        predictions[name] = [binary(baseline[r["case_id"]]["y_pred"]) for r in cases]
        assert all(p is not None for p in predictions[name])
        sources.append(dict(method=name, source=record(source), scope=scope,
                            **stat([{**r, "prediction": p} for r, p in zip(cases, predictions[name])])))
    ci, tests, _, _, _ = uncertainty.analyse(cases, predictions, "Unverified agent-format candidate",
                                            resamples=10000, seed=20261005)
    write(out, "gdp_UNVERIFIED_candidate_metrics.csv", sources)
    write(out, "gdp_UNVERIFIED_candidate_intervals.csv", [dict(scope=scope, **r) for r in ci])
    write(out, "gdp_UNVERIFIED_candidate_tests.csv", [dict(scope=scope, **r) for r in tests])


def external(out):
    results, missing = [], []
    for task, path, expected, baseline_path in (
        ("drishti", RESULTS / "drishti/drishti_test_agentic_cfp_predictions.csv", 51,
         RESULTS / "drishti/drishti_retfound_cfp_predictions.csv"),
        ("refuge2", RESULTS / "refuge/refuge_test_agentic_cfp_predictions.csv", 400,
         RESULTS / "refuge/refuge_test_retfound_cfp_predictions.csv")):
        rows = index(read(path))
        baseline = index(read(baseline_path))
        assert len(baseline) == expected and set(rows) <= set(baseline)
        valid = []
        for key, base in baseline.items():
            truth = binary(base["Ground_Truth"])
            row = rows.get(key)
            if row and binary(row["Ground_Truth"]) != truth:
                raise ValueError(f"External truth mismatch {key}")
            prediction = binary(row["Pred_GL"]) if row else None
            if prediction is None:
                missing.append(dict(task=task, case_id=key, truth=truth,
                                    reason="invalid prediction" if row else "missing row"))
            else:
                valid.append(dict(truth=truth, prediction=prediction))
        results.append(dict(task=task, source=record(path), expected_n=expected, saved_rows=len(rows),
                            invalid_saved_rows=len(rows)-len(valid), absent_rows=expected-len(rows),
                            full_cohort_result="NOT FOUND", **stat(valid)))
    write(out, "external_complete_response_metrics.csv", results)
    write(out, "external_missing.csv", missing)


def counterfactual(out):
    path = ROOT / "equi-agent/outputs/fairvision_reliability_selective_arbitration/metadata_counterfactuals.csv"
    rows = read(path)
    grouped = defaultdict(list)
    for row in rows:
        for task in (row["task"], "pooled"):
            grouped[(task, row["attribute"], row["original_value"], row["counterfactual_value"])].append(row)
            grouped[(task, row["attribute"], "ALL", "ALL")].append(row)
            grouped[(task, "ALL", "ALL", "ALL")].append(row)
    output = []
    for (task, attr, old, new), part in sorted(grouped.items()):
        increased = sum(r["base_escalate"].lower() == "false" and r["counterfactual_escalate"].lower() == "true" for r in part)
        decreased = sum(r["base_escalate"].lower() == "true" and r["counterfactual_escalate"].lower() == "false" for r in part)
        output.append(dict(task=task, attribute=attr, original=old, replacement=new, n_perturbations=len(part),
            label_changes=sum(int(r["label_flipped"]) for r in part), increased=increased, decreased=decreased,
            increased_percent=100*increased/len(part), decreased_percent=100*decreased/len(part)))
    write(out, "counterfactual_directions.csv", output)


def natural_prevalence(methods, out):
    rows = read(ROOT / "equi-agent/outputs/predictions/fairvision_oct_retfound_test_thresholded.csv")
    results = []
    for task in TASKS:
        full = [r for r in rows if r["task"] == task]
        assert len(full) == 3000 and len(index(full)) == 3000
        prevalence = sum(int(r["y_true"]) for r in full) / len(full)
        for method in (("RetinAgent", "visionfm_oct", "urfound_oct") if task == "amd" else ("RetinAgent", "RETFound")):
            source = methods[task][method]
            m = stat(source)
            tp, fn = prevalence*m["sensitivity"], prevalence*(1-m["sensitivity"])
            tn, fp = (1-prevalence)*m["specificity"], (1-prevalence)*(1-m["specificity"])
            f1 = (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))/2
            results.append(dict(task=task, method=method, n_source=len(source), test_prevalence=prevalence,
                sensitivity=m["sensitivity"], specificity=m["specificity"], estimated_macro_f1=f1,
                status="prevalence transport assumption" if len(source) == 250 else
                "PARTIAL 210 responses: conditional estimate only; full-cohort estimate NOT FOUND"))
    write(out, "natural_prevalence.csv", results)


def weight_analysis(locked, out):
    priors_path = ROOT / sensitivity.DEFAULT_PRIORS_JSON
    support_path = ROOT / sensitivity.DEFAULT_SUPPORT_CSV
    saved_path = ROOT / "OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv"
    for path in (priors_path, support_path, saved_path):
        record(path)
    priors = sensitivity.load_json_rows(priors_path)
    lookup = sensitivity.prior_lookup(priors)
    support_rows = read(support_path)
    support = sensitivity.ranking.support_tables(support_rows, None)
    models = {task: sensitivity.ranking.available_models([r for r in priors if r["task"] == task], "foundation") for task in TASKS}
    canonical = sensitivity.score(lookup, support, models, sensitivity.scenarios()[0])
    verification = sensitivity.verify_precomputed(canonical, saved_path)
    weights = [("equal", dict.fromkeys(sensitivity.CANONICAL, .2)),
               ("fnr_fpr_preserved_ratio", dict(fnr=7/12, fpr=5/12, ece=0, auroc=0, f1=0)),
               ("fnr_fpr_historical", dict(fnr=.85, fpr=.15, ece=0, auroc=0, f1=0))]
    rng = np.random.default_rng(20261005)
    weights.extend((f"dirichlet_{i:03d}", dict(zip(sensitivity.CANONICAL, map(float, vector))))
                   for i, vector in enumerate(rng.dirichlet(np.ones(5), size=200), 1))
    outputs, score_rows, draws = [], [], []
    for name, w in weights:
        alternative = sensitivity.score(lookup, support, models, (name, w, 50., False))
        draws.append(dict(scenario=name, **w))
        for combo, values in alternative.items():
            score_rows.extend(dict(scenario=name, task=combo[0], age_group=combo[1], race=combo[2],
                                   sex_gender=combo[3], model=model, risk=risk)
                              for model, risk in values.items())
        for layout in ("saved_score_age50_70", "live_lookup_age40_60"):
            for cohort in ("full_test_3000", "locked_250"):
                for task in TASKS:
                    if cohort == "full_test_3000":
                        people = [dict(age=float(r["age"]), race=r["race"], sex_gender=r["sex_gender"])
                                  for r in support_rows if r["task"] == task]
                    else:
                        people = list(locked[task].values())
                    changed = tied = 0
                    for p in people:
                        low, high = (40, 60) if layout == "live_lookup_age40_60" else (50, 70)
                        age = "younger" if p["age"] < low else "middle-aged" if p["age"] < high else "older"
                        key = (task, age, p["race"], p["sex_gender"])
                        before, after = sensitivity.winners(canonical[key]), sensitivity.winners(alternative[key])
                        changed += before != after
                        tied += len(before) > 1 or len(after) > 1
                    outputs.append(dict(scenario=name, task=task, cohort=cohort, age_mapping=layout,
                                        n=len(people), unchanged=len(people)-changed,
                                        unchanged_percent=100*(1-changed/len(people)), tied=tied))
    summaries = []
    for task in TASKS:
        for cohort in ("full_test_3000", "locked_250"):
            for layout in ("saved_score_age50_70", "live_lookup_age40_60"):
                relevant = [r for r in outputs if r["task"] == task and r["cohort"] == cohort and r["age_mapping"] == layout]
                for name in ("equal", "fnr_fpr_preserved_ratio", "fnr_fpr_historical", "dirichlet"):
                    values = [r["unchanged_percent"] for r in relevant if r["scenario"].startswith(name)]
                    summaries.append(dict(task=task, cohort=cohort, age_mapping=layout, scenario=name,
                        draws=len(values), mean_unchanged_percent=float(np.mean(values)),
                        min_unchanged_percent=min(values), max_unchanged_percent=max(values),
                        p025=float(np.quantile(values,.025)), p975=float(np.quantile(values,.975))))
    write(out, "weight_draws.csv", draws)
    write(out, "weight_scores.csv", score_rows)
    write(out, "weight_patient_ranking.csv", outputs)
    write(out, "weight_summary.csv", summaries)
    return verification


def prompt_export(out):
    folders = ("BioProfilerAgent", "VisionAgent", "Orchestrator", "EquityAgent", "GuidelinesAgent",
               "SafetyAgent", "FunctionalInterpretationAgent", "CounterfactualAgent")
    paths = [p for folder in folders for p in (ROOT / "OphthalmicAgent" / folder).glob("*.py")
             if not p.name.startswith(("test", "train", "finetune", "linear_probing", "compute_"))]
    paths += [ROOT / "OphthalmicAgent/Progression/prompts.py", ROOT / "OphthalmicAgent/llm_baseline_utils.py"]
    paths += list((ROOT / "OphthalmicAgent").glob("evaluate_*_baseline.py"))
    paths += [ROOT / "equi-agent/scripts/run_gdp_progression_llm_multitarget_baseline.py",
              ROOT / "equi-agent/scripts/run_gdp_progression_llm_baseline.py"]
    blocks = ["RETINAGENT SUPPLEMENT: SOURCE-EXTRACTED PROMPTS AND TEMPLATES\n"
              "Current checkout export, not proof of prompts used in historical runs.\n"
              "No imports, inference or API calls. Static strings are decoded; dynamic f-strings\n"
              "retain Python interpolation expressions. Source locations and hashes are included.\n"
              "Includes legacy/unused agents for disclosure, NOT a claim they all executed.\n"
              "Equity/Guideline/Safety modules are not called in the inspected FairVision path.\n"
              "Historical frozen canonical FairVision request snapshots: NOT FOUND.\n"]
    catalogue = []
    for path in sorted(set(paths)):
        text = path.read_text()
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                      and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
        entries = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)) or id(node) in docstrings:
                continue
            if isinstance(parents.get(node), (ast.JoinedStr, ast.BinOp, ast.FormattedValue)):
                continue
            if isinstance(node, ast.Constant):
                value = node.value
                if not isinstance(value, str):
                    continue
            elif isinstance(node, ast.JoinedStr):
                value = ast.get_source_segment(text, node)
            elif isinstance(node.op, ast.Add):
                value = ast.get_source_segment(text, node)
            else:
                continue
            parent = parents.get(node)
            prompt_context = (isinstance(parent, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value in {"content", "system"} and v is node
                for k, v in zip(parent.keys, parent.values)))
            if len(value) < 100 and not prompt_context:
                continue
            entries.append((node.lineno, node.end_lineno, value))
        if not entries:
            continue
        source = record(path)
        blocks.append(f"\n{'='*78}\nSOURCE: {source}\nSHA256: {SOURCES[source]}\n")
        for line, end, value in sorted(entries):
            blocks.append(f"\n[Lines {line}-{end}; source text/template]\n{value}\n")
            catalogue.append(dict(source=source, line=line, end_line=end))
    (out / "supplementary_prompts.txt").write_text("\n".join(blocks))
    write(out, "prompt_index.csv", catalogue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "equi-agent/outputs/audits/manuscript_20_questions_20261005")
    parser.add_argument("--ablation-inputs", type=Path,
                        default=Path("/private/tmp/fairvision_ablation_transfer.ftrks6"),
                        help="Saved prepared_cases/config/offline_predictions bundle; never launches agents.")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    locked = {t: {} for t in TASKS}
    for r in read(RECOVERY / "manifest_recovered.csv"):
        task = r["Task_Folder"].lower()
        key = case_id(r)
        assert key not in locked[task]
        age = float(r["Age"])
        locked[task][key] = dict(case_id=key, truth=binary(r["Ground_Truth"]), age=age,
                                **uncertainty.runner.demographic(r))
    assert all(len(r) == 250 and all(x["truth"] is not None for x in r.values()) for r in locked.values())
    splits = []
    for task, folder in (("glaucoma", "Glaucoma"), ("amd", "AMD"), ("dr", "DR")):
        path = ROOT / f"OphthalmicAgent/data/{folder}/data_summary_{task}.csv"
        rows = read(path)
        for split, n in Counter(r["use"] for r in rows).items():
            splits.append(dict(task=task, split=split, n=n, source=record(path)))
    write(out, "split_counts.csv", splits)
    counts = []
    seed_summary = {r["task"]: r for r in read(ROOT / "OphthalmicAgent/data/fairvision_balanced_250_summary.csv")}
    for task, people in locked.items():
        selected = index(read(ROOT / seed_summary[task]["path"]))
        same = set(selected) == set(people)
        if same:
            assert all(binary(selected[k]["Ground_Truth"]) == p["truth"] for k, p in people.items())
        counts.append(dict(task=task, n=250, positive=sum(r["truth"] for r in people.values()),
            negative=sum(1-r["truth"] for r in people.values()), seed=seed_summary[task]["seed"] if same else "NOT FOUND",
            seed_receipt_same_ids=same, alternate_seed2026_overlap=len(set(selected) & set(people))))
    write(out, "locked_counts.csv", counts)
    methods = load_methods(locked, out)
    print("Aligned historical predictions; retained missing/invalid cases as missing.", flush=True)
    paired_tests(methods, out)
    gdp_candidate(out)
    external(out)
    counterfactual(out)
    natural_prevalence(methods, out)
    print("Completed paired bootstraps, external counts, counterfactual directions and prevalence estimates.", flush=True)
    verification = weight_analysis(locked, out)
    prompt_export(out)
    unavailable = []
    for task in TASKS:
        unavailable.append(dict(item=16, task=task, status="NOT FOUND",
            reason="No explicit canonical live escalation flag; 2000-replicate flagged/unflagged bootstrap not run."))
    for task in ("amd", "dr"):
        for variant in ("agents_without_reliability", "paired_full_control"):
            unavailable.append(dict(item=17, task=task, variant=variant, status="NOT FOUND",
                reason="No complete paired agent outputs identified in local saved artifacts; no API calls made."))
    write(out, "unavailable_analyses.csv", unavailable)
    ablation = args.ablation_inputs
    if ablation.exists():
        config = read_json(ablation / "config.json")
        predictions = read_json(ablation / "offline_predictions.json")
        prepared = read_json(ablation / "prepared_cases.json")
        assert uncertainty.runner.digest(predictions) == config["offline_sha256"]
        assert uncertainty.runner.digest(prepared) == config["prepared_sha256"]
        rows = []
        lookup = {(c["task"], c["case_id"]): c for c in prepared}
        for task in TASKS:
            assert set(locked[task]) == {k[1] for k in lookup if k[0] == task}
            for variant in uncertainty.runner.VARIANTS[:2]:
                selected = []
                for r in predictions:
                    if r["task"] == task and r["variant"] == variant:
                        meta = lookup[(task, r["case_id"])]
                        assert meta["truth"] == locked[task][r["case_id"]]["truth"]
                        selected.append({**meta, "prediction": r["prediction"]})
                assert len(selected) == 250
                groups = subgroup_rows(selected, task, variant)
                rows.append(dict(task=task, variant=variant, **stat(selected),
                                 worst_group_macro_f1=min(g["f1_macro"] for g in groups)))
        write(out, "ablation_offline_recomputed.csv", rows)
    record(Path(__file__))
    record(Path(uncertainty.__file__))
    record(Path(metrics.__file__))
    record(Path(sensitivity.__file__))
    record(Path(sensitivity.ranking.__file__))
    record(Path(sensitivity.ranking.risk_score.__code__.co_filename))
    (out / "provenance.json").write_text(json.dumps(dict(api_calls=0, training_runs=0,
        bootstrap_replicates=10000, bootstrap_seed=20261005, dirichlet_seed=20261005,
        dirichlet_draws=200, canonical_score_verification=verification, source_sha256=SOURCES,
        limitations=["Missing escalation flags are not inferred from narrative text or invalid predictions.",
                     "Partial AMD pairings are diagnostic, not full-cohort comparisons.",
                     "Historical saved risks use test covariate counts for support; no labels enter this sensitivity audit.",
                     "Rank stability does not establish stability of live-agent decisions.",
                     "Recovered glaucoma reference label retains the prior audit's NPZ-verification limitation."]
        ), indent=2) + "\n")
    print(f"Completed saved-output audit: {out}")


if __name__ == "__main__":
    main()

"""Paired, complete-cohort reporting; no fabricated probabilities or hidden exclusions."""
from __future__ import annotations

import json
from collections import defaultdict

from .evidence import clean, native
from .prompts import ENDPOINTS
from .workflow import binary, validate, write_json


def classification(rows):
    from sklearn.metrics import f1_score
    y = [native.label(r) for r in rows]
    pred = [binary(r["y_pred"]) for r in rows]
    tp = sum(a == b == 1 for a, b in zip(y, pred))
    tn = sum(a == b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": len(y), "positives": sum(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "f1_positive": f1_score(y, pred, average="binary", zero_division=0),
            "f1_macro": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
            "f1_weighted": f1_score(y, pred, labels=[0, 1], average="weighted", zero_division=0),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None}


def with_subgroups(rows):
    values = classification(rows)
    eligible = []
    for attribute in ("race", "sex_gender", "age_group"):
        groups = defaultdict(list)
        for row in rows:
            name = row.get(attribute, "").strip().lower()
            if name not in {"", "unknown", "missing", "nan", "none", "-1"}:
                groups[name].append(row)
        sufficient = [g for g in groups.values() if sum(native.label(r) for r in g) >= 20
                      and sum(1 - native.label(r) for r in g) >= 20]
        if len(sufficient) >= 2:
            eligible.extend(classification(g) for g in sufficient)
    values["worst_group_f1_positive"] = min((g["f1_positive"] for g in eligible), default=None)
    values["worst_group_f1_macro"] = min((g["f1_macro"] for g in eligible), default=None)
    return values


def completed_cases(args, case_ids, fingerprint):
    results = {}
    for case_id in case_ids:
        path = args.out_dir / "cases" / f"{case_id}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("run_fingerprint") != fingerprint or result.get("case_id") != case_id:
                raise ValueError(f"Changed provenance in {path}")
            validate("orchestrator", {"predictions": result["predictions"]})
            results[case_id] = result
    return results


def export_predictions(args, answers, cases):
    for target, expected in answers.items():
        rows = []
        for meta in expected:
            result = cases.get(native.case_id(meta))
            if result is None:
                continue
            final = result["predictions"][target]
            # The orchestrator's label is the prediction, including when review is required.
            rows.append({k: meta.get(k, "") for k in (
                "patient_id", "eye_id", "image_id", "dataset", "task", "split", "y_true", "race", "sex_gender", "age_group")})
            rows[-1].update(model_name="ophthalmic_agent_gdp_progression_staged", progression_target=target,
                            y_pred=binary(final["prediction"]), review_required=final["review_required"],
                            reasoning=final["reasoning"], run_fingerprint=result["run_fingerprint"])
        if rows:
            native.write_csv(args.out_dir / f"predictions_{target}.csv", rows)


def collect(args, answers, cases):
    expected_ids = {native.case_id(r) for r in answers[native.TARGET]}
    if len(expected_ids) != 200 or set(cases) != expected_ids:
        raise ValueError("Cannot report a main table until all 200 cases have six valid endpoint predictions")
    all_results, paired, skipped = [], [], []
    markdown = ["# OphthalmicAgent GDP Progression", "",
                "All methods use the same 200 held-out cases per endpoint. Final orchestrator labels are preserved, including review flags.",
                "Both positive-class and macro-F1 are reported. Worst-group F1 below is positive-class F1; groups require >=20 positives and >=20 negatives and at least two eligible groups per attribute.",
                "Reliability age groups: <40, 40-59, >=60 (OphthalmicAgent convention). Evaluation age groups retain the locked manifests, matching the clean-suite comparison.",
                "Agent probabilities and AUROC are not reported: the final stage emits labels, not calibrated probabilities.",
                ("Additional OCT input: this is an expanded-input comparison, not input-matched to the native helper or standalone LLMs."
                 if args.include_oct else "RNFLT and baseline visual field are input-matched to the native helper; demographics are restricted to descriptive/reliability context."), ""]
    latex = []
    for target, expected in answers.items():
        prefix = f"gdp_progression_forecasting_{target}"
        helper = clean.align_predictions(args.clean_root / "predictions" / f"{prefix}_{native.MODEL}.csv", expected)
        ours = [{**meta, "y_pred": cases[native.case_id(meta)]["predictions"][target]["prediction"]} for meta in expected]
        methods = [("Native helper", helper), ("OphthalmicAgent (staged)", ours)]
        for slug, name in clean.LLMS.items():
            path = args.llm_root / slug
            try:
                summary = clean.read_json(path / "summary.json")
                if summary.get("dry_run") is not False or not summary.get("complete_locked_cohort"):
                    raise ValueError("Incomplete or dry-run LLM")
                methods.append((name, clean.align_predictions(path / f"predictions_{target}.csv", expected)))
            except (OSError, ValueError, KeyError) as error:
                skipped.append({"target": target, "model": name, "reason": str(error)})
        transitions = {"target": target, "n": len(expected), "changed": 0, "rescued": 0, "regressed": 0}
        for h, a in zip(helper, ours):
            if h["y_pred"] != a["y_pred"]:
                transitions["changed"] += 1
                transitions["rescued" if a["y_pred"] == native.label(a) else "regressed"] += 1
        paired.append(transitions)
        markdown.extend([f"## {target}", "", f"N=200; positives={sum(native.label(r) for r in expected)}.", "",
                         "| Method | Positive F1 | Macro-F1 | Worst-group positive F1 | Sensitivity | Specificity | Balanced accuracy |",
                         "|---|---:|---:|---:|---:|---:|---:|"])
        escaped = target.replace("_", r"\_")
        modality_note = (" The agent additionally receives OCT; this is an expanded-input comparison."
                         if args.include_oct else " The agent and native helper receive baseline RNFLT and visual-field measurements; demographics inform agent reliability context.")
        latex.extend([r"\begin{table}[t]", r"\centering", r"\caption{GDP progression: \texttt{" + escaped
                      + r"}. Same 200 cases per method, with all forced labels. Both F1 definitions are shown; worst-group F1 is positive-class F1. Eligible subgroups require at least 20 positives and 20 negatives, with at least two eligible groups per attribute."
                      + modality_note + "}",
                      r"\begin{tabular}{lcccccc}", r"\toprule",
                      r"Method & Positive F1 & Macro-F1 & Worst-group F1 & Sensitivity & Specificity & Balanced Acc. \\", r"\midrule"])
        for name, rows in methods:
            metrics = with_subgroups(rows)
            all_results.append({"target": target, "method": name, **metrics})
            cells = ["N/A" if metrics[k] is None else f"{metrics[k]:.4f}" for k in (
                "f1_positive", "f1_macro", "worst_group_f1_positive", "sensitivity", "specificity", "balanced_accuracy")]
            markdown.append("| " + name + " | " + " | ".join(cells) + " |")
            latex.append(name + " & " + " & ".join(cells) + r" \\")
        markdown.extend(["", f"Compared with native: {transitions['changed']} changed labels, {transitions['rescued']} corrections, {transitions['regressed']} regressions.", ""])
        latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    destination = args.out_dir / "results"
    native.write_csv(destination / "results.csv", all_results)
    native.write_csv(destination / "paired_changes.csv", paired)
    write_json(destination / "unavailable_llm_baselines.json", skipped)
    if skipped:
        markdown.extend(["## Unavailable LLM Rows", "", *[f"- {r['target']} / {r['model']}: {r['reason']}" for r in skipped]])
    (destination / "results.md").write_text("\n".join(markdown) + "\n")
    (destination / "results.tex").write_text("\n".join(latex) + "\n")
    print("\n".join(markdown), flush=True)

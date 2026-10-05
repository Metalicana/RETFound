"""Complete-cohort metrics, explicit selective prediction and live perturbations."""
import json
from collections import defaultdict

import numpy as np
import run_fairvision_ablation as base
from Confirmation.contract import parse_final
from Confirmation.design import demographics_profiles, read_protocol


def selective(rows):
    rng = np.random.default_rng(20261006)
    output = []
    for flag in (False, True):
        subset = [r for r in rows if r["escalation_required"] is flag]
        result = dict(escalation_required=flag, n=len(subset), fraction=len(subset)/len(rows),
                      macro_f1=None, lower=None, upper=None, bootstrap_replicates=2000)
        if subset:
            result["macro_f1"] = base.audit.metrics(subset)["f1_macro"]
            strata = [[r for r in subset if r["truth"] == y] for y in (0, 1)]
            values = []
            for _ in range(2000):
                sample = [group[i] for group in strata if group for i in rng.integers(len(group), size=len(group))]
                values.append(base.audit.metrics(sample)["f1_macro"])
            result["lower"], result["upper"] = map(float, np.quantile(values, [.025, .975]))
        output.append(result)
    return output


def validate_saved(saved, config, case, receipt, arm=None, scenario="original"):
    base.require(saved["fingerprint"] == config["fingerprint"], "Mixed experiment generations")
    base.require(saved["receipt"] == receipt and saved["case_id"] == case["case_id"], "Changed case/input receipt")
    if arm is not None:
        base.require(saved["task"] == case["task"] and saved["variant"] == arm and saved["scenario"] == scenario,
                     "Task/arm/scenario mismatch")
    else:
        base.require(saved["dataset"] == case["dataset"], "External dataset mismatch")
    decision = parse_final(saved["decision"])
    base.require(decision["diagnosis"] == saved["prediction"] and
                 decision["escalation_required"] is saved["escalation_required"], "Decision/flag mismatch")


def collect_complete(root, config, cases, external):
    base.write_json(root / "completion.json", dict(complete=False, status="validating"))
    protocol = read_protocol()
    base.require(len({(c["task"], c["case_id"]) for c in cases}) == len(cases) == 750, "Need 750 unique FairVision cases")
    for task in base.TASKS:
        cohort = [c for c in cases if c["task"] == task]
        base.require(len(cohort) == 250 and sum(c["truth"] for c in cohort) == 125, "Different locked class counts")
    for dataset, count in protocol["external_counts"].items():
        cohort = [c for c in external if c["dataset"] == dataset]
        base.require(len(cohort) == len({c["case_id"] for c in cohort}) == count, "External cohort incomplete/duplicated")
    base.require(len(external) == sum(protocol["external_counts"].values()), "Unexpected external cohort")
    receipt = base.digest(json.loads((root / "input_receipt.json").read_text()))
    missing = []
    for c in cases:
        for arm in base.VARIANTS[2:]:
            p = root / "agent" / c["task"] / arm / f"{c['case_id']}.json"
            if not p.exists():
                missing.append(str(p.relative_to(root)))
        for profile in demographics_profiles(c):
            for suffix in (".json", "_profile.json"):
                p = root / "demographic_audit" / c["task"] / c["case_id"] / (profile['name'] + suffix)
                if not p.exists():
                    missing.append(str(p.relative_to(root)))
    for c in external:
        p = root / "external" / c["dataset"] / f"{c['case_id']}.json"
        if not p.exists():
            missing.append(str(p.relative_to(root)))
    base.write_json(root / "completion.json", dict(complete=False, status="validating" if not missing else "incomplete",
                                                   missing_count=len(missing), missing=missing))
    base.require(not missing, f"Incomplete experiment: {len(missing)} missing outputs; no partial-cohort table produced")
    flags, directions = [], []
    for task in base.TASKS:
        cohort = [c for c in cases if c["task"] == task]
        for arm in base.VARIANTS[2:]:
            scored = []
            for c in cohort:
                saved = json.loads((root / "agent" / task / arm / f"{c['case_id']}.json").read_text())
                validate_saved(saved, config, c, receipt, arm)
                scored.append({**c, "prediction": saved["prediction"], "escalation_required": saved["escalation_required"]})
                if arm == "retinagent_full":
                    for profile in demographics_profiles(c):
                        cf = json.loads((root / "demographic_audit" / task / c["case_id"] / f"{profile['name']}.json").read_text())
                        validate_saved(cf, config, c, receipt, arm, profile["name"])
                        proof = json.loads((root / "demographic_audit" / task / c["case_id"] /
                                            f"{profile['name']}_profile.json").read_text())
                        evidence = json.loads((root / "shared" / task / f"{c['case_id']}.json").read_text())["evidence"]
                        base.require(proof["profile"] == profile and proof["decision_fingerprint"] == cf["evidence_fingerprint"]
                                     and proof["image_evidence_sha256"] == base.digest(
                                         {k: v for k, v in evidence.items() if k != "narrative"}), "Demographic audit evidence changed")
                        directions.append(dict(task=task, case_id=c["case_id"], attribute=profile["field"],
                            original=profile["original"], replacement=profile["replacement"],
                            label_changed=cf["prediction"] != saved["prediction"],
                            escalation_increased=cf["escalation_required"] and not saved["escalation_required"],
                            escalation_decreased=saved["escalation_required"] and not cf["escalation_required"]))
            flags.extend(dict(task=task, variant=arm, **r) for r in selective(scored))
    external_metrics = []
    for dataset in sorted({c["dataset"] for c in external}):
        scored = []
        baseline = []
        for c in (c for c in external if c["dataset"] == dataset):
            saved = json.loads((root / "external" / dataset / f"{c['case_id']}.json").read_text())
            validate_saved(saved, config, c, receipt)
            scored.append(dict(truth=c["truth"], prediction=saved["prediction"], escalation_required=saved["escalation_required"]))
            baseline.append(dict(truth=c["truth"], prediction=int(c["probability"] >= .5)))
        external_metrics.extend([dict(dataset=dataset, method="RetinAgent", **base.audit.metrics(scored)),
                                 dict(dataset=dataset, method="RETFound raw threshold 0.5", **base.audit.metrics(baseline))])
        flags.extend(dict(task=dataset, variant="retinagent_full", **r) for r in selective(scored))
    base.collect(root)
    base.audit.write_csv(root / "live_escalation.csv", flags)
    base.audit.write_csv(root / "live_demographic_counterfactuals.csv", directions)
    base.audit.write_csv(root / "external_metrics.csv", external_metrics)
    grouped = defaultdict(list)
    for row in directions:
        grouped[(row["task"], row["attribute"], str(row["original"]), str(row["replacement"]))].append(row)
    base.audit.write_csv(root / "live_demographic_directions.csv", [dict(task=k[0], attribute=k[1], original=k[2], replacement=k[3],
        n=len(rows), **{field: sum(r[field] for r in rows)/len(rows) for field in
                       ("label_changed", "escalation_increased", "escalation_decreased")}) for k, rows in grouped.items()])
    base.write_json(root / "completion.json", dict(complete=True, status="validated", missing_count=0,
        fairvision_cases=len(cases), paired_agent_decisions=2*len(cases), live_perturbations=len(directions),
        external_cases=len(external), fingerprint=config["fingerprint"]))
    print("Complete frozen experiment collected; all forced diagnoses retained, including escalations.")

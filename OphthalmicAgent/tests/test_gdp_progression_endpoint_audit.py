from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "OphthalmicAgent/scripts/audit_gdp_progression_endpoint_errors.py"
spec = importlib.util.spec_from_file_location("endpoint_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, n=4):
    ids = [f"data_{i:04d}" for i in range(1, n + 1)]
    prepared, labels, finals = [], {t: [] for t in audit.TARGETS}, {}
    for i, case_id in enumerate(ids, 1):
        helpers = {t: {"prediction": 0, "probability": .1, "source_threshold": .5} for t in audit.TARGETS}
        if i in (1, 2):
            helpers[audit.PRIMARY].update(prediction=1, probability=.9)
        if i == 3:
            helpers["md"].update(prediction=1, probability=.8)
        predictions = {t: {"prediction": 0, "review_required": True, "reasoning": f"Synthetic {case_id}/{t}"}
                       for t in audit.TARGETS}
        if i in (1, 2, 3):
            predictions[audit.PRIMARY]["prediction"] = 1
        if i in (1, 2):
            predictions["md"]["prediction"] = 1
        if i == 1:
            predictions["vfi"]["prediction"] = 1
        if i == 4:
            predictions["td_pointwise"]["prediction"] = 1
        scenarios = {s: {"predictions": {t: 0 for t in audit.TARGETS}, "reasoning": "Synthetic scenario"}
                     for s in audit.SCENARIOS}
        if i == 1:
            scenarios["full_evidence"]["predictions"]["md"] = 1
        finals[case_id] = {"case_id": case_id, "predictions": predictions,
                          "counterfactual": {"scenarios": scenarios, "interpretation": "Synthetic interpretation"}}
        reliability = {"source": "development_oof", "development_cases": 300, "trust_score": .6,
                       "global": {"auroc": .7, "f1": .5, "ece": .2, "fpr": .1, "fnr": .4}}
        prepared.append({"case_id": case_id, "helper_predictions": helpers,
                         "reliability": {t: copy.deepcopy(reliability) for t in audit.TARGETS}})
        for target in audit.TARGETS:
            labels[target].append({"image_id": case_id + ".npz", "split": "test", "task": "progression_forecasting",
                                   "y_true": "1.0" if i == 4 or (i == 1 and target == audit.PRIMARY) else "0.0"})
    config = {"targets": list(audit.TARGETS), "case_ids": ids, "evidence_sha256": audit.digest(prepared),
              "prompt_version": "synthetic_fixture"}
    config["fingerprint"] = audit.digest(config)
    save(root / "resolved_config.json", config)
    save(root / "prepared_cases.json", prepared)
    save(root / "evaluation_labels.json", labels)
    save(root / "summary.json", {"run_fingerprint": config["fingerprint"], "dry_run": False,
                                "complete_live_cohort": True, "completed_cases": n, "expected_cases": n,
                                "missing_cases": []})
    for case_id, result in finals.items():
        save(root / "cases" / f"{case_id}.json", {**result, "run_fingerprint": config["fingerprint"]})


class EndpointAuditTest(unittest.TestCase):
    def test_overlap_stage_attribution_and_existing_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            config, records, sources = audit.load_run(root, 4)
            self.assertEqual(len(records), 24)
            self.assertEqual(len(sources), 8)
            summaries, overlaps, patterns, cases = audit.analyze(records)
            by_target = {r["target"]: r for r in summaries}
            md = by_target["md"]
            self.assertEqual((md["corrections"], md["regressions"]), (1, 2))
            self.assertEqual(md["new_false_positives"], 2)
            self.assertEqual(md["new_fp_already_positive_in_cf_full"], 1)
            self.assertEqual(md["new_fp_flipped_only_at_final_stage"], 1)
            self.assertEqual(md["new_fp_primary_helper_positive"], 2)
            self.assertEqual(md["new_fp_primary_truth_positive"], 1)
            primary = by_target[audit.PRIMARY]
            self.assertEqual(primary["agent_fp"], 2)
            self.assertEqual(primary["new_false_positives"], 1)
            self.assertEqual(primary["agent_tp"], 1)
            self.assertEqual(primary["agent_positive_f1"], .4)
            self.assertEqual(by_target["td_pointwise"]["corrections"], 1)
            shared = next(r for r in overlaps if (r["target_a"], r["target_b"]) == ("md", "vfi"))
            self.assertEqual(shared["shared_case_ids"], "data_0001")
            self.assertEqual(shared["jaccard"], .5)
            empty = next(r for r in overlaps if (r["target_a"], r["target_b"]) == ("md_fast", "md_fast_no_p_cut"))
            self.assertIsNone(empty["jaccard"])
            self.assertEqual(patterns[0]["case_id"], "data_0001")
            self.assertEqual(patterns[0]["new_fp_endpoints"], 2)
            overview, report = audit.report(config, summaries, overlaps, patterns, cases, 1)
            self.assertIn("does not establish", overview)
            self.assertIn("### data_0001", report)
            self.assertNotIn("### data_0002", report)
            self.assertIn("Synthetic data_0001/md", report)

    def test_rejects_missing_cases_bad_provenance_and_label_misalignment(self):
        for alteration in ("missing_case", "case_hash", "evidence", "config", "duplicate_label", "missing_endpoint", "partial", "dry"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                if alteration == "missing_case":
                    (root / "cases/data_0001.json").unlink()
                elif alteration == "case_hash":
                    path = root / "cases/data_0001.json"
                    result = audit.read_json(path)
                    result["run_fingerprint"] = "different"
                    save(path, result)
                elif alteration == "evidence":
                    path = root / "prepared_cases.json"
                    result = audit.read_json(path)
                    result[0]["helper_predictions"]["md"]["probability"] = .3
                    save(path, result)
                elif alteration == "config":
                    path = root / "resolved_config.json"
                    result = audit.read_json(path)
                    result["prompt_version"] = "tampered"
                    save(path, result)
                elif alteration in ("duplicate_label", "missing_endpoint"):
                    path = root / "evaluation_labels.json"
                    result = audit.read_json(path)
                    if alteration == "duplicate_label":
                        result["md"].append(result["md"][0])
                    else:
                        del result["vfi"]
                    save(path, result)
                else:
                    path = root / "summary.json"
                    result = audit.read_json(path)
                    result["dry_run" if alteration == "dry" else "complete_live_cohort"] = alteration == "dry"
                    save(path, result)
                with self.assertRaises(ValueError):
                    audit.load_run(root, 4)

    def test_cli_uses_only_saved_files_and_does_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            fixture(root, 200)
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            result = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(root)],
                                    capture_output=True, text=True, check=True)
            self.assertIn("No API calls", result.stdout)
            out = root / "audit_endpoint_errors"
            summary = audit.read_json(out / "summary.json")
            self.assertEqual(summary["case_count"], 200)
            self.assertEqual(summary["api_calls"], 0)
            self.assertEqual(summary["prediction_changes"], 0)
            with (out / "case_endpoint_audit.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1200)
            with (out / "false_positive_overlap.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 15)
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_binary_validation_and_absent_class_metrics(self):
        self.assertEqual(audit.binary("1.0"), 1)
        for value in (True, .9, -1, "nan"):
            with self.assertRaises(ValueError):
                audit.binary(value)
        result = audit.metrics([{"y_true": 0, "agent_prediction": 0}], "agent_prediction")
        self.assertEqual(result["positive_f1"], 0)
        self.assertEqual(result["macro_f1"], .5)
        self.assertIsNone(result["balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent"))
sys.path.insert(0, str(ROOT / "equi-agent/tests"))

from Progression import evidence, prompt_review, reporting, workflow
from Progression.prompts import ENDPOINTS, SCENARIOS, SYSTEM_PROMPTS
import test_gdp_progression_clean_suite as fixtures

spec = importlib.util.spec_from_file_location("staged_runner", ROOT / "OphthalmicAgent/scripts/run_gdp_progression_ophthalmic_agent.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def valid_response(stage, prediction=1):
    if stage == "orchestrator":
        return {"predictions": {t: {"prediction": prediction, "review_required": True,
                                   "reasoning": "Synthetic test response"} for t in ENDPOINTS}}
    if stage == "counterfactual":
        return {"scenarios": {s: {"predictions": {t: 0 for t in ENDPOINTS},
                                  "reasoning": "Synthetic audit"} for s in SCENARIOS},
                "interpretation": "Synthetic evidence-dependence trace"}
    return {"report": "Synthetic specialist report, not a real clinical result"}


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.requests = []
        self.fail_final = False

    def create(self, **request):
        self.requests.append(request)
        stage = next(k for k, v in SYSTEM_PROMPTS.items() if v == request["messages"][0]["content"])
        value = valid_response(stage)
        if stage == "orchestrator" and self.fail_final:
            del value["predictions"]["vfi"]
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(value)))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30))

    def close(self):
        pass


def case(root):
    image = root / "image.png"
    image.write_bytes(b"synthetic image fixture; fake client does not decode it")
    return {"case_id": "data_0301", "demographics": {"age": 70, "sex_gender": "female"},
            "modalities": ["RNFLT", "TDS"], "rnflt_image": str(image),
            "rnflt_statistics": {"mean": 70}, "td_values": {"td1": -3}, "td_summary": {"mean_db": -3},
            "helper_predictions": {t: {"probability": 0.01, "prediction": 0} for t in ENDPOINTS},
            "reliability": {t: {"source": "development_oof"} for t in ENDPOINTS},
            "y_true": "SECRET_TEST_LABEL", "progression_md": "SECRET_FUTURE_OUTCOME"}


class StagedProgressionTest(unittest.TestCase):
    def test_separate_stages_blinded_specialists_and_no_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeClient()
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, retry_sleep=0)
            result = workflow.run_case(case(root), caller)
            self.assertEqual(len(fake.requests), 5)
            expected = ["bio_profiler", "rnflt_specialist", "functional_specialist", "counterfactual", "orchestrator"]
            self.assertEqual([r["messages"][0]["content"] for r in fake.requests], [SYSTEM_PROMPTS[k] for k in expected])
            transmitted = json.dumps(fake.requests)
            self.assertNotIn("SECRET_TEST_LABEL", transmitted)
            self.assertNotIn("SECRET_FUTURE_OUTCOME", transmitted)
            self.assertNotIn("helper_predictions", json.dumps(fake.requests[:3]))
            self.assertIn("image_url", json.dumps(fake.requests[1]))
            self.assertNotIn("image_url", json.dumps(fake.requests[2]))
            self.assertTrue(all(r["prediction"] == 1 for r in result["predictions"].values()))
            # Strong negative helper scores are not silently forced onto the final labels.
            self.assertTrue(all(r["review_required"] for r in result["predictions"].values()))

    def test_resume_reuses_valid_stages_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeClient()
            patient = case(root)
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, retry_sleep=0)
            first = workflow.run_case(patient, caller)
            reused = workflow.CachedCaller(root, "gpt-5.1", lambda: self.fail("Cached stages must not construct a client"))
            self.assertEqual(first, workflow.run_case(patient, reused))
            patient["demographics"]["age"] = 80
            with self.assertRaisesRegex(ValueError, "Cached evidence changed"):
                workflow.run_case(patient, reused)

    def test_failed_final_stage_not_cached_or_replaced_by_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient = case(root)
            fake = FakeClient()
            fake.fail_final = True
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, max_attempts=2, retry_sleep=0)
            with self.assertRaisesRegex(RuntimeError, "orchestrator failed"):
                workflow.run_case(patient, caller)
            self.assertEqual(len(fake.requests), 6)
            self.assertFalse((root / "stage_cache/data_0301/orchestrator.json").exists())
            fake.fail_final = False
            workflow.run_case(patient, caller)
            self.assertEqual(len(fake.requests), 7)

    def test_actual_evidence_removed_in_each_ablation_packet(self):
        full = {"patient_narrative": "demographics", "helper_predictions": "probabilities",
                "reliability": "metrics", "structural_reports": "images", "functional_report": "field"}
        packets = workflow.ablation_packets(full)
        self.assertNotIn("helper_predictions", packets["without_helper_predictions"])
        self.assertNotIn("structural_reports", packets["without_structural_interpretation"])
        self.assertNotIn("functional_report", packets["without_functional_interpretation"])
        self.assertNotIn("patient_narrative", packets["without_demographic_reliability"])
        self.assertNotIn("reliability", packets["without_demographic_reliability"])
        self.assertEqual(packets["full_evidence"], full)

    def test_truncation_and_auth_failure_do_not_create_cached_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeClient()
            truncated = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                        message=SimpleNamespace(content=json.dumps(valid_response("bio_profiler"))))], usage=None)
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, max_attempts=2, retry_sleep=0)
            with patch.object(fake, "create", return_value=truncated) as request:
                with self.assertRaisesRegex(RuntimeError, "Incomplete response: length"):
                    caller.call("fixture", "bio_profiler", {})
                self.assertEqual(request.call_count, 2)
            error = RuntimeError("Synthetic authentication failure")
            error.status_code = 401
            with patch.object(fake, "create", side_effect=error) as request:
                with self.assertRaisesRegex(RuntimeError, "authentication failure"):
                    caller.call("fixture", "bio_profiler", {})
                self.assertEqual(request.call_count, 1)
            self.assertFalse((root / "stage_cache/fixture/bio_profiler.json").exists())

    def test_runner_smoke_marks_partial_cohort_and_reuses_completed_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(stage="smoke", out_dir=root, include_oct=False,
                                   deployment="gpt-5.1", max_attempts=2)
            first = case(root)
            second = {**first, "case_id": "data_0302"}
            answers = {t: [{"image_id": c["case_id"], "y_true": "0", "split": "test"}
                           for c in [first, second]] for t in ENDPOINTS}
            fake = FakeClient()
            with patch.object(evidence, "prepare", return_value=([first, second], answers, {})), \
                 patch.object(runner, "freeze_run", return_value="fixture"), \
                 patch.object(runner, "make_client", return_value=fake), redirect_stdout(StringIO()):
                runner.execute(args)
                runner.execute(args)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(len(fake.requests), 5)
            self.assertEqual(summary["completed_cases"], 1)
            self.assertEqual(summary["missing_cases"], ["data_0302"])
            self.assertFalse(summary["complete_live_cohort"])
            self.assertFalse((root / "results/results.md").exists())

    def test_schema_rejects_missing_endpoints_nonbinary_and_missing_review(self):
        for bad in (True, "1", -1, 0.5):
            value = valid_response("orchestrator")
            value["predictions"]["md"]["prediction"] = bad
            with self.assertRaises(ValueError):
                workflow.validate("orchestrator", value)
        value = valid_response("orchestrator")
        del value["predictions"]["md"]["review_required"]
        with self.assertRaises(ValueError):
            workflow.validate("orchestrator", value)
        value = valid_response("counterfactual")
        del value["scenarios"]["full_evidence"]["predictions"]["vfi"]
        with self.assertRaises(ValueError):
            workflow.validate("counterfactual", value)

    def test_optional_oct_is_a_real_separate_image_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient = case(root)
            patient["oct_image"] = patient["rnflt_image"]
            fake = FakeClient()
            workflow.run_case(patient, workflow.CachedCaller(root, "gpt-5.1", lambda: fake))
            self.assertEqual(len(fake.requests), 6)
            self.assertEqual(fake.requests[2]["messages"][0]["content"], SYSTEM_PROMPTS["oct_specialist"])
            self.assertIn("image_url", json.dumps(fake.requests[2]))

    def test_source_prompt_export_uses_active_ast_and_does_not_change_originals(self):
        before = {p: (prompt_review.ROOT / p).read_bytes() for p, _, _ in prompt_review.SOURCE_PROMPTS.values()}
        with tempfile.TemporaryDirectory() as tmp:
            result = prompt_review.export(Path(tmp) / "comparison.html").read_text()
            for role in SYSTEM_PROMPTS:
                original, source, sha = prompt_review.original_prompt(role)
                self.assertTrue(original)
                self.assertTrue(source)
                self.assertEqual(len(sha), 64)
                self.assertIn(f'id="{role}"', result)
            self.assertIn("1,000 calls", result)
        for p, contents in before.items():
            self.assertEqual((prompt_review.ROOT / p).read_bytes(), contents)

    def test_reliability_matches_original_risk_and_sparse_fallback(self):
        rows = fixtures.predictions(fixtures.cohort("md", 300, 0), "fixture", 1, "oof")
        for r in rows:
            r["age"] = "65"
            r["race"] = "white"
        tool = evidence.Reliability(rows)
        packet = tool.packet({"age": "90", "race": "unrepresented", "sex_gender": "unknown"})
        risk = evidence.reliability_tool.risk_score(tool.global_metrics, None, *evidence.RISK_WEIGHTS)
        self.assertAlmostEqual(packet["trust_score"], 1 - risk)
        self.assertEqual(packet["source"], "development_oof")
        self.assertFalse(packet["subgroups"]["race"]["eligible"])
        with self.assertRaises(ValueError):
            evidence.Reliability([{**r, "split": "test"} for r in rows])
        self.assertEqual(evidence.group_values({"age": "39"})["age_group"], "younger")
        self.assertEqual(evidence.group_values({"age": "60"})["age_group"], "older")

    def test_macro_and_positive_f1_are_distinct_and_rare_groups_unavailable(self):
        rows = [{"y_true": 1, "y_pred": 0} for _ in range(4)]
        rows += [{"y_true": 0, "y_pred": 1} for _ in range(3)]
        rows += [{"y_true": 0, "y_pred": 0} for _ in range(193)]
        values = reporting.with_subgroups(rows)
        self.assertEqual(values["f1_positive"], 0)
        self.assertAlmostEqual(values["f1_macro"], 193 / 393)
        self.assertEqual(values["sensitivity"], 0)
        self.assertIsNone(values["worst_group_f1_positive"])
        self.assertNotIn("auroc", values)

    def test_preflight_complete_reporting_and_resume_guards_without_api(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixtures.arguments(root)
            fixtures.write_manifests(args)
            image = root / "baseline.npz"
            np.savez(image, rnflt=np.arange(25, dtype=np.float32).reshape(5, 5), progression=np.array([1, 0, 1, 0, 0, 1]))
            for target in ENDPOINTS:
                path = evidence.clean.target_manifest(args, target)
                rows = evidence.native.read_csv(path)
                for r in rows:
                    r.update(rnflt_path=str(image), rnflt_key="rnflt", age="65", md="SECRET_MD")
                    r.update({col: "-3" for col in evidence.baseline.GDP_TD_COLUMNS})
                evidence.native.write_csv(path, rows)
            cohorts = evidence.clean.check_manifests(args)
            for target in ENDPOINTS:
                fixtures.write_bundle(args, target, cohorts[target])
            evidence.clean.stage_inputs(args, cohorts)
            args.clean_root = args.run_root
            args.out_dir = root / "staged"
            args.out_dir.mkdir()
            args.include_oct = False
            args.path_prefix_from = args.path_prefix_to = ""
            with patch.object(evidence.baseline, "render_rnflt_png", return_value=b"synthetic PNG"):
                cases, answers, sources = evidence.prepare(args)
            self.assertEqual(len(cases), 200)
            encoded = json.dumps(cases)
            self.assertNotIn("SECRET_MD", encoded)
            self.assertNotIn('"y_true"', encoded)
            self.assertNotIn('"progression"', encoded)
            fingerprint = runner.freeze_run(args, cases, answers, sources)
            self.assertEqual(fingerprint, runner.freeze_run(args, cases, answers, sources))
            changed = copy.deepcopy(cases)
            changed[0]["td_values"]["td1"] = 50
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.freeze_run(args, changed, answers, sources)
            results = {c["case_id"]: {"case_id": c["case_id"], "run_fingerprint": fingerprint,
                                      **valid_response("orchestrator")} for c in cases}
            with self.assertRaisesRegex(ValueError, "200 cases"):
                reporting.collect(args, answers, dict(list(results.items())[:199]))
            with redirect_stdout(StringIO()):
                reporting.export_predictions(args, answers, results)
                reporting.collect(args, answers, results)
            values = evidence.native.read_csv(args.out_dir / "results/results.csv")
            self.assertEqual(len(values), 12)
            self.assertEqual({r["target"] for r in values}, set(ENDPOINTS))
            pred = evidence.native.read_csv(args.out_dir / "predictions_md.csv")
            self.assertEqual(len(pred), 200)
            self.assertTrue(all(r["y_pred"] == "1" for r in pred))
            self.assertNotIn("y_prob", pred[0])
            workflow.write_json(args.out_dir / "cases" / f'{cases[0]["case_id"]}.json', results[cases[0]["case_id"]])
            with self.assertRaisesRegex(ValueError, "provenance"):
                reporting.completed_cases(args, [cases[0]["case_id"]], "wrong fingerprint")


if __name__ == "__main__":
    unittest.main()

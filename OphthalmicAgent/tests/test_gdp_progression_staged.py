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
from Progression.prompts import DEFAULT_RUN_NAME, ENDPOINTS, SCENARIOS, SYSTEM_PROMPTS, VERSION
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
        self.fail_counterfactual = False

    def create(self, **request):
        self.requests.append(request)
        stage = next(k for k, v in SYSTEM_PROMPTS.items() if v == request["messages"][0]["content"])
        value = valid_response(stage)
        if stage == "orchestrator" and self.fail_final:
            del value["predictions"]["vfi"]
        if stage == "counterfactual" and self.fail_counterfactual:
            del value["scenarios"]["full_evidence"]
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
    def test_concise_role_prompts_preserve_forecasting_and_output_contract(self):
        self.assertEqual(VERSION, "ophthalmic_progression_staged_v2")
        word_limits = {"bio_profiler": 90, "rnflt_specialist": 130, "oct_specialist": 130,
                       "functional_specialist": 130, "counterfactual": 180, "orchestrator": 275}
        for stage, prompt in SYSTEM_PROMPTS.items():
            with self.subTest(stage=stage):
                self.assertLessEqual(len(prompt.split()), word_limits[stage])
                self.assertIn("Return JSON", prompt)
        for stage in ("counterfactual", "orchestrator"):
            self.assertIn("1 = progression predicted; 0 = non-progression predicted", SYSTEM_PROMPTS[stage])
            self.assertIn("Predict from baseline data; follow-up examinations are not provided", SYSTEM_PROMPTS[stage])
        self.assertIn("without treating uncertainty as a negative prediction", SYSTEM_PROMPTS["orchestrator"])
        self.assertIn("dependence on that source, not necessarily an error", SYSTEM_PROMPTS["counterfactual"])
        with patch.object(sys, "argv", [str(spec.origin)]):
            args = runner.parse_args()
        self.assertEqual(args.out_dir, ROOT / "OphthalmicAgent/outputs" / DEFAULT_RUN_NAME)
        self.assertEqual(DEFAULT_RUN_NAME, "gdp_progression_staged_v2")

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

    def test_previous_prompt_or_version_cache_cannot_be_reused(self):
        for alteration in ("prompt", "version"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fake = FakeClient()
                patient = case(root)
                caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake)
                previous = (patch.dict(SYSTEM_PROMPTS, {"bio_profiler": "Synthetic earlier prompt"})
                            if alteration == "prompt" else
                            patch.object(workflow, "VERSION", "ophthalmic_progression_staged_v1"))
                with previous:
                    caller.call(patient["case_id"], "bio_profiler", {})
                path = root / "stage_cache" / patient["case_id"] / "bio_profiler.json"
                before = path.read_bytes()
                reused = workflow.CachedCaller(root, "gpt-5.1", lambda: self.fail("Changed prompts must fail before API access"))
                with self.assertRaisesRegex(ValueError, "Cached evidence changed"):
                    reused.call(patient["case_id"], "bio_profiler", {})
                self.assertEqual(path.read_bytes(), before)

    def test_v1_prompt_run_requires_new_directory_even_with_contract_upgrade(self):
        for upgrade in (False, True):
            with self.subTest(upgrade=upgrade), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = SimpleNamespace(out_dir=root, deployment="gpt-5.1", include_oct=False,
                                       upgrade_output_contract=upgrade)
                patient = case(root)
                runner.freeze_run(args, [patient], {}, {})
                path = root / "resolved_config.json"
                saved = json.loads(path.read_text())
                saved.pop("fingerprint")
                saved["prompt_version"] = "ophthalmic_progression_staged_v1"
                saved["system_prompts"]["orchestrator"] = "Synthetic earlier clinical prompt"
                saved["fingerprint"] = workflow.digest(saved)
                workflow.write_json(path, saved)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    runner.freeze_run(args, [patient], {}, {})
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse((root / "resolved_config.before_output_contract.json").exists())

    def test_final_request_requires_exact_six_endpoint_schema(self):
        output = workflow.response_format("orchestrator")
        self.assertEqual(output["type"], "json_schema")
        self.assertIs(output["json_schema"]["strict"], True)
        root = output["json_schema"]["schema"]
        self.assertEqual(root["required"], ["predictions"])
        endpoints = root["properties"]["predictions"]
        self.assertEqual(set(endpoints["required"]), set(ENDPOINTS))
        self.assertEqual(set(endpoints["properties"]), set(ENDPOINTS))
        for schema in [root, endpoints, *endpoints["properties"].values()]:
            self.assertEqual(schema["type"], "object")
            self.assertIs(schema["additionalProperties"], False)
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        for item in endpoints["properties"].values():
            self.assertEqual(item["properties"]["prediction"], {"type": "integer", "enum": [0, 1]})
            self.assertEqual(item["properties"]["review_required"], {"type": "boolean"})
        for stage in set(SYSTEM_PROMPTS) - {"orchestrator", "counterfactual"}:
            self.assertEqual(workflow.response_format(stage), {"type": "json_object"})
        with self.assertRaisesRegex(ValueError, "top_level_keys"):
            workflow.validate("orchestrator", valid_response("orchestrator")["predictions"])

    def test_counterfactual_request_requires_five_scenarios_and_six_binary_endpoints(self):
        output = workflow.response_format("counterfactual")
        self.assertEqual(output["type"], "json_schema")
        self.assertTrue(output["json_schema"]["strict"])
        self.assertEqual(output["json_schema"]["name"], "gdp_progression_counterfactual")
        root = output["json_schema"]["schema"]
        self.assertEqual(root["required"], ["scenarios", "interpretation"])
        scenarios = root["properties"]["scenarios"]
        self.assertEqual(scenarios["required"], list(SCENARIOS))
        objects = [root, scenarios]
        for scenario in scenarios["properties"].values():
            self.assertEqual(scenario["required"], ["predictions", "reasoning"])
            endpoints = scenario["properties"]["predictions"]
            self.assertEqual(endpoints["required"], list(ENDPOINTS))
            for label in endpoints["properties"].values():
                self.assertEqual(label, {"type": "integer", "enum": [0, 1]})
            objects.extend([scenario, endpoints])
        for schema in objects:
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        with self.assertRaisesRegex(ValueError, "top_level_keys"):
            workflow.validate("counterfactual", valid_response("counterfactual")["scenarios"])

    def test_api_refusal_is_logged_without_retry_or_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeClient()
            refusal = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                      message=SimpleNamespace(content=None, refusal="Synthetic refusal"))], usage=None)
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, retry_sleep=0)
            with patch.object(fake, "create", return_value=refusal) as request:
                with self.assertRaisesRegex(workflow.ResponseRefusal, "Synthetic refusal"):
                    caller.call("fixture", "orchestrator", {})
                self.assertEqual(request.call_count, 1)
            self.assertFalse((root / "stage_cache/fixture/orchestrator.json").exists())
            record = json.loads((root / "attempts.jsonl").read_text())
            self.assertEqual(record["refusal"], "Synthetic refusal")
            self.assertEqual(record["response_format"]["type"], "json_schema")

    def legacy_configuration(self, args, patients, *, counterfactual=False):
        runner.freeze_run(args, patients, {}, {})
        current = json.loads((args.out_dir / "resolved_config.json").read_text())
        legacy = {k: v for k, v in current.items() if k not in {"response_formats", "fingerprint"}}
        legacy["implementation"] = {**legacy["implementation"], **runner.LEGACY_OUTPUT_IMPLEMENTATION}
        if counterfactual:
            legacy["implementation"] = {**current["implementation"], **runner.LEGACY_COUNTERFACTUAL_IMPLEMENTATION}
            legacy["response_formats"] = {**current["response_formats"], "counterfactual": {"type": "json_object"}}
        legacy["fingerprint"] = workflow.digest(legacy)
        workflow.write_json(args.out_dir / "resolved_config.json", legacy)
        return legacy

    def test_failed_legacy_smoke_upgrades_both_contracts_and_preserves_specialists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(out_dir=root, deployment="gpt-5.1", include_oct=False)
            patient = case(root)
            legacy = self.legacy_configuration(args, [patient])
            fake = FakeClient()
            fake.fail_final = True
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, max_attempts=1, retry_sleep=0)
            with patch.object(workflow, "response_format", return_value={"type": "json_object"}):
                with self.assertRaisesRegex(RuntimeError, "orchestrator failed"):
                    workflow.run_case(patient, caller)
            caches = {p: p.read_bytes() for p in (root / "stage_cache").glob("*/*.json")}
            self.assertEqual(len(caches), 4)
            with self.assertRaisesRegex(ValueError, "upgrade-output-contract"):
                runner.freeze_run(args, [patient], {}, {})
            args.upgrade_output_contract = True
            with redirect_stdout(StringIO()):
                fingerprint = runner.freeze_run(args, [patient], {}, {})
            self.assertNotEqual(fingerprint, legacy["fingerprint"])
            self.assertEqual(json.loads((root / "resolved_config.before_output_contract.json").read_text()), legacy)
            audit = json.loads((root / "output_contract_upgrade.json").read_text())
            self.assertEqual(len(audit["retained_upstream_cache_files"]), 3)
            self.assertEqual(set(audit["response_format_changes"]), {"counterfactual", "orchestrator"})
            self.assertEqual(len(audit["archived_cache_files"]), 1)
            fake.fail_final = False
            before = len(fake.requests)
            result = workflow.run_case(patient, caller)
            self.assertEqual(len(fake.requests) - before, 2)
            self.assertEqual(fake.requests[-1]["response_format"]["type"], "json_schema")
            self.assertEqual(set(result["predictions"]), set(ENDPOINTS))
            for path, raw in caches.items():
                if path.stem == "counterfactual":
                    archived = root / "stage_cache.before_output_contract" / path.relative_to(root / "stage_cache")
                    self.assertEqual(archived.read_bytes(), raw)
                    self.assertEqual(json.loads(path.read_text())["response_format"]["type"], "json_schema")
                else:
                    self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(runner.freeze_run(args, [patient], {}, {}), fingerprint)

    def test_failed_counterfactual_smoke_upgrades_and_reuses_three_specialists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(out_dir=root, deployment="gpt-5.1", include_oct=False)
            patient = case(root)
            legacy = self.legacy_configuration(args, [patient], counterfactual=True)
            fake = FakeClient()
            fake.fail_counterfactual = True
            caller = workflow.CachedCaller(root, "gpt-5.1", lambda: fake, max_attempts=3, retry_sleep=0)
            current_format = workflow.response_format
            with patch.object(workflow, "response_format", side_effect=lambda stage:
                              {"type": "json_object"} if stage == "counterfactual" else current_format(stage)):
                with self.assertRaisesRegex(RuntimeError, "counterfactual failed after 3 attempts"):
                    workflow.run_case(patient, caller)
            self.assertEqual(len(fake.requests), 6)
            caches = {p: p.read_bytes() for p in (root / "stage_cache").glob("*/*.json")}
            self.assertEqual(len(caches), 3)
            original_attempts = (root / "attempts.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "upgrade-output-contract"):
                runner.freeze_run(args, [patient], {}, {})
            args.upgrade_output_contract = True
            with redirect_stdout(StringIO()):
                fingerprint = runner.freeze_run(args, [patient], {}, {})
            self.assertNotEqual(fingerprint, legacy["fingerprint"])
            self.assertEqual(json.loads((root / "resolved_config.before_output_contract.json").read_text()), legacy)
            audit = json.loads((root / "output_contract_upgrade.json").read_text())
            self.assertEqual(set(audit["response_format_changes"]), {"counterfactual"})
            self.assertFalse(audit["clinical_prompts_changed"])
            self.assertEqual(len(audit["retained_upstream_cache_files"]), 3)
            self.assertEqual(audit["archived_cache_files"], [])
            fake.fail_counterfactual = False
            before = len(fake.requests)
            workflow.run_case(patient, caller)
            self.assertEqual(len(fake.requests) - before, 2)
            for request in fake.requests[-2:]:
                self.assertEqual(request["response_format"]["type"], "json_schema")
            for path, raw in caches.items():
                self.assertEqual(path.read_bytes(), raw)
            self.assertTrue((root / "attempts.jsonl").read_bytes().startswith(original_attempts))
            self.assertEqual(runner.freeze_run(args, [patient], {}, {}), fingerprint)

    def test_contract_upgrade_cannot_bypass_other_provenance_guards(self):
        alterations = ("evidence", "prompt", "implementation", "fingerprint", "completed_case", "cached_final", "summary")
        for counterfactual, alteration in ((c, a) for c in (False, True) for a in alterations):
            with self.subTest(counterfactual=counterfactual, alteration=alteration), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = SimpleNamespace(out_dir=root, deployment="gpt-5.1", include_oct=False,
                                       upgrade_output_contract=True)
                patient = case(root)
                saved = self.legacy_configuration(args, [patient], counterfactual=counterfactual)
                if alteration == "evidence":
                    patient["demographics"]["age"] = 50
                elif alteration in {"prompt", "implementation", "fingerprint"}:
                    if alteration == "prompt":
                        saved["system_prompts"]["orchestrator"] = "Different clinical instructions"
                    elif alteration == "implementation":
                        saved["implementation"]["OphthalmicAgent/Progression/evidence.py"] = "different"
                    saved["fingerprint"] = ("invalid" if alteration == "fingerprint" else
                                              workflow.digest({k: v for k, v in saved.items() if k != "fingerprint"}))
                    workflow.write_json(root / "resolved_config.json", saved)
                elif alteration == "completed_case":
                    workflow.write_json(root / "cases/data_0301.json", {})
                elif alteration == "cached_final":
                    workflow.write_json(root / "stage_cache/data_0301/orchestrator.json", {})
                else:
                    workflow.write_json(root / "summary.json", {"completed_cases": 1})
                with self.assertRaises(ValueError):
                    runner.freeze_run(args, [patient], {}, {})
                self.assertFalse((root / "resolved_config.before_output_contract.json").exists())

    def test_five_case_runner_resumes_after_counterfactual_contract_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = ["data_0344", "data_0371", "data_0397", "data_0411", "data_0432"]
            args = SimpleNamespace(stage="smoke", case_ids=selected, out_dir=root, include_oct=False,
                                   deployment="gpt-5.1", max_attempts=1, upgrade_output_contract=True)
            first = case(root)
            patients = [{**first, "case_id": case_id} for case_id in ["data_0301", *selected]]
            answers = {t: [{"image_id": c["case_id"], "y_true": "0", "split": "test"}
                           for c in patients] for t in ENDPOINTS}
            self.legacy_configuration(args, patients, counterfactual=True)
            fake = FakeClient()
            fake.fail_counterfactual = True
            caller = workflow.CachedCaller(root, args.deployment, lambda: fake, max_attempts=1)
            current_format = workflow.response_format
            with patch.object(workflow, "response_format", side_effect=lambda stage:
                              {"type": "json_object"} if stage == "counterfactual" else current_format(stage)):
                with self.assertRaisesRegex(RuntimeError, "counterfactual failed"):
                    workflow.run_case(patients[1], caller)
            before = len(fake.requests)
            fake.fail_counterfactual = False
            with patch.object(evidence, "prepare", return_value=(patients, answers, {})), \
                 patch.object(runner, "make_client", return_value=fake), \
                 patch.object(reporting, "collect") as collect, redirect_stdout(StringIO()):
                runner.execute(args)
                # First patient's three specialist calls are reused; four patients run fully.
                self.assertEqual(len(fake.requests) - before, 22)
                runner.execute(args)
                self.assertEqual(len(fake.requests) - before, 22)
                collect.assert_not_called()
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["completed_requested_cases"], 5)
            self.assertEqual(summary["missing_cases"], ["data_0301"])
            self.assertFalse(summary["complete_live_cohort"])
            for target in ENDPOINTS:
                rows = evidence.native.read_csv(root / f"predictions_{target}.csv")
                self.assertEqual({r["image_id"] for r in rows}, set(selected))

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

    def test_selected_five_case_smoke_and_full_run_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = ["data_0344", "data_0371", "data_0397", "data_0411", "data_0432"]
            args = SimpleNamespace(stage="smoke", case_ids=selected, out_dir=root, include_oct=False,
                                   deployment="gpt-5.1", max_attempts=2)
            first = case(root)
            patients = [first] + [{**first, "case_id": case_id} for case_id in selected]
            answers = {t: [{"image_id": c["case_id"], "y_true": "0", "split": "test"}
                           for c in patients] for t in ENDPOINTS}
            fake = FakeClient()
            with patch.object(evidence, "prepare", return_value=(patients, answers, {})), \
                 patch.object(runner, "make_client", return_value=fake), \
                 patch.object(reporting, "collect") as collect, redirect_stdout(StringIO()):
                runner.execute(args)
                fingerprint = json.loads((root / "resolved_config.json").read_text())["fingerprint"]
                runner.execute(args)
                self.assertEqual(len(fake.requests), 25)
                summary = json.loads((root / "summary.json").read_text())
                self.assertEqual(summary["requested_case_ids"], selected)
                self.assertEqual(summary["completed_requested_cases"], 5)
                self.assertEqual(summary["completed_cases"], 5)
                self.assertEqual(summary["expected_cases"], 6)
                self.assertEqual(summary["missing_cases"], ["data_0301"])
                self.assertFalse(summary["complete_live_cohort"])
                self.assertFalse((root / "cases/data_0301.json").exists())
                self.assertFalse((root / "stage_cache/data_0301").exists())
                self.assertFalse((root / "results/results.md").exists())
                collect.assert_not_called()
                for target in ENDPOINTS:
                    rows = evidence.native.read_csv(root / f"predictions_{target}.csv")
                    self.assertEqual({r["image_id"] for r in rows}, set(selected))
                args.stage, args.case_ids = "run", None
                runner.execute(args)
                self.assertEqual(len(fake.requests), 30)
                collect.assert_called_once()
                self.assertEqual(json.loads((root / "resolved_config.json").read_text())["fingerprint"], fingerprint)
                summary = json.loads((root / "summary.json").read_text())
                self.assertEqual(summary["completed_requested_cases"], 6)
                self.assertTrue(summary["complete_live_cohort"])

    def test_bad_smoke_selection_fails_before_freezing_or_api(self):
        for selected in (["data_9999"], ["data_0301", "data_0301"], []):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = SimpleNamespace(stage="smoke", case_ids=selected, out_dir=root)
                with patch.object(evidence, "prepare", return_value=([case(root)], {}, {})), \
                     patch.object(runner, "freeze_run") as freeze, \
                     patch.object(runner, "make_client") as client, redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(ValueError, "case-ids"):
                        runner.execute(args)
                    freeze.assert_not_called()
                    client.assert_not_called()

    def test_case_selection_only_allowed_for_smoke(self):
        with patch.object(sys, "argv", [str(spec.origin), "--stage", "smoke", "--case-ids", "data_0344", "data_0371"]):
            self.assertEqual(runner.parse_args().case_ids, ["data_0344", "data_0371"])
        for stage in ("prepare", "run", "collect", "status", "prompts"):
            with self.subTest(stage=stage), \
                 patch.object(sys, "argv", [str(spec.origin), "--stage", stage, "--case-ids", "data_0344"]), \
                 patch("sys.stderr", new=StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.parse_args()
                self.assertEqual(raised.exception.code, 2)

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
            self.assertIn(VERSION, result)
            self.assertIn(f"outputs/{DEFAULT_RUN_NAME}", result)
            self.assertNotIn("outputs/gdp_progression_staged_v1", result)
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

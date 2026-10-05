"""No-network tests of the frozen experiment, with synthetic API responses."""
import ast
import importlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_frozen_confirmation as runner
from Confirmation import contract, design, reporting, workflow
from EquityAgent.demographics import fairvision_age_group
from test_fairvision_ablation import FakeClient, FakeOrchestrator, trace

base, live = runner.base, workflow.live


def final(label=1, flag=True):
    return json.dumps(dict(diagnosis=label, escalation_required=flag, reasoning="Synthetic evidence",
                           overview="Synthetic overview", escalation_reason="Synthetic review decision"))


def config():
    return dict(fingerprint="frozen-test", deployment="gpt-5.1",
                frozen_generation=design.read_protocol()["generation"], oct_weights="unused.pth")


def case(task="glaucoma", key="synthetic", truth=0):
    metadata = dict(Age="55", Gender="female", Race="white", Ethnicity="non-hispanic")
    return dict(task=task, case_id=key, truth=truth, filename=f"data/{base.DISEASE_FOLDERS[task]}/Test/{key}.npz",
                metadata=metadata, **base.demographic(metadata))


class FrozenTests(unittest.TestCase):
    def test_age_boundaries_and_entrypoints(self):
        for age, group in ((0, "younger"), (49.9, "younger"), (50, "middle-aged"),
                           (60, "middle-aged"), (69.9, "middle-aged"), (70, "older"), (100, "older")):
            self.assertEqual(fairvision_age_group(age), group)
            self.assertEqual(base.demographic(dict(Age=age))["age_group"], group)
        for age in (-1, "nan", "inf", None, "missing"):
            with self.assertRaises(ValueError):
                fairvision_age_group(age)
        for name in ("main_new.py", *(f"evaluate_fairvision_{t}_agentic.py" for t in base.TASKS)):
            tree = ast.parse((ROOT / "OphthalmicAgent" / name).read_text())
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                and n.func.id == "fairvision_age_group" for n in ast.walk(tree)))

    def test_required_explicit_flag_and_binary_label(self):
        for flag in (True, False):
            self.assertIs(contract.parse_final(final(0, flag))["escalation_required"], flag)
        original = json.loads(final())
        for field, value in (("diagnosis", True), ("diagnosis", -1), ("escalation_required", "false"),
                             ("escalation_required", 1), ("escalation_reason", "")):
            with self.assertRaises(ValueError):
                contract.parse_final(json.dumps({**original, field: value}))
        for field in original:
            with self.assertRaises(ValueError):
                contract.parse_final(json.dumps({k: v for k, v in original.items() if k != field}))

    def test_cohorts_are_reproducible_balanced_and_exclude_all_recorded_attempts(self):
        paths = [design.DESIGN / n for n in ("cohorts.csv", "selection_receipt.json")]
        before = [p.read_bytes() for p in paths]
        with redirect_stdout(StringIO()):
            design.select_cohorts()
        self.assertEqual(before, [p.read_bytes() for p in paths])
        receipt = json.loads(paths[1].read_text())
        locked = base.load_locked(paths[0])
        for task in base.TASKS:
            self.assertEqual(Counter(c["truth"] for c in locked[task].values()), {0: 125, 1: 125})
            self.assertFalse(set(locked[task]) & set(receipt["exclusions"][task]))
            for c in locked[task].values():
                self.assertEqual(len(design.demographics_profiles(c)), 6)

    def test_demographic_changes_are_single_attribute_and_age_aligned(self):
        original = case()
        profiles = design.demographics_profiles(original)
        self.assertEqual(Counter(p["field"] for p in profiles), dict(Race=2, Gender=1, Age=2, Ethnicity=1))
        for p in profiles:
            self.assertEqual([k for k in original["metadata"] if original["metadata"][k] != p["metadata"][k]], [p["field"]])
            self.assertEqual(p["age_group"], fairvision_age_group(p["metadata"]["Age"]))
        self.assertEqual(original["metadata"]["Age"], "55")

    def test_request_settings_override_legacy_and_cache_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient([final()])
            api = live.CachedClient(client, Path(tmp), config(), "dr", "orchestrator", contract.parse_final)
            request = dict(model="wrong-deployment", temperature=.9, top_p=.4, seed=123, max_tokens=1,
                           messages=[dict(role="system", content="Legacy instructions")])
            # Deprecated max_tokens must not coexist with max_completion_tokens.
            api.create(**request)
            api.create(**request)
            self.assertEqual(len(client.requests), 1)
            sent = client.requests[0]
            for field in ("temperature", "top_p", "max_completion_tokens"):
                self.assertEqual(sent[field], config()["frozen_generation"][field])
            self.assertEqual(sent["model"], "gpt-5.1")
            self.assertNotIn("seed", sent)
            self.assertNotIn("max_tokens", sent)
            self.assertIn("vision-threatening DR", sent["messages"][0]["content"])
            self.assertIn("escalation_required", sent["response_format"]["json_schema"]["schema"]["required"])
            record = json.loads(next(Path(tmp).glob("*.json")).read_text())
            self.assertIn("request_started_utc", record)
            self.assertIn("completed_utc", record)

    def test_invalid_responses_not_cached_as_negative(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(live.time, "sleep"):
            client = FakeClient(['{"diagnosis":0}']*3)
            api = live.CachedClient(client, Path(tmp), config(), "amd", "orchestrator", contract.parse_final)
            with self.assertRaises(RuntimeError):
                api.create(messages=[dict(role="system", content="Test")])
            self.assertEqual(len(client.requests), 3)
            self.assertFalse(list(Path(tmp).glob("*.json")))
            self.assertEqual(len(list((Path(tmp)/"failed").glob("*.json"))), 3)

    def test_api_budget_and_returned_identity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = FakeClient(["one", "two"])
            client = workflow.BudgetClient(raw, Path(tmp), 1)
            client.create(model="gpt-5.1")
            with self.assertRaises(ValueError):
                client.create(model="gpt-5.1")
            self.assertEqual(len(raw.requests), 1)
        with tempfile.TemporaryDirectory() as tmp:
            raw = FakeClient(["one", "two"])
            client = workflow.BudgetClient(raw, Path(tmp), 10)
            base.write_json(Path(tmp)/"model_identity.json", dict(deployment="gpt-5.1", returned_model="different"))
            with self.assertRaises(RuntimeError):
                client.create(model="gpt-5.1")
            with self.assertRaises(ValueError):
                client.create(model="gpt-5.1")
            self.assertEqual(len(raw.requests), 1)

    def test_paired_decisions_are_cached_and_do_not_send_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = dict(narrative="Patient", oct_report="OCT", slo_report="SLO", cdr="Not Available", probability_percent=65.)
            client = FakeClient([json.dumps(trace()), final()] * 2)
            for _ in range(2):
                for arm in base.VARIANTS[2:]:
                    workflow.decide(root, config(), case(), evidence, .7 if arm == "retinagent_full" else None,
                                    FakeOrchestrator(), client, "receipt", arm)
            self.assertEqual(len(client.requests), 4)
            for request in client.requests:
                self.assertNotIn("truth", json.dumps(request))
            no_priors = [r for r in client.requests if r.get("response_format", {}).get("json_schema", {}).get("name") == "retinagent_final_v2"][0]
            self.assertNotIn("**Trust Score**", json.dumps(no_priors))
            target = root/"agent/glaucoma/retinagent_full/synthetic.json"
            saved = json.loads(target.read_text())
            saved["escalation_required"] = False
            base.write_json(target, saved)
            with self.assertRaises(ValueError):
                workflow.decide(root, config(), case(), evidence, .7, FakeOrchestrator(), client, "receipt", "retinagent_full")

    def test_actual_three_task_orchestrators_accept_the_frozen_contract(self):
        evidence = dict(narrative="Patient", oct_report="OCT", slo_report="SLO", cdr="Not Available", probability_percent=65.)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                "openai": SimpleNamespace(AzureOpenAI=None), "dotenv": SimpleNamespace(load_dotenv=lambda: None)}):
            for task in base.TASKS:
                module = importlib.import_module("Orchestrator.fairvision_" + task)
                orchestrator = module.Orchestrator.__new__(module.Orchestrator)
                client = FakeClient([json.dumps(trace()), final()] * 2)
                for arm in base.VARIANTS[2:]:
                    result = workflow.decide(Path(tmp), config(), case(task), evidence, .7 if arm == "retinagent_full" else None,
                                             orchestrator, client, "receipt", arm)
                    self.assertIs(result["escalation_required"], True)
                self.assertEqual(len(client.requests), 4)

    def test_all_six_live_profiles_rerun_metadata_stages_only(self):
        class Bio:
            def generate_narrative(self, metadata):
                return self.model_client.create(messages=[dict(role="system", content="Scribe"),
                    dict(role="user", content=json.dumps(metadata))]).choices[0].message.content
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = dict(narrative="Original", oct_report="OCT unchanged", slo_report="SLO unchanged", cdr=.5, probability_percent=65.)
            responses = [json.dumps(trace()), final()]*2
            for _ in range(6):
                responses.extend(["Changed narrative", json.dumps(trace()), final(0, False)])
            client = FakeClient(responses)
            revision = dict(model="test", revision="test")
            base.write_json(root/"external_cdr_model.json", revision)
            slo = SimpleNamespace(cdr_model_name="test", model_cdr=SimpleNamespace(config=SimpleNamespace(_commit_hash="test")))
            modules = {
                "VisionAgent.vision_slo_glaucoma": SimpleNamespace(VisionSpecialistSlo=lambda *a, **kw: slo),
                "BioProfilerAgent.bio_profiler_glaucoma": SimpleNamespace(BioProfiler=Bio),
                "Orchestrator.fairvision_glaucoma": SimpleNamespace(Orchestrator=FakeOrchestrator),
            }
            with patch.dict(sys.modules, {"data.loader": SimpleNamespace(ExcelEyeLoader=lambda path: None)}), \
                    patch.object(base, "TASKS", ("glaucoma",)), \
                    patch.object(workflow.importlib, "import_module", side_effect=modules.__getitem__), \
                    patch.object(live, "anchor_priors", return_value={"glaucoma": {"synthetic": .7}}), \
                    patch.object(live, "make_oct"), patch.object(live, "shared_evidence", return_value=evidence) as shared, \
                    patch.object(workflow, "anchor_fitted", return_value="fitted"), \
                    patch.object(base, "trust_for", return_value=.6) as trust, redirect_stdout(StringIO()):
                workflow.fairvision(root, config(), [case()], [], "receipt", client, "cpu")
                self.assertEqual(len(client.requests), 22)  # Shared three calls are mocked out.
                self.assertEqual(trust.call_count, 6)
                self.assertEqual(shared.call_count, 1)
                workflow.fairvision(root, config(), [case()], [], "receipt", client, "cpu")
                self.assertEqual(len(client.requests), 22)
            self.assertEqual(len(list((root/"demographic_audit/glaucoma/synthetic").glob("*_profile.json"))), 6)

    def test_real_external_sources_have_all_451_cases(self):
        external, _ = runner.external_cases(SimpleNamespace(external_root=ROOT/"OphthalmicAgent"))
        self.assertEqual(Counter(c["dataset"] for c in external), dict(drishti=51, refuge2=400))
        self.assertEqual(sum(c["truth"] for c in external if c["dataset"] == "drishti"), 34)
        self.assertEqual(sum(c["truth"] for c in external if c["dataset"] == "refuge2"), 40)

    def test_external_pipeline_uses_raw_score_explicit_flags_and_resumes(self):
        from CounterfactualAgent.counterfactual_cfp import SCENARIO_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = [dict(dataset="drishti", case_id="test", truth=0, probability=.6, cfp_path="unused.png")]
            base.write_json(root/"external_cdr/drishti/test.json", dict(cdr=None))
            cf = dict(scenarios={name: dict(diagnosis=1, confidence="uncertain", reasoning="Synthetic")
                                for name in SCENARIO_NAMES}, interpretation="Synthetic audit")
            client = FakeClient(["Synthetic CFP report", json.dumps(cf), final()])
            with patch.dict(sys.modules, {"openai": SimpleNamespace(AzureOpenAI=None)}), \
                    patch("run_external_glaucoma_agent.data_url", return_value="data:image/jpeg;base64,SYNTHETIC"), \
                    redirect_stdout(StringIO()):
                workflow.external(root, config(), external, "receipt", client)
                workflow.external(root, config(), external, "receipt", client)
            self.assertEqual(len(client.requests), 3)
            self.assertNotIn("truth", json.dumps(client.requests))
            self.assertIn("raw glaucoma probability: 60.0%", json.dumps(client.requests[-1]))
            saved = json.loads((root/"external/drishti/test.json").read_text())
            reporting.validate_saved(saved, config(), external[0], "receipt")

    def test_npz_preflight_checks_reference_before_any_inference(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = case()
            image = root/row["filename"]
            image.parent.mkdir(parents=True)
            np.savez(image, oct_bscans=np.zeros((8, 2, 2)), slo_fundus=np.zeros((2, 2)), glaucoma=0)
            weights = root/"weights.pth"
            weights.write_bytes(b"synthetic checkpoint")
            cfg = {**config(), "oct_weights": str(weights), "data_root": str(root), "confirmation_source_lock": "lock"}
            with patch.object(runner, "read_run", return_value=(cfg, [row], [], [])), redirect_stdout(StringIO()):
                runner.preflight(SimpleNamespace(run_root=root))
                np.savez(image, oct_bscans=np.zeros((8, 2, 2)), slo_fundus=np.zeros((2, 2)), glaucoma=1)
                with self.assertRaisesRegex(ValueError, "label mismatch"):
                    runner.preflight(SimpleNamespace(run_root=root))
            self.assertFalse((root/"api").exists())

    def test_complete_collection_exports_all_forced_and_selective_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [case(t, str(i), i % 2) for t in base.TASKS for i in range(250)]
            external = [dict(dataset=d, case_id=str(i), truth=i % 2, probability=.8) for d, count in
                        design.read_protocol()["external_counts"].items() for i in range(count)]
            base.write_json(root/"input_receipt.json", {})
            receipt = base.digest({})
            for c in cases:
                evidence = dict(narrative="synthetic", oct_report="image only", slo_report="image only", cdr=.5)
                base.write_json(root/"shared"/c["task"]/f"{c['case_id']}.json", dict(evidence=evidence))
                for arm in base.VARIANTS[2:]:
                    saved = dict(fingerprint=config()["fingerprint"], receipt=receipt, task=c["task"], case_id=c["case_id"],
                        variant=arm, scenario="original", prediction=c["truth"], escalation_required=False,
                        decision=final(c["truth"], False), shared_evidence_sha256=base.digest(evidence))
                    base.write_json(root/"agent"/c["task"]/arm/f"{c['case_id']}.json", saved)
                for profile in design.demographics_profiles(c):
                    folder = root/"demographic_audit"/c["task"]/c["case_id"]
                    changed = {**saved, "scenario": profile["name"], "evidence_fingerprint": "test"}
                    base.write_json(folder/f"{profile['name']}.json", changed)
                    base.write_json(folder/f"{profile['name']}_profile.json", dict(profile=profile,
                        decision_fingerprint="test", image_evidence_sha256=base.digest({k:v for k,v in evidence.items() if k != "narrative"})))
            for c in external:
                base.write_json(root/"external"/c["dataset"]/f"{c['case_id']}.json", dict(
                    fingerprint=config()["fingerprint"], receipt=receipt, dataset=c["dataset"], case_id=c["case_id"],
                    prediction=c["truth"], escalation_required=False, decision=final(c["truth"], False)))
            with patch.object(base, "collect") as collect, redirect_stdout(StringIO()):
                reporting.collect_complete(root, config(), cases, external)
            collect.assert_called_once_with(root)
            completion = json.loads((root/"completion.json").read_text())
            self.assertTrue(completion["complete"])
            self.assertEqual(completion["live_perturbations"], 4500)
            self.assertEqual(len(base.read_csv(root/"live_escalation.csv")), 16)
            self.assertEqual(len(base.read_csv(root/"external_metrics.csv")), 4)

    def test_complete_cohort_guard_and_unambiguous_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base.write_json(root/"input_receipt.json", {})
            cases = [case(t, str(i), i % 2) for t in base.TASKS for i in range(250)]
            external = [dict(dataset=d, case_id=str(i)) for d, count in design.read_protocol()["external_counts"].items() for i in range(count)]
            with self.assertRaisesRegex(ValueError, "Incomplete experiment"):
                reporting.collect_complete(root, config(), cases, external)
            self.assertFalse(json.loads((root/"completion.json").read_text())["complete"])
            self.assertFalse((root/"results.csv").exists())
            with self.assertRaisesRegex(ValueError, "750 unique"):
                reporting.collect_complete(root, config(), cases[:-1], external)
        saved = dict(fingerprint=config()["fingerprint"], receipt="receipt", task="glaucoma", variant="retinagent_full",
                     scenario="original", case_id="wrong", prediction=1, escalation_required=True, decision=final())
        with self.assertRaises(ValueError):
            reporting.validate_saved(saved, config(), case(), "receipt", "retinagent_full")

    def test_selective_bootstrap_retains_forced_predictions_and_empty_group(self):
        rows = [dict(truth=y, prediction=y, escalation_required=False) for y in (0, 1)] * 3
        result = reporting.selective(rows)
        self.assertEqual(result[0]["macro_f1"], 1.)
        self.assertEqual(result[0]["lower"], 1.)
        self.assertEqual(result[0]["upper"], 1.)
        self.assertEqual(result[1]["n"], 0)
        self.assertIsNone(result[1]["macro_f1"])
        rows[0]["prediction"] = 1
        self.assertLess(reporting.selective(rows)[0]["macro_f1"], 1.)

    def test_source_and_prompt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"synthetic.py"
            source.write_text('PROMPT = "' + 'Synthetic text. '*10 + '"\n')
            with patch.object(design, "ROOT", root), patch.object(design, "DESIGN", root), \
                    patch.object(design, "freeze_paths", return_value=[source]), \
                    patch.object(design, "read_protocol", return_value={"version": "test"}), \
                    patch.object(design.subprocess, "check_output", return_value="fake-commit\n"), redirect_stdout(StringIO()):
                design.freeze()
                design.verify_freeze(require_commit=False)
                original = source.read_text()
                source.write_text(original + "CHANGED = True\n")
                with self.assertRaises(ValueError):
                    design.verify_freeze(require_commit=False)
                source.write_text(original)
                (root/"prompts_snapshot.txt").write_text("Changed")
                with self.assertRaises(ValueError):
                    design.verify_freeze(require_commit=False)


if __name__ == "__main__":
    unittest.main()

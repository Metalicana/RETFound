from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_fairvision_ablation as runner
from Ablation import fairvision_live as live


def trace():
    return dict(scenarios=[dict(name=s, diagnosis=1, confidence="uncertain", reasoning="Synthetic test")
                           for s in live.SCENARIOS], interpretation="Synthetic audit")


class FakeClient:
    def __init__(self, contents, finish="stop"):
        self.contents = iter(contents)
        self.requests = []
        self.finish = finish
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(model="fake-test-only", usage=None,
            choices=[SimpleNamespace(finish_reason=self.finish,
                                     message=SimpleNamespace(content=next(self.contents)))])


class FakeOrchestrator:
    def analyze(self, state, probability, cdr, trust, cf):
        messages = [dict(role="system", content="Test-only orchestrator"),
                    dict(role="user", content=f"{state}\nCDR: {cdr}\n- **Trust Score**: {trust}\n{cf}")]
        response = self.model_client.chat.completions.create(messages=messages)
        return {"decision": response.choices[0].message.content}


class AblationTests(unittest.TestCase):
    def cdr_fixture(self, root):
        cases = [dict(task="glaucoma", case_id="valid", truth=1),
                 dict(task="amd", case_id="missing", truth=0)]
        config = dict(deployment="test", version=runner.VERSION,
                      prepared_sha256=runner.digest(cases), offline_sha256=runner.digest([]),
                      validation_sha256=runner.digest([]),
                      source_code={**runner.current_code(), **runner.LEGACY_CDR_CODE})
        config["fingerprint"] = runner.digest(config)
        for name, value in (("config", config), ("prepared_cases", cases),
                            ("offline_predictions", []), ("validation_cases", [])):
            runner.write_json(root / (name + ".json"), value)
        runner.write_json(root / "anchor_trust.json", {"unchanged": True})
        runner.write_json(root / "live_receipt.json", {"unchanged": True})
        for case, cdr in zip(cases, (.514, -1.)):
            t, c = case["task"], case["case_id"]
            evidence = dict(narrative="Patient", oct_report="OCT", slo_report="SLO",
                            cdr=cdr, probability_percent=78.)
            runner.write_json(root / "shared" / t / f"{c}.json", dict(fingerprint="shared", evidence=evidence))
            runner.write_json(root / "api" / t / c / "shared/report.json", {"paid_report": "unchanged"})
            runner.write_json(root / "anchor_validation" / t / "a.json", {"prediction": .8})
            client = FakeClient([json.dumps(trace()), "[LABELS]" + t.upper() + "_DETECTED: 1[/LABELS]"] * 2)
            live.run_pair(root, config, case, evidence, .7, FakeOrchestrator(), client, "receipt")
        return config, cases

    def test_missing_cdr_is_not_a_zero_or_negative_measurement(self):
        for raw in (None, "Not Available", " not available ", "N/A", "", -1, "-1", float("nan"),
                    float("inf"), -.4, 1.4):
            with self.subTest(raw=raw):
                self.assertEqual(runner.cdr_value(raw), "Not Available")
        for raw in (0, .426, 1, "0.514"):
            self.assertEqual(runner.cdr_value(raw), float(raw))
        for raw in (True, False, "unexpected tool output"):
            with self.assertRaises(ValueError):
                runner.cdr_value(raw)

    def test_failed_shared_evidence_resumes_paid_reports_with_missing_cdr(self):
        class Bio:
            def generate_narrative(self, metadata):
                return self.model_client.create(messages=[dict(role="user", content="narrative")]).choices[0].message.content

        class OCT:
            def analyze(self, image, middle, state):
                state["oct_diagnosis"] = {"Glaucoma": {"Prob_Pct": 78}}
                r = self.model_client.create(messages=[dict(role="user", content="OCT")])
                return None, r.choices[0].message.content

        class SLO:
            def analyze(self, image, state):
                r = self.model_client.create(messages=[dict(role="user", content="SLO")])
                return r.choices[0].message.content, "Not Available"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(fingerprint="run", deployment="test")
            case = dict(task="glaucoma", case_id="failed", metadata={})
            client = FakeClient(["narrative", "OCT report", "SLO report"])
            components = (OCT(), SLO(), Bio())
            with patch.object(live, "load_images", return_value=dict(oct_img=None, middle_oct=None, fundus_img=None)):
                with patch.object(live, "cdr_value", side_effect=float), self.assertRaises(ValueError):
                    live.shared_evidence(root, config, case, None, *components, client, "receipt")
                self.assertEqual(len(client.requests), 3)
                self.assertFalse((root / "shared/glaucoma/failed.json").exists())
                evidence = live.shared_evidence(root, config, case, None, *components, client, "receipt")
                self.assertEqual(evidence["cdr"], "Not Available")
                self.assertEqual(len(client.requests), 3)
                # The resumed case and both arms receive the same explicit missing value.
                for trust in (None, .7):
                    messages = live.counterfactual_messages("glaucoma", evidence, trust)
                    payload = json.loads(messages[1]["content"].split("EVIDENCE_JSON:\n")[1])
                    self.assertEqual(payload["vertical_cup_to_disc_ratio"], "Not Available")

    def test_cdr_repair_preserves_valid_cases_and_shared_paid_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases = self.cdr_fixture(root)
            originals = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.json")}
            with self.assertRaisesRegex(ValueError, "--stage repair-cdr"):
                runner.validate_run_code(root, config)
            with redirect_stdout(StringIO()):
                runner.repair_cdr(SimpleNamespace(run_root=root))
            runner.validate_run_code(root, config)
            self.assertEqual((root / "config.json").read_bytes(), originals["config.json"])
            fixed = json.loads((root / "shared/amd/missing.json").read_text())
            self.assertEqual(fixed["evidence"]["cdr"], "Not Available")
            receipt = json.loads((root / runner.CDR_REPAIR / "receipt.json").read_text())
            self.assertEqual(len(receipt["updates"]), 1)
            self.assertEqual(receipt["status"], "complete")
            for relative, data in originals.items():
                if relative in receipt["artifacts"]:
                    self.assertEqual((root / runner.CDR_REPAIR / "original" / relative).read_bytes(), data)
                    if relative not in receipt["updates"]:
                        self.assertFalse((root / relative).exists())
                else:
                    self.assertEqual((root / relative).read_bytes(), data)
            # Both corrected arms rerun, not their upstream reports or any valid case.
            client = FakeClient([json.dumps(trace()), "[LABELS]AMD_DETECTED: 1[/LABELS]"] * 2)
            live.run_pair(root, config, cases[1], fixed["evidence"], .7, FakeOrchestrator(), client, "receipt")
            self.assertEqual(len(client.requests), 4)
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            with redirect_stdout(StringIO()):
                runner.repair_cdr(SimpleNamespace(run_root=root))
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.json")})

    def test_cdr_repair_recovers_interrupted_archiving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self.cdr_fixture(root)
            replace = Path.replace
            moved = []

            def interrupt(path, destination):
                result = replace(path, destination)
                if "original" in destination.parts:
                    moved.append(destination)
                    if len(moved) == 2:
                        raise RuntimeError("Simulated interruption")
                return result

            with patch.object(Path, "replace", interrupt), self.assertRaisesRegex(RuntimeError, "interruption"):
                runner.repair_cdr(SimpleNamespace(run_root=root))
            with self.assertRaisesRegex(ValueError, "repair incomplete"):
                runner.validate_run_code(root, config)
            with redirect_stdout(StringIO()):
                runner.repair_cdr(SimpleNamespace(run_root=root))
            runner.validate_run_code(root, config)
            self.assertEqual(json.loads((root / "shared/amd/missing.json").read_text())["evidence"]["cdr"], "Not Available")

    def test_cdr_repair_rejects_other_changes_and_active_run(self):
        import fcntl

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self.cdr_fixture(root)
            config["source_code"]["OphthalmicAgent/data/loader.py"] = "unknown-code"
            config["fingerprint"] = runner.digest({k: v for k, v in config.items() if k != "fingerprint"})
            runner.write_json(root / "config.json", config)
            with self.assertRaisesRegex(ValueError, "Not a recognized CDR-only"):
                runner.repair_cdr(SimpleNamespace(run_root=root))
            self.assertFalse((root / runner.CDR_REPAIR).exists())
            with (root / "run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "while this run is active"):
                    runner.repair_cdr(SimpleNamespace(run_root=root))

    def test_supported_dataset_layouts_preserve_task_and_split(self):
        row = dict(task="glaucoma", filename="data/Glaucoma/Test/data_07001.npz")
        for relative in ("data/Glaucoma/Test/data_07001.npz", "Glaucoma/Test/data_07001.npz",
                         "Test/Glaucoma/data_07001.npz", "Test/data_07001.npz",
                         "HarvardFairVision30k/Glaucoma/Test/data_07001.npz"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                expected = root / relative
                expected.parent.mkdir(parents=True)
                expected.touch()
                self.assertEqual(runner.resolve_image_path(root, row), expected.resolve())
                self.assertEqual(live.image_path({"data_root": str(root)}, row), expected.resolve())
                self.assertEqual(len(runner.require_images(root, [row])), 1)
                validation = dict(task="glaucoma", split="val", filename="data/Glaucoma/Validation/data_07001.npz")
                with self.assertRaisesRegex(ValueError, "0/1 resolved"):
                    runner.require_images(root, [validation])

    def test_path_resolver_does_not_pick_a_different_disease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "AMD/Test/data_07001.npz"
            other.parent.mkdir(parents=True)
            other.touch()
            row = dict(task="glaucoma", filename="data/Glaucoma/Test/data_07001.npz")
            with self.assertRaisesRegex(ValueError, "0/1 resolved"):
                runner.require_images(root, [row])
            with self.assertRaisesRegex(ValueError, "Task/path mismatch"):
                runner.resolve_image_path(root, {**row, "task": "amd"})
            with self.assertRaisesRegex(ValueError, "Split/path mismatch"):
                runner.resolve_image_path(root, {**row, "split": "val"})

    def test_path_resolver_accepts_alias_but_rejects_ambiguous_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "Glaucoma/Test/a.npz"
            first.parent.mkdir(parents=True)
            first.touch()
            alias = root / "Test/a.npz"
            alias.parent.mkdir(parents=True)
            alias.symlink_to(first)
            row = dict(task="glaucoma", filename="data/Glaucoma/Test/a.npz")
            self.assertEqual(runner.resolve_image_path(root, row), first.resolve())
            alias.unlink()
            alias.touch()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                runner.resolve_image_path(root, row)

    def test_strict_labels_probabilities_and_cohort(self):
        for value in (-1, "", "nan", .3, 2):
            with self.assertRaises(ValueError):
                runner.binary(value)
        for value in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runner.probability(value)
        cases = [dict(case_id="a", truth=0, race="white", sex_gender="male", age_group="older")]
        with self.assertRaisesRegex(ValueError, "Partial"):
            runner.score_rows(cases, [])
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            runner.score_rows(cases, [dict(case_id="b", prediction=0)])
        m, _ = runner.score_rows(cases, [dict(case_id="a", prediction=0)])
        self.assertEqual(m["f1_macro"], .5)
        self.assertEqual(m["f1_weighted"], 1.)

    def test_validation_only_threshold_and_prior(self):
        rows = [dict(split="val", y_true=0, y_prob=.1), dict(split="val", y_true=1, y_prob=.3)]
        self.assertEqual(runner.select_threshold(rows), .3)
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            runner.select_threshold(rows)
        with self.assertRaisesRegex(ValueError, "validation"):
            runner.fit_reliability(rows, "amd", "retfound_oct")

    def test_validation_support_is_not_test_covariate_support(self):
        rows = [dict(task="amd", split="val", y_true=i % 2, y_prob=.2 if i % 2 == 0 else .8,
                     age_group="older", race="white", sex_gender="female") for i in range(10)]
        fitted = runner.fit_reliability(rows, "amd", "retfound_oct")
        self.assertEqual(fitted[1]["n_total"]["amd"], 10)
        original = runner.trust_for(rows[0], "amd", "retfound_oct", fitted)
        changed_test_label = {**rows[0], "y_true": 1, "split": "test"}
        self.assertEqual(runner.trust_for(changed_test_label, "amd", "retfound_oct", fitted), original)
        self.assertGreaterEqual(original, 0)
        self.assertLessEqual(original, 1)

    def test_final_parser_has_no_negative_fallback(self):
        self.assertEqual(live.final_label("[LABELS]\nAMD_DETECTED: 1\n[/LABELS]\nReasoning", "amd"), 1)
        for raw in ("not parseable", "[LABELS]AMD_DETECTED: -1[/LABELS]",
                    "[LABELS]DR_DETECTED: 1[/LABELS]", "[LABELS]AMD_DETECTED: 1 0[/LABELS]"):
            with self.assertRaises(ValueError):
                live.final_label(raw, "amd")

    def test_counterfactual_contract_and_task(self):
        live.validate_trace(json.dumps(trace()))
        invalid = trace()
        invalid["scenarios"][1]["name"] = "full_evidence"
        with self.assertRaises(ValueError):
            live.validate_trace(json.dumps(invalid))
        e = dict(narrative="A patient", probability_percent=78, oct_report="OCT report", slo_report="SLO report", cdr=.4)
        full = live.counterfactual_messages("amd", e, .712345)
        removed = live.counterfactual_messages("amd", e, None)
        self.assertIn("age-related macular degeneration", full[0]["content"])
        self.assertNotIn("glaucoma", json.dumps(full))
        self.assertIn("0.712345", json.dumps(full))
        self.assertNotIn("0.712345", json.dumps(removed))
        payload = json.loads(removed[1]["content"].split("EVIDENCE_JSON:\n")[1])
        self.assertNotIn("demographic_reliability_trust_score", payload)
        self.assertFalse({"truth", "y_true", "Ground_Truth"} & payload.keys())

    def test_structured_contracts_cover_all_scenarios_and_binary_labels(self):
        schema = live.response_format("counterfactual")["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(set(schema["schema"]["properties"]["scenarios"]["required"]), set(live.SCENARIOS))
        value = dict(scenarios={r["name"]: {k: v for k, v in r.items() if k != "name"} for r in trace()["scenarios"]},
                     interpretation="Synthetic")
        self.assertEqual(len(live.validate_trace(json.dumps(value))["scenarios"]), 5)
        self.assertEqual(live.final_label(json.dumps(dict(diagnosis=1, reasoning="Synthetic", overview="Synthetic")), "dr"), 1)
        with self.assertRaises(ValueError):
            live.final_label(json.dumps(dict(diagnosis=-1, reasoning="Synthetic", overview="Synthetic")), "dr")

    def test_retry_then_cache_valid_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(["unparseable", "[LABELS]AMD_DETECTED: 1[/LABELS]"])
            adapter = live.CachedClient(client, Path(tmp), dict(deployment="test", fingerprint="run"),
                                       "amd", "orchestrator", lambda raw: live.final_label(raw, "amd"), True)
            messages = [dict(role="system", content="Test"), dict(role="user", content="Reports\n- **Trust Score**: None\nAudit")]
            with patch.object(live.time, "sleep"):
                adapter.create(messages=messages)
                adapter.create(messages=messages)
            self.assertEqual(len(client.requests), 2)
            self.assertNotIn("**Trust Score**", client.requests[0]["messages"][1]["content"])
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)
            self.assertEqual(len(list((Path(tmp) / "failed").glob("*.json"))), 1)

    def test_truncation_is_not_cached_as_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(["[LABELS]AMD_DETECTED: 0[/LABELS]"] * 3, finish="length")
            adapter = live.CachedClient(client, Path(tmp), dict(deployment="test", fingerprint="run"), "amd", "orchestrator")
            with patch.object(live.time, "sleep"), self.assertRaises(RuntimeError):
                adapter.create(messages=[])
            self.assertFalse(list(Path(tmp).glob("*.json")))

    def test_two_arms_share_evidence_but_not_trust_or_counterfactual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(fingerprint="run", deployment="test")
            case = dict(task="amd", case_id="a", truth=1)
            evidence = dict(narrative="A patient", probability_percent=78., oct_report="OCT", slo_report="SLO", cdr=.4)
            client = FakeClient([json.dumps(trace()), "[LABELS]AMD_DETECTED: 1[/LABELS]"] * 2)
            live.run_pair(root, config, case, evidence, .712345, FakeOrchestrator(), client, "receipt")
            live.run_pair(root, config, case, evidence, .712345, FakeOrchestrator(), client, "receipt")
            self.assertEqual(len(client.requests), 4)
            records = [json.loads(p.read_text()) for p in (root / "agent/amd").glob("*/*.json")]
            self.assertEqual(len({r["shared_evidence_sha256"] for r in records}), 1)
            full_requests = list((root / "api/amd/a/retinagent_full").glob("*.json"))
            absent_requests = list((root / "api/amd/a/agents_without_reliability").glob("*.json"))
            self.assertEqual(len(full_requests), 2)
            self.assertEqual(len(absent_requests), 2)
            for path in full_requests:
                self.assertIn("0.712345", json.dumps(json.loads(path.read_text())["request"]))
            for path in absent_requests:
                text = json.dumps(json.loads(path.read_text())["request"])
                self.assertNotIn("0.712345", text)
                self.assertNotIn("truth", text)
                self.assertNotIn("y_true", text)

    def test_collector_waits_for_both_complete_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [dict(case_id=f"c{i}", task=t, truth=i % 2, race="white", sex_gender="female", age_group="older")
                     for t in runner.TASKS for i in range(250)]
            offline = [dict(case_id=c["case_id"], task=c["task"], variant=v, prediction=c["truth"])
                       for c in cases for v in runner.VARIANTS[:2]]
            config = dict(prepared_sha256=runner.digest(cases), offline_sha256=runner.digest(offline))
            config["fingerprint"] = runner.digest(config)
            for name, value in (("config", config), ("prepared_cases", cases), ("offline_predictions", offline)):
                runner.write_json(root / (name + ".json"), value)
            for c in cases:
                for v in runner.VARIANTS[2:]:
                    if v == "agents_without_reliability" and c["case_id"] == "c249":
                        continue
                    runner.write_json(root / "agent" / c["task"] / v / (c["case_id"] + ".json"),
                                      dict(task=c["task"], case_id=c["case_id"], variant=v, prediction=c["truth"],
                                           fingerprint=config["fingerprint"], shared_evidence_sha256="same"))
            with redirect_stdout(StringIO()):
                runner.collect(root)
            rows = runner.read_csv(root / "results.csv")
            self.assertTrue(all(r["status"] == "incomplete" for r in rows if r["variant"] in runner.VARIANTS[2:]))
            for c in cases:
                v = "agents_without_reliability"
                runner.write_json(root / "agent" / c["task"] / v / (c["case_id"] + ".json"),
                                  dict(task=c["task"], case_id=c["case_id"], variant=v, prediction=c["truth"],
                                       fingerprint=config["fingerprint"], shared_evidence_sha256="same"))
            with redirect_stdout(StringIO()):
                runner.collect(root)
            self.assertNotIn("PENDING", (root / "table4.md").read_text())
            self.assertIn("1.0000 | 1.0000 | 1.0000 | 1.0000", (root / "table4.md").read_text())

    def test_configure_preserves_inputs_and_refuses_started_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, offline, validation = [], [], []
            config = dict(prepared_sha256=runner.digest(cases), offline_sha256=runner.digest(offline),
                          validation_sha256=runner.digest(validation), source_code={})
            config["fingerprint"] = runner.digest(config)
            for name, value in (("config", config), ("prepared_cases", cases),
                                ("offline_predictions", offline), ("validation_cases", validation)):
                runner.write_json(root / (name + ".json"), value)
            args = SimpleNamespace(run_root=root, data_root=Path("/cluster/data"), oct_weights=Path("/cluster/model.pth"))
            with redirect_stdout(StringIO()):
                runner.configure(args)
            updated, _, _ = runner.load_prepared(root)
            self.assertEqual(updated["data_root"], "/cluster/data")
            self.assertEqual(updated["prepared_sha256"], config["prepared_sha256"])
            self.assertNotEqual(updated["fingerprint"], config["fingerprint"])
            runner.write_json(root / "live_receipt.json", {})
            with self.assertRaisesRegex(ValueError, "after inference"):
                runner.configure(args)

    def test_path_upgrade_is_limited_to_original_code_and_unused_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [dict(task="glaucoma", filename="data/Glaucoma/Test/a.npz")]
            image = root / "dataset/Test/a.npz"
            image.parent.mkdir(parents=True)
            image.touch()
            config = dict(prepared_sha256=runner.digest(cases), offline_sha256=runner.digest([]),
                          validation_sha256=runner.digest([]),
                          source_code={str(p.relative_to(runner.ROOT)): runner.sha(p) for p in runner.code_paths()})
            config["source_code"].update(runner.LEGACY_PATH_CODE)
            config["fingerprint"] = runner.digest(config)
            for name, value in (("config", config), ("prepared_cases", cases),
                                ("offline_predictions", []), ("validation_cases", [])):
                runner.write_json(root / (name + ".json"), value)
            args = SimpleNamespace(run_root=root, data_root=root / "dataset", oct_weights=root / "model.pth",
                                   upgrade_path_layout=False)
            with self.assertRaisesRegex(ValueError, "--upgrade-path-layout"):
                runner.configure(args)
            args.upgrade_path_layout = True
            bad = {**config, "source_code": {**config["source_code"], "OphthalmicAgent/data/loader.py": "unknown-code"}}
            bad["fingerprint"] = runner.digest({k: v for k, v in bad.items() if k != "fingerprint"})
            runner.write_json(root / "config.json", bad)
            with self.assertRaisesRegex(ValueError, "Not a recognized"):
                runner.configure(args)
            runner.write_json(root / "config.json", config)
            args.data_root = root / "wrong-layout"
            with self.assertRaisesRegex(ValueError, "0/1 resolved"):
                runner.configure(args)
            self.assertEqual(json.loads((root / "config.json").read_text()), config)
            self.assertFalse((root / "config_before_path_layout_update.json").exists())
            args.data_root = root / "dataset"
            with redirect_stdout(StringIO()):
                runner.configure(args)
            upgraded, saved_cases, _ = runner.load_prepared(root)
            self.assertEqual(saved_cases, cases)
            self.assertEqual(upgraded["prepared_sha256"], config["prepared_sha256"])
            self.assertEqual(upgraded["offline_sha256"], config["offline_sha256"])
            self.assertEqual(upgraded["path_layout_version"], "fairvision_paths_v1")
            self.assertEqual(json.loads((root / "config_before_path_layout_update.json").read_text()), config)
            (root / "api").mkdir()
            with self.assertRaisesRegex(ValueError, "after inference"):
                runner.configure(args)


if __name__ == "__main__":
    unittest.main()

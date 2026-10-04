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
                    dict(role="user", content=f"{state}\n- **Trust Score**: {trust}\n{cf}")]
        response = self.model_client.chat.completions.create(messages=messages)
        return {"decision": response.choices[0].message.content}


class AblationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

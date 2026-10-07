"""Offline regression tests. Clients below are synthetic and never use a network."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_glaucoma_v2_replay as replay


def final(label=1):
    return json.dumps(dict(diagnosis=label, reasoning="Synthetic evidence only", overview="Synthetic overview",
                           escalation_required=True, escalation_reason="Unverified measurement"))


def trace(label=0, cdr=1.):
    return dict(fingerprint="synthetic-evidence", evidence=dict(patient_narrative="Synthetic narrative",
        retfound_glaucoma_probability_percent=20., oct_specialist_report="Not assessable",
        slo_specialist_report="Synthetic report", vertical_cup_to_disc_ratio=cdr,
        demographic_reliability_trust_score=.77), full_evidence_diagnosis=label,
        scenarios=[dict(name="full_evidence", diagnosis=label, reasoning="Synthetic rationale")],
        label_flip_scenarios=[], evidence_sensitive=False, interpretation="Synthetic interpretation")


def bundle():
    cases = [dict(case_id=f"data/Glaucoma/Test/{i}.npz", truth=t) for i, t in enumerate((0, 1, 1, 0))]
    predictions = dict(RetinAgent=[1, 0, 1, 0], RETFound=[0, 0, 0, 0])
    traces = {cases[i]["case_id"]: [dict(source_line=2, record=trace(1)), dict(source_line=1, record=trace(0))]
              for i in (0, 2, 3)}
    return replay.make_bundle(cases, predictions, traces,
        "Original reasoning.\nOutput EXACTLY in the following format:\n[LABELS]\nGLAUCOMA_DETECTED: [0 or 1]\n[/LABELS]",
        {"synthetic_source": "test"})


class FakeResponse:
    def __init__(self, raw, model="fake-model", finish="stop"):
        self.raw, self.model, self.finish = raw, model, finish
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason=finish)]

    def model_dump(self, mode):
        return dict(model=self.model, id="fake-response", usage=dict(total_tokens=1),
                    choices=[dict(finish_reason=self.finish, message=dict(content=self.raw))])


class FakeClient:
    def __init__(self, bad=False, drift=False, error=False):
        self.requests, self.bad, self.drift, self.error = [], bad, drift, error
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise TimeoutError("synthetic timeout")
        raw = final() if "response_format" in request else "[LABELS]\nGLAUCOMA_DETECTED: 1\n[/LABELS]\nReasoning: synthetic"
        if self.bad:
            raw = "Malformed; never infer a negative"
        model = "changed-model" if self.drift and len(self.requests) > 1 else "fake-model"
        return FakeResponse(raw, model)


class ReplayTests(unittest.TestCase):
    def run_fake(self, b, root, client, limit=None):
        with redirect_stdout(StringIO()):
            replay.execute(b, root, lambda _: client, limit)

    def test_selection_retains_missing_and_first_trace_not_best_label(self):
        b = bundle()
        self.assertEqual(len(b["evaluation"]), 3)
        self.assertEqual(len(b["cases"]), 2)
        self.assertEqual(b["api_budget"], 4)
        self.assertEqual(sum(not c["eligible"] for c in b["evaluation"]), 1)
        for c in b["cases"]:
            self.assertEqual(c["source_line"], 1)
            self.assertEqual(c["trace"]["full_evidence_diagnosis"], 0)

    def test_requests_are_label_blind_and_evidence_is_shared(self):
        b = bundle()
        case = b["cases"][0]
        original = copy.deepcopy(case)
        case.update(truth="SECRET_LABEL", historical_prediction="SECRET_PREDICTION", group="SECRET_GROUP")
        requests = replay.requests_for(case, b["settings"], b["systems"])
        self.assertEqual(requests, original["requests"])
        for request in requests.values():
            text = json.dumps(request)
            self.assertNotIn("SECRET", text)
            self.assertNotIn(case["case_id"], text)
            self.assertEqual(request["max_completion_tokens"], 2000)
        legacy = requests[replay.ARMS[0]]["messages"][1]["content"]
        self.assertTrue(requests["v2"]["messages"][1]["content"].startswith(legacy))
        self.assertNotIn("Output EXACTLY", b["systems"]["v2"])
        self.assertIn("mask", b["systems"]["v2"].lower())

    def test_cdr_checks_do_not_relabel_or_clamp(self):
        for raw, status in ((None, "missing"), ("Not Available", "missing"), (1., "boundary_unverified"),
                            (0., "boundary_unverified"), (.63, "unverified"), (1.5, "invalid")):
            e = trace(cdr=raw)["evidence"]
            self.assertEqual(replay.measurement_checks(e)["cdr_status"], status)
            self.assertEqual(e["vertical_cup_to_disc_ratio"], raw)

    def test_strict_parsers_do_not_default_negative(self):
        for raw in ("Reasoning only", "[LABELS]GLAUCOMA_DETECTED: -1[/LABELS]",
                    "[LABELS]GLAUCOMA_DETECTED: 10[/LABELS]", "[LABELS]GLAUCOMA_DETECTED: 1 or 0[/LABELS]",
                    "[LABELS]GLAUCOMA_DETECTED: 1[/LABELS]" * 2):
            with self.assertRaises(ValueError):
                replay.parse_response(replay.ARMS[0], raw)
        self.assertEqual(replay.parse_response("v2", final(0))["diagnosis"], 0)
        for field, value in (("diagnosis", True), ("escalation_required", "false"), ("reasoning", "")):
            item = json.loads(final())
            item[field] = value
            with self.assertRaises(ValueError):
                replay.parse_response("v2", json.dumps(item))

    def test_successful_resume_makes_no_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient()
            self.run_fake(b, root, client, 1)
            self.assertEqual(len(client.requests), 2)
            self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 4)
            self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 4)
            with redirect_stdout(StringIO()):
                report = replay.collect(b, root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["groups"]["historical_error"]["expected"], 2)
            self.assertEqual(report["groups"]["historical_error"]["paired_valid"], 1)
            rows = replay.base.read_csv(root / "case_results.csv")
            missing = next(r for r in rows if r["eligible"] == "False")
            self.assertEqual(missing["v2_prediction"], "")
            self.assertEqual(missing["v2_status"], "missing_evidence")
            receipts = [json.loads(r) for r in (root / "api_receipts.jsonl").read_text().splitlines()]
            self.assertEqual(len(receipts), 4)
            self.assertTrue(all(r["raw"] and r["request_json"] and r["response_json"]["usage"] and r["started_utc"] for r in receipts))

    def test_invalid_response_stops_without_retry_or_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient(bad=True)
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 1)
            with redirect_stdout(StringIO()):
                report = replay.collect(b, root)
            self.assertFalse(report["complete"])
            ledger = replay.Ledger(root, b)
            row = ledger.rows()[0]
            ledger.close()
            self.assertEqual(row["status"], "invalid")
            self.assertIsNone(row["parsed_json"])
            self.assertTrue(row["raw"])

    def test_timeout_consumes_budget_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient(error=True)
            with self.assertRaises(RuntimeError):
                self.run_fake(b, root, client)
            ledger = replay.Ledger(root, b)
            row = ledger.rows()[0]
            self.assertEqual(row["status"], "api_error")
            self.assertFalse(ledger.reserve(row["case_id"], row["arm"], json.loads(row["request_json"])))
            ledger.close()
            self.assertEqual(len(client.requests), 1)

    def test_budget_persists_and_cannot_be_raised_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient()
            b["api_budget"] = 1
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 1)
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 1)
            b["api_budget"] = 250
            with self.assertRaises(ValueError):
                replay.Ledger(root, b)

    def test_crash_after_reservation_is_never_automatically_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient()
            ledger = replay.Ledger(root, b)
            case = b["cases"][0]
            self.assertTrue(ledger.reserve(case["case_id"], "v2", case["requests"]["v2"]))
            ledger.close()
            self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 3)
            with redirect_stdout(StringIO()):
                self.assertFalse(replay.collect(b, root)["complete"])

    def test_model_drift_is_saved_and_stops_future_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient(drift=True)
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 2)
            with self.assertRaises(ValueError):
                self.run_fake(b, root, client)
            self.assertEqual(len(client.requests), 2)
            ledger = replay.Ledger(root, b)
            self.assertEqual(ledger.rows()[-1]["status"], "model_drift")
            self.assertTrue(ledger.rows()[-1]["raw"])
            ledger.close()

    def test_lock_rejects_concurrent_execution_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeClient()
            with replay.run_lock(root), self.assertRaises(ValueError):
                self.run_fake(bundle(), root, client)
            self.assertEqual(client.requests, [])

    def test_frozen_bundle_and_changed_requests_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b = Path(tmp), bundle()
            path = root / "bundle.json"
            replay.base.write_json(path, b)
            self.assertEqual(replay.load_bundle(path), b)
            b["cases"][0]["requests"]["v2"]["temperature"] = .9
            replay.base.write_json(path, b)
            with self.assertRaises(ValueError):
                replay.load_bundle(path)

    def test_preflight_and_collect_need_no_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(StringIO()):
                report = replay.collect(bundle(), Path(tmp))
            self.assertEqual(report["attempts_reserved"], 0)
            self.assertFalse((Path(tmp) / "ledger.sqlite3").exists())
            self.assertFalse(report["complete"])

    def test_existing_non_experiment_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "report.txt"
            original.write_text("Historical result, do not replace")
            with self.assertRaises(ValueError):
                replay.collect(bundle(), root)
            self.assertEqual(original.read_text(), "Historical result, do not replace")
            self.assertFalse((root / "ledger.sqlite3").exists())

    def test_sdk_retries_disabled_and_credentials_not_saved(self):
        captured = {}
        def constructor(**kwargs):
            captured.update(kwargs)
            return FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            root, b = Path(tmp), bundle()
            ledger = replay.Ledger(root, b)
            with patch.dict(sys.modules, {"openai": SimpleNamespace(AzureOpenAI=constructor)}), patch.dict(
                    replay.os.environ, {"AZURE_OPENAI_ENDPOINT": "https://synthetic.invalid", "AZURE_OPENAI_API_KEY": "secret-fixture"}):
                replay.azure_factory(b)(ledger)
            self.assertEqual(captured["max_retries"], 0)
            self.assertEqual(captured["timeout"], 180)
            saved = list(ledger.db.execute("SELECT * FROM metadata"))
            self.assertNotIn("secret-fixture", str([tuple(r) for r in saved]))
            ledger.close()

    def test_cli_run_requires_explicit_api_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bundle.json"
            replay.base.write_json(path, bundle())
            result = subprocess.run([sys.executable, str(Path(replay.__file__)), "--stage", "run",
                                     "--bundle", str(path), "--run-root", str(root / "run")],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Paid run requires --allow-api", result.stderr)
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()

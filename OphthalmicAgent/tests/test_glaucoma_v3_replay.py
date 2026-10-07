"""Synthetic offline V3 tests. No patient diagnosis or API calls."""
import copy
from contextlib import redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_glaucoma_v3_replay as v3

np = v3.oct_view.np


def fake_volume(depth=200, height=64, width=96):
    rng = np.random.default_rng(123)
    return rng.integers(0, 255, (depth, height, width), dtype=np.uint8)


def final(label=1):
    return json.dumps(dict(diagnosis=label, reasoning="Synthetic final", overview="Synthetic overview",
                           escalation_required=False, escalation_reason="Synthetic only"))


def counterfactual(label=1):
    return json.dumps(dict(scenarios=[dict(name=s, diagnosis=label, confidence="uncertain", reasoning="New OCT used")
                                     for s in v3.cf.SCENARIOS], interpretation="Synthetic new audit"))


class FakeClient:
    def __init__(self, finish="stop", error=False, drift=False):
        self.requests, self.finish, self.error, self.drift = [], finish, error, drift
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise TimeoutError("Synthetic timeout")
        kind = request.get("response_format", {}).get("type")
        raw = final() if kind == "json_schema" else counterfactual() if kind == "json_object" else "NEW OCT REPORT"
        model = "changed" if self.drift and len(self.requests) > 1 else "fake-model"
        data = dict(model=model, choices=[dict(finish_reason=self.finish, message=dict(content=raw))], usage={})
        return SimpleNamespace(model=model, choices=[SimpleNamespace(finish_reason=self.finish,
            message=SimpleNamespace(content=raw))], model_dump=lambda mode: data)


def make_bundle(root, count=2):
    source = json.loads(v3.replay.DEFAULT_BUNDLE.read_text())
    settings, systems = source["settings"], source["systems"]
    cases, evaluation = [], []
    (root / "images").mkdir()
    for i in range(count):
        case_id = f"data/Glaucoma/Test/data_{i:05}.npz"
        image, meta = v3.oct_view.render(fake_volume())
        payload = v3.oct_view.jpeg_bytes(image)
        path = f"images/data_{i:05}.jpg"
        (root / path).write_bytes(payload)
        cases.append(dict(case_id=case_id, evidence=dict(patient_narrative="Synthetic narrative",
            oct_specialist_report="OLD OCT MUST BE REPLACED", slo_specialist_report="SLO UNCHANGED",
            retfound_glaucoma_probability_percent=23., demographic_reliability_trust_score=.8,
            vertical_cup_to_disc_ratio=.51), image=dict(path=path, metadata=meta,
                sha256=hashlib.sha256(payload).hexdigest(), npz_sha256="synthetic")))
        evaluation.append(dict(case_id=case_id, truth=1, historical_prediction=0, group="historical_error",
                               eligible=True, v2_status="valid", v2_prediction=0))
    evaluation.append(dict(case_id="data/Glaucoma/Test/data_99999.npz", truth=0, historical_prediction=1,
                           group="historical_error", eligible=False, v2_status="missing_evidence", v2_prediction=None))
    b = dict(version=v3.VERSION, scope=v3.SCOPE, cases=cases, evaluation=evaluation, settings=settings,
             systems=systems, api_budget=3 * count, runtime_code_sha256=v3.code_hashes(),
             oct_system=v3.oct_view.SYSTEM_PROMPT)
    for case in cases:
        case["oct_request_sha256"] = v3.base.digest(v3.request_for("oct", case, b, root, {}))
    b["fingerprint"] = v3.base.digest(b)
    v3.base.write_json(root / "bundle.json", b)
    return b


class PresentationTests(unittest.TestCase):
    def test_indices_derive_from_volume_not_legacy_captions(self):
        for depth in (8, 64, 127, 128, 200, 201):
            meta = v3.oct_view.slice_metadata((depth, 200, 200))
            sampled = np.linspace(0, depth - 1, 8, dtype=int).tolist()
            self.assertEqual(meta["central_index"], depth // 2)
            self.assertEqual(meta["classifier_sampled_indices"], sampled)
            self.assertEqual(meta["context_indices"], [sampled[i] for i in (1, 2, 4, 5)])
            self.assertEqual(v3.oct_view.captions(meta)[1], [f"Slice {sampled[i]}" for i in (1, 2, 4, 5)])
        meta = v3.oct_view.slice_metadata((200, 200, 200))
        self.assertEqual(meta["context_indices"], [28, 56, 113, 142])
        self.assertIn("Central slice 100", v3.oct_view.captions(meta)[0])

    def test_montage_keeps_legacy_pixels_and_slot_order(self):
        volume = fake_volume()
        original = volume.copy()
        image, meta = v3.oct_view.render(volume)
        height, width = volume.shape[1:]
        pixels = np.asarray(image)
        self.assertEqual(image.size, (4 * width + 36, 3 * height + 90))
        for slot, index in enumerate(meta["context_indices"]):
            x, y = slot * (width + 12), 70 + 2 * height + 20
            expected = v3.oct_view.enhance(volume[index])
            np.testing.assert_array_equal(pixels[y:y + height, x:x + width], expected)
        expected = v3.oct_view.cv2.resize(v3.oct_view.enhance(volume[100]), (2 * width, 2 * height),
                                         interpolation=v3.oct_view.cv2.INTER_CUBIC)
        x = (image.width - 2 * width) // 2
        np.testing.assert_array_equal(pixels[35:35 + 2 * height, x:x + 2 * width], expected)
        np.testing.assert_array_equal(volume, original)

    def test_prompt_framing_is_not_a_diagnosis_or_verified_metadata(self):
        image, meta = v3.oct_view.render(fake_volume())
        request = v3.oct_view.request(v3.oct_view.jpeg_bytes(image), meta, "gpt-5.1")
        text = request["messages"][0]["content"]
        self.assertIn("expected optic-nerve-head context", text)
        self.assertIn("not\nadjacent slices", text)
        self.assertIn("Do not infer glaucoma from an excavation alone", text)
        self.assertIn("not\nverified acquisition metadata", text)
        self.assertNotIn("classifier was uncertain", text)
        self.assertNotIn("* Foveal contour.", text)
        self.assertNotIn("* Optic disc appearance.", text)
        self.assertEqual(request["max_completion_tokens"], 500)
        self.assertNotIn("temperature", request)
        self.assertFalse(meta["acquisition_metadata_verified"])

    def test_invalid_volume_rejected_not_silently_rescaled(self):
        for volume in (np.zeros((7, 200, 200)), np.zeros((200, 200)),
                       np.full((8, 64, 64), np.nan), np.full((8, 64, 64), 256)):
            with self.assertRaises(ValueError):
                v3.oct_view.render(volume)


class ReplayTests(unittest.TestCase):
    def execute(self, b, root, client, count=2):
        with redirect_stdout(StringIO()):
            v3.execute(b, root, lambda _: client, count)

    def collect(self, b, root):
        with redirect_stdout(StringIO()):
            return v3.collect(b, root)

    def test_new_oct_and_new_trace_reach_downstream_with_other_evidence_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeClient()
            b = make_bundle(root)
            self.execute(b, root, client, 1)
            self.assertEqual(len(client.requests), 3)
            oct_req, cf_req, final_req = client.requests
            self.assertEqual(oct_req["messages"][0]["content"], v3.oct_view.SYSTEM_PROMPT)
            cf_evidence = json.loads(cf_req["messages"][1]["content"].split("EVIDENCE_JSON:\n")[1])
            self.assertEqual(cf_evidence, {**b["cases"][0]["evidence"], "oct_specialist_report": "NEW OCT REPORT"})
            text = final_req["messages"][1]["content"]
            self.assertIn("NEW OCT REPORT", text)
            self.assertIn("Synthetic new audit", text)
            self.assertIn("SLO UNCHANGED", text)
            self.assertIn('"oct_images_reassessed": true', text)
            self.assertIn('"scan_location_verified": false', text)
            self.assertNotIn("OLD OCT MUST BE REPLACED", text)
            self.assertEqual(final_req["messages"][0]["content"], b["systems"]["v2"])
            self.assertEqual(cf_req["temperature"], 0)

    def test_labels_ids_and_error_group_do_not_enter_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            case = b["cases"][0]
            parsed = dict(oct=dict(report="New report"),
                          counterfactual=v3.parse_response("counterfactual", counterfactual(), case["case_id"]))
            requests = [v3.request_for(s, case, b, root, parsed) for s in v3.STAGES]
            case.update(truth="SECRET_LABEL", historical_prediction="SECRET_OLD", group="SECRET_GROUP")
            for s, before in zip(v3.STAGES, requests):
                after = v3.request_for(s, case, b, root, parsed)
                self.assertEqual(before, after)
                self.assertNotIn("SECRET", json.dumps(after))
                self.assertNotIn(case["case_id"], json.dumps(after))

    def test_resume_uses_one_ledger_and_collect_retains_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeClient()
            b = make_bundle(root)
            self.assertEqual(self.collect(b, root)["attempts_reserved"], 0)
            self.execute(b, root, client, 1)
            self.execute(b, root, client)
            self.execute(b, root, client)
            self.assertEqual(len(client.requests), 6)
            report = self.collect(b, root)
            self.assertEqual((report["valid"], report["repaired"], report["unassessed"]), (2, 2, 1))
            self.assertEqual(report["additional_repairs_vs_saved_v2"], 2)
            rows = v3.base.read_csv(root / "run/case_results.csv")
            self.assertEqual(rows[-1]["v3_prediction"], "")
            self.assertEqual(rows[-1]["final_status"], "missing_saved_evidence")

    def test_truncation_timeout_and_drift_stop_dependency_chain_and_resume(self):
        for client, exc, expected in ((FakeClient(finish="length"), ValueError, 1),
                                     (FakeClient(error=True), RuntimeError, 1),
                                     (FakeClient(drift=True), ValueError, 2)):
            with self.subTest(expected=expected, exc=exc), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b = make_bundle(root)
                with self.assertRaises(exc):
                    self.execute(b, root, client)
                with self.assertRaises(ValueError):
                    self.execute(b, root, client)
                self.assertEqual(len(client.requests), expected)
                report = self.collect(b, root)
                self.assertEqual(report["valid"], 0)
                self.assertEqual(report["attempts_reserved"], expected)

    def test_interrupted_reservation_not_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeClient()
            b = make_bundle(root)
            ledger = v3.Ledger(root / "run", b)
            case = b["cases"][0]
            ledger.reserve(case["case_id"], "oct", v3.request_for("oct", case, b, root, {}))
            ledger.close()
            with self.assertRaises(ValueError):
                self.execute(b, root, client)
            self.assertEqual(client.requests, [])

    def test_budget_hard_cap_and_persistent_limit(self):
        source = json.loads(v3.replay.DEFAULT_BUNDLE.read_text())
        cases, rows = v3.selected_errors(source)
        self.assertEqual((len(cases), len(rows), 3 * len(cases)), (49, 57, 147))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            ledger = v3.Ledger(root / "run", b)
            for i in range(b["api_budget"]):
                self.assertTrue(ledger.reserve(str(i), "oct", {}))
            with self.assertRaises(ValueError):
                ledger.reserve("extra", "oct", {})
            ledger.close()
            with self.assertRaises(ValueError):
                v3.Ledger(root / "run", {**b, "api_budget": 250})
            b["api_budget"] = 150
            b["fingerprint"] = v3.base.digest({k: v for k, v in b.items() if k != "fingerprint"})
            v3.base.write_json(root / "bundle.json", b)
            with self.assertRaises(ValueError):
                v3.load_bundle(root)

    def test_changed_image_or_code_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            self.assertEqual(v3.load_bundle(root), b)
            with patch.object(v3, "code_hashes", return_value={}):
                with self.assertRaises(ValueError):
                    v3.load_bundle(root)
            (root / b["cases"][0]["image"]["path"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                v3.load_bundle(root)

    def test_dependency_receipt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeClient()
            b = make_bundle(root)
            self.execute(b, root, client, 1)
            ledger = v3.Ledger(root / "run", b)
            with ledger.db:
                ledger.db.execute("UPDATE attempts SET request_json='{}' WHERE arm='counterfactual'")
            ledger.close()
            with self.assertRaises(ValueError):
                self.collect(b, root)

    def test_counterfactual_parser_rejects_duplicate_and_boolean_labels(self):
        raw = json.loads(counterfactual())
        raw["scenarios"][0]["diagnosis"] = True
        with self.assertRaises(ValueError):
            v3.parse_response("counterfactual", json.dumps(raw), "test")
        raw = json.loads(counterfactual())
        raw["scenarios"][-1] = raw["scenarios"][0]
        with self.assertRaises(ValueError):
            v3.parse_response("counterfactual", json.dumps(raw), "test")

    def test_cli_preflight_is_offline_and_run_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            command = [sys.executable, str(Path(v3.__file__)), "--experiment-dir", str(root)]
            result = subprocess.run(command + ["--stage", "preflight"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No API client created", result.stdout)
            self.assertFalse((root / "run").exists())
            result = subprocess.run(command + ["--stage", "run", "--max-cases", "1"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require --allow-api", result.stderr)
            self.assertFalse((root / "run").exists())

    def test_sources_are_task_qualified_no_cross_task_fallback(self):
        for path in ("data/AMD/Test/data_07001.npz", "data/Glaucoma/Test/../data_07001.npz", "data_07001.npz"):
            with self.assertRaises(ValueError):
                v3.filename(path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("Datasets/FairVision/AMD/Test/data_07001.npz", "../data_07001.npz"):
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    entry = tarfile.TarInfo(name)
                    entry.size = 3
                    tar.addfile(entry, BytesIO(b"bad"))
                args = SimpleNamespace(archive=archive)
                with self.assertRaises(ValueError):
                    list(v3.source_payloads(args, ["data/Glaucoma/Test/data_07001.npz"]))


if __name__ == "__main__":
    unittest.main()

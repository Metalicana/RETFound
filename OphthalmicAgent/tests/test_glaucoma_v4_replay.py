"""Offline tests: mechanical evidence retention and bounded execution, not medical accuracy."""
import ast
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_glaucoma_v4_replay as v4


def make_bundle(root):
    cases = []
    evaluation = []
    for i in range(2):
        case_id = f"data/Glaucoma/Test/data_{i:05d}.npz"
        evidence = dict(patient_narrative="Synthetic adult", retfound_glaucoma_probability_percent=45,
                        oct_specialist_report="Excavation is visible.\nPresence or degree cannot be determined.",
                        slo_specialist_report="A rim is visible.\nPossible thinning; not assessable reliably.",
                        vertical_cup_to_disc_ratio=.5, demographic_reliability_trust_score=.8)
        cases.append(dict(case_id=case_id, evidence=evidence, oct_acquisition=dict(acquisition_metadata_verified=False)))
        evaluation.append(dict(case_id=case_id, truth=i, historical_prediction=1-i, v3_prediction=1-i,
                               eligible=True, v4_selected=True))
    evaluation.append(dict(case_id="missing", truth=1, historical_prediction=0, v3_prediction=None,
                           eligible=False, v4_selected=False))
    b = dict(version=v4.VERSION, scope=v4.SCOPE, api_budget=4, cases=cases, evaluation=evaluation,
             settings=dict(deployment="gpt-5.1", temperature=.3, api_version="2024-12-01-preview", timeout_seconds=180),
             source_snapshot=dict(v2_attempts=100, v3_attempts=7,
                                  v3_metadata=dict(returned_model="synthetic-model", endpoint_sha256="endpoint")),
             source_paths=dict(v2_root=str(root / "v2"), v3_dir=str(root / "v3"), v2_bundle="unused"),
             runtime_code_sha256=v4.code_hashes())
    for case in cases:
        case["review_request_sha256"] = v4.base.digest(v4.request_for("review", case, b, {}))
    b["fingerprint"] = v4.base.digest(b)
    v4.base.write_json(root / "bundle.json", b)
    return b


def response_value(data, final=False, diagnosis=1):
    sources = {}
    for name, units in data["sources"].items():
        limits = data["protected_limitation_ids"][name]
        if final:
            limits = data["qualification_review"]["source_review"][name]["limitation_ids"]
        sources[name] = dict(finding_ids=[units[0]["id"]], limitation_ids=list(limits),
                             interpretation="The observed finding does not resolve the stated uncertainty.")
    value = dict(source_review=sources)
    if final:
        value.update(cdr_use="context_only_unverified", dependence_note="Each pair shares an image source.",
                     diagnosis=diagnosis, reasoning="A forced choice, not a confirmed finding.",
                     overview="Limited evidence.", escalation_required=True, escalation_reason="Uncertain evidence.")
    return value


class FakeClient:
    def __init__(self, mutate=None, fail_at=None):
        self.chat = SimpleNamespace(completions=self)
        self.requests = []
        self.mutate = mutate
        self.fail_at = fail_at
        self.closed = False

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        n = len(self.requests)
        if n == self.fail_at:
            raise TimeoutError("synthetic failure")
        data = json.loads(request["messages"][1]["content"])
        final = "qualification_review" in data
        raw = json.dumps(response_value(data, final))
        envelope = dict(model="synthetic-model", choices=[dict(finish_reason="stop",
                        message=dict(content=raw, refusal=None, tool_calls=None))])
        if self.mutate:
            self.mutate(envelope, n)
        choice = envelope["choices"][0]
        return SimpleNamespace(model=envelope["model"], choices=[SimpleNamespace(
            finish_reason=choice["finish_reason"], message=SimpleNamespace(**choice["message"]))],
            model_dump=lambda **_: copy.deepcopy(envelope))

    def close(self):
        self.closed = True


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = make_bundle(self.root)
        self.case = self.bundle["cases"][0]
        self.data = v4.payload(self.case)
        self.review = response_value(self.data)

    def test_preserves_presence_uncertainty_and_possible_positive_findings(self):
        parsed = v4.parse("review", json.dumps(self.review), self.case, {})
        self.assertEqual(parsed, self.review)
        for source in ("oct", "slo"):
            broken = copy.deepcopy(self.review)
            broken["source_review"][source]["limitation_ids"] = []
            with self.assertRaisesRegex(ValueError, "dropped"):
                v4.parse("review", json.dumps(broken), self.case, {})

    def test_unknown_cross_source_and_duplicate_citations_rejected(self):
        for refs in (["oct:999"], ["slo:000"], ["oct:000", "oct:000"]):
            value = copy.deepcopy(self.review)
            value["source_review"]["oct"]["finding_ids"] = refs
            with self.assertRaises(ValueError):
                v4.parse("review", json.dumps(value), self.case, {})

    def test_final_must_retain_extra_review_limitations_not_just_lexical_ones(self):
        self.review["source_review"]["oct"]["limitation_ids"].append("oct:000")
        self.data["qualification_review"] = self.review
        final = response_value(self.data, True)
        final["source_review"]["oct"]["limitation_ids"].remove("oct:000")
        with self.assertRaisesRegex(ValueError, "dropped"):
            v4.parse("final", json.dumps(final), self.case, dict(review=self.review))

    def test_both_diagnostic_directions_allowed_and_unverified_cdr_not_certified(self):
        self.data["qualification_review"] = self.review
        for diagnosis in (0, 1):
            value = response_value(self.data, True, diagnosis)
            self.assertEqual(v4.parse("final", json.dumps(value), self.case, dict(review=self.review))["diagnosis"], diagnosis)
        for diagnosis in (True, -1, "1"):
            value = response_value(self.data, True, diagnosis)
            with self.assertRaises(ValueError):
                v4.parse("final", json.dumps(value), self.case, dict(review=self.review))
        value = response_value(self.data, True)
        value["cdr_use"] = "verified"
        with self.assertRaises(ValueError):
            v4.parse("final", json.dumps(value), self.case, dict(review=self.review))

    def test_missing_cdr_requires_unavailable_not_negative_diagnosis(self):
        self.case["evidence"]["vertical_cup_to_disc_ratio"] = "Not Available"
        self.data["qualification_review"] = self.review
        value = response_value(self.data, True, 1)
        with self.assertRaises(ValueError):
            v4.parse("final", json.dumps(value), self.case, dict(review=self.review))
        value["cdr_use"] = "unavailable"
        self.assertEqual(v4.parse("final", json.dumps(value), self.case, dict(review=self.review))["diagnosis"], 1)

    def test_final_cannot_rewrite_source_interpretation_even_with_valid_citations(self):
        self.data["qualification_review"] = self.review
        value = response_value(self.data, True)
        value["source_review"]["oct"]["interpretation"] = "Disease presence is certain; severity is not."
        with self.assertRaisesRegex(ValueError, "rewrote"):
            v4.parse("final", json.dumps(value), self.case, dict(review=self.review))

    def test_labels_old_decisions_and_counterfactuals_never_enter_requests(self):
        parsed = dict(review=self.review)
        before = [v4.request_for(s, self.case, self.bundle, parsed) for s in v4.STAGES]
        self.case.update(truth="SECRET", historical_prediction="SECRET", trace="SECRET", v3_prediction="SECRET")
        self.bundle["evaluation"][0]["truth"] = "SECRET"
        after = [v4.request_for(s, self.case, self.bundle, parsed) for s in v4.STAGES]
        self.assertEqual(before, after)
        self.assertNotIn("SECRET", json.dumps(after))
        review = json.loads(after[0]["messages"][1]["content"])
        self.assertNotIn("other_evidence", review)
        final = json.loads(after[1]["messages"][1]["content"])
        self.assertEqual(final["other_evidence"]["retfound_glaucoma_probability_percent"], 45)
        self.assertTrue(final["measurement_provenance"]["oct_images_reassessed_in_saved_v3"])
        self.assertNotIn(self.case["case_id"], json.dumps(after))

    def test_schema_fields_match_and_extra_output_not_allowed(self):
        self.review["diagnosis"] = 1
        with self.assertRaises(ValueError):
            v4.parse("review", json.dumps(self.review), self.case, {})
        for stage in v4.STAGES:
            schema = v4.contract.schema(stage)["json_schema"]["schema"]
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            self.assertFalse(schema["additionalProperties"])


class ExecutionTests(unittest.TestCase):
    def execute(self, bundle, root, client, count=2):
        with patch.object(v4, "check_sources"), redirect_stdout(StringIO()):
            v4.execute(bundle, root, lambda _: client, count)

    def collect(self, bundle, root):
        with redirect_stdout(StringIO()):
            return v4.collect(bundle, root)

    def test_resume_four_call_cap_and_all_evaluation_rows_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, client = make_bundle(root), FakeClient()
            self.execute(b, root, client, 1)
            self.execute(b, root, client, 2)
            self.execute(b, root, client, 2)
            self.assertEqual(len(client.requests), 4)
            self.assertTrue(client.closed)
            final = json.loads(client.requests[1]["messages"][1]["content"])
            self.assertEqual(final["qualification_review"], response_value(v4.payload(b["cases"][0])))
            report = self.collect(b, root)
            self.assertEqual((report["valid"], report["repaired"], report["unassessed"]), (2, 1, 1))
            self.assertEqual(report["known_total_attempts"], 111)
            rows = v4.base.read_csv(root / "run/case_results.csv")
            self.assertEqual(rows[-1]["v4_prediction"], "")
            ledger = v4.Ledger(root / "run", b)
            try:
                with self.assertRaisesRegex(ValueError, "budget"):
                    ledger.reserve("extra", "review", {})
            finally:
                ledger.close()

    def test_failure_refusal_truncation_drift_and_lost_qualification_stop(self):
        def lost(e, n):
            value = json.loads(e["choices"][0]["message"]["content"])
            value["source_review"]["oct"]["limitation_ids"] = []
            e["choices"][0]["message"]["content"] = json.dumps(value)
        mutations = [lambda e,n: e["choices"][0].update(finish_reason="length"),
                     lambda e,n: e["choices"][0]["message"].update(refusal="Refused"),
                     lambda e,n: e.update(model="changed"), lost]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b, client = make_bundle(root), FakeClient(mutate=mutation)
                with self.assertRaises(ValueError):
                    self.execute(b, root, client)
                with self.assertRaises(ValueError):
                    self.execute(b, root, client)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(self.collect(b, root)["valid"], 0)

    def test_api_exception_and_interrupted_reservation_cannot_retry(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                b, client = make_bundle(root), FakeClient(fail_at=1)
                if interrupted:
                    ledger = v4.Ledger(root / "run", b)
                    c = b["cases"][0]
                    ledger.reserve(c["case_id"], "review", v4.request_for("review", c, b, {}))
                    ledger.close()
                else:
                    with self.assertRaises(RuntimeError):
                        self.execute(b, root, client)
                with self.assertRaises(ValueError):
                    self.execute(b, root, client)
                self.assertEqual(len(client.requests), 0 if interrupted else 1)

    def test_second_stage_failure_does_not_send_next_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, client = make_bundle(root), FakeClient(fail_at=2)
            with self.assertRaises(RuntimeError):
                self.execute(b, root, client)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(self.collect(b, root)["valid"], 0)

    def test_case_expansion_or_upstream_change_blocks_before_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, client = make_bundle(root), FakeClient()
            with self.assertRaises(ValueError):
                self.execute(b, root, client, 49)
            with patch.object(v4, "check_sources", side_effect=ValueError("Upstream changed")):
                with self.assertRaises(ValueError):
                    v4.execute(b, root, lambda _: client, 2)
            self.assertEqual(client.requests, [])

    def test_request_or_runtime_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b = make_bundle(root)
            self.assertEqual(v4.load_bundle(root), b)
            b["cases"][0]["evidence"]["slo_specialist_report"] = "Changed"
            v4.base.write_json(root / "bundle.json", b)
            with self.assertRaises(ValueError):
                v4.load_bundle(root)

    def test_ledger_metadata_and_dependent_request_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, client = make_bundle(root), FakeClient()
            self.execute(b, root, client, 1)
            rows = v4.read_run(b, root)
            rows[1]["request_json"] = "{}"
            with self.assertRaises(ValueError):
                v4.validate_attempts(b, rows)
            ledger = v4.Ledger(root / "run", b)
            with ledger.db:
                ledger.db.execute("UPDATE metadata SET value='999' WHERE name='budget'")
            ledger.close()
            with self.assertRaises(ValueError):
                v4.read_run(b, root)


class CDRCodeTests(unittest.TestCase):
    def test_existing_formula_includes_cup_but_does_not_validate_mask_quality(self):
        import numpy as np
        from scipy.ndimage import find_objects
        path = ROOT / "OphthalmicAgent/VisionAgent/vision_slo_glaucoma.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "calculate_cdr_from_mask")
        scope = dict(np=np, find_objects=find_objects)
        # Execute only the pure method, not constructors/imports that load models.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        calc = lambda mask: scope["calculate_cdr_from_mask"](None, mask)
        mask = np.zeros((12, 12), dtype=np.uint8)
        self.assertEqual(calc(mask), (-1, -1))
        mask[1:11, 1:11] = 1
        mask[5:7, 5:7] = 2
        self.assertEqual(calc(mask)[0], .2)
        mask[2, 5] = 2  # Disconnected outlier changes bounding box; no QC rejection.
        self.assertEqual(calc(mask)[0], .5)
        mask[1:11, 1:11] = 2
        self.assertEqual(calc(mask), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()

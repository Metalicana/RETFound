"""Downloaded-export review with synthetic clients only."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import audit_glaucoma_v2_replay as review
from test_glaucoma_v2_replay import bundle, FakeClient

replay = review.replay


class ExportReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.bundle = bundle()
        self.bundle_path = self.root / "frozen/bundle.json"
        replay.base.write_json(self.bundle_path, self.bundle)
        with redirect_stdout(StringIO()):
            replay.execute(self.bundle, self.run, lambda _: FakeClient())
            review.errors.collect(self.bundle, self.run)
        self.rows = replay.base.read_csv(self.run / "failed_case_results.csv")
        self.receipts = [json.loads(s) for s in (self.run / "api_receipts.jsonl").read_text().splitlines()]

    def test_offline_outputs_preserve_inputs_and_missing_rows(self):
        paths = [self.bundle_path, self.run / "failed_case_results.csv", self.run / "api_receipts.jsonl"]
        before = [p.read_bytes() for p in paths]
        with patch.object(replay, "azure_factory", side_effect=AssertionError("No API")), \
                patch.object(replay, "Ledger", side_effect=AssertionError("No ledger")):
            result = review.audit(self.bundle_path, self.run, self.root / "audit")
        self.assertEqual(before, [p.read_bytes() for p in paths])
        self.assertEqual(result["historical_errors"], 2)
        self.assertEqual(result["arms"]["v2"]["unassessed"], 1)
        self.assertEqual(result["arms"]["v2"]["still_wrong"], 1)
        self.assertEqual(result["trace_agreement"]["v2"]["matches"], 0)
        self.assertEqual(result["correction_controls"]["v2"]["correct"], 1)
        self.assertEqual(result["v2_review_flags"]["still_wrong"], dict(valid=1, flagged=1))
        exported = replay.base.read_csv(self.root / "audit/cases.csv")
        self.assertEqual(len(exported), 3)
        missing = next(r for r in exported if r["eligible"] == "False")
        self.assertEqual(missing["v2_prediction"], "")

    def test_tampered_csv_prediction_reason_or_truth_rejected(self):
        valid = next(i for i, r in enumerate(self.rows) if r["eligible"] == "True")
        for field, replacement in (("v2_prediction", "0"), ("v2_reasoning", "altered"),
                                   ("truth", "1"), ("v2_escalation_required", "False"),
                                   ("saved_full_evidence_label", "1")):
            rows = copy.deepcopy(self.rows)
            rows[valid][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CSV/receipt/bundle mismatch"):
                review.validate_exports(self.bundle, rows, self.receipts)

    def test_duplicate_receipts_and_missing_csv_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate receipt"):
            review.validate_exports(self.bundle, self.rows, self.receipts + self.receipts[:1])
        with self.assertRaisesRegex(ValueError, "retain all historical errors"):
            review.validate_exports(self.bundle, self.rows[:1], self.receipts)

    def test_request_and_saved_parse_integrity(self):
        for field, replacement, message in (("request_hash", "changed", "Frozen request"),
                                             ("parsed_json", {}, "Saved parse")):
            receipts = copy.deepcopy(self.receipts)
            receipts[0][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                review.validate_exports(self.bundle, self.rows, receipts)

    def test_truncated_response_cannot_count_as_valid(self):
        receipts = copy.deepcopy(self.receipts)
        receipts[0]["response_json"]["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "Invalid response envelope"):
            review.validate_exports(self.bundle, self.rows, receipts)

    def test_invalid_attempt_is_unassessed_not_negative(self):
        receipt = next(r for r in self.receipts if r["arm"] == "v2" and
                       any(x["case_id"] == r["case_id"] for x in self.rows))
        receipt.update(status="invalid", parsed_json=None)
        row = next(r for r in self.rows if r["case_id"] == receipt["case_id"])
        row["v2_status"] = "invalid"
        for field in ("prediction", "reasoning", "escalation_required", "escalation_reason"):
            row["v2_" + field] = ""
        rows, _ = review.validate_exports(self.bundle, self.rows, self.receipts)
        result = review.summarize(self.bundle, rows, self.receipts)
        self.assertEqual(result["arms"]["v2"]["unassessed"], 2)
        self.assertEqual(result["arms"]["v2"]["still_wrong"], 0)
        self.assertEqual(result["paired_valid"], 0)

    def test_reject_output_inside_run_or_frozen_inputs(self):
        for out in (self.run, self.run / "audit", self.bundle_path.parent / "audit", self.root):
            with self.subTest(out=out), self.assertRaisesRegex(ValueError, "separate from frozen inputs"):
                review.audit(self.bundle_path, self.run, out)


if __name__ == "__main__":
    unittest.main()

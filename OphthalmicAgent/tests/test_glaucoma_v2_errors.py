"""Failure-only continuation tests using the existing synthetic replay client."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import run_glaucoma_v2_errors as errors
from test_glaucoma_v2_replay import bundle, FakeClient

replay = errors.replay


class FailureOnlyTests(unittest.TestCase):
    def test_pilot_is_reused_and_correct_cases_are_not_scheduled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient()
            before = replay.base.digest(b)
            cases = errors.error_cases(b)
            # A completed error pair must be reused without scheduling the
            # eligible historical-correction case.
            c = cases[0]
            pilot = {**b, "cases": [c]}
            with redirect_stdout(StringIO()):
                replay.execute(pilot, root, lambda _: client)
                report = errors.collect(b, root)
                errors.execute(b, root, lambda _: client)
                errors.execute(b, root, lambda _: client)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(before, replay.base.digest(b))
            self.assertTrue(report["eligible_error_pairs_complete"])
            self.assertFalse(report["all_historical_errors_assessed_in_both_arms"])
            self.assertEqual(errors.plan(b, root)["remaining_error_requests"], 0)
            self.assertEqual(report["arms"]["v2"]["unassessed"], 1)

    def test_correct_case_pilot_preserved_but_no_new_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient()
            ids = {c["case_id"] for c in errors.error_cases(b)}
            control = next(c for c in b["cases"] if c["case_id"] not in ids)
            with redirect_stdout(StringIO()):
                replay.execute({**b, "cases": [control]}, root, lambda _: client)
                ledger = replay.Ledger(root, b)
                pilot_rows = ledger.rows()
                ledger.close()
                errors.execute(b, root, lambda _: client)
                report = errors.collect(b, root)
            self.assertEqual(len(client.requests), 4)
            ledger = replay.Ledger(root, b)
            preserved = [r for r in ledger.rows() if r["case_id"] == control["case_id"]]
            ledger.close()
            self.assertEqual(pilot_rows, preserved)
            self.assertEqual(report["plan"]["prior_nonerror_requests_retained"], 2)
            self.assertEqual(len(replay.base.read_csv(root / "failed_case_results.csv")), 2)
            self.assertEqual(len(replay.base.read_csv(root / "case_results.csv")), 3)

    def test_invalid_is_unassessed_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b, client = Path(tmp), bundle(), FakeClient(bad=True)
            with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                errors.execute(b, root, lambda _: client)
            client.bad = False
            with redirect_stdout(StringIO()):
                errors.execute(b, root, lambda _: client)
                report = errors.collect(b, root)
            self.assertEqual(len(client.requests), 2)
            self.assertFalse(report["eligible_error_pairs_complete"])
            self.assertEqual(report["paired_valid"], 0)
            self.assertEqual(sum(v["valid"] for v in report["arms"].values()), 1)
            self.assertEqual(sum(v["statuses"].get("invalid", 0) for v in report["arms"].values()), 1)

    def test_preflight_is_read_only_and_run_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b = Path(tmp), bundle()
            p = root / "bundle.json"
            replay.base.write_json(p, b)
            target = root / "run"
            self.assertEqual(errors.plan(b, target)["remaining_error_requests"], 2)
            self.assertFalse(target.exists())
            result = subprocess.run([sys.executable, errors.__file__, "--stage", "run", "--bundle", str(p),
                                     "--run-root", str(target)], capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Paid run requires --allow-api", result.stderr)
            self.assertFalse(target.exists())

    def test_selection_does_not_depend_on_pilot_results(self):
        b = bundle()
        selected = errors.error_cases(b)
        self.assertEqual(len(selected), 1)
        expected = next(r for r in b["evaluation"] if r["group"] == "historical_error" and r["eligible"])
        self.assertEqual(selected[0]["case_id"], expected["case_id"])
        self.assertEqual(selected[0]["requests"], next(c["requests"] for c in b["cases"] if c["case_id"] == expected["case_id"]))

    def test_analysis_counts_repairs_wrong_and_missing_separately(self):
        rows = []
        for i, (truth, legacy, new) in enumerate(((1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1))):
            row = dict(case_id=str(i), truth=truth, group="historical_error", eligible=True)
            for arm, label in zip(replay.ARMS, (legacy, new)):
                row.update({arm + "_prediction": label, arm + "_status": "valid"})
            rows.append(row)
        _, report = errors.analyse(rows)
        self.assertEqual(report["paired_outcomes"], dict(both_repaired=1, v2_only_repaired=1,
                                                        legacy_only_repaired=1, neither_repaired=1))
        self.assertEqual(report["arms"]["v2"]["repair_rate_among_valid"], .5)
        self.assertEqual(report["arms"]["v2"]["error_types"]["historical_false_negatives"]["repaired"], 2)
        self.assertEqual(report["arms"]["v2"]["error_types"]["historical_false_positives"]["still_wrong"], 2)


if __name__ == "__main__":
    unittest.main()

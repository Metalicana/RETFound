import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gap_audit", Path(__file__).resolve().parents[1] / "scripts/audit_manuscript_result_gaps.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ManuscriptAuditTests(unittest.TestCase):
    def test_f1_definitions_are_distinct(self):
        rows = [dict(truth=0, prediction=0)] * 9 + [dict(truth=1, prediction=0)]
        result = audit.metrics(rows)
        self.assertEqual(result["f1_positive"], 0)
        self.assertAlmostEqual(result["f1_macro"], 9 / 19)
        self.assertAlmostEqual(result["f1_weighted"], .9 * 18 / 19)
        self.assertEqual(result["balanced_accuracy"], .5)

    def test_invalid_and_missing_cases_not_filled(self):
        expected = {"a": {"truth": 1}, "b": {"truth": 0}}
        row = dict(id="a", truth=1, prediction=None)
        result, _ = audit.summarize([row], expected)
        self.assertEqual(result["missing_cases"], 1)
        self.assertEqual(result["invalid_prediction"], 1)
        self.assertFalse(result["complete_locked_cohort"])
        self.assertNotIn("f1_macro", result)

    def test_duplicates_and_label_conflicts_block_completion(self):
        row = dict(id="a", truth=1, prediction=1)
        result, _ = audit.summarize([row, row], {"a": {"truth": 0}})
        self.assertFalse(result["complete_locked_cohort"])
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(result["label_conflicts"], 2)
        self.assertNotIn("f1_macro", result)

    def test_progression_is_not_detection(self):
        self.assertIsNone(audit.dataset_task({"task": "progression_forecasting"}, Path("gdp.csv")))
        self.assertEqual(audit.dataset_task({"Task_Folder": "AMD"}, Path("predictions.csv")), ("fairvision", "amd"))


if __name__ == "__main__":
    unittest.main()

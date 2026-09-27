"""Small, dependency-free regression tests for the read-only figure audit."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_recovered_experiment_figures import binary, flag, metrics, paired_rows, percentile, unique_rows


class RecoveredFigureTests(unittest.TestCase):
    def test_f1_definitions_are_not_interchangeable(self):
        pairs = [(0, 0)] * 48 + [(0, 1)] * 18 + [(1, 0)] * 8 + [(1, 1)] * 7
        result = metrics(pairs)
        self.assertEqual(result["n"], 81)
        self.assertAlmostEqual(result["f1"], .35)
        self.assertAlmostEqual(result["weighted_f1"], .7059805707346691)

    def test_invalid_labels_are_not_silently_scored(self):
        for value in (-1, 2, .5, "nan", "True", ""):
            with self.assertRaises(ValueError):
                binary(value)
        with self.assertRaises(ValueError):
            metrics([])

    def test_boolean_acceptance_flags(self):
        self.assertTrue(flag("True"))
        self.assertFalse(flag("False"))
        self.assertTrue(flag("1"))
        with self.assertRaises(ValueError):
            flag("unknown")

    def test_single_class_metrics_are_explicit(self):
        result = metrics([(0, 0)])
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["weighted_f1"], 1)
        self.assertIsNone(result["sensitivity"])
        self.assertIsNone(result["balanced_accuracy"])

    def test_duplicate_baseline_rejected(self):
        with self.assertRaises(ValueError):
            unique_rows([{"case_id": "a"}, {"case_id": "a"}], "case_id")

    def test_patient_grouping_and_last_attempt(self):
        base = [dict(case_id=k, patient_id="p", split="test", y_true="1", y_pred="1")
                for k in ("left", "right")]
        attempts = [dict(case_id=k, split="test", Ground_Truth="1", Pred_GL="0")
                    for k in ("left", "right")]
        attempts.append(dict(attempts[0], Pred_GL="1"))
        groups = paired_rows(base, attempts)
        self.assertEqual(groups, [[(1, 1, 1), (1, 1, 0)]])
        attempts.append(dict(attempts[0], Pred_GL="-1"))
        with self.assertRaises(ValueError):
            paired_rows(base, attempts)

    def test_pairing_rejects_different_labels_or_cohorts(self):
        base = [dict(case_id="a", split="test", y_true="1", y_pred="1")]
        with self.assertRaises(ValueError):
            paired_rows(base, [])
        attempts = [dict(case_id="a", split="test", Ground_Truth="0", Pred_GL="0")]
        with self.assertRaises(ValueError):
            paired_rows(base, attempts)

    def test_percentile_interpolation(self):
        self.assertAlmostEqual(percentile([4, 0, 2], .25), 1)


if __name__ == "__main__":
    unittest.main()

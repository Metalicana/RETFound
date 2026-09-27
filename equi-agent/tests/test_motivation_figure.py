"""Verify that paired error counts cannot be mistaken for net model gains."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_motivation_figure import compare


class PairedErrorTests(unittest.TestCase):
    def test_all_four_outcomes(self):
        base = {str(i): dict(y_true=1, y_pred=p) for i, p in enumerate([1, 0, 1, 0])}
        other = {str(i): dict(y_true=1, y_pred=p) for i, p in enumerate([1, 0, 0, 1])}
        self.assertEqual(compare(base, other), dict(both_correct=1, both_wrong=1,
                         reference_only_correct=1, alternative_only_correct=1))

    def test_identity_has_no_discordant_pairs(self):
        base = {"a": dict(y_true=1, y_pred=0), "b": dict(y_true=0, y_pred=0)}
        counts = compare(base, base)
        self.assertEqual(counts["reference_only_correct"], 0)
        self.assertEqual(counts["alternative_only_correct"], 0)
        self.assertEqual(sum(counts.values()), 2)

    def test_mismatched_cohort_rejected(self):
        with self.assertRaises(ValueError):
            compare({"a": dict(y_true=1, y_pred=1)}, {})

    def test_mismatched_labels_rejected(self):
        with self.assertRaises(ValueError):
            compare({"a": dict(y_true=1, y_pred=1)}, {"a": dict(y_true=0, y_pred=1)})

    def test_invalid_decision_rejected(self):
        with self.assertRaises(ValueError):
            compare({"a": dict(y_true=1, y_pred=1)}, {"a": dict(y_true=1, y_pred=-1)})


if __name__ == "__main__":
    unittest.main()

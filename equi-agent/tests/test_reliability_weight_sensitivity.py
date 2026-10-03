import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("weight_audit", SCRIPTS / "audit_reliability_weight_sensitivity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class WeightSensitivityTests(unittest.TestCase):
    def test_scenarios_are_normalized_and_named(self):
        variants = audit.scenarios()
        self.assertEqual(len(variants), 21)
        self.assertEqual(len({r[0] for r in variants}), len(variants))
        for _, weights, k, _ in variants:
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertGreaterEqual(min(weights.values()), 0)
            self.assertGreaterEqual(k, 0)

    def test_ties_are_not_arbitrarily_broken(self):
        self.assertEqual(audit.winners({"a": .2, "b": .2}), {"a", "b"})

    def test_saved_table_must_match_and_cannot_duplicate(self):
        combo = ("amd", "older", "white", "male")
        row = dict(zip(("task", "age_group", "race", "gender"), combo))
        row.update(model_name="retfound_oct", final_R_bad=.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            def write(rows):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
            write([row])
            self.assertEqual(audit.verify_precomputed({combo: {"retfound_oct": .2}}, path)["rows"], 1)
            with self.assertRaisesRegex(ValueError, "differs"):
                audit.verify_precomputed({combo: {"retfound_oct": .3}}, path)
            write([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit.verify_precomputed({combo: {"retfound_oct": .2}}, path)

    def test_baseline_and_weighted_comparison(self):
        keys = [("amd", "older", "white", "male"), ("amd", "older", "black", "male")]
        baseline = {k: {"retfound_oct": .1, "flair_slo": .2} for k in keys}
        support = {"combo_counts": dict(zip(keys, [9, 1]))}
        unchanged, _ = audit.compare(baseline, baseline, support)
        self.assertEqual(unchanged[0]["max_absolute_trust_change"], 0)
        alternate = {keys[0]: baseline[keys[0]], keys[1]: {"retfound_oct": .3, "flair_slo": .2}}
        changed, _ = audit.compare(baseline, alternate, support)
        self.assertEqual(changed[0]["changed_winner_sets"], 1)
        self.assertEqual(changed[0]["support_weighted_winner_change"], .1)
        self.assertAlmostEqual(changed[0]["retfound_max_absolute_trust_change"], .2)


if __name__ == "__main__":
    unittest.main()

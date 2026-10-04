from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "OphthalmicAgent/scripts"))
import audit_fairvision_glaucoma_traces as audit


def trace(label=0, cdr="Not Available"):
    row = dict(case_id="data/Glaucoma/Test/a.npz", deployment="test", prompt_version="test",
        task="glaucoma", evidence=dict(retfound_glaucoma_probability_percent=78.83,
            demographic_reliability_trust_score=.7, vertical_cup_to_disc_ratio=cdr,
            patient_narrative="Test narrative", oct_specialist_report="Limited image",
            slo_specialist_report="Test visual description"),
        scenarios=[dict(name=name, diagnosis=label, reasoning=f"Test {name}") for name in audit.SCENARIOS],
        full_evidence_diagnosis=label, interpretation="Test interpretation")
    row["fingerprint"] = audit.runner.digest({k: row[k] for k in ("case_id", "deployment", "evidence", "prompt_version")})
    return row


def baseline():
    return {"data/Glaucoma/Test/a.npz": dict(Probability_Positive=".7883", Decision_Threshold=".5")}


class TraceAuditTests(unittest.TestCase):
    def load(self, records, base=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            text = "\n".join(json.dumps(r) for r in records) + "\n"
            path.write_text(text)
            result = audit.load_traces(path, set(baseline()), baseline() if base is None else base)
            self.assertEqual(path.read_text(), text)
            return result

    def test_all_versions_retained_including_conflicting_labels(self):
        records = [trace(0), trace(1, .5)]
        loaded = self.load(records)
        key = records[0]["case_id"]
        cases = [dict(case_id=key, truth=1)]
        rows, packets = audit.build_review(cases, dict(RetinAgent=[0], RETFound=[1]), loaded, baseline())
        self.assertEqual(rows[0]["outcome"], "regression")
        self.assertEqual(rows[0]["trace_versions"], 2)
        self.assertEqual(rows[0]["full_evidence_labels"], "0,1")
        self.assertFalse(rows[0]["all_saved_full_labels_differ_from_final"])
        self.assertEqual([r["record"] for r in packets[0]["saved_traces"]], records)
        report = audit.case_report(packets[0], Path("/tmp/traces.jsonl"))
        self.assertIn("Saved Trace Line 1", report)
        self.assertIn("Saved Trace Line 2", report)
        self.assertIn("not the final orchestrator response", report)

    def test_missing_traces_are_not_dropped_or_scored_as_no_flip(self):
        key = next(iter(baseline()))
        rows, packets = audit.build_review([dict(case_id=key, truth=0)], dict(RetinAgent=[0], RETFound=[1]), {}, baseline())
        self.assertEqual(rows[0]["outcome"], "correction")
        self.assertEqual(rows[0]["trace_records"], 0)
        self.assertIsNone(rows[0]["any_trace_flip_without_visual_interpretation"])
        self.assertIn("No trace available", audit.case_report(packets[0], Path("/tmp/traces.jsonl")))

    def test_correct_counterfactual_and_wrong_final_are_distinguished(self):
        key = next(iter(baseline()))
        loaded = self.load([trace(1)])
        rows, _ = audit.build_review([dict(case_id=key, truth=1)], dict(RetinAgent=[0], RETFound=[1]), loaded, baseline())
        self.assertTrue(rows[0]["all_saved_full_labels_differ_from_final"])

    def test_corrupt_trace_and_probability_mismatch_rejected(self):
        for fault in ("fingerprint", "probability", "scenarios", "label", "full", "task"):
            with self.subTest(fault=fault):
                row = trace()
                if fault == "fingerprint":
                    row["fingerprint"] = "wrong"
                elif fault == "probability":
                    row["evidence"]["retfound_glaucoma_probability_percent"] = 22.
                    row["fingerprint"] = audit.runner.digest({k: row[k] for k in ("case_id", "deployment", "evidence", "prompt_version")})
                elif fault == "scenarios":
                    row["scenarios"].pop()
                elif fault == "label":
                    row["scenarios"][1]["diagnosis"] = 2
                elif fault == "full":
                    row["full_evidence_diagnosis"] = 1
                else:
                    row["task"] = "amd"
                with self.assertRaises(ValueError):
                    self.load([row])

    def test_recovered_rounded_score_is_not_given_fake_precision(self):
        base = baseline()
        row = base[next(iter(base))]
        row.update(Probability_Positive="", Probability_Percent_Rounded="78.83")
        self.assertTrue(self.load([trace()], base))
        row["Probability_Percent_Rounded"] = "12"
        with self.assertRaises(ValueError):
            self.load([trace()], base)

    def test_minus_one_is_an_explicit_hypothetical_flip_not_final_label(self):
        row = trace(0)
        row["scenarios"][3]["diagnosis"] = -1
        loaded = self.load([row])
        key = row["case_id"]
        rows, _ = audit.build_review([dict(case_id=key, truth=1)], dict(RetinAgent=[0], RETFound=[1]), loaded, baseline())
        self.assertTrue(rows[0]["any_trace_flip_without_visual_interpretation"])
        self.assertEqual(rows[0]["agent_prediction"], 0)
        self.assertIn("not independent reruns", audit.summary_report(rows))


if __name__ == "__main__":
    unittest.main()

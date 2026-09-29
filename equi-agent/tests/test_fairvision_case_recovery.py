"""Saved-evidence recovery tests; no model inference or API clients."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recover_fairvision_glaucoma_case import CASE, recover, rounded_score_decision


def fixture():
    manifest, source, agent, baseline, traces = [], [], [], [], []
    for i, (key, truth) in enumerate((("data_07001.npz", 0), ("data_07002.npz", 1), (CASE, 0))):
        path = f"data/Glaucoma/Test/{key}"
        label = "-1" if key == CASE else str(truth)
        meta = {"Age": "65.98", "Gender": "male", "Race": "white"}
        manifest.append({"filename": path, "Task_Folder": "Glaucoma", "Ground_Truth": label, **meta})
        source.append({"filename": key, "glaucoma": "yes" if truth else "no", "use": "test",
                       "age": "65.98", "gender": "male", "race": "white"})
        agent.append({"Filename": path, "Task_Folder": "Glaucoma", "Ground_Truth": label,
                      "Pred_GL": str(truth), "Is_Correct": "-1" if key == CASE else "1", **meta})
        if key != CASE:
            baseline.append({"Row_Index": str(i), "Filename": path, "Model": "RETFound", "Modality": "OCT",
                             "Disease": "GLAUCOMA", "Ground_Truth": label, "Probability_Positive": str(.8 if truth else .2),
                             "Decision_Threshold": "0.5", "Prediction": str(truth), "Is_Correct": "1",
                             "Age_Group": "middle", **meta})
        traces.append({"case_id": path, "task": "glaucoma", "fingerprint": f"trace-{i}",
                       "evidence": {"retfound_glaucoma_probability_percent": 38.75 if key == CASE else 80 if truth else 20}})
    return manifest, source, agent, baseline, traces


class FairVisionCaseRecoveryTests(unittest.TestCase):
    def test_recovery_preserves_inputs_and_does_not_invent_exact_probability(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        manifest, agent, baseline, report = recover(*inputs, expected_cases=3)
        self.assertEqual(inputs, before)
        self.assertEqual(manifest[-1]["Ground_Truth"], "0")
        self.assertEqual(agent[-1]["Pred_GL"], "0")
        self.assertEqual(agent[-1]["Is_Correct"], "1")
        self.assertEqual(baseline[-1]["Prediction"], "0")
        self.assertEqual(baseline[-1]["Probability_Positive"], "")
        self.assertEqual(baseline[-1]["Probability_Percent_Rounded"], "38.75")
        self.assertEqual(report["matching_trace_cases"], 2)
        self.assertEqual(report["before"]["Ours"]["n"], 2)
        self.assertEqual(report["after"]["Ours"]["n"], 3)

    def test_recovery_uses_source_label_even_when_models_are_wrong(self):
        inputs = fixture()
        inputs[1][-1]["glaucoma"] = "yes"
        _, agent, baseline, report = recover(*inputs, expected_cases=3)
        self.assertEqual(agent[-1]["Ground_Truth"], "1")
        self.assertEqual(agent[-1]["Is_Correct"], "0")
        self.assertEqual(baseline[-1]["Is_Correct"], "0")
        self.assertEqual(report["after"]["Ours"]["fn"], 1)

    def test_rounding_must_not_change_thresholded_decision(self):
        self.assertEqual(rounded_score_decision(38.75, .5)[0], 0)
        self.assertEqual(rounded_score_decision(75, .5)[0], 1)
        for percent, threshold in ((50, .5), (38.75, .3875), ("nan", .5), (101, .5)):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                rounded_score_decision(percent, threshold)

    def test_different_model_scores_cannot_fill_a_missing_row(self):
        inputs = fixture()
        inputs[4][0]["evidence"]["retfound_glaucoma_probability_percent"] = 40
        with self.assertRaisesRegex(ValueError, "probabilities differ"):
            recover(*inputs, expected_cases=3)

    def test_conflicting_attempts_are_not_selected_by_correctness(self):
        inputs = fixture()
        extra = copy.deepcopy(inputs[4][-1])
        extra["evidence"]["retfound_glaucoma_probability_percent"] = 70
        inputs[4].append(extra)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            recover(*inputs, expected_cases=3)

    def test_label_and_metadata_conflicts_are_rejected(self):
        for field, value in (("Ground_Truth", "1"), ("Age", "20"), ("Pred_GL", "-1")):
            inputs = fixture()
            inputs[2][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                recover(*inputs, expected_cases=3)

    def test_duplicate_cases_and_missing_extra_cases_are_rejected(self):
        inputs = fixture()
        inputs[2].append(dict(inputs[2][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            recover(*inputs, expected_cases=3)
        inputs = fixture()
        inputs[3].pop()
        with self.assertRaisesRegex(ValueError, "only the specified case"):
            recover(*inputs, expected_cases=3)

    def test_missing_evidence_is_not_replaced_with_assumed_prediction(self):
        inputs = fixture()
        inputs[4].pop()
        with self.assertRaisesRegex(ValueError, "No saved RETFound"):
            recover(*inputs, expected_cases=3)


if __name__ == "__main__":
    unittest.main()

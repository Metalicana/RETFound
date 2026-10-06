from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "equi-agent/scripts"))
import audit_saved_evidence_ablation as audit


def record(task="glaucoma", version=0, full=1, removed=0):
    values = {name: full for name in audit.NAMES}
    values["without_demographic_reliability"] = removed
    scenarios = [dict(name=k, diagnosis=v) for k, v in values.items()]
    changed = [k for k in audit.REMOVED if values[k] != full]
    row = dict(case_id=f"data/{task}/Test/case.npz", task="glaucoma", deployment="fake",
        prompt_version="glaucoma_evidence_ablation_v1", evidence=dict(version=version),
        scenarios=scenarios, full_evidence_diagnosis=full, label_flip_scenarios=changed,
        label_flip_count=len(changed), evidence_sensitive=bool(changed),
        raw_response=json.dumps(dict(scenarios=scenarios)))
    row["fingerprint"] = audit.fingerprint(row)
    return row


def load(rows):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "traces.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return audit.load_traces(path, "test", {task: {"case", "missing"} for task in audit.TASKS})


class EvidenceAblationTests(unittest.TestCase):
    def test_task_comes_from_path_not_shared_glaucoma_field(self):
        rows = load([record(task="amd"), record(task="dr")])
        self.assertEqual([r["task"] for r in rows], ["amd", "dr"])
        self.assertTrue(all(r["recorded_task"] == "glaucoma" for r in rows))
        cases, _, inventory, _ = audit.summarize(rows, {t: {"case", "missing"} for t in audit.TASKS})
        self.assertEqual(len(cases), 8)
        self.assertEqual([r["missing_cases"] for r in inventory], [1, 1])

    def test_versions_do_not_inflate_patient_count_and_inconclusive_is_not_binary_flip(self):
        rows = load([record(removed=1), record(version=1, removed=-1)])
        cases, summary, inventory, _ = audit.summarize(rows, {"glaucoma": {"case", "missing"}})
        demo = next(c for c in cases if c["scenario"] == audit.REMOVED[0])
        self.assertEqual((demo["first_change"], demo["last_change"], demo["any_change"], demo["all_change"]),
                         (False, True, True, False))
        self.assertFalse(demo["any_binary_flip"])
        self.assertTrue(demo["any_inconclusive_transition"])
        self.assertEqual(summary[0]["cases"], 1)
        self.assertEqual(summary[0]["any_change_percent"], 100.)
        self.assertEqual(inventory[0]["cases_with_multiple_versions"], 1)
        self.assertEqual(inventory[0]["missing_cases"], 1)

    def test_exact_duplicates_retained_but_not_counted_as_versions(self):
        r = record()
        rows = load([r, r])
        _, _, inventory, transitions = audit.summarize(rows, {"glaucoma": {"case"}})
        self.assertEqual((inventory[0]["records"], inventory[0]["unique_versions"], inventory[0]["cases"]), (2, 1, 1))
        self.assertTrue(all(r["trace_versions"] == 1 for r in transitions))

    def test_conflicting_response_for_same_fingerprint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting responses"):
            load([record(removed=0), record(removed=1)])

    def test_fingerprint_raw_labels_and_summary_are_checked(self):
        faults = []
        r = record()
        r["evidence"]["version"] = 999
        faults.append(r)
        r = record()
        raw = json.loads(r["raw_response"])
        raw["scenarios"][0]["diagnosis"] = 0
        r["raw_response"] = json.dumps(raw)
        faults.append(r)
        r = record()
        r["label_flip_count"] = 999
        faults.append(r)
        for r in faults:
            with self.subTest(row=r), self.assertRaises(ValueError):
                load([r])

    def test_strict_scenarios_and_recorded_raw_string_extras(self):
        r = record()
        for value in (True, "1", 2, None):
            bad = copy.deepcopy(r["scenarios"])
            bad[0]["diagnosis"] = value
            with self.assertRaises(ValueError):
                audit.labels(bad)
        for bad in (r["scenarios"][:-1], r["scenarios"] + [r["scenarios"][0]]):
            with self.assertRaises(ValueError):
                audit.labels(bad)
        raw = json.loads(r["raw_response"])
        raw["scenarios"].append("extra generated text")
        r["raw_response"] = json.dumps(raw)
        self.assertEqual(load([r])[0]["raw_extra_string_entries"], 1)

    def test_incomplete_paired_run_is_not_given_a_significance_result(self):
        with tempfile.TemporaryDirectory() as temp:
            status = audit.ablation_availability(Path(temp))
        self.assertEqual(list(status["decision_file_counts"].values()), [0, 0])
        self.assertIn("not_available", status["status"])
        self.assertNotIn("p_value", status)
        self.assertIn("--resamples 10000", status["command"])

    @unittest.skipUnless(all(p.exists() for p in audit.SOURCES.values()) and audit.MANIFEST.exists(),
                         "Historical local artifacts unavailable")
    def test_saved_snapshot_counts_and_denominators(self):
        locked = audit.load_locked(audit.MANIFEST)
        rows = [r for source, path in audit.SOURCES.items() for r in audit.load_traces(path, source, locked)]
        _, summaries, inventory, _ = audit.summarize(rows, locked)
        expected = {"glaucoma": [2, 37, 17, 143], "amd": [0, 89, 12, 27], "dr": [3, 72, 20, 43]}
        for task, counts in expected.items():
            lookup = {r["scenario"]: r for r in summaries if r["source"] == "shared_cache" and r["task"] == task}
            self.assertEqual([lookup[s]["any_change_n"] for s in audit.REMOVED], counts)
            self.assertTrue(all(r["cases"] == 250 for r in lookup.values()))
        run = [r for r in summaries if r["source"] == "glaucoma_run_cache"]
        self.assertEqual([r["any_change_n"] for r in run], [0, 12, 5, 78])
        self.assertTrue(all(r["cases"] == 156 for r in run))
        self.assertEqual(sum(r["records"] for r in inventory), 1824)
        self.assertEqual(sum(r["records_with_raw_extra_strings"] for r in inventory), 10)


if __name__ == "__main__":
    unittest.main()

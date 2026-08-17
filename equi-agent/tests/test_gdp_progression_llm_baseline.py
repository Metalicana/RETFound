from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "equi-agent" / "scripts" / "run_gdp_progression_llm_baseline.py"
MULTITARGET_SCRIPT = (
    REPO_ROOT
    / "equi-agent"
    / "scripts"
    / "run_gdp_progression_llm_multitarget_baseline.py"
)
NATIVE_MULTITARGET_SCRIPT = (
    REPO_ROOT / "equi-agent" / "scripts" / "predict_gdp_native_multitarget.py"
)
AGENT_MULTITARGET_SCRIPT = (
    REPO_ROOT
    / "equi-agent"
    / "scripts"
    / "run_equi_agent_gdp_progression_multitarget_live.py"
)
TARGETS = [
    "md",
    "vfi",
    "td_pointwise",
    "md_fast",
    "md_fast_no_p_cut",
    "td_pointwise_no_p_cut",
]
TD_COLUMNS = [
    *[f"td{index}" for index in range(1, 25)],
    *[f"td{index}" for index in range(26, 34)],
    *[f"td{index}" for index in range(35, 55)],
]


class GDPProgressionLLMBaselineTest(unittest.TestCase):
    def test_dry_run_validates_inputs_without_importing_api_clients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rnflt_path = root / "case.npz"
            np.savez(rnflt_path, rnflt=np.arange(25, dtype=np.float32).reshape(5, 5))

            row = {
                "patient_id": "patient-1",
                "eye_id": "",
                "visit_id": "",
                "image_id": "case.npz",
                "dataset": "harvard_gdp",
                "task": "progression_forecasting",
                "split": "test",
                "label_raw": "1",
                "y_true": "1",
                "progression_target": "md",
                "rnflt_path": str(rnflt_path),
                "rnflt_key": "rnflt",
                "race": "white",
                "ethnicity": "non-hispanic",
                "sex_gender": "female",
                "age": "65",
                "age_group": "60-69",
                "metadata_missing_flag": "False",
            }
            row.update({column: str(-index / 10) for index, column in enumerate(TD_COLUMNS)})
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            out_dir = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                    "--model",
                    "gpt-5.1",
                    "--progression-target",
                    "md",
                    "--allow-cohort-mismatch",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            snapshot = json.loads((out_dir / "prompt_snapshot.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["cohort_audit"]["cases"], 1)
            self.assertEqual(summary["cohort_audit"]["td_point_count"], 52)
            self.assertIn("mean-deviation-based progression", snapshot["first_case_user_prompt"])
            self.assertNotIn("progression_md", snapshot["first_case_user_prompt"])
            self.assertNotIn("label_raw", snapshot["first_case_user_prompt"])
            self.assertIn("td54", snapshot["first_case_user_prompt"])

    def test_multitarget_dry_run_validates_all_manifests_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests_root = root / "manifests"
            manifests_root.mkdir()
            rnflt_path = root / "case.npz"
            np.savez(rnflt_path, rnflt=np.arange(25, dtype=np.float32).reshape(5, 5))

            for target_index, target in enumerate(TARGETS):
                row = {
                    "patient_id": "patient-1",
                    "eye_id": "",
                    "visit_id": "",
                    "image_id": "case.npz",
                    "dataset": "harvard_gdp",
                    "task": "progression_forecasting",
                    "split": "test",
                    "label_raw": str(target_index % 2),
                    "y_true": str(target_index % 2),
                    "progression_target": target,
                    "rnflt_path": str(rnflt_path),
                    "rnflt_key": "rnflt",
                    "race": "white",
                    "ethnicity": "non-hispanic",
                    "sex_gender": "female",
                    "age": "65",
                    "age_group": "60-69",
                    "metadata_missing_flag": "False",
                }
                row.update(
                    {column: str(-index / 10) for index, column in enumerate(TD_COLUMNS)}
                )
                manifest_path = manifests_root / f"gdp_progression_forecasting_{target}.csv"
                with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)

            out_dir = root / "multi-out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MULTITARGET_SCRIPT),
                    "--manifests-root",
                    str(manifests_root),
                    "--out-dir",
                    str(out_dir),
                    "--model",
                    "gpt-5.1",
                    "--allow-cohort-mismatch",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            snapshot = json.loads((out_dir / "prompt_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["cohort_audit"]["cases"], 1)
            self.assertEqual(summary["cohort_audit"]["predictions_per_call"], 6)
            self.assertEqual(summary["cohort_audit"]["targets"], TARGETS)
            for target in TARGETS:
                self.assertIn(target, snapshot["first_case_user_prompt"])
            self.assertNotIn("label_raw", snapshot["first_case_user_prompt"])
            self.assertNotIn("y_true", snapshot["first_case_user_prompt"])

    def test_native_multitarget_audit_validates_shared_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests_root = root / "manifests"
            manifests_root.mkdir()
            fieldnames = None
            rows_by_target = {target: [] for target in TARGETS}
            for case_index, split in enumerate(["train", "train", "test", "test"]):
                rnflt_path = root / f"case-{case_index}.npz"
                np.savez(
                    rnflt_path,
                    rnflt=np.arange(225 * 225, dtype=np.float32).reshape(225, 225),
                )
                for target_index, target in enumerate(TARGETS):
                    row = {
                        "patient_id": f"patient-{case_index}",
                        "eye_id": "",
                        "visit_id": "",
                        "image_id": rnflt_path.name,
                        "dataset": "harvard_gdp",
                        "task": "progression_forecasting",
                        "split": split,
                        "y_true": str((case_index + target_index) % 2),
                        "progression_target": target,
                        "rnflt_path": str(rnflt_path),
                        "rnflt_key": "rnflt",
                        "race": "white",
                        "ethnicity": "non-hispanic",
                        "sex_gender": "female",
                        "age": "65",
                        "age_group": "60-69",
                        "metadata_missing_flag": "False",
                    }
                    row.update(
                        {column: str(-index / 10) for index, column in enumerate(TD_COLUMNS)}
                    )
                    fieldnames = list(row)
                    rows_by_target[target].append(row)
            for target, rows in rows_by_target.items():
                path = manifests_root / f"gdp_progression_forecasting_{target}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            summary_path = root / "native-audit.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(NATIVE_MULTITARGET_SCRIPT),
                    "--manifests-root",
                    str(manifests_root),
                    "--summary",
                    str(summary_path),
                    "--expected-train-cases",
                    "2",
                    "--expected-test-cases",
                    "2",
                    "--audit-only",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["cohort_audit"]["cases"], 4)
            self.assertEqual(summary["cohort_audit"]["split_counts"], {"test": 2, "train": 2})

    def test_multitarget_agent_dry_run_writes_six_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions_root = root / "predictions"
            metrics_root = root / "metrics"
            predictions_root.mkdir()
            for target_index, target in enumerate(TARGETS):
                rows = []
                for case_index in range(2):
                    rows.append(
                        {
                            "patient_id": f"patient-{case_index}",
                            "eye_id": "",
                            "visit_id": "",
                            "image_id": f"case-{case_index}.npz",
                            "dataset": "harvard_gdp",
                            "task": "progression_forecasting",
                            "model_name": "rnflt_logreg",
                            "y_true": str((case_index + target_index) % 2),
                            "y_prob": str(0.2 + 0.6 * case_index),
                            "y_pred": str(case_index),
                            "applied_threshold": "0.5",
                            "split": "test",
                            "race": "white",
                            "ethnicity": "non-hispanic",
                            "sex_gender": "female",
                            "age": "65",
                            "age_group": "60-69",
                            "metadata_missing_flag": "False",
                        }
                    )
                prefix = f"gdp_progression_forecasting_{target}"
                prediction_path = predictions_root / f"{prefix}_rnflt_logreg.csv"
                with prediction_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                metric_dir = metrics_root / f"exp8_{prefix}_rnflt"
                metric_dir.mkdir(parents=True)
                aggregate = metric_dir / f"{prefix}_rnflt_logreg_aggregate.csv"
                aggregate.write_text(
                    "n,f1,auroc,balanced_accuracy,ece,fpr,fnr\n"
                    "2,0.8,0.8,0.8,0.1,0.2,0.2\n",
                    encoding="utf-8",
                )

            out_dir = root / "agent"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AGENT_MULTITARGET_SCRIPT),
                    "--predictions-root",
                    str(predictions_root),
                    "--metrics-root",
                    str(metrics_root),
                    "--out-dir",
                    str(out_dir),
                    "--models",
                    "rnflt_logreg",
                    "--expected-cases",
                    "2",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete_locked_cohort"])
            self.assertEqual(summary["completed_calls"], 2)
            for target in TARGETS:
                with (out_dir / f"predictions_{target}.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()

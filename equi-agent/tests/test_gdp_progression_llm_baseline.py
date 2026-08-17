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


if __name__ == "__main__":
    unittest.main()

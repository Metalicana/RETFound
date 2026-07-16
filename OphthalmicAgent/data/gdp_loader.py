"""GDP test-split loading shared by RETFound and agentic evaluators."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

class GDPTestLoader:
    def __init__(
        self,
        csv_path,
        bscan_dir,
        rnflt_dir,
        oct_slices=8,
        limit=0,
        require_rnflt=True,
    ):
        frame = pd.read_csv(csv_path)
        required = {"filename", "gender", "age", "race", "glaucoma", "glaucoma_detection_use"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"GDP CSV is missing columns: {sorted(missing)}")
        test_frame = frame[
            frame["glaucoma_detection_use"].astype(str).str.lower() == "test"
        ].reset_index(drop=True)
        if test_frame.empty:
            raise ValueError("No GDP glaucoma_detection_use=test cases found")
        self.bscan_dir = Path(bscan_dir)
        self.rnflt_dir = Path(rnflt_dir)
        self.oct_slices = oct_slices
        self.require_rnflt = require_rnflt

        available_rows = []
        self.skipped_missing_files = []
        for _, row in test_frame.iterrows():
            npz_name = self._npz_name(row["filename"])
            bscan_path = self.bscan_dir / npz_name
            rnflt_path = self.rnflt_dir / npz_name
            missing = []
            if not bscan_path.is_file():
                missing.append(str(bscan_path))
            if require_rnflt and not rnflt_path.is_file():
                missing.append(str(rnflt_path))
            if missing:
                self.skipped_missing_files.append(
                    {"patient_id": str(row["filename"]), "missing": missing}
                )
            else:
                available_rows.append(row)

        self.frame = pd.DataFrame(available_rows).reset_index(drop=True)
        if limit > 0:
            self.frame = self.frame.head(limit)
        if self.frame.empty:
            raise ValueError("No GDP Test cases have all required files")

        if self.skipped_missing_files:
            print(
                f"Skipping {len(self.skipped_missing_files)} GDP Test cases with "
                "missing required files:"
            )
            for skipped in self.skipped_missing_files:
                print(
                    f"  {skipped['patient_id']}: "
                    + ", ".join(skipped["missing"])
                )
        print(f"GDP Test cases available for evaluation: {len(self.frame)}")

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def _npz_name(filename):
        name = str(filename)
        return name if name.lower().endswith(".npz") else f"{name}.npz"

    def load(self, index, include_rnflt=True):
        row = self.frame.iloc[index]
        npz_name = self._npz_name(row["filename"])
        bscan_path = self.bscan_dir / npz_name
        rnflt_path = self.rnflt_dir / npz_name
        with np.load(bscan_path) as data:
            volume = np.asarray(data["bscans"])
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D bscans, got {volume.shape}: {bscan_path}")
        # GDP stores B-scans as [height, width, slices] = [200, 200, 300].
        volume = np.moveaxis(volume, -1, 0)
        indices = np.linspace(0, volume.shape[0] - 1, self.oct_slices, dtype=int)
        selected = volume[indices]
        middle = volume[volume.shape[0] // 2]
        rnflt = None
        if include_rnflt:
            with np.load(rnflt_path) as data:
                rnflt = np.asarray(data["rnflt"], dtype=np.float32)
        return {
            "patient_id": str(row["filename"]),
            "bscan_path": bscan_path,
            "rnflt_path": rnflt_path,
            "oct_slices": selected,
            "middle_oct": middle,
            "slice_indices": indices.tolist(),
            "rnflt": rnflt,
            "ground_truth": int(row["glaucoma"]),
            "age": float(row["age"]),
            "gender": str(row["gender"]),
            "race": str(row["race"]),
        }

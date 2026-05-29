from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FairVisionNPZ(Dataset):
    def __init__(
        self,
        root_dir,
        split="Training",
        transform=None,
        image_kind="oct",
        oct_representation="center",
    ):
        self.files = []
        self.transform = transform
        self.image_kind = image_kind
        self.oct_representation = oct_representation
        self.strict_alignment = self._env_flag("FAIRVISION_STRICT_ALIGNMENT", True)
        self.sources = ["AMD", "DR", "Glaucoma"]
        self.root_dir = Path(root_dir).expanduser()
        self.split_folder = self._normalize_split(split)
        self.csv_base_path = self._metadata_root()
        self.image_base_path = self._image_root()
        self.required_npz_keys = {
            "AMD": "amd_condition",
            "DR": "dr_subtype",
            "Glaucoma": "glaucoma",
        }

        self.amd_map = {
            "not.in.icd.table": 0.0,
            "no.amd.diagnosis": 0.0,
            "normal": 0.0,
            "no amd": 0.0,
            "early.dry": 1.0,
            "early amd": 1.0,
            "intermediate.dry": 2.0,
            "intermediate amd": 2.0,
            "late amd": 3.0,
            "advanced.atrophic.dry.with.subfoveal.involvement": 3.0,
            "advanced.atrophic.dry.without.subfoveal.involvement": 3.0,
            "wet.amd.active.choroidal.neovascularization": 3.0,
            "wet.amd.inactive.choroidal.neovascularization": 3.0,
            "wet.amd.inactive.scar": 3.0,
        }
        self.dr_map = {
            "not.in.icd.table": 0.0,
            "no.dr.diagnosis": 0.0,
            "no dr": 0.0,
            "non-vision threatening dr": 0.0,
            "mild.npdr": 0.0,
            "moderate.npdr": 0.0,
            "severe.npdr": 1.0,
            "pdr": 1.0,
            "vision threatening dr": 1.0,
        }

        self.metadata_lookup = self._load_all_metadata()

        print(f"Scanning {self.split_folder} data in {self.root_dir}...")
        for source in self.sources:
            records = self._metadata_records_for_split(source)
            if records:
                self._append_records(source, records)
                continue
            self._append_legacy_files(source)
        print(f"Found {len(self.files)} images with metadata for {self.split_folder}.")
        self._print_path_diagnostics()
        self._validate_alignment()

    def label_summary(self):
        summary = {}
        for source in self.sources:
            rows = [item["meta"] for item in self.files if item["source"] == source]
            if not rows:
                summary[source] = {"rows": 0}
                continue

            if source == "AMD":
                raw_values = [self._scalar_to_string(row.get("amd", "")) for row in rows]
                stages = [int(self.amd_map.get(value, 0.0)) for value in raw_values]
                summary[source] = {
                    "rows": len(rows),
                    "raw_counts": self._counts(raw_values),
                    "stage_counts": self._counts(stages),
                    "stage>=1_rate": self._rate(stage >= 1 for stage in stages),
                    "stage>=2_rate": self._rate(stage >= 2 for stage in stages),
                    "stage>=3_rate": self._rate(stage >= 3 for stage in stages),
                }
            elif source == "DR":
                raw_values = [self._scalar_to_string(row.get("dr", "")) for row in rows]
                labels = [int(self.dr_map.get(value, 0.0)) for value in raw_values]
                summary[source] = {
                    "rows": len(rows),
                    "raw_counts": self._counts(raw_values),
                    "positive_rate": self._rate(label == 1 for label in labels),
                }
            else:
                raw_values = [self._scalar_to_string(row.get("glaucoma", "")) for row in rows]
                labels = [1 if value in {"1", "1.0", "true", "yes", "y"} else 0 for value in raw_values]
                summary[source] = {
                    "rows": len(rows),
                    "raw_counts": self._counts(raw_values),
                    "positive_rate": self._rate(label == 1 for label in labels),
                }
        return summary

    @staticmethod
    def _counts(values):
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: str(item[0])))

    @staticmethod
    def _rate(values):
        values = list(values)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _env_flag(name, default=True):
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def path_summary(self):
        paths = [item["path"] for item in self.files]
        filenames = [Path(item["path"]).name for item in self.files]
        by_source = {
            source: len({item["path"] for item in self.files if item["source"] == source})
            for source in self.sources
        }
        path_sources = {}
        filename_sources = {}
        for item in self.files:
            path_sources.setdefault(item["path"], set()).add(item["source"])
            filename_sources.setdefault(Path(item["path"]).name, set()).add(item["source"])
        reused_paths = {
            Path(path).name: sorted(sources)
            for path, sources in path_sources.items()
            if len(sources) > 1
        }
        reused_filenames = {
            filename: sorted(sources)
            for filename, sources in filename_sources.items()
            if len(sources) > 1
        }
        return {
            "rows": len(paths),
            "unique_paths": len(set(paths)),
            "unique_filenames": len(set(filenames)),
            "unique_paths_by_source": by_source,
            "paths_reused_across_sources": len(reused_paths),
            "filenames_reused_across_sources": len(reused_filenames),
            "reused_path_examples": dict(list(sorted(reused_paths.items()))[:5]),
            "reused_filename_examples": dict(list(sorted(reused_filenames.items()))[:5]),
        }

    def _print_path_diagnostics(self):
        if os.environ.get("FAIRVISION_PATH_DIAGNOSTICS", "1").strip().lower() in {"0", "false", "no"}:
            return
        summary = self.path_summary()
        print(
            f"{self.split_folder} path summary: rows={summary['rows']} "
            f"unique_paths={summary['unique_paths']} unique_filenames={summary['unique_filenames']} "
            f"unique_paths_by_source={summary['unique_paths_by_source']}"
        )
        if summary["paths_reused_across_sources"]:
            print(
                f"{self.split_folder} paths reused across disease CSVs: "
                f"{summary['paths_reused_across_sources']} examples={summary['reused_path_examples']}"
            )
        elif summary["filenames_reused_across_sources"]:
            print(
                f"{self.split_folder} filenames reused across disease CSVs: "
                f"{summary['filenames_reused_across_sources']} examples={summary['reused_filename_examples']}"
            )

    def _validate_alignment(self):
        if not self.strict_alignment:
            return

        summary = self.path_summary()
        if summary["paths_reused_across_sources"]:
            raise RuntimeError(
                f"FairVision layout is ambiguous for split={self.split_folder}: "
                f"{summary['paths_reused_across_sources']} NPZ paths are reused across disease CSVs. "
                f"Examples: {summary['reused_path_examples']}. "
                "AMD, DR, and Glaucoma use overlapping filenames for different cases, so the data must be "
                "stored in disease-specific split folders such as AMD/Training/data_00001.npz, "
                "DR/Training/data_00001.npz, and Glaucoma/Training/data_00001.npz. "
                "Set FAIRVISION_STRICT_ALIGNMENT=0 only for debugging."
            )

        missing = []
        checked_by_source = {source: 0 for source in self.sources}
        max_checks_per_source = int(os.environ.get("FAIRVISION_ALIGNMENT_CHECK_SAMPLES", 50))
        for item in self.files:
            source = item["source"]
            if checked_by_source[source] >= max_checks_per_source:
                continue
            checked_by_source[source] += 1
            required_key = self.required_npz_keys[source]
            try:
                with np.load(item["path"], allow_pickle=True) as data:
                    if required_key not in data.files:
                        missing.append((source, Path(item["path"]).name, required_key, tuple(data.files)))
            except Exception as exc:
                missing.append((source, Path(item["path"]).name, required_key, f"load_error={exc}"))
            if len(missing) >= 5:
                break

        if missing:
            examples = [
                f"{source}/{filename} missing {required_key}; keys={keys}"
                for source, filename, required_key, keys in missing
            ]
            raise RuntimeError(
                f"FairVision NPZ/CSV alignment check failed for split={self.split_folder}. "
                f"Examples: {examples}. "
                "This usually means the loader is pointing an AMD or DR CSV row at a Glaucoma NPZ, "
                "or the flat dataset lost the disease-specific folder context."
            )

    def _append_records(self, source, records):
        for file_meta in records:
            fname = str(file_meta["filename"])
            f_path = self._find_npz_path(source, self.split_folder, fname)
            if f_path is None:
                print(f"Warning: NPZ not found for {source}/{self.split_folder}/{fname}")
                continue
            self.files.append({"path": str(f_path), "source": source, "meta": file_meta})

    def _append_legacy_files(self, source):
        legacy_dir = self.root_dir / source / self.split_folder
        if not legacy_dir.exists():
            print(f"Warning: No metadata or legacy directory found for {source}: {legacy_dir}")
            return
        files_found = sorted(path for path in legacy_dir.iterdir() if path.suffix == ".npz")
        for f_path in files_found:
            fname = f_path.name
            file_meta = self.metadata_lookup.get((source, fname), self.metadata_lookup.get(fname, {}))
            self.files.append({"path": str(f_path), "source": source, "meta": file_meta})

    @staticmethod
    def _normalize_split(split):
        key = str(split).strip().lower()
        return {
            "train": "Training",
            "training": "Training",
            "val": "Validation",
            "valid": "Validation",
            "validation": "Validation",
            "test": "Test",
        }.get(key, str(split))

    @staticmethod
    def _scalar_to_string(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip().lower()

    def _metadata_root(self):
        if (self.root_dir / "HarvardFairVision30k").exists():
            return self.root_dir / "HarvardFairVision30k"
        return self.root_dir

    def _image_root(self):
        if all((self.root_dir / split).exists() for split in ("Training", "Validation", "Test")):
            return self.root_dir
        if all((self.root_dir.parent / split).exists() for split in ("Training", "Validation", "Test")):
            return self.root_dir.parent
        return self.root_dir

    def _metadata_csv_path(self, source):
        task = source.lower()
        candidates = [
            self.csv_base_path / source / "ReadMe" / f"data_summary_{task}.csv",
            self.csv_base_path / source / f"data_summary_{task}.csv",
            self.root_dir / source / "ReadMe" / f"data_summary_{task}.csv",
            self.root_dir / source / f"data_summary_{task}.csv",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _metadata_records_for_split(self, source):
        csv_path = self._metadata_csv_path(source)
        if csv_path is None:
            return []
        df = pd.read_csv(csv_path)
        if "use" in df.columns:
            split_mask = df["use"].map(self._normalize_split) == self.split_folder
            df = df[split_mask].copy()
        return df.to_dict("records")

    def _find_npz_path(self, source, split_folder, filename):
        candidates = [
            self.root_dir / source / split_folder / filename,
            self.root_dir / split_folder / source / filename,
            self.image_base_path / source / split_folder / filename,
            self.image_base_path / split_folder / source / filename,
            self.csv_base_path / source / split_folder / filename,
            self.root_dir / split_folder / filename,
            self.image_base_path / split_folder / filename,
        ]
        return next((path for path in candidates if path.exists()), None)

    def _load_all_metadata(self):
        combined_meta = {}
        for source in self.sources:
            csv_path = self._metadata_csv_path(source)
            if csv_path is not None:
                records = pd.read_csv(csv_path).to_dict("records")
                for rec in records:
                    combined_meta[(source, rec["filename"])] = rec
                    combined_meta.setdefault(rec["filename"], rec)
            else:
                print(f"Warning: Metadata CSV not found for {source} under {self.root_dir}")
        return combined_meta

    def _build_label_and_metadata(self, item, data):
        label = torch.full((5,), -1.0)
        source = item["source"]
        csv_meta = item["meta"]

        if source == "AMD":
            raw_value = data["amd_condition"] if "amd_condition" in data.files else csv_meta.get("amd", "")
            severity = int(self.amd_map.get(self._scalar_to_string(raw_value), 0.0))
            label[0] = 1.0 if severity >= 1 else 0.0
            label[1] = 1.0 if severity >= 2 else 0.0
            label[2] = 1.0 if severity >= 3 else 0.0
        elif source == "DR":
            raw_value = data["dr_subtype"] if "dr_subtype" in data.files else csv_meta.get("dr", "")
            severity = int(self.dr_map.get(self._scalar_to_string(raw_value), 0.0))
            label[3] = 1.0 if severity >= 1.0 else 0.0
        elif source == "Glaucoma":
            raw_value = data["glaucoma"] if "glaucoma" in data.files else csv_meta.get("glaucoma", 0)
            key = self._scalar_to_string(raw_value)
            severity = 1 if key in {"1", "1.0", "true", "yes", "y"} else 0
            label[4] = 1.0 if severity == 1 else 0.0
        else:
            severity = -1

        metadata = {
            "disease": source,
            "age": csv_meta.get("age", "unknown"),
            "gender": csv_meta.get("gender", "unknown"),
            "race": csv_meta.get("race", "unknown"),
            "ethnicity": csv_meta.get("ethnicity", "unknown"),
            "language": csv_meta.get("language", "unknown"),
            "maritalstatus": csv_meta.get("maritalstatus", "unknown"),
            "filename": os.path.basename(item["path"]),
            "groundtruth": severity,
        }
        return label, metadata

    @staticmethod
    def _normalize_to_uint8(image):
        image = np.asarray(image, dtype="float32")
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros(image.shape, dtype="uint8")
        lo, hi = np.percentile(image[finite], [1.0, 99.0])
        if hi <= lo:
            lo = float(np.nanmin(image[finite]))
            hi = float(np.nanmax(image[finite]))
        if hi <= lo:
            return np.zeros(image.shape, dtype="uint8")
        image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        return (image * 255.0).astype("uint8")

    @staticmethod
    def _center_slice(volume):
        volume = np.asarray(volume)
        if volume.ndim == 2:
            return volume
        if volume.ndim != 3:
            raise ValueError(f"Expected 2D or 3D OCT array, got shape={volume.shape}")
        slice_axis = min(range(3), key=lambda axis: volume.shape[axis])
        return np.take(volume, volume.shape[slice_axis] // 2, axis=slice_axis)

    @staticmethod
    def _slice_axis(volume):
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D OCT array, got shape={volume.shape}")
        return min(range(3), key=lambda axis: volume.shape[axis])

    @staticmethod
    def _take_fractional_slice(volume, axis, fraction):
        index = int(round((volume.shape[axis] - 1) * fraction))
        return np.take(volume, index, axis=axis)

    def _oct_array(self, volume):
        volume = np.asarray(volume, dtype="float32")
        if volume.ndim == 2:
            return volume
        axis = self._slice_axis(volume)
        if self.oct_representation == "center":
            return self._take_fractional_slice(volume, axis, 0.5)
        if self.oct_representation == "mean":
            return np.mean(volume, axis=axis)
        if self.oct_representation == "max":
            return np.max(volume, axis=axis)
        if self.oct_representation == "three_slices":
            return np.stack(
                [
                    self._take_fractional_slice(volume, axis, 0.25),
                    self._take_fractional_slice(volume, axis, 0.5),
                    self._take_fractional_slice(volume, axis, 0.75),
                ],
                axis=-1,
            )
        if self.oct_representation == "mean_max_center":
            return np.stack(
                [
                    self._take_fractional_slice(volume, axis, 0.5),
                    np.mean(volume, axis=axis),
                    np.max(volume, axis=axis),
                ],
                axis=-1,
            )
        raise ValueError(f"Unknown OCT representation: {self.oct_representation}")

    def _oct_image(self, data):
        image = self._oct_array(data["oct_bscans"])
        image = self._normalize_to_uint8(image)
        return Image.fromarray(image).convert("RGB")

    def _slo_image(self, data):
        fundus_img = data["slo_fundus"]
        fundus_img = self._normalize_to_uint8(fundus_img)
        return Image.fromarray(fundus_img).convert("L")

    def __getitem__(self, idx):
        item = self.files[idx]
        try:
            data = np.load(item["path"])
            image = self._slo_image(data) if self.image_kind == "slo" else self._oct_image(data)
            if self.transform:
                image = self.transform(image)
            label, metadata = self._build_label_and_metadata(item, data)
            return image, label, metadata
        except Exception as exc:
            print(f"Error at index {idx}: {exc}")
            return self.__getitem__(idx - 1 if idx > 0 else 0)

    def __len__(self):
        return len(self.files)

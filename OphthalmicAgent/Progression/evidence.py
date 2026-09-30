"""Strict cohort/provenance checks and whitelisted baseline evidence preparation."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "equi-agent/scripts"))
import run_gdp_progression_clean_suite as clean
import run_gdp_progression_llm_baseline as baseline
from EquityAgent import compute_demographic_reliability_score as reliability_tool

from .prompts import ENDPOINTS

native = clean.native
UNKNOWN = {"", "unknown", "missing", "nan", "none", "-1"}
RISK_WEIGHTS = (0.35, 0.25, 0.15, 0.15, 0.10)


def demographics(row):
    result = {}
    for field in ("race", "ethnicity", "sex_gender"):
        value = row.get(field, "").strip().lower()
        if value not in UNKNOWN:
            result[field] = value
    try:
        age = float(row.get("age", ""))
        if math.isfinite(age) and 0 <= age <= 120:
            result["age"] = age
    except (ValueError, TypeError):
        pass
    return result


def group_values(row):
    meta = demographics(row)
    age = meta.get("age")
    return {"age_group": "unknown" if age is None else "older" if age >= 60 else "younger" if age < 40 else "middle-aged",
            "race": reliability_tool.norm_race(meta.get("race", "")),
            "sex_gender": reliability_tool.norm_gender(meta.get("sex_gender", ""))}


def finite_metrics(rows):
    values = native.metrics(rows, "oof")
    fields = ("n", "tp", "tn", "fp", "fn", "f1", "auroc", "balanced_accuracy", "ece", "fpr", "fnr")
    return {key: float(values[key]) if math.isfinite(float(values[key])) else None for key in fields}


class Reliability:
    """Reuse OphthalmicAgent's risk function with development-only subgroup support."""
    def __init__(self, rows):
        if len(rows) != 300 or any(r.get("split") != "oof" for r in rows):
            raise ValueError("Reliability requires exactly 300 development-OOF predictions")
        self.rows = [{**r, **group_values(r)} for r in rows]
        self.global_metrics = finite_metrics(rows)
        self.groups = {}
        for field in ("age_group", "race", "sex_gender"):
            for value in {r[field] for r in self.rows} - UNKNOWN:
                members = [r for r in self.rows if r[field] == value]
                positive = sum(native.label(r) for r in members)
                eligible = positive >= 20 and len(members) - positive >= 20
                self.groups[field, value] = {"n": len(members), "positives": positive,
                                            "eligible": eligible,
                                            "metrics": finite_metrics(members) if eligible else None}

    def packet(self, row):
        profile = group_values(row)
        global_risk = reliability_tool.risk_score(self.global_metrics, None, *RISK_WEIGHTS)
        groups, weighted, support = {}, 0.0, 0
        for field, value in profile.items():
            group = self.groups.get((field, value), {"n": 0, "positives": 0, "eligible": False, "metrics": None})
            groups[field] = {"value": value or "unavailable", **group}
            if group["eligible"]:
                weighted += group["n"] * reliability_tool.risk_score(group["metrics"], self.global_metrics, *RISK_WEIGHTS)
                support += group["n"]
        intersection = sum(all(r[k] == v and v not in UNKNOWN for k, v in profile.items()) for r in self.rows)
        mixing = intersection / (intersection + 50) if support else 0.0
        local_risk = weighted / support if support else global_risk
        return {"source": "development_oof", "f1_average": "binary", "development_cases": 300,
                "global": self.global_metrics, "subgroups": groups, "intersection_n": intersection,
                "trust_score": 1 - (mixing * local_risk + (1 - mixing) * global_risk),
                "formula": "1 - R_bad; R_bad uses OphthalmicAgent risk_score weights 0.35 FNR, 0.25 FPR, 0.15 ECE, 0.15 (1-AUROC), 0.10 (1-F1); support-weighted eligible subgroups, intersection shrinkage k=50",
                "limitations": "Sparse/unknown subgroups use global risk. This score is not a correctness probability or a guarantee against bias."}


def save_asset(directory, name, data):
    import hashlib
    path = directory / f"{name}_{hashlib.sha256(data).hexdigest()[:16]}.png"
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def oct_montage(path, key):
    import io
    import numpy as np
    from PIL import Image, ImageDraw

    with np.load(path, allow_pickle=False) as archive:
        volume = np.asarray(archive[key], dtype=np.float32)
    if volume.ndim != 3 or min(volume.shape) < 5:
        raise ValueError(f"Invalid GDP B-scan shape {volume.shape}: {path}")
    # GDP Bscan files use height x width x slice, as in data/gdp_loader.py.
    indices = np.linspace(0, volume.shape[2] - 1, 5).astype(int)
    selected = volume[:, :, indices]
    finite = selected[np.isfinite(selected)]
    if not finite.size:
        raise ValueError(f"No finite OCT pixels: {path}")
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        raise ValueError(f"Constant/unreadable OCT input: {path}")
    image = Image.new("RGB", (5 * 224, 250), "white")
    draw = ImageDraw.Draw(image)
    for col, index in enumerate(indices):
        pixels = np.nan_to_num(volume[:, :, index], nan=low, posinf=high, neginf=low)
        pixels = np.clip((pixels - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        image.paste(Image.fromarray(pixels).resize((224, 224)), (col * 224, 26))
        draw.text((col * 224 + 8, 6), f"Baseline slice {index}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare(args):
    """Return baseline-only cases separately from evaluation labels."""
    old = SimpleNamespace(run_root=args.clean_root, primary_run=args.primary_run,
                          manifests_root=args.manifests_root)
    cohorts = clean.check_manifests(old)
    sources, models, reliabilities = {}, {}, {}
    for target in ENDPOINTS:
        prediction, prior = clean.checked_bundle(old, target, cohorts[target])
        staged_prediction = args.clean_root / "predictions" / prediction.name
        staged_prior = args.clean_root / "priors" / prior.parent.name / prior.name
        for staged, original in ((staged_prediction, prediction), (staged_prior, prior)):
            if native.sha256(staged) != native.sha256(original):
                raise ValueError(f"Staged clean evidence differs from verified native output: {staged}")
            sources[str(staged)] = native.sha256(staged)
        sources[str(clean.target_manifest(old, target))] = native.sha256(clean.target_manifest(old, target))
        models[target] = {native.case_id(r): r for r in clean.align_predictions(staged_prediction, cohorts[target][1])}
        directory = clean.target_run(old, target)
        oof = [r for fold in range(1, 6) for r in native.read_csv(directory / f"fold_{fold}/predictions.csv")]
        reliabilities[target] = Reliability(oof)
    cases, answers = [], {target: cohorts[target][1] for target in ENDPOINTS}
    indexes = {t: {native.case_id(r): r for r in rows} for t, rows in answers.items()}
    for row in cohorts[native.TARGET][1]:
        case_id = native.case_id(row)
        feature_keys = ("rnflt_path", "rnflt_key", "bscan_path", "bscan_key", "age", "race", "ethnicity", "sex_gender", *baseline.GDP_TD_COLUMNS)
        for target in ENDPOINTS:
            if any(row.get(k, "") != indexes[target][case_id].get(k, "") for k in feature_keys):
                raise ValueError(f"Baseline evidence differs across endpoint manifests: {case_id}/{target}")
        rnflt_path = baseline.resolve_data_path(row["rnflt_path"], args.path_prefix_from, args.path_prefix_to)
        sources[str(rnflt_path)] = native.sha256(rnflt_path)
        array, stats = baseline.load_rnflt({**row, "resolved_rnflt_path": str(rnflt_path)})
        td, td_summary = baseline.td_evidence(row)
        if not all(math.isfinite(v) for v in td.values()):
            raise ValueError(f"Incomplete baseline visual field: {case_id}")
        directory = args.out_dir / "assets" / case_id
        case = {"case_id": case_id, "demographics": demographics(row),
                "modalities": ["baseline RNFLT map", "baseline 52-point visual field"],
                "rnflt_image": save_asset(directory, "rnflt", baseline.render_rnflt_png(array)),
                "rnflt_statistics": stats, "td_values": td, "td_summary": td_summary,
                "helper_predictions": {}, "reliability": {}}
        if args.include_oct:
            bscan_path = baseline.resolve_data_path(row["bscan_path"], args.path_prefix_from, args.path_prefix_to)
            sources[str(bscan_path)] = native.sha256(bscan_path)
            case["oct_image"] = save_asset(directory, "oct", oct_montage(bscan_path, row.get("bscan_key") or "bscans"))
            case["modalities"].append("baseline OCT volume (additional input beyond the native helper)")
        for target in ENDPOINTS:
            prediction = models[target][case_id]
            case["helper_predictions"][target] = {
                "model": native.MODEL, "probability": prediction["y_prob"],
                "prediction": prediction["y_pred"], "source_threshold": 0.5,
            }
            case["reliability"][target] = reliabilities[target].packet(row)
        cases.append(case)
    return cases, answers, sources

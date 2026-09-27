"""Inventory paper result files without running models or copying patient rows/secrets."""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import zipfile


SKIP_DIRS = {"feature_cache", "checkpoints", "weights", "raw", "__pycache__", ".git"}
SUMMARY_KEYS = ("dataset", "task", "tasks", "dry_run", "cases", "models_requested",
                "test_common_cases", "validation_common_cases", "complete_locked_cohort")


def category(path):
    name = path.name.lower()
    if path.suffix.lower() == ".csv":
        if "coverage" in name:
            return "coverage"
        if "predict" in name or "result" in name or name.startswith(("fairvision_", "gdp_")):
            return "predictions_or_metrics"
        if any(word in name for word in ("aggregate", "subgroup", "disparit", "ablation", "prior", "reliability")):
            return "metrics_or_priors"
    if path.suffix.lower() == ".json" and "summary" in name:
        return "run_summary"
    return None


def inventory(root):
    results = []
    for rel in ("equi-agent/outputs", "OphthalmicAgent/outputs"):
        base = root / rel
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(folder) / d).is_symlink())
            for name in sorted(files):
                path = Path(folder) / name
                kind = category(path)
                if kind is None or path.is_symlink():
                    continue
                item = {"path": str(path.relative_to(root)), "kind": kind,
                        "bytes": path.stat().st_size,
                        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        "dry_run_path_hint": any(word in str(path).lower() for word in ("dryrun", "dry_run", "smoke")),
                        "columns": [], "summary": {}, "error": ""}
                try:
                    if path.suffix.lower() == ".csv":
                        with path.open(newline="", encoding="utf-8-sig") as handle:
                            item["columns"] = next(csv.reader(handle), [])
                    elif item["bytes"] <= 5_000_000:
                        payload = json.loads(path.read_text())
                        if isinstance(payload, dict):
                            item["summary"] = {key: payload[key] for key in SUMMARY_KEYS if key in payload}
                except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                    item["error"] = type(exc).__name__
                results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/retinagent_figure_audit"))
    args = parser.parse_args()
    records = inventory(args.repo_root.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "file_count": len(records), "counts": dict(Counter(r["kind"] for r in records)),
               "includes_patient_rows": False, "files": records}
    report = args.out_dir / "inventory.json"
    report.write_text(json.dumps(payload, indent=2) + "\n")
    table = args.out_dir / "inventory.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "kind", "bytes", "modified_utc", "dry_run_path_hint", "columns", "summary", "error"))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "columns": json.dumps(record["columns"]), "summary": json.dumps(record["summary"])})
    archive = args.out_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (report, table):
            bundle.write(path, path.name)
    print(json.dumps({"files_found": len(records), "archive": str(archive),
                      "scope": "File paths, CSV headers and whitelisted run-summary fields only; no prediction rows."}, indent=2))


if __name__ == "__main__":
    main()

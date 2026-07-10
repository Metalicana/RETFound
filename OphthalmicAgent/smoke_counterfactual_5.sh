#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p outputs/smoke_counterfactual_5

export MAX_CASES=5
export OUTPUT_CSV="outputs/smoke_counterfactual_5/predictions.csv"
export COUNTERFACTUAL_CACHE_PATH="outputs/smoke_counterfactual_5/counterfactual_traces.jsonl"

python -u -m main_new 2>&1 | tee outputs/smoke_counterfactual_5/run.log

python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/smoke_counterfactual_5")
predictions_path = root / "predictions.csv"
traces_path = root / "counterfactual_traces.jsonl"

if not predictions_path.exists():
    raise SystemExit("Smoke test failed: predictions.csv was not created")

with predictions_path.open(newline="", encoding="utf-8") as handle:
    predictions = list(csv.DictReader(handle))

if len(predictions) != 5:
    raise SystemExit(f"Smoke test failed: expected 5 prediction rows, found {len(predictions)}")

failed = [row for row in predictions if row.get("Pred_GL") in {"", "-1", None}]
if failed:
    raise SystemExit(f"Smoke test failed: {len(failed)} cases have no valid glaucoma prediction")

traces = []
if traces_path.exists():
    with traces_path.open(encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]

print(
    json.dumps(
        {
            "prediction_rows": len(predictions),
            "valid_glaucoma_predictions": len(predictions) - len(failed),
            "counterfactual_trace_rows": len(traces),
            "note": (
                "Counterfactual rows are created only for cases with RETFound probability "
                "between 10% and 90%; high-confidence cases retain the existing bypass."
            ),
            "outputs": {
                "predictions": str(predictions_path),
                "counterfactual_traces": str(traces_path),
                "log": str(root / "run.log"),
            },
        },
        indent=2,
    )
)
PY

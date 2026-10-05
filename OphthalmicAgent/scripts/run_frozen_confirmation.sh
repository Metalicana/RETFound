#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:-$REPO/OphthalmicAgent/outputs/fairvision_confirmation_v1}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${PYTHON_BIN:-python}"
SCRIPT="$REPO/OphthalmicAgent/scripts/run_frozen_confirmation.py"

mkdir -p "$RUN"
"$PYTHON" -u "$SCRIPT" --stage prepare --run-root "$RUN" "$@"
# run repeats all immutable input checks before constructing its API client.
exec "$PYTHON" -u "$SCRIPT" --stage run --run-root "$RUN" "$@"

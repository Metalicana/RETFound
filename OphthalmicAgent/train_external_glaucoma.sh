#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${1:?usage: bash train_external_glaucoma.sh gamma|papila}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "$DATASET" in
  gamma)
    MANIFEST="data_gamma/manifest.csv"
    python scripts/validate_external_glaucoma_manifest.py --manifest "$MANIFEST" --require-cfp --require-oct
    python scripts/train_retfound_external_glaucoma.py --manifest "$MANIFEST" --dataset gamma --modality cfp --out-dir outputs/gamma/retfound_cfp --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
    python scripts/train_retfound_external_glaucoma.py --manifest "$MANIFEST" --dataset gamma --modality oct --out-dir outputs/gamma/retfound_oct --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
    python scripts/precompute_external_cdr.py --manifest "$MANIFEST" --out outputs/gamma/cdr.csv
    ;;
  papila)
    MANIFEST="data_papila/manifest.csv"
    python scripts/validate_external_glaucoma_manifest.py --manifest "$MANIFEST" --require-cfp
    python scripts/train_retfound_external_glaucoma.py --manifest "$MANIFEST" --dataset papila --modality cfp --out-dir outputs/papila/retfound_cfp --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
    python scripts/precompute_external_cdr.py --manifest "$MANIFEST" --out outputs/papila/cdr.csv
    ;;
  *)
    echo "unknown dataset: $DATASET (expected gamma or papila)" >&2
    exit 2
    ;;
esac

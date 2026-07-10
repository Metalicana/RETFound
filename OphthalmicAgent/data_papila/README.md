# PAPILA scaffold

PAPILA supplies bilateral CFP images, glaucoma diagnoses, clinical data, and
expert optic-disc/cup annotations.

Cluster-only images and masks belong under `raw/`. Convert downloaded metadata
to `manifest.csv` using `manifest.example.csv`. Patient separation is mandatory
because both eyes are provided. Resolve the suspicious-glaucoma class with a
documented exclusion or label policy before training.

```bash
python scripts/prepare_papila.py
python scripts/validate_external_glaucoma_manifest.py --manifest data_papila/manifest.csv --require-cfp
python scripts/train_retfound_external_glaucoma.py --manifest data_papila/manifest.csv --modality cfp --dataset papila --out-dir outputs/papila/retfound_cfp
python scripts/precompute_external_cdr.py --manifest data_papila/manifest.csv --out outputs/papila/cdr.csv
```

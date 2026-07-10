# GAMMA scaffold

GAMMA is the paired CFP + 3D OCT external glaucoma dataset.

Cluster-only images belong under `raw/` and must not be committed. Convert the
downloaded metadata to `manifest.csv` using `manifest.example.csv`.

Required columns are `dataset`, `case_id`, `patient_id`, `split`, `label`,
`cfp_path`, and `oct_path`. Labels are binary (`0` normal, `1` any glaucoma).
OCT paths may point to a volume file or a directory of ordered B-scans.

Never tune thresholds on the test split. If official test labels are withheld,
make a patient-separated validation split from released training cases.

```bash
python scripts/validate_external_glaucoma_manifest.py --manifest data_gamma/manifest.csv --require-cfp --require-oct
python scripts/train_retfound_external_glaucoma.py --manifest data_gamma/manifest.csv --modality cfp --dataset gamma --out-dir outputs/gamma/retfound_cfp
python scripts/train_retfound_external_glaucoma.py --manifest data_gamma/manifest.csv --modality oct --dataset gamma --out-dir outputs/gamma/retfound_oct
python scripts/precompute_external_cdr.py --manifest data_gamma/manifest.csv --out outputs/gamma/cdr.csv
```

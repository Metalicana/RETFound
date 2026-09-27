# Motivation Figure

**Overall performance hides complementary errors.** (a) Observed balanced accuracy of nine foundation-model/modality combinations on the same 3,000 held-out FairVision glaucoma cases. RETFound has the highest observed score. (b) Paired correctness relative to RETFound, using the same row order as panel a. Teal counts cases where the alternative is correct and RETFound is wrong; coral counts the reverse. Counts overlap across models and must not be added across rows. Of RETFound's 815 errors, 745 have at least one correct alternative prediction. This retrospective observation motivates studying case-specific source trust; it does not establish a deployable selector, achievable improvement or clinical benefit.

## Use In The Paper

Use this as the **problem/motivation figure**, before the architecture. The message is not that every model is equally good: lower-ranked models can contribute correct evidence, but also introduce errors. The proposed method must separately demonstrate that it can distinguish those situations without access to test labels.

## Provenance

This is a descriptive reanalysis of the saved full 3,000-case test predictions used by the historical deterministic-arbitration experiment. It is **not** the manuscript's 250-case agent evaluation. Model names, source paths and SHA-256 hashes are recorded in `motivation_source_data.json`. Saved thresholded decisions are used unchanged; no fitting, rethresholding, LLM calls, case selection or prompt changes were performed. All nine available models are shown, ordered by observed balanced accuracy for readability. These are point estimates, not tests of statistical superiority. Case IDs and binary labels are checked for exact agreement across files. Balanced accuracy is used to avoid mixing the binary and support-weighted F1 conventions of earlier tables.

## Rebuild

```bash
python equi-agent/scripts/build_motivation_figure.py \
  --inputs-root /path/to/separate/extracted/archive
```

The draw.io file is editable. Rebuilding overwrites generated files; keep manual edits in a separately named copy.

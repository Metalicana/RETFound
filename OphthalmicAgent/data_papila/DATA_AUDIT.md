# PAPILA Data Audit

Audited on 2026-07-10 from the Hugging Face mirror
`ai4ophth/PAPILA-dataset` downloaded as Parquet shards. The authoritative
dataset release remains the original PAPILA/Figshare distribution.

## Local layout

The cluster stores the physical download under:

```text
/data/ab575577/ophthalmic_datasets/papila
```

The repository accesses it through:

```text
OphthalmicAgent/data_papila/raw
```

which is a symlink to the physical directory.

The Hugging Face mirror contains 11 Parquet shards under `raw/data/`. Each row
contains:

- `rich text`: string, empty in the inspected examples
- `sparse text`: diagnostic label
- `retina`: retinal image bytes and original path
- `cup_exp1`, `cup_exp2`: two expert cup-mask images
- `disc_exp1`, `disc_exp2`: two expert disc-mask images
- `opht_cont`: additional ophthalmic image bytes
- `metadata`: comma-separated clinical metadata string

Observed clinical metadata includes:

- eye side (`OS` or `OD`)
- age
- gender code
- refraction/dioptre values
- astigmatism
- phakic/pseudophakic status
- pneumatic and Perkins pressure measurements
- pachymetry
- axial length
- visual-field mean deviation

Race is not present in the inspected schema and must not be inferred.

## Verified cohort structure

The Parquet shards contain:

| Quantity | Count |
| --- | ---: |
| Rows / eyes | 488 |
| Patients | 244 |
| OS eyes | 244 |
| OD eyes | 244 |
| Valid OS/OD patient pairs | 244 |
| Invalid or incomplete pairs | 0 |

Each retinal filename occurs exactly twice: once for `OS` and once for `OD`.
The retinal filename stem is therefore the patient identifier. Any train,
validation, and test split must be performed at this patient level so the two
eyes never cross splits.

## Eye-level diagnostic labels

| Raw label | Eyes |
| --- | ---: |
| `healthy` | 170 |
| `no glaucoma/healthy eye` | 163 |
| `glaucoma present` | 74 |
| `glaucoma suspect` | 47 |
| `glaucoma-suspicious` | 34 |

The two suspect spellings are distinct raw strings but represent the same
uncertain/suspect concept for normalization.

## Patient-level label combinations

| Bilateral label combination | Patients |
| --- | ---: |
| `healthy` + `no glaucoma/healthy eye` | 163 |
| `glaucoma present` + `glaucoma suspect` | 40 |
| `glaucoma present` + `glaucoma-suspicious` | 34 |
| `glaucoma suspect` + `healthy` | 7 |

Notably, every patient with a `glaucoma present` eye has a suspect fellow eye;
there are no observed patients with two `glaucoma present` labels in this
mirror.

## Locked initial evaluation policy

For the first binary RETFound-CFP head:

```text
healthy                    -> 0
no glaucoma/healthy eye    -> 0
glaucoma present           -> 1
glaucoma suspect           -> excluded from binary head fitting/scoring
glaucoma-suspicious        -> excluded from binary head fitting/scoring
```

Suspect eyes must not be forced into the negative or positive class. Preserve
them as a separate uncertainty/selective-escalation cohort for OphthalmicAgent.

Because all positive eyes belong to patients whose fellow eyes are suspects,
patient-level splitting must occur before filtering or exporting individual
eyes. Otherwise the fellow eye could leak patient-specific information across
splits.

## Intended evidence lanes

PAPILA can supply the following OphthalmicAgent evidence:

1. RETFound-CFP glaucoma probability.
2. CFP vision-agent morphology report.
3. SegFormer-estimated CDR.
4. Expert-mask-derived CDR from two annotators.
5. Bilateral/fellow-eye evidence.
6. Age and gender context.
7. Non-demographic clinical evidence such as IOP, pachymetry, axial length,
   refraction, and visual-field MD.

Counterfactual ablations should distinguish demographic context from clinical
measurements. For example, removing age/gender is not the same intervention as
removing IOP or visual-field evidence.

## Remaining checks before training

- Parse and validate every metadata field and missing-value convention.
- Confirm the meaning of the gender codes from authoritative PAPILA
  documentation rather than inferring it.
- Extract retinal images and both expert mask sets with eye-specific filenames.
- Generate deterministic patient-grouped train/validation/test splits.
- Report class counts after the patient split and suspect-eye exclusion.
- Compare SegFormer CDR against each expert annotation and inter-expert CDR.


# Clean Six-Endpoint GDP Rerun

Endpoints: `md`, `vfi`, `td_pointwise`, `md_fast`, `md_fast_no_p_cut`, and
`td_pointwise_no_p_cut`. They describe the same patients, not six different cohorts.

## Scope

- Six **independently trained, single-output** RNFLT+TDS native helpers using the
  recovered preprocessing/model and fixed 60-epoch schedule. The source dataset's
  label index is extended to all six labels; images and architecture are unchanged.
- Five development-only OOF folds and a 300-development-case final fit per target.
  The completed primary OOF run is reused only if its exact known predecessor
  fingerprint and every other source/data/config field match. Existing results
  are not relabeled as a joint six-output model.
- One **existing multitarget agent request per test case**, containing six separate
  evidence packets: approximately 200 API requests plus retries, not 1,200. This
  uses the existing compact multitarget prompt unchanged. It is not identical to
  the historical single-target prompting protocol; report that distinction.
- Strict OOF-only priors for the native helper. RETFound and classical sources are
  deliberately not agent inputs in this run because their OOF priors have not been
  established. This is a native-helper-plus-agent experiment, not proof of multi-FM
  arbitration or a rerun of every paper method.
- Existing standalone LLM baselines are **not rerun**. The collector uses them only
  when the run is non-dry, complete, and exact IDs/labels match for the endpoint.
  Missing/invalid rows are listed explicitly; cohort checks alone do not certify
  the entire historical baseline protocol.

No thresholds, epochs or seeds are searched. Fixed threshold 0.5 applies to the
helpers. The agent retains its existing threshold actions. Main metrics use all
200 forced predictions, including escalated cases, never an accepted-only subset.
These test cases were inspected previously: describe this as a corrected retrospective
rerun, not a newly untouched confirmatory test cohort.

## Run on CECSL

Push locally, then pull on CECSL. Do not run the older `everything` script for this repair.

```bash
cd ~/RETFound
conda activate retfound
source equi-agent/scripts/gdp_llm_api_preamble.sh

RUN="$HOME/RETFound/equi-agent/outputs/gdp_progression_clean_v1"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  python equi-agent/scripts/run_gdp_progression_clean_suite.py \
  --stage all --run-root "$RUN" --device cuda:0 --deployment gpt-5.1 \
  > "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/run.pid"
```

`AZURE_OPENAI_API_KEY` must be available in the environment or existing `.env`.
The initial API client check does not send a query. Training runs on physical GPU 1.
`--stage all` **does make paid agent calls after training**. For GPU work only, use
`--stage train`; later `--stage agent` validates/stages results and runs the API phase.
Completed native folds/final fits and completed agent cases resume with strict
provenance checks. Partial native fits restart from scratch. Three consecutive failed
API cases stop the run, preserving completed cases; no fallback labels are invented.

```bash
python equi-agent/scripts/run_gdp_progression_clean_suite.py --stage status
tail -n 40 "$RUN/run.log"
```

## Outputs

- Primary native run: `equi-agent/outputs/gdp_native_oof_v1` (existing directory).
- Other native runs: `$RUN/native/<target>/`.
- Agent evidence: `$RUN/predictions/`, `$RUN/priors/`, `$RUN/input_sources.json`.
- Dry join check: `$RUN/agent_join_check/`, explicitly separate from real outputs.
- Live agent: `$RUN/agent/predictions_<target>.csv`, saved requests, responses and
  resume fingerprint. The six target prompts and existing numerical policies are unchanged.
- Six tables: `$RUN/results/results.md`, `results.tex`, `results.csv`.
- Subgroup supports/eligibility: `$RUN/results/subgroups.csv`.
- Missing historical LLM rows: `$RUN/results/unavailable_llm_baselines.json`.

F1 is binary positive-class F1 throughout these new tables. Worst-group F1 uses
race, sex/gender and age groups with >=20 positives and >=20 negatives per group,
and requires >=2 eligible groups in an attribute. There is no unstable-group fallback.
The sparse endpoints will often have N/A worst-group F1; subgroup counts remain
available. Changing that definition requires an explicitly separate reporting rule,
not silently mixing old and new worst-group values.

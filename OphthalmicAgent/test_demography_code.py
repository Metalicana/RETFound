from types import SimpleNamespace
from compute_demographic_reliability_score import main
from pathlib import Path

args = SimpleNamespace(
    subgroup_reliability_csv=Path("/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv"),
    task="glaucoma",
    model="retfound_oct",
    age_group="older",
    race="white",
    gender="female",
    score_column="final_R_bad",
    json=False,
    priors_json=None,
    support_csv=None,
    k=50.0,
    fnr_weight=0.35,
    fpr_weight=0.25,
    ece_weight=0.15,
    auroc_weight=0.15,
    f1_weight=0.10,
    raw_local_lambdas=False,
)

score = main(args)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

# --- CONFIGURATION ---
CSV_PATH = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/outputs/glaucoma_counterfactual_250/predictions.csv"  # Path to your main results output file

def print_subset_report(df_subset, title):

    if len(df_subset) == 0:
        print(f"\n{title}: No samples.")
        return

    y_true = df_subset["Ground_Truth"].astype(int).values

    y_pred = np.where(
        df_subset["Is_Correct"] == 1,
        y_true,
        1 - y_true
    )

    print("\n" + "="*75)
    print(title)
    print(f"Number of samples: {len(df_subset)}")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Healthy (0)",
                "Pathological (1)"
            ],
            digits=4,
            zero_division=0
        )
    )


def calculate_metrics(csv_path):

    df = pd.read_csv(csv_path)

    # ---------------- Column checking ----------------

    if 'Is_Correct' not in df.columns or 'Ground_Truth' not in df.columns:

        df.columns = [c.strip().lower() for c in df.columns]

        if 'is_correct' in df.columns and 'ground_truth' in df.columns:
            df = df.rename(columns={
                'is_correct': 'Is_Correct',
                'ground_truth': 'Ground_Truth'
            })
        else:
            raise KeyError(
                "Could not find Is_Correct or Ground_Truth columns."
            )

    # ---------------- Filter invalid rows ----------------

    filtered_df = df[
        (df["Ground_Truth"] != -1) &
        (df["Is_Correct"] != -1)
    ].copy()

    print(f"Total valid rows: {len(filtered_df)}")

    #########################################################
    # Overall
    #########################################################

    print("\n" + "#"*30)
    print("OVERALL PERFORMANCE")
    print("#"*30)

    print_subset_report(
        filtered_df,
        f"Overall (N={len(filtered_df)})"
    )

    #########################################################
    # Gender
    #########################################################

    print("\n" + "#"*25)
    print("Gender Breakdown")
    print("#"*25)

    for gender in ["male", "female"]:

        subset = filtered_df[
            filtered_df["Gender"].str.lower() == gender
        ]

        print_subset_report(
            subset,
            f"Gender: {gender} (N={len(subset)})"
        )

    #########################################################
    # Race
    #########################################################

    print("\n" + "#"*25)
    print("Race Breakdown")
    print("#"*25)

    for race in ["asian", "white", "black"]:

        subset = filtered_df[
            filtered_df["Race"].str.lower() == race
        ]

        print_subset_report(
            subset,
            f"Race: {race} (N={len(subset)})"
        )

    #########################################################
    # Age Group
    #########################################################

    print("\n" + "#"*25)
    print("Age Group Breakdown")
    print("#"*25)

    age = filtered_df["Age"]

    groups = {
        "young": age < 50,
        "middle": (age >= 50) & (age < 70),
        "older": age >= 70
    }

    for group_name, mask in groups.items():

        subset = filtered_df[mask]

        print_subset_report(
            subset,
            f"Age Group: {group_name} (N={len(subset)})"
        )

    print("\n" + "#"*90)
    
if __name__ == "__main__":
    calculate_metrics(CSV_PATH)
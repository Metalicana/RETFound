import pandas as pd
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score

# --- Define thresholds ---
THRESHOLDS = {
    'AMD_L1': 0.5,
    'AMD_L2': 0.5,
    'AMD_L3': 0.5,
    'DR': 0.5,
    'Glaucoma': 0.5
}

# Load CSV
df = pd.read_csv('ophthalmic_performance_results_apr06.csv')

# --- Map predictions based on Task_Folder ---
def get_relevant_pred(row):
    task = str(row['Task_Folder']).strip()
    if task == 'Glaucoma':
        return row['Pred_GL']
    elif task == 'AMD':
        return row['Pred_AMD']  # we'll handle stages separately
    elif task == 'DR':
        return row['Pred_DR']
    else:
        return None

df['Final_Pred'] = df.apply(get_relevant_pred, axis=1)
df = df.dropna()

# --- AMD multi-stage functions ---
def get_amd_stage(row):
    return row['Ground_Truth']

def get_predicted_amd_stage(row, thresholds):
    return row['Pred_AMD']

# --- Metrics calculation ---
def calculate_metrics(y_true, y_pred):
    """Always use macro averaging."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, acc

# --- Print header ---
#print(f"{'Task Folder':<16} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Acc':<7}")
print(f"{'Task':<16} | {'Prec':<7} | {'Rec':<7} | {'F1':<6} | {'Count':<6} | {'Correct':<7}")
print("-" * 65)

for task in df['Task_Folder'].unique():
    subset = df[df['Task_Folder'] == task]

    if task == 'AMD':
        # Compute AMD stages
        mask = subset['Ground_Truth'] != -1
        if mask.any():
            y_true = subset.loc[mask, 'Ground_Truth']
            y_pred = subset.loc[mask, 'Final_Pred']
#            p, r, f1, acc = calculate_metrics(y_true, y_pred)
#            print(f"{task:<16} | {p:.4f}  | {r:.4f}  | {f1:.4f}  | {acc:.4f}")

            # Per-stage metrics
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            for stage_key in sorted(report.keys()):
                if stage_key.replace('.', '').isdigit():
                    metrics = report[stage_key]
                    stage_label = f"AMD -> Stage {stage_key}"
                    stage_mask = y_true == int(float(stage_key))
                    stage_count = stage_mask.sum()
                    stage_correct = (y_pred[stage_mask] == y_true[stage_mask]).sum()
#                    print(f"{stage_label:<16} | {metrics['precision']:.4f}  | {metrics['recall']:.4f}  | {metrics['f1-score']:.4f}  | {'-':<7}")
                    print(f"{stage_label:<16} | {metrics['precision']:.4f}  | {metrics['recall']:.4f}  | {metrics['f1-score']:.4f} | {stage_count:<6} | {stage_correct:<7}")
    else:
        # Binary tasks using thresholds
        target_col = f'Ground_Truth'
        pred_col = f'Final_Pred'
        mask = subset[target_col] != -1
        if mask.any():
            y_true = subset.loc[mask, target_col]
            y_pred = (subset.loc[mask, pred_col] >= THRESHOLDS[task]).astype(int)
            
            total_count = len(y_true)
            correct_count = (y_true == y_pred).sum()
              
            p, r, f1, acc = calculate_metrics(y_true, y_pred)
#            print(f"{task:<16} | {p:.4f}  | {r:.4f}  | {f1:.4f}  | {acc:.4f}")
            print(f"{task:<16} | {p:.4f}  | {r:.4f}  | {f1:.4f} | {total_count:<6} | {correct_count:<7}")
print("=" * 65)
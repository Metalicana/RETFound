#new metrics table for the generated csv 

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, cohen_kappa_score

# 1. Load your CSV
df = pd.read_csv('ophthalmic_performance_results_may03.csv') 

# 2. Filter for the first 100 examples from each disease folder
df_subset = df.groupby('Task_Folder').head(100).copy()

print(f"\n{'='*25} PRIMARY PERFORMANCE METRICS {'='*25}")
print(f"Evaluating first 100 samples per task. (Total: {len(df_subset)})\n")
print(f"{'Task/Stage':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<6} | {'Support':<8} | {'Correct':<7}")
print("-" * 80)

# 3. Primary Evaluation Loop
for task in ['AMD', 'DR', 'Glaucoma']:
    subset = df_subset[df_subset['Task_Folder'] == task]
    if subset.empty: continue

    pred_col = f'Pred_{"GL" if task == "Glaucoma" else task}' 
    mask = (subset['Ground_Truth'] != -1) & (subset[pred_col] != -1)
    y_true = subset.loc[mask, 'Ground_Truth'].astype(int)
    y_pred = subset.loc[mask, pred_col].astype(int)

    if task == 'AMD':
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        for stage in ['0', '1', '2', '3']:
            s_key = str(float(stage)) if str(float(stage)) in report else stage
            if s_key in report:
                m = report[s_key]
                correct = ((y_true == int(float(stage))) & (y_pred == int(float(stage)))).sum()
                print(f"AMD Stage {stage:<9} | {m['precision']:.4f}  | {m['recall']:.4f}  | {m['f1-score']:.4f} | {int(m['support']):<8} | {correct:<7}")
        print(f"{'AMD Weighted Kappa':<20} | {kappa:.4f} (Quadratic)")
        print("-" * 80)
    else:
        p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        correct = (y_true == y_pred).sum()
        print(f"{task:<20} | {p:.4f}  | {r:.4f}  | {f1:.4f} | {len(y_true):<8} | {correct:<7}")

# 4. EQUITY TABLE: Performance by Race (Sensitivity focuses on not missing disease)
print(f"\n\n{'='*25} EQUITY AUDIT: SENSITIVITY BY Race {'='*25}")
print(f"{'Race':<20} | {'AMD Rec':<8} | {'DR Rec':<8} | {'GL Rec':<8}")
print("-" * 60)

Races = df_subset['Race'].unique()
for Race in sorted(Races):
    Race_results = []
    for task in ['AMD', 'DR', 'Glaucoma']:
        sub = df_subset[(df_subset['Task_Folder'] == task) & (df_subset['Race'] == Race)]
        if sub.empty:
            Race_results.append(0.0)
            continue
        
        pred_col = f'Pred_{"GL" if task == "Glaucoma" else task}'
        # For AMD sensitivity, we treat any Stage > 0 as positive
        y_t = (sub['Ground_Truth'] > 0).astype(int) if task == 'AMD' else sub['Ground_Truth']
        y_p = (sub[pred_col] > 0).astype(int) if task == 'AMD' else sub[pred_col]
        
        # Filter abstentions
        valid = (y_t != -1) & (y_p != -1)
        rec = recall_score(y_t[valid], y_p[valid], pos_label=1, zero_division=0) if any(y_t[valid] == 1) else 1.0
        Race_results.append(rec)
        
    print(f"{Race[:20]:<20} | {Race_results[0]:.4f}   | {Race_results[1]:.4f}   | {Race_results[2]:.4f}")

# 5. DEMOGRAPHIC TABLE: Reliability by Gender
print(f"\n\n{'='*25} CLINICAL RELIABILITY: F1 BY Gender {'='*25}")
print(f"{'Gender':<20} | {'Overall F1 Score (Macro)':<20}")
print("-" * 50)

for Gender in df_subset['Gender'].unique():
    sub = df_subset[df_subset['Gender'] == Gender]
    all_true, all_pred = [], []
    for task in ['AMD', 'DR', 'Glaucoma']:
        pred_col = f'Pred_{"GL" if task == "Glaucoma" else task}'
        v = (sub['Task_Folder'] == task) & (sub['Ground_Truth'] != -1) & (sub[pred_col] != -1)
        all_true.extend(sub.loc[v, 'Ground_Truth'].tolist())
        all_pred.extend(sub.loc[v, pred_col].tolist())
    
    f1_g = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"{Gender:<20} | {f1_g:.4f}")

print("\n" + "="*80)

#import pandas as pd
#import numpy as np
#from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, cohen_kappa_score
#
## 1. Load your CSV
#df = pd.read_csv('ophthalmic_performance_results_apr30.csv') 
#
## 2. Filter for the first 100 examples from each disease folder
#df_subset = df.groupby('Task_Folder').head(100).copy()
#
#print(f"\n{'='*25} PRIMARY PERFORMANCE METRICS {'='*25}")
#print(f"Evaluating first 100 samples per task. (Total: {len(df_subset)})\n")
#print(f"{'Task/Stage':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<6} | {'Support':<8} | {'Correct':<7}")
#print("-" * 85)
#
## 3. Primary Evaluation Loop
#for task in ['AMD', 'DR', 'Glaucoma']:
#    subset = df_subset[df_subset['Task_Folder'] == task]
#    if subset.empty: continue
#
#    pred_col = f'Pred_{"GL" if task == "Glaucoma" else "DR" if task == "DR" else "AMD"}' 
#    mask = (subset['Ground_Truth'] != -1) & (subset[pred_col] != -1)
#    y_true = subset.loc[mask, 'Ground_Truth'].astype(int)
#    y_pred = subset.loc[mask, pred_col].astype(int)
#
#    if task == 'AMD':
#        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
#        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
#        
#        for stage in ['0', '1', '2', '3']:
#            s_key = str(float(stage)) if str(float(stage)) in report else stage
#            if s_key in report:
#                m = report[s_key]
#                correct = ((y_true == int(float(stage))) & (y_pred == int(float(stage)))).sum()
#                print(f"AMD Stage {stage:<9} | {m['precision']:.4f}  | {m['recall']:.4f}  | {m['f1-score']:.4f} | {int(m['support']):<8} | {correct:<7}")
#        print(f"{'AMD Weighted Kappa':<20} | {kappa:.4f} (Quadratic)")
#        print("-" * 85)
#    else:
#        p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
#        r = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
#        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
#        correct = (y_true == y_pred).sum()
#        print(f"{task:<20} | {p:.4f}  | {r:.4f}  | {f1:.4f} | {len(y_true):<8} | {correct:<7}")
#
## 4. EQUITY TABLE: Performance by Race with "Positive Case" support
#print(f"\n\n{'='*25} EQUITY AUDIT: SENSITIVITY BY RACE {'='*25}")
#print(f"{'Race':<15} | {'AMD Rec (pos/total)':<20} | {'DR Rec (pos/total)':<20} | {'GL Rec (pos/total)':<20}")
#print("-" * 85)
#
#def get_recall_str(y_true, y_pred):
#    n_pos = (y_true == 1).sum()
#    n_total = len(y_true)
#    if n_pos == 0:
#        return f"N/A (0/{n_total})"
#    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
#    return f"{rec:.4f} ({int(rec*n_pos)}/{n_pos})"
#
#races = df_subset['Race'].unique()
#for race in sorted(races):
#    row_str = f"{race[:15]:<15} | "
#    for task in ['AMD', 'DR', 'Glaucoma']:
#        sub = df_subset[(df_subset['Task_Folder'] == task) & (df_subset['Race'] == race)]
#        pred_col = f'Pred_{"GL" if task == "Glaucoma" else "DR" if task == "DR" else "AMD"}'
#        
#        mask = (sub['Ground_Truth'] != -1) & (sub[pred_col] != -1)
#        y_t = sub.loc[mask, 'Ground_Truth']
#        y_p = sub.loc[mask, pred_col]
#        
#        # Binarize AMD for "Any Disease" sensitivity
#        if task == 'AMD':
#            y_t = (y_t > 0).astype(int)
#            y_p = (y_p > 0).astype(int)
#            
#        row_str += f"{get_recall_str(y_t, y_p):<20} | "
#    print(row_str.rstrip(" | "))
#
## 5. DEMOGRAPHIC TABLE: Reliability by Gender with Support
#print(f"\n\n{'='*25} CLINICAL RELIABILITY: F1 BY GENDER {'='*25}")
#print(f"{'Gender':<15} | {'Overall F1 (Macro)':<20} | {'Support (n)':<12}")
#print("-" * 55)
#
#for gender in df_subset['Gender'].unique():
#    sub = df_subset[df_subset['Gender'] == gender]
#    all_true, all_pred = [], []
#    for task in ['AMD', 'DR', 'Glaucoma']:
#        pred_col = f'Pred_{"GL" if task == "Glaucoma" else "DR" if task == "DR" else "AMD"}'
#        mask = (sub['Task_Folder'] == task) & (sub['Ground_Truth'] != -1) & (sub[pred_col] != -1)
#        all_true.extend(sub.loc[mask, 'Ground_Truth'].tolist())
#        all_pred.extend(sub.loc[mask, pred_col].tolist())
#    
#    if len(all_true) > 0:
#        f1_g = f1_score(all_true, all_pred, average='macro', zero_division=0)
#        print(f"{gender:<15} | {f1_g:.4f}               | {len(all_true):<12}")
#    else:
#        print(f"{gender:<15} | 0.0000               | 0")
#
#print("\n" + "="*85)
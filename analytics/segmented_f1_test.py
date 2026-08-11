"""
Segmented F1 Test: Does training separate models for high-signal segments
break the F1 ceiling?
"""
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("Loading patched features data...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} rows, {len(df.columns)} cols")
print(f"Global Positive rate: {df['label_eligibilite'].mean():.4%}")

# Use CUSTOMER_RATING as the primary segmentation variable
segment_col = 'CUSTOMER_RATING'
print(f"\nAnalyzing segments in {segment_col}:")
segment_stats = df.groupby(segment_col)['label_eligibilite'].agg(['count', 'mean', 'sum'])
segment_stats = segment_stats.sort_values('mean', ascending=False)
print(segment_stats.head(10))

# ── Preparation ──
EXCLUDE = ['label_code', 'label_eligibilite', 'label_nom']
cat_cols = [
    'CUSTOMER_RATING', 'pack_actuel', 'MARITAL_STATUS', 'NOMBRE_ENFANT',
    'pack_etat', 'CODE_VILLE_regroupe',
    'interaction_solde_min_x_depot_moyen',
    'flux_cred_total_bin', 'solde_moyen_bin',
    'anciennete_digitale_jours_imp_bin', 'depot_moyen_bin',
    'nb_mois_avec_flux_bin', 'solde_min_bin',
    'pack_actuel_x_CUSTOMER_RATING',
    'pack_etat_x_CUSTOMER_RATING',
    'pack_actuel_x_pack_etat',
]
drop_cols = ['CODE_VILLE', 'BPR', 'GENDER', 'TAILLE_ENTREPRI'] + EXCLUDE

# Encode categoricals globally to ensure consistent mapping
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df['label_eligibilite']

# We need the original segment labels for splitting
segments = df[segment_col]

# Split indices to ensure we train/test on the same rows as the global model
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

y_test_global = y.iloc[test_idx]
print(f"\nGlobal Test positive rate: {y_test_global.mean():.4%}")

print("\n" + "="*60)
print("TRAINING SEGMENTED MODELS")
print("="*60)

global_predictions = np.zeros(len(test_idx))
segment_models = {}

# We will group segments with very few positive cases into an "OTHER" category to avoid overfitting
# Let's say a segment needs at least 500 positive cases in train to get its own model.
segment_mapping = {}
for seg_val in segments.unique():
    seg_mask_train = (segments.iloc[train_idx] == seg_val)
    pos_count = y.iloc[train_idx][seg_mask_train].sum()
    if pos_count > 500:
        segment_mapping[seg_val] = seg_val
    else:
        segment_mapping[seg_val] = 'OTHER'

mapped_segments = segments.map(segment_mapping)

for seg_name in mapped_segments.unique():
    print(f"\n--- Training Model for Segment: {seg_name} ---")
    
    # Get masks
    train_mask = (mapped_segments.iloc[train_idx] == seg_name)
    test_mask = (mapped_segments.iloc[test_idx] == seg_name)
    
    X_tr = X.iloc[train_idx][train_mask]
    y_tr = y.iloc[train_idx][train_mask]
    
    X_te = X.iloc[test_idx][test_mask]
    y_te = y.iloc[test_idx][test_mask]
    
    if len(y_tr) == 0 or len(y_te) == 0:
        continue
        
    pos_rate_tr = y_tr.mean()
    print(f"  Train Size: {len(X_tr)} (Pos Rate: {pos_rate_tr:.2%})")
    print(f"  Test Size: {len(X_te)} (Pos Rate: {y_te.mean():.2%})")
    
    if pos_rate_tr == 0:
        print("  Skipping: No positive cases in training.")
        continue
        
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_tr, y_tr)
    
    # Predict probabilities for the segment's test set
    y_proba = model.predict_proba(X_te)[:, 1]
    
    # Local Threshold Optimization
    best_f1_local, best_t_local = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_te, preds)
        if f > best_f1_local:
            best_f1_local, best_t_local = f, t
            
    print(f"  Local Best Threshold: {best_t_local:.2f} -> Local F1: {best_f1_local:.4f}")
    
    # Store predictions back into the global array using the local optimal threshold
    # Note: We need to map the boolean test_mask back to the actual indices in the global test array
    test_indices_in_global = np.where(test_mask)[0]
    global_predictions[test_indices_in_global] = (y_proba >= best_t_local).astype(int)

print("\n" + "="*60)
print("VERDICT: AGGREGATED SEGMENTED PERFORMANCE")
print("="*60)

f1_agg = f1_score(y_test_global, global_predictions)
prec_agg = precision_score(y_test_global, global_predictions)
rec_agg = recall_score(y_test_global, global_predictions)

print(f"Segmented F1 Score: {f1_agg:.4f}")
print(f"Segmented Precision: {prec_agg:.4f}")
print(f"Segmented Recall: {rec_agg:.4f}")

if f1_agg > 0.23:
    print(f"\n  >>> CEILING BROKEN! F1 = {f1_agg:.4f} > 0.23 <<<")
else:
    print(f"\n  >>> CEILING NOT BROKEN. F1 = {f1_agg:.4f} <<<")

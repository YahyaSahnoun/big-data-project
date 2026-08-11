import pandas as pd
import numpy as np
import glob
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Force flush prints immediately
def log(msg):
    print(msg, flush=True)

out_dir = '/tmp'
output_img_path = os.path.join(out_dir, 'unbiased_threshold_metrics.png')

log("Loading data...")
files = sorted(glob.glob('features/*.parquet')) or sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
log(f"Total: {len(df)} rows, {len(df.columns)} cols")

# Subsample if dataset is very large (>200k rows) to keep runtime reasonable
MAX_ROWS = 200000
if len(df) > MAX_ROWS:
    log(f"Subsampling from {len(df)} to {MAX_ROWS} rows (stratified)...")
    df, _ = train_test_split(df, train_size=MAX_ROWS, stratify=df['label_eligibilite'], random_state=42)
    log(f"After subsampling: {len(df)} rows")

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

for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0)
y = df['label_eligibilite']

log("Splitting into Train (60%), Validation (20%), Test (20%)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
log(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
log(f"scale_pos_weight: {scale_pos:.2f}")

model = lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

log("Training LightGBM...")
model.fit(X_train, y_train)
log("Training complete.")

log("Selecting threshold on Validation set...")
y_val_proba = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.05, 0.95, 50)
val_f1_scores = []
for t in thresholds:
    preds = (y_val_proba >= t).astype(int)
    val_f1_scores.append(f1_score(y_val, preds, zero_division=0))

best_idx = np.argmax(val_f1_scores)
best_threshold = thresholds[best_idx]
best_val_f1 = val_f1_scores[best_idx]
log(f"Optimal Threshold (Validation): {best_threshold:.2f}, Val F1: {best_val_f1:.4f}")

log("Evaluating on unseen Test set...")
y_test_proba = model.predict_proba(X_test)[:, 1]
test_preds_best = (y_test_proba >= best_threshold).astype(int)
test_preds_default = (y_test_proba >= 0.50).astype(int)

best_test_f1 = f1_score(y_test, test_preds_best, zero_division=0)
best_test_prec = precision_score(y_test, test_preds_best, zero_division=0)
best_test_rec = recall_score(y_test, test_preds_best, zero_division=0)
default_test_f1 = f1_score(y_test, test_preds_default, zero_division=0)

log(f"=== UNBIASED TEST RESULTS ===")
log(f"Optimal Threshold: {best_threshold:.2f}")
log(f"Test F1:        {best_test_f1:.4f}")
log(f"Test Precision: {best_test_prec:.4f}")
log(f"Test Recall:    {best_test_rec:.4f}")
log(f"Default (0.50) Test F1: {default_test_f1:.4f}")

log("Generating plots...")
# Metrics on test set for visualization
precisions = []
recalls = []
f1s = []
for t in thresholds:
    preds = (y_test_proba >= t).astype(int)
    precisions.append(precision_score(y_test, preds, zero_division=0))
    recalls.append(recall_score(y_test, preds, zero_division=0))
    f1s.append(f1_score(y_test, preds, zero_division=0))

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.plot(thresholds, precisions, label='Precision', color='#3B82F6', linewidth=2.5)
ax1.plot(thresholds, recalls, label='Recall', color='#10B981', linewidth=2.5)
ax1.plot(thresholds, f1s, label='F1-Score', color='#8B5CF6', linewidth=3)
ax1.axvline(x=best_threshold, color='#EF4444', linestyle='--', linewidth=1.5,
            label=f'Optimal (T={best_threshold:.2f})')
ax1.axvline(x=0.50, color='#F59E0B', linestyle=':', linewidth=1.5,
            label='Default (T=0.50)')
ax1.scatter([best_threshold], [best_test_f1], color='#EF4444', s=100, zorder=5)
ax1.annotate(f'Test F1 = {best_test_f1:.4f}\n(T = {best_threshold:.2f})',
             xy=(best_threshold, best_test_f1),
             xytext=(best_threshold + 0.15, best_test_f1 - 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Unbiased Test Metrics vs. Threshold\n(Threshold selected on Validation set)', fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='lower left', frameon=True, shadow=True)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
ax2.plot(recall_curve, precision_curve, color='#4F46E5', linewidth=3, label='LGBM Model')
ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold', pad=15)

specific_ts = [0.80, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05]
colors_ts = ['#DF80AC', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899', '#6B7280']
for t, col in zip(specific_ts, colors_ts):
    p_t = precision_score(y_test, (y_test_proba >= t).astype(int), zero_division=0)
    r_t = recall_score(y_test, (y_test_proba >= t).astype(int), zero_division=0)
    ax2.scatter([r_t], [p_t], color=col, s=80, zorder=5, label=f'T = {t:.2f}')

ax2.legend(loc='lower left', frameon=True, shadow=True)
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, 1.05)

plt.suptitle('Unbiased Threshold Moving Analysis\n(No Data Leakage: Train/Val/Test Split)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_img_path, dpi=200, bbox_inches='tight')
plt.close()

log(f"Saved to {output_img_path}")
log(f"File exists: {os.path.exists(output_img_path)}")
log(f"File size: {os.path.getsize(output_img_path)} bytes")
log("DONE")

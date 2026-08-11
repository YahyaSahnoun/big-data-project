import pandas as pd
import numpy as np
import glob
import os
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

# Define paths
out_dir = '/tmp'
output_img_path = os.path.join(out_dir, 'threshold_moving_metrics.png')

print("Loading data...")
files = sorted(glob.glob('features/*.parquet')) or sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} rows, {len(df.columns)} cols")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

print("Training LightGBM model...")
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics across thresholds
thresholds = np.linspace(0.01, 0.99, 100)
precisions = []
recalls = []
f1_scores = []

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, preds, zero_division=0))
    recalls.append(recall_score(y_test, preds, zero_division=0))
    f1_scores.append(f1_score(y_test, preds, zero_division=0))

precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_prec = precisions[best_idx]
best_rec = recalls[best_idx]

print(f"Optimal Threshold (Max F1): {best_threshold:.2f}")
print(f"F1-Score: {best_f1:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f}")

# Set style
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Metrics vs Thresholds
ax1.plot(thresholds, precisions, label='Precision', color='#3B82F6', linewidth=2.5)
ax1.plot(thresholds, recalls, label='Recall', color='#10B981', linewidth=2.5)
ax1.plot(thresholds, f1_scores, label='F1-Score', color='#8B5CF6', linewidth=3)

# Highlight Best F1 and Default 0.5
ax1.axvline(x=best_threshold, color='#EF4444', linestyle='--', linewidth=1.5,
            label=f'Optimal F1 (T={best_threshold:.2f})')
ax1.axvline(x=0.50, color='#F59E0B', linestyle=':', linewidth=1.5,
            label='Default (T=0.50)')

ax1.scatter([best_threshold], [best_f1], color='#EF4444', s=100, zorder=5)
ax1.annotate(f'Max F1 = {best_f1:.4f}\n(T = {best_threshold:.2f})', 
             xy=(best_threshold, best_f1), 
             xytext=(best_threshold - 0.25, best_f1 - 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Metrics vs. Decision Threshold (Threshold Moving)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower left', frameon=True, shadow=True)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)

# Plot 2: Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
ax2.plot(recall_curve, precision_curve, color='#4F46E5', linewidth=3, label='LGBM Model')
ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)

# Plot points on PR-Curve for specific thresholds
specific_ts = [0.80, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05]
colors_ts = ['#DF80AC', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899', '#6B7280']
for t, col in zip(specific_ts, colors_ts):
    p_t = precision_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    r_t = recall_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    ax2.scatter([r_t], [p_t], color=col, s=80, zorder=5, label=f'Threshold = {t:.2f}')

ax2.legend(loc='lower left', frameon=True, shadow=True)
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, 1.05)

plt.suptitle('Threshold Moving Analysis & Decision Trade-offs', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_img_path, dpi=300)
plt.close()

print(f"Visual representation saved successfully to {output_img_path}")
print("Exists post-save:", os.path.exists(output_img_path))

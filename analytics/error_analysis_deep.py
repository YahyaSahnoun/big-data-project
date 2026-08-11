"""
Deep Error Analysis script to identify data limits.
Outputs CSVs and statistics.
"""
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs('/workspace/error_analysis_results', exist_ok=True)

# 1. LOAD DATA
print("Loading patched features data...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

EXCLUDE = ['label_code', 'label_eligibilite', 'label_nom']
cat_cols = [
    'CUSTOMER_RATING', 'pack_actuel', 'MARITAL_STATUS', 'NOMBRE_ENFANT',
    'pack_etat', 'CODE_VILLE_regroupe',
    'interaction_solde_min_x_depot_moyen',
    'flux_cred_total_bin', 'solde_moyen_bin',
    'anciennete_digitale_jours_imp_bin', 'depot_moyen_bin',
    'nb_mois_avec_flux_bin', 'solde_min_bin',
    'pack_actuel_x_CUSTOMER_RATING', 'pack_etat_x_CUSTOMER_RATING', 'pack_actuel_x_pack_etat',
]
drop_cols = ['CODE_VILLE', 'BPR', 'GENDER', 'TAILLE_ENTREPRI'] + EXCLUDE

for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df['label_eligibilite']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. TRAIN GLOBAL MODEL
print("Training Baseline LightGBM...")
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1
)
model.fit(X_train, y_train)

print("Generating Test Predictions...")
y_proba = model.predict_proba(X_test)[:, 1]

# Choose an optimal threshold (for analysis purposes, we'll pick the one that maxes F1)
from sklearn.metrics import f1_score
best_f1, best_t = 0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    f = f1_score(y_test, (y_proba >= t).astype(int))
    if f > best_f1:
        best_f1, best_t = f, t

y_pred = (y_proba >= best_t).astype(int)
print(f"Optimal Threshold: {best_t:.2f} (F1: {best_f1:.4f})")

# 3. SEPARABILITY & CONFIDENCE
print("\n--- Separability & Confidence ---")
pos_proba = y_proba[y_test == 1]
neg_proba = y_proba[y_test == 0]

# Save probability distributions
prob_df = pd.DataFrame({
    'Probability': y_proba,
    'True_Label': y_test.values
})
prob_df.to_csv('/workspace/error_analysis_results/probability_distribution.csv', index=False)
print("Saved probability distributions to CSV.")

# Define buckets
test_df = X_test.copy()
test_df['True_Label'] = y_test.values
test_df['Predicted_Label'] = y_pred
test_df['Probability'] = y_proba

# Classify
test_df['Category'] = 'TN'
test_df.loc[(test_df['True_Label'] == 1) & (test_df['Predicted_Label'] == 1), 'Category'] = 'TP'
test_df.loc[(test_df['True_Label'] == 1) & (test_df['Predicted_Label'] == 0), 'Category'] = 'FN'
test_df.loc[(test_df['True_Label'] == 0) & (test_df['Predicted_Label'] == 1), 'Category'] = 'FP'

counts = test_df['Category'].value_counts()
print("\nConfusion Matrix Breakdown:")
print(counts)

fn_df = test_df[test_df['Category'] == 'FN']
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
fn_df['Conf_Bucket'] = pd.cut(fn_df['Probability'], bins=bins, labels=labels, right=False)
print("\nFalse Negative Confidence Buckets (Where are we missing them?):")
print(fn_df['Conf_Bucket'].value_counts().sort_index())

# 4. STATISTICAL COMPARISON (Cohen's d: TP vs FN)
print("\n--- Statistical Feature Comparison (TP vs FN) ---")
tp_df = test_df[test_df['Category'] == 'TP']
top_features = [x[0] for x in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20]]

def cohen_d(x, y):
    if len(x) == 0 or len(y) == 0: return np.nan
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_sd == 0: return 0
    return (np.mean(x) - np.mean(y)) / pool_sd

d_results = []
for col in top_features:
    d = cohen_d(tp_df[col].values, fn_df[col].values)
    d_results.append({'Feature': col, 'Cohen_d': abs(d)})

d_df = pd.DataFrame(d_results).sort_values('Cohen_d', ascending=True)
d_df.to_csv('/workspace/error_analysis_results/cohens_d_tp_vs_fn.csv', index=False)
print("Saved Cohen's d results.")
print("\nTop 5 WEAKEST separators (TP vs FN look identical):")
print(d_df.head(5))

# 5. CLUSTERING FALSE NEGATIVES
print("\n--- Clustering False Negatives ---")
# Use a subset of important numeric features for clustering
clust_cols = ['age_client', 'solde_moyen', 'flux_cred_total', 'anciennete_digitale_jours_imp', 'ratio_solde_age']
X_fn_cluster = fn_df[clust_cols].copy()

scaler = StandardScaler()
X_fn_scaled = scaler.fit_transform(X_fn_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
fn_df['Cluster'] = kmeans.fit_predict(X_fn_scaled)

# Calculate centroids in original space
centroids = fn_df.groupby('Cluster')[clust_cols].mean()
centroids['Count'] = fn_df['Cluster'].value_counts()
centroids.to_csv('/workspace/error_analysis_results/fn_cluster_centroids.csv')
print("Saved False Negative Cluster Centroids.")
print(centroids)

# 6. SHAP on FALSE NEGATIVES
print("\n--- SHAP Analysis on False Negatives ---")
try:
    import shap
    # Take a sample of FNs to speed up SHAP
    fn_sample = fn_df.sample(n=min(2000, len(fn_df)), random_state=42)
    X_fn_shap = fn_sample[feature_cols]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fn_shap)
    
    # LightGBM binary classification shap_values might be a list (one for each class)
    # or an array of shape (n_samples, n_features) if it's the raw margin
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] # positive class
    else:
        shap_vals = shap_values
        
    # Calculate mean absolute SHAP values for these FNs
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    # Calculate directional SHAP (are features pushing prediction DOWN towards 0?)
    mean_shap = shap_vals.mean(axis=0)
    
    shap_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP_Direction': mean_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    shap_df.to_csv('/workspace/error_analysis_results/fn_shap_summary.csv', index=False)
    print("Saved SHAP summary for False Negatives.")
    print("\nTop 5 features dragging False Negatives down (Negative Direction):")
    print(shap_df.sort_values('Mean_SHAP_Direction').head(5))
except ImportError:
    print("SHAP library not installed. Skipping SHAP analysis.")

print("\nError Analysis Complete. Artifacts saved in /workspace/error_analysis_results/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import glob
import os

# Create artifacts directory for images
out_dir = '/workspace/error_analysis_results/visuals'
os.makedirs(out_dir, exist_ok=True)

print("Loading data...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# 1. VISUALIZATION
print("\nGenerating Visualizations...")
# Take a random sample for visualization to make it manageable
np.random.seed(42)
pos_df = df[df['label_eligibilite'] == 1].sample(n=min(10000, sum(df['label_eligibilite'] == 1)), random_state=42)
neg_df = df[df['label_eligibilite'] == 0].sample(n=min(10000, sum(df['label_eligibilite'] == 0)), random_state=42)
sample_df = pd.concat([pos_df, neg_df])

# Top features identified earlier
top_features = ['flux_cred_total', 'solde_moyen', 'depot_moyen', 'age_client']

# KDE Plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(data=sample_df, x=feature, hue='label_eligibilite', fill=True, common_norm=False, log_scale=True if feature != 'age_client' else False)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(f'{out_dir}/feature_distributions.png')
plt.close()
print("Saved feature distributions plot.")

# PCA Plot
print("Computing PCA for 2D visualization...")
numeric_cols = sample_df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('label_eligibilite')
if 'label_code' in numeric_cols: numeric_cols.remove('label_code')
if 'label_nom' in numeric_cols: numeric_cols.remove('label_nom')

X_vis = sample_df[numeric_cols].fillna(0)
from sklearn.preprocessing import StandardScaler
X_vis_scaled = StandardScaler().fit_transform(X_vis)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_vis_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=sample_df['label_eligibilite'], alpha=0.5, s=10)
plt.title('PCA 2D Projection of Data (Class 0 vs 1)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.savefig(f'{out_dir}/pca_projection.png')
plt.close()
print("Saved PCA projection plot.")

# 2. SMOTE EXPERIMENT
print("\nRunning SMOTE / Tomek Experiment...")
# Prepare data (subset for speed, but maintaining true distribution for test)
# Encode categoricals quickly
cat_cols = sample_df.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    df[c] = df[c].astype('category')

X = df[numeric_cols].fillna(0)
y = df['label_eligibilite']

# Stratified split to keep true test distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Test Set Pos Rate: {y_test.mean():.4%} (True Distribution)")

# Apply SMOTE to training data only
print("Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.5) # Upsample minority to 50% of majority
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE Train Pos Rate: {y_train_smote.mean():.4%}")

# Train model on SMOTE data
print("Training LightGBM on SMOTE data...")
model_smote = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model_smote.fit(X_train_smote, y_train_smote)

# Evaluate on TRUE distribution test set
y_proba_smote = model_smote.predict_proba(X_test)[:, 1]
best_f1_smote, best_t_smote = 0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    preds = (y_proba_smote >= t).astype(int)
    f = f1_score(y_test, preds)
    if f > best_f1_smote:
        best_f1_smote, best_t_smote = f, t

y_pred = (y_proba_smote >= best_t_smote).astype(int)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"\nSMOTE Results (Evaluated on True Distribution):")
print(f"Best Threshold: {best_t_smote:.2f}")
print(f"F1 Score:       {best_f1_smote:.4f}")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")

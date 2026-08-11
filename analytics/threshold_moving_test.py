import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
# Support running both locally or inside the docker container
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
X = df[feature_cols].fillna(0)
y = df['label_eligibilite']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

print("\nTraining LightGBM model...")
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("THRESHOLD MOVING EXPERIMENT")
print("="*60)
print(f"{'Threshold':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 60)

thresholds = [0.80, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05]

for t in thresholds:
    preds = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    
    print(f"P > {t:.2f}{' (Default)' if t == 0.50 else '':<9} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

print("="*60)
print("Conclusion: Notice how lowering the threshold massively increases Recall")
print("but drops Precision. You can achieve undersampling-like behavior")
print("purely by picking a lower threshold (e.g., P > 0.10 or 0.15).")

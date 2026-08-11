import pandas as pd
import numpy as np
import glob

files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

###############################################
# A. LABEL ANALYSIS
###############################################
print("\n" + "="*60)
print("A. LABEL ANALYSIS")
print("="*60)
print(df['label_eligibilite'].value_counts())
print(f"Positive rate: {df['label_eligibilite'].mean():.4%}")

print("\nLabel vs label_nom cross-tab:")
cross = df.groupby(['label_eligibilite', 'label_nom']).size().reset_index(name='count')
print(cross.to_string())

print("\nLabel vs label_code cross-tab:")
cross2 = df.groupby(['label_eligibilite', 'label_code']).size().reset_index(name='count')
print(cross2.to_string())

###############################################
# B. ZERO-DOMINATED FEATURES (massive signal loss)
###############################################
print("\n" + "="*60)
print("B. ZERO-DOMINATED FEATURES")
print("="*60)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    zero_pct = (df[c] == 0).mean()
    if zero_pct > 0.50:
        # Check if the feature is useful AMONG non-zero values
        non_zero = df[df[c] != 0]
        if len(non_zero) > 100:
            rate_nonzero = non_zero['label_eligibilite'].mean()
            rate_zero = df[df[c] == 0]['label_eligibilite'].mean()
            print(f"  {c:45s}: {zero_pct:.1%} zeros | elig rate: zero={rate_zero:.3%} nonzero={rate_nonzero:.3%} | LIFT={rate_nonzero/max(rate_zero,0.0001):.2f}x")

###############################################
# C. CONSTANT / NEAR-CONSTANT FEATURES
###############################################
print("\n" + "="*60)
print("C. CONSTANT / NEAR-CONSTANT")
print("="*60)
for c in df.columns:
    nuniq = df[c].nunique(dropna=False)
    if nuniq <= 3:
        print(f"  {c}: {nuniq} unique values -> {df[c].value_counts(dropna=False).head(5).to_dict()}")

###############################################
# D. FEATURE SEPARATION POWER (quick IV proxy)
###############################################
print("\n" + "="*60)
print("D. TOP FEATURE SEPARATION POWER (Eligible vs Non-eligible means)")
print("="*60)
result = []
for c in num_cols:
    if c == 'label_eligibilite':
        continue
    mean_0 = df[df['label_eligibilite']==0][c].mean()
    mean_1 = df[df['label_eligibilite']==1][c].mean()
    std_all = df[c].std()
    if std_all > 0:
        sep = abs(mean_1 - mean_0) / std_all
    else:
        sep = 0
    result.append((c, sep, mean_0, mean_1))

result.sort(key=lambda x: x[1], reverse=True)
print(f"{'Feature':45s} {'Separation':>10s} {'Mean(0)':>12s} {'Mean(1)':>12s}")
for name, sep, m0, m1 in result[:25]:
    print(f"  {name:43s} {sep:10.4f} {m0:12.2f} {m1:12.2f}")

###############################################
# E. CHECK IF PIPELINE LABEL_NOM IS EXCLUDED
###############################################
print("\n" + "="*60)
print("E. PIPELINE EXCLUSION CHECK")
print("="*60)
# These are in COLS_A_EXCLURE_DES_FEATURES from the pipeline
excluded = ["label_code", "label_eligibilite", "label_nom", "RADICAL"]
for c in excluded:
    if c in df.columns:
        print(f"  {c}: PRESENT in dataset (should be excluded by pipeline)")
    else:
        print(f"  {c}: not in dataset")

# Check if RADICAL is present (it shouldn't be)
print(f"\n  RADICAL in columns: {'RADICAL' in df.columns}")

###############################################
# F. CHECK WHAT THE PIPELINE ACTUALLY SEES
###############################################
print("\n" + "="*60)
print("F. PIPELINE FEATURE DIMENSIONS")
print("="*60)
# Reproduce the pipeline's colonnes_features_numeriques logic
COLS_CATEGORIELLES_BASSE_CARDINALITE = [
    "CUSTOMER_RATING", "pack_actuel", "MARITAL_STATUS", "NOMBRE_ENFANT", "pack_etat",
]
COL_HAUTE_CARDINALITE = "CODE_VILLE_regroupe"
COLS_A_EXCLURE_DES_FEATURES = ["label_code", "label_eligibilite", "label_nom", "RADICAL"]

cols_categorielles_brutes = set(COLS_CATEGORIELLES_BASSE_CARDINALITE + [COL_HAUTE_CARDINALITE])

pipeline_num_features = [
    c for c, t in zip(df.columns, df.dtypes)
    if t in ('int32', 'int64', 'float64')
    and c not in COLS_A_EXCLURE_DES_FEATURES
    and c not in cols_categorielles_brutes
]
print(f"Numeric features the Spark pipeline would use: {len(pipeline_num_features)}")
print(pipeline_num_features)

# Check what categoricals/bins the pipeline uses vs what's in the new features dataset
print(f"\nCategorical columns in features dataset:")
cat_cols_in_data = [c for c in df.columns if df[c].dtype == 'object']
print(cat_cols_in_data)

# Check which categorical columns from the FEATURES dataset are NOT covered by the pipeline config
pipeline_cat_covered = set(COLS_CATEGORIELLES_BASSE_CARDINALITE + [COL_HAUTE_CARDINALITE])
COLONNES_BINNEES = [
    "flux_cred_total_bin", "flux_cred_total_etait_extreme_bin", "solde_moyen_bin",
    "flux_cred_moyen_etait_extreme_bin", "anciennete_digitale_jours_imp_bin", "depot_moyen_bin",
    "nb_mois_avec_flux_bin", "nb_mois_observes_solde_bin", "solde_min_bin",
]
COLONNE_INTERACTION = "interaction_solde_min_x_depot_moyen"
pipeline_cat_covered |= set(COLONNES_BINNEES) | {COLONNE_INTERACTION}
# Exclude identifiers
identifiers = set(["label_code", "label_nom", "BPR", "CODE_VILLE", "GENDER", "TAILLE_ENTREPRI"])

cat_not_covered = [c for c in cat_cols_in_data if c not in pipeline_cat_covered and c not in identifiers]
if cat_not_covered:
    print(f"\nCATEGORICAL COLUMNS IN DATA BUT *NOT* USED BY PIPELINE:")
    for c in cat_not_covered:
        print(f"  {c}: {df[c].nunique()} unique values")
        top_rates = df.groupby(c)['label_eligibilite'].mean().sort_values(ascending=False).head(5)
        print(f"    Top 5 by eligibility rate: {top_rates.to_dict()}")
else:
    print("\nAll categorical columns are covered by the pipeline config.")

import pandas as pd
import numpy as np
import glob

files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Compare features vs dataset_final to see what columns differ
files2 = sorted(glob.glob('/workspace/dataset final parquet/*.parquet'))
dfs2 = [pd.read_parquet(f) for f in files2]
df_base = pd.concat(dfs2, ignore_index=True)

print("=== 11. SCHEMA DIFFERENCE: features vs dataset_final ===")
cols_features = set(df.columns)
cols_base = set(df_base.columns)
print(f"Columns only in features: {cols_features - cols_base}")
print(f"Columns only in dataset_final: {cols_base - cols_features}")

print("\n=== 12. ROW COUNT DIFFERENCE ===")
print(f"Features: {len(df)}")
print(f"Dataset final: {len(df_base)}")

print("\n=== 13. LABEL DISTRIBUTION COMPARISON ===")
print("Features:")
print(df['label_eligibilite'].value_counts().to_dict())
print(f"Positive rate: {df['label_eligibilite'].mean():.4%}")
print("Dataset final:")
print(df_base['label_eligibilite'].value_counts().to_dict())
print(f"Positive rate: {df_base['label_eligibilite'].mean():.4%}")

# Check what percentage of eligible customers have None products
print("\n=== 14. ELIGIBLE CUSTOMERS WITH NO PRODUCT INFO ===")
elig_features = df[df['label_eligibilite'] == 1]
none_product_count = elig_features[elig_features['label_nom'].isna() | (elig_features['label_nom'] == 'None') | (elig_features['label_nom'] == '')].shape[0]
print(f"Eligible with label_nom=None: {none_product_count} / {len(elig_features)} ({none_product_count/len(elig_features):.2%})")

none_code = elig_features[elig_features['label_code'].isna() | (elig_features['label_code'] == 'None') | (elig_features['label_code'] == '')].shape[0]
print(f"Eligible with label_code=None: {none_code} / {len(elig_features)} ({none_code/len(elig_features):.2%})")

print("\n=== 15. NON-ELIGIBLE WITH PRODUCT INFO (LEAKAGE CHECK) ===")
non_elig = df[df['label_eligibilite'] == 0]
has_product = non_elig[~(non_elig['label_nom'].isna() | (non_elig['label_nom'] == 'None') | (non_elig['label_nom'] == ''))].shape[0]
print(f"Non-eligible with label_nom != None: {has_product} / {len(non_elig)} ({has_product/len(non_elig):.2%})")

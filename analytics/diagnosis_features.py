import pandas as pd
import numpy as np
import glob

files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("=== 6. FEATURE QUALITY CHECK: Constants, Near-constants, Inf/NaN ===")
for c in df.columns:
    nuniq = df[c].nunique(dropna=False)
    na_count = df[c].isna().sum()
    if nuniq <= 2:
        print(f"  {c}: nunique={nuniq}, na={na_count}, values={df[c].value_counts(dropna=False).to_dict()}")

print("\n=== 7. NUMERIC FEATURE STATISTICS (new features only) ===")
new_num = ['ratio_retraits_flux_mois', 'ratio_flux_mois', 'ratio_volatilite_solde', 'ratio_solde_age']
print(df[new_num].describe().to_string())

print("\n=== 8. INF VALUES CHECK ===")
num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    n_inf = np.isinf(df[c]).sum()
    if n_inf > 0:
        print(f"  {c}: {n_inf} inf values")

print("\n=== 9. ZERO-DOMINANT FEATURES ===")
for c in num_cols:
    zero_pct = (df[c] == 0).mean()
    if zero_pct > 0.80:
        print(f"  {c}: {zero_pct:.2%} zeros")

print("\n=== 10. CROSS-FEATURE CARDINALITIES ===")
cat_crosses = ['pack_actuel_x_CUSTOMER_RATING', 'pack_etat_x_CUSTOMER_RATING', 'pack_actuel_x_pack_etat']
for c in cat_crosses:
    print(f"  {c}: {df[c].nunique()} unique values")
    top5 = df.groupby(c)['label_eligibilite'].agg(['count','mean']).sort_values('mean', ascending=False).head(5)
    print(f"    Top 5 by positive rate:")
    print(top5.to_string())
    print()

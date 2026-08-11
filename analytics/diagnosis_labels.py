import pandas as pd
import numpy as np
import glob

# Load ALL features data
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n=== 1. LABEL ANALYSIS ===")
print(df['label_eligibilite'].value_counts())
print(f"Positive rate: {df['label_eligibilite'].mean():.4%}")

print("\n=== 2. LABEL vs LABEL_NOM (checking consistency) ===")
cross = df.groupby(['label_eligibilite', 'label_nom']).size().reset_index(name='count')
print(cross.to_string())

print("\n=== 3. LABEL vs LABEL_CODE ===")
cross2 = df.groupby(['label_eligibilite', 'label_code']).size().reset_index(name='count')
print(cross2.to_string())

print("\n=== 4. Distribution of label_nom among ELIGIBLE (label=1) ===")
elig = df[df['label_eligibilite'] == 1]
print(elig['label_nom'].value_counts())
print(f"Total eligible: {len(elig)}")

print("\n=== 5. Rows where label_eligibilite=1 but label_nom is None ===")
weird = df[(df['label_eligibilite'] == 1) & (df['label_nom'].isna() | (df['label_nom'] == 'None'))]
print(f"Count: {len(weird)}")

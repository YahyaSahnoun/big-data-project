import glob, pandas as pd
files = sorted(glob.glob('/workspace/features/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files])
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(df['label_eligibilite'].value_counts())

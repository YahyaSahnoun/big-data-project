import pandas as pd
import numpy as np
import glob
import os

print("Starting Parquet Patching...")
feature_files = sorted(glob.glob('/workspace/features/*.parquet'))

for file in feature_files:
    print(f"Processing {file}...")
    df = pd.read_parquet(file)
    
    # Calculate the new ratio_volatilite_solde
    # The original buggy formula: ("solde_volatilite_indefinie", "solde_moyen")
    # New corrected formula: ("solde_volatilite_relative_imp", "solde_moyen")
    
    if 'solde_volatilite_relative_imp' in df.columns and 'solde_moyen' in df.columns:
        df['ratio_volatilite_solde'] = np.where(
            df['solde_moyen'] > 0, 
            df['solde_volatilite_relative_imp'] / df['solde_moyen'], 
            0.0
        )
        print("  - ratio_volatilite_solde calculated.")
    else:
        print("  - WARNING: Required columns not found for ratio_volatilite_solde.")
        
    # Overwrite the parquet file
    df.to_parquet(file, index=False)
    print(f"  - Saved patched file to {file}")

print("Patching complete.")

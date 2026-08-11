"""
diagnose_dataset_gap.py
========================
Investigates WHY dataset_eligibilite gave F1=0.26 vs dataset_eligibilite_final's
0.22-0.25, before you commit to using it. Three things get checked, in order:

  A) Population check      -- same clients? same label rate? any duplicate RADICAL?
  B) Column/leakage check  -- any target-derived columns present as features?
                               (label_code, label_nom leak the answer almost directly)
                               how different are the winsorized columns between the two?
  C) Fair controlled test  -- SAME RADICAL-grouped split, SAME minimal features,
                               SAME model/hyperparams, trained separately on each
                               dataset. If the gap survives this, it's probably real
                               signal. If it evaporates, it was an artifact.

USAGE (WSL / bash):
  pip install s3fs pyarrow pandas lightgbm scikit-learn --break-system-packages

  export MINIO_ENDPOINT="http://localhost:9000"   # or http://minio:9000 if run inside the docker network
  export MINIO_ACCESS_KEY="minioadmin"
  export MINIO_SECRET_KEY="minioadmin123"

  python3 diagnose_dataset_gap.py \
      --path_a s3a://processed-data/dataset_eligibilite \
      --path_b s3a://processed-data/dataset_eligibilite_final \
      --label_col label_eligibilite \
      --sample_frac 1.0
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_recall_curve

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# columns that are structurally derived from / define the target -- NEVER features
TARGET_LEAK_COLS = {"label_eligibilite", "label_code", "label_nom"}

# raw identifier / non-predictive columns per your report (section 6.2, 6.6)
ID_COLS = {"RADICAL", "BANQUE", "AGENCE", "GENERIC", "PLURAL", "CCLE",
           "DATE_OF_BIRTH", "LIBELLE_VILLE", "digital_date_activation",
           "derniere_operation_gab"}

WINSORIZED_COLS = ["solde_moyen", "solde_min", "solde_max", "depot_moyen",
                    "flux_cred_moyen", "flux_cred_total", "montant_total_gab",
                    "montant_moyen_gab", "montant_total_retraits",
                    "montant_total_payfac", "montant_total_vignette",
                    "nb_mois_observes_solde"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_s3fs():
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
    return pafs.S3FileSystem(
        endpoint_override=endpoint.replace("http://", "").replace("https://", ""),
        access_key=key, secret_key=secret,
        scheme="http" if endpoint.startswith("http://") else "https",
    )


def load_dataset(path, s3, sample_frac=1.0):
    """path like s3a://bucket/prefix or s3://bucket/prefix"""
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    dataset = ds.dataset(bucket_path, filesystem=s3, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED)
    return df


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
def check_population(name, df, label_col):
    log(f"[{name}] rows={len(df):,} cols={df.shape[1]}")
    if "RADICAL" in df.columns:
        n_dupe = df["RADICAL"].duplicated().sum()
        log(f"[{name}] distinct RADICAL={df['RADICAL'].nunique():,}  "
            f"duplicated RADICAL rows={n_dupe:,}"
            + ("  <-- LEAKAGE RISK if unsplit-aware" if n_dupe > 0 else ""))
    pos_rate = df[label_col].mean()
    log(f"[{name}] label positive rate = {pos_rate:.4%}  (n_pos={int(df[label_col].sum()):,})")
    return {"n_rows": len(df), "n_dupe_radical": int(df["RADICAL"].duplicated().sum())
            if "RADICAL" in df.columns else None, "pos_rate": pos_rate}


def check_columns(name, df):
    leak_present = TARGET_LEAK_COLS.intersection(df.columns) - {"label_eligibilite"}
    if leak_present:
        log(f"[{name}] !!! TARGET-DERIVED COLUMNS PRESENT AS RAW COLUMNS: {leak_present} "
            f"-- these must be EXCLUDED from any feature set, confirm they were not "
            f"one-hot-encoded into the model your friend trained.")
    else:
        log(f"[{name}] OK: no target-derived columns beyond label_eligibilite itself.")


def compare_winsorized_columns(df_a, name_a, df_b, name_b):
    section("B. Winsorized-column comparison (range/spread) -- did cleaning remove signal or noise?")
    rows = []
    for col in WINSORIZED_COLS:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        a = df_a[col].dropna()
        b = df_b[col].dropna()
        rows.append({
            "column": col,
            f"{name_a}_min": a.min(), f"{name_a}_max": a.max(), f"{name_a}_std": a.std(),
            f"{name_b}_min": b.min(), f"{name_b}_max": b.max(), f"{name_b}_std": b.std(),
        })
    comp = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print(comp.to_string(index=False))
    return comp


def population_overlap(df_a, name_a, df_b, name_b):
    if "RADICAL" not in df_a.columns or "RADICAL" not in df_b.columns:
        log("RADICAL not present in both -- skipping overlap check")
        return
    set_a, set_b = set(df_a["RADICAL"]), set(df_b["RADICAL"])
    inter = set_a & set_b
    log(f"RADICAL overlap: {name_a}={len(set_a):,}  {name_b}={len(set_b):,}  "
        f"intersection={len(inter):,} "
        f"({len(inter)/max(len(set_a),1)*100:.1f}% of {name_a} also in {name_b})")
    if len(inter) < 0.95 * min(len(set_a), len(set_b)):
        log("  -> Populations differ meaningfully. The two datasets are NOT just "
            "cleaned/uncleaned versions of the same rows -- treat any F1 comparison "
            "between them with caution, they may represent different label definitions "
            "or filtering criteria.")


# ----------------------------------------------------------------------
def build_minimal_features(df, label_col):
    """Minimal, leakage-safe feature prep shared identically across both datasets."""
    drop_cols = TARGET_LEAK_COLS | ID_COLS
    drop_cols = {c for c in drop_cols if c != label_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = X.pop(label_col)
    groups = df["RADICAL"] if "RADICAL" in df.columns else pd.Series(np.arange(len(df)))

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(-999)

    return X, y.astype(int), groups, cat_cols


def fair_controlled_test(name, df, label_col):
    section(f"C. Fair controlled test -- {name}")
    from lightgbm import LGBMClassifier

    X, y, groups, cat_cols = build_minimal_features(df, label_col)
    log(f"[{name}] using {X.shape[1]} raw columns as features "
        f"({len(cat_cols)} categorical, {X.shape[1]-len(cat_cols)} numeric), "
        f"groups by RADICAL to prevent client leakage across split")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    fit_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
    y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    tune_idx, final_idx = next(gss2.split(X_val, y_val, groups=groups.iloc[val_idx]))
    X_val_tune, X_val_final = X_val.iloc[tune_idx], X_val.iloc[final_idx]
    y_val_tune, y_val_final = y_val.iloc[tune_idx], y_val.iloc[final_idx]

    scale_pos = np.sqrt((len(y_fit) - y_fit.sum()) / y_fit.sum())
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=50, scale_pos_weight=scale_pos,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=2, verbose=-1
    )
    model.fit(X_fit, y_fit, categorical_feature=cat_cols)

    probas_tune = model.predict_proba(X_val_tune)[:, 1]
    probas_final = model.predict_proba(X_val_final)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val_tune, probas_tune)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_thr = thresholds[np.argmax(f1s[:-1])]

    preds_final = (probas_final >= best_thr).astype(int)
    f1_final = f1_score(y_val_final, preds_final)
    log(f"[{name}] RESULT: threshold={best_thr:.4f}  F1(final, group-split, "
        f"leak-safe)={f1_final:.4f}")
    return f1_final


# ----------------------------------------------------------------------
def main(args):
    s3 = get_s3fs()

    section(f"Loading datasets (sample_frac={args.sample_frac})")
    df_a = load_dataset(args.path_a, s3, args.sample_frac)
    name_a = args.path_a.rstrip("/").split("/")[-1]
    df_b = load_dataset(args.path_b, s3, args.sample_frac)
    name_b = args.path_b.rstrip("/").split("/")[-1]

    section("A. Population checks")
    check_population(name_a, df_a, args.label_col)
    check_population(name_b, df_b, args.label_col)
    population_overlap(df_a, name_a, df_b, name_b)

    section("A2. Target-derived column check")
    check_columns(name_a, df_a)
    check_columns(name_b, df_b)

    compare_winsorized_columns(df_a, name_a, df_b, name_b)

    f1_a = fair_controlled_test(name_a, df_a, args.label_col)
    f1_b = fair_controlled_test(name_b, df_b, args.label_col)

    section("VERDICT")
    log(f"{name_a}: F1={f1_a:.4f}")
    log(f"{name_b}: F1={f1_b:.4f}")
    gap = f1_a - f1_b
    if abs(gap) < 0.01:
        log("Gap is small (<0.01) under a leak-safe, apples-to-apples protocol -- "
            "the 0.26 your friend saw was very likely an artifact of their pipeline "
            "(split, feature list, or unwinsorized noise), not real signal from "
            "dataset_eligibilite itself.")
    elif gap > 0:
        log(f"{name_a} genuinely outperforms {name_b} by {gap:.4f} even under this "
            "controlled test. Worth investigating further -- but note this test used "
            "RAW columns only (no engineered features_v3), so re-run your actual EDA "
            "feature pipeline on both before concluding dataset_eligibilite should "
            "replace dataset_eligibilite_final for production.")
    else:
        log(f"{name_b} (final) is equal or better under this controlled test -- "
            "the earlier 0.26 result likely doesn't replicate once the split is made "
            "leakage-safe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_a", required=True, help="s3a://bucket/dataset_eligibilite")
    parser.add_argument("--path_b", required=True, help="s3a://bucket/dataset_eligibilite_final")
    parser.add_argument("--label_col", default="label_eligibilite")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                         help="Use <1.0 (e.g. 0.3) for a faster first pass on 2 cores")
    args = parser.parse_args()
    main(args)

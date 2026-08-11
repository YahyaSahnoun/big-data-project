"""
build_clean_test_dataset.py
=============================
Fixes the two data-quality issues found in dataset_eligibilite and writes a
cleaned copy to MinIO as "dataset_eligibilite_test", then trains the SAME
leak-safe model on both the original and the cleaned version so you can see
whether the fix actually moves F1 or not.

FIX 1 -- "0/0 division" fake-NaNs (e.g. montant_moyen_gab):
  For each *_moyen_gab / *_moyen_* style ratio column, if its corresponding
  *_total_* column is 0 (i.e. zero activity, not missing data), the NaN
  average gets set to 0 instead of being left for a downstream imputer to
  fill with something misleading like the population median.
  A companion flag column *_had_zero_activity is added so the model can still
  distinguish "genuinely inactive" from "had activity but average is small".

FIX 2 -- repeated-sentinel values in solde_moyen / solde_min / solde_max:
  Detects numeric values that repeat FAR more often than plausible for a
  continuous financial variable (statistical outlier in the *frequency*
  distribution itself, not the value distribution). These get replaced with
  NaN (true missing) and then imputed with the TRAIN-ONLY median -- never
  the population statistic, to avoid the leakage your report already flagged
  as a risk (section 6.5.4). A *_was_sentinel flag column is kept.

USAGE (WSL / bash):
  pip install s3fs pyarrow pandas scikit-learn lightgbm --break-system-packages

  export MINIO_ENDPOINT="http://localhost:9000"
  export MINIO_ACCESS_KEY="minioadmin"
  export MINIO_SECRET_KEY="minioadmin123"

  python3 build_clean_test_dataset.py \
      --source_path s3a://processed-data/dataset_eligibilite \
      --dest_path s3a://processed-data/dataset_eligibilite_test \
      --label_col label_eligibilite \
      --sample_frac 1.0
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_recall_curve

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

TARGET_LEAK_COLS = {"label_eligibilite", "label_code", "label_nom"}
ID_COLS = {"RADICAL", "BANQUE", "AGENCE", "GENERIC", "PLURAL", "CCLE",
           "DATE_OF_BIRTH", "LIBELLE_VILLE", "digital_date_activation",
           "derniere_operation_gab"}

# ratio_column -> the "total"/count column that, when 0, explains a 0/0 NaN
RATIO_TO_TOTAL = {
    "montant_moyen_gab": "montant_total_gab",
    "depot_moyen": None,  # genuinely no matching total col in this schema; left for review
}

SENTINEL_CANDIDATE_COLS = ["solde_moyen", "solde_min", "solde_max"]
# a value's frequency must exceed this multiple of the column's median
# non-unique frequency to be flagged as a likely sentinel
SENTINEL_FREQ_MULTIPLIER = 15
SENTINEL_MIN_COUNT = 200  # never flag a value repeating fewer than this many times


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


def load_dataset(path, s3):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    dataset = ds.dataset(bucket_path, filesystem=s3, format="parquet")
    return dataset.to_table().to_pandas()


def write_dataset(df, path, s3):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    table = pa.Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(table, bucket_path, filesystem=s3, format="parquet",
                      existing_data_behavior="overwrite_or_ignore")


# ----------------------------------------------------------------------
def fix_zero_division_nans(df):
    log("FIX 1: zero-division fake-NaNs...")
    for ratio_col, total_col in RATIO_TO_TOTAL.items():
        if ratio_col not in df.columns:
            continue
        n_nan_before = df[ratio_col].isna().sum()
        if total_col is not None and total_col in df.columns:
            zero_activity_mask = (df[total_col] == 0) & df[ratio_col].isna()
            df[ratio_col] = df[ratio_col].fillna(
                pd.Series(np.where(zero_activity_mask, 0.0, np.nan), index=df.index)
            )
            flag_col = f"{ratio_col}_had_zero_activity"
            df[flag_col] = zero_activity_mask.astype(int)
            n_fixed = zero_activity_mask.sum()
            n_nan_after = df[ratio_col].isna().sum()
            log(f"  {ratio_col}: {n_nan_before} NaN -> {n_fixed} explained by "
                f"{total_col}==0 and set to 0.0, {n_nan_after} genuine NaN remain "
                f"(left for train-only median imputation)")
        else:
            log(f"  {ratio_col}: no matching total column configured -- SKIPPED, "
                f"{n_nan_before} NaN left as-is. Review manually.")
    return df


def fix_sentinel_values(df, fit_mask):
    """fit_mask marks which rows count as 'train' for computing the median used
    to replace flagged sentinels -- never use validation/test rows for this."""
    log("FIX 2: repeated-sentinel detection in solde_* columns...")
    for col in SENTINEL_CANDIDATE_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        vc_multi = vc[vc > 1]
        if len(vc_multi) == 0:
            continue
        median_repeat_count = vc_multi.median()
        threshold = max(SENTINEL_MIN_COUNT, median_repeat_count * SENTINEL_FREQ_MULTIPLIER)
        # exclude 0.0: a genuinely common, meaningful value (e.g. min/max balance
        # legitimately hitting zero), not a placeholder -- don't treat it as suspicious
        suspicious_values = [v for v in vc[vc >= threshold].index.tolist() if v != 0]

        if not suspicious_values:
            log(f"  {col}: no values repeat >= {threshold:.0f}x -- nothing flagged")
            continue

        mask = df[col].isin(suspicious_values)
        n_flagged = mask.sum()
        log(f"  {col}: flagged {len(suspicious_values)} suspicious value(s) "
            f"{suspicious_values[:5]}{'...' if len(suspicious_values) > 5 else ''} "
            f"covering {n_flagged} rows ({n_flagged/len(df)*100:.2f}%)")

        flag_col = f"{col}_was_sentinel"
        df[flag_col] = mask.astype(int)

        train_median = df.loc[fit_mask & ~mask, col].median()
        df.loc[mask, col] = train_median
        log(f"    -> replaced with train-only median ({train_median:,.2f})")
    return df


# ----------------------------------------------------------------------
def build_minimal_features(df, label_col):
    drop_cols = (TARGET_LEAK_COLS | ID_COLS) - {label_col}
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
    from lightgbm import LGBMClassifier

    log(f"\n--- Fair controlled test: {name} ---")
    X, y, groups, cat_cols = build_minimal_features(df, label_col)

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
    log(f"[{name}] threshold={best_thr:.4f}  F1(final, group-split, leak-safe)={f1_final:.4f}")
    return f1_final


# ----------------------------------------------------------------------
def main(args):
    s3 = get_s3fs()

    log(f"Loading source dataset: {args.source_path}")
    df_orig = load_dataset(args.source_path, s3)
    if args.sample_frac < 1.0:
        df_orig = df_orig.sample(frac=args.sample_frac, random_state=RANDOM_SEED).reset_index(drop=True)
    log(f"  shape={df_orig.shape}")

    df_clean = df_orig.copy()

    df_clean = fix_zero_division_nans(df_clean)

    # fit_mask: use a stable, non-leaking 70% split by RADICAL if available,
    # purely to decide which rows are allowed to inform the sentinel-replacement
    # median. This is NOT the same split used for model evaluation below.
    if "RADICAL" in df_clean.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
        fit_idx, _ = next(gss.split(df_clean, groups=df_clean["RADICAL"]))
        fit_mask = pd.Series(False, index=df_clean.index)
        fit_mask.iloc[fit_idx] = True
    else:
        fit_mask = pd.Series(True, index=df_clean.index)

    df_clean = fix_sentinel_values(df_clean, fit_mask)

    log(f"\nWriting cleaned dataset to {args.dest_path} ...")
    write_dataset(df_clean, args.dest_path, s3)
    log("  done.")

    log("\n=== A/B TEST: original vs cleaned ===")
    f1_orig = fair_controlled_test("original (dataset_eligibilite)", df_orig, args.label_col)
    f1_clean = fair_controlled_test("cleaned (dataset_eligibilite_test)", df_clean, args.label_col)

    log("\n=== VERDICT ===")
    log(f"original: F1={f1_orig:.4f}")
    log(f"cleaned : F1={f1_clean:.4f}")
    gap = f1_clean - f1_orig
    if gap > 0.01:
        log(f"Cleaning improved F1 by {gap:.4f} -- the fake-null / sentinel fixes "
            f"were masking real signal. Worth propagating this fix upstream into "
            f"clean_dataset.py itself, not just this test copy.")
    elif gap < -0.01:
        log(f"Cleaning DECREASED F1 by {abs(gap):.4f} -- this can happen if the model "
            f"was inadvertently exploiting the sentinel/NaN pattern as a proxy signal "
            f"(e.g. 'has a fake-looking solde value' correlating with some segment). "
            f"That would mean the original F1 was partly an artifact too -- worth "
            f"treating the cleaned number as the more honest one even though it's lower.")
    else:
        log("No meaningful difference (<0.01) -- these particular data-quality issues "
            "don't appear to be a material driver of your F1 ceiling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True,
                         help="s3a://processed-data/dataset_eligibilite")
    parser.add_argument("--dest_path", required=True,
                         help="s3a://processed-data/dataset_eligibilite_test")
    parser.add_argument("--label_col", default="label_eligibilite")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()
    main(args)

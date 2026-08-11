"""
test_cleaned_dataset_battery.py  (v2 -- fixes two bugs from the first run)
============================================================================
FIX 1 (confidence bins): the first version hardcoded bins assuming a
threshold near 0.75 (chapter 7's original run). This run's threshold was
0.37, so the fixed 0.5/0.7 bins were guaranteed empty -- not a real finding,
just a mismatch between the bin edges and this run's actual threshold. Bins
are now defined as FRACTIONS of the threshold actually found by this run
(0.25x, 0.5x, 0.75x, 1.0x), so they're meaningful regardless of where the
threshold lands.

FIX 2 (invalid 6/25 vs 63/66 comparison): those numbers came from two
different feature sets (this script's ~25-30 minimal raw columns vs.
chapter 7's ~66 engineered columns) and only one dataset (cleaned) was ever
run through the diagnostic -- there was no within-script baseline. This
version runs the SAME deep error analysis, with the SAME minimal feature
set, on BOTH the original and cleaned datasets, so "does cleaning reduce
indistinguishability" is finally a real, controlled comparison instead of
a comparison against a different chapter's different feature set.

Everything else (leak-safe GroupShuffleSplit by RADICAL, threshold tuned on
a held-out slice, F1 measured on a disjoint final slice, XGBoost as a
second algorithm family, the 2-model ensemble) is unchanged from the run
that already produced the trustworthy "+0.0005, cleaning doesn't move F1"
result -- that result stands and does not need to be re-run.

USAGE:
  pip install s3fs pyarrow pandas scikit-learn lightgbm xgboost --break-system-packages

  export MINIO_ENDPOINT="http://localhost:9000"
  export MINIO_ACCESS_KEY="minioadmin"
  export MINIO_SECRET_KEY="minioadmin123"

  python3 -u test_cleaned_dataset_battery.py \
      --original_path s3a://processed-data/dataset_eligibilite \
      --clean_path s3a://processed-data/dataset_eligibilite_test \
      --label_col label_eligibilite \
      --out_json battery_results_v2.json 2>&1 | tee battery_run_v2.log
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

TARGET_LEAK_COLS = {"label_eligibilite", "label_code", "label_nom"}
ID_COLS = {"RADICAL", "BANQUE", "AGENCE", "GENERIC", "PLURAL", "CCLE",
           "DATE_OF_BIRTH", "LIBELLE_VILLE", "digital_date_activation",
           "derniere_operation_gab"}

F1_SIGNIFICANCE_THRESHOLD = 0.01
COHENS_D_INDISTINGUISHABLE = 0.1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_s3fs():
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
    return pafs.S3FileSystem(
        endpoint_override=endpoint.replace("http://", "").replace("https://", ""),
        access_key=key, secret_key=secret, request_timeout=300, connect_timeout=300,
        scheme="http" if endpoint.startswith("http://") else "https",
    )


def load_dataset(path, s3):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    dataset = ds.dataset(bucket_path, filesystem=s3, format="parquet")
    return dataset.to_table().to_pandas()


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

    return X, y.astype(int), groups, cat_cols, num_cols


def make_model(kind, scale_pos):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63,
            min_child_samples=50, scale_pos_weight=scale_pos,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=2, verbose=-1,
        )
    elif kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            min_child_weight=10, scale_pos_weight=scale_pos,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=2,
        )
    raise ValueError(f"unknown model kind: {kind}")


def encode_categoricals_ordinal(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].cat.codes
    return X


def fit_predict(kind, X_fit, y_fit, X_val_tune, X_val_final, cat_cols):
    scale_pos = np.sqrt((len(y_fit) - y_fit.sum()) / y_fit.sum())
    model = make_model(kind, scale_pos)

    if kind == "lightgbm":
        X_fit_m, X_tune_m, X_final_m = X_fit, X_val_tune, X_val_final
    else:
        X_fit_m = encode_categoricals_ordinal(X_fit, cat_cols)
        X_tune_m = encode_categoricals_ordinal(X_val_tune, cat_cols)
        X_final_m = encode_categoricals_ordinal(X_val_final, cat_cols)

    log(f"    fitting {kind} on {len(X_fit_m):,} rows / {X_fit_m.shape[1]} columns...")
    if kind == "lightgbm":
        model.fit(X_fit_m, y_fit, categorical_feature=cat_cols)
    else:
        model.fit(X_fit_m, y_fit)
    log(f"    {kind} fit done, scoring...")

    p_tune = model.predict_proba(X_tune_m)[:, 1]
    p_final = model.predict_proba(X_final_m)[:, 1]
    return p_tune, p_final


def threshold_and_score(y_val_tune, probas_tune, y_val_final, probas_final):
    precisions, recalls, thresholds = precision_recall_curve(y_val_tune, probas_tune)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_thr = thresholds[np.argmax(f1s[:-1])]
    preds_final = (probas_final >= best_thr).astype(int)
    f1_final = f1_score(y_val_final, preds_final)
    prec_final = precision_score(y_val_final, preds_final, zero_division=0)
    rec_final = recall_score(y_val_final, preds_final, zero_division=0)
    return best_thr, f1_final, prec_final, rec_final, preds_final


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return abs(a.mean() - b.mean()) / pooled_std


def deep_error_analysis(label, X_val_final, y_val_final, probas_final, preds_final,
                         num_cols, threshold):
    """FIX 1 applied here: bins are now fractions of the ACTUAL threshold
    found by this run, not hardcoded absolute values borrowed from a
    different run with a different threshold."""
    log(f"\n--- Deep error analysis: {label} ---")

    y_val_final = np.asarray(y_val_final)
    is_fn = (y_val_final == 1) & (preds_final == 0)
    is_tp = (y_val_final == 1) & (preds_final == 1)
    n_fn, n_tp = is_fn.sum(), is_tp.sum()
    log(f"  {n_fn} false negatives, {n_tp} true positives at threshold {threshold:.4f}")

    fn_probas = probas_final[is_fn]
    # bins scaled to THIS run's threshold, not a borrowed absolute scale
    bin_edges = [0.0, 0.25 * threshold, 0.50 * threshold, 0.75 * threshold, threshold]
    log(f"  Confidence distribution of missed (false negative) clients "
        f"(bins are fractions of this run's threshold={threshold:.4f}):")
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        count = int(((fn_probas >= lo) & (fn_probas < hi)).sum())
        log(f"    {lo:.4f} - {hi:.4f} ({(lo/threshold)*100:.0f}%-{(hi/threshold)*100:.0f}% of thr) : {count}")
    near_miss = int((fn_probas >= 0.5 * threshold).sum())
    log(f"    -> {near_miss}/{n_fn} false negatives ({near_miss / max(n_fn, 1) * 100:.1f}%) "
        f"were already at >= 50% of the decision threshold (\"near misses\" relative to "
        f"THIS run's own threshold -- not compared against an absolute 0.5 borrowed from "
        f"a different run)")

    d_values = {}
    X_num = X_val_final[num_cols]
    for col in num_cols:
        d = cohens_d(X_num.loc[is_tp, col], X_num.loc[is_fn, col])
        d_values[col] = d
    d_series = pd.Series(d_values).dropna().sort_values()
    n_indistinguishable = int((d_series < COHENS_D_INDISTINGUISHABLE).sum())
    log(f"  {n_indistinguishable}/{len(d_series)} numeric features have "
        f"Cohen's d < {COHENS_D_INDISTINGUISHABLE} (TP vs FN)")
    log("  5 weakest (most indistinguishable) features:")
    for col, d in d_series.head(5).items():
        log(f"    {col:45s} d={d:.4f}")

    return {
        "n_fn": int(n_fn), "n_tp": int(n_tp),
        "fn_near_miss_pct": float(near_miss / max(n_fn, 1) * 100),
        "n_features_indistinguishable": n_indistinguishable,
        "n_features_total": int(len(d_series)),
        "weakest_features": d_series.head(10).to_dict(),
    }


# ----------------------------------------------------------------------
def controlled_split(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    fit_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
    y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    tune_idx, final_idx = next(gss2.split(X_val, y_val, groups=groups.iloc[val_idx]))
    return (X_fit, y_fit,
            X_val.iloc[tune_idx], y_val.iloc[tune_idx],
            X_val.iloc[final_idx], y_val.iloc[final_idx])


def run_one(name, kind, df, label_col, do_error_analysis=False):
    X, y, groups, cat_cols, num_cols = build_minimal_features(df, label_col)
    X_fit, y_fit, X_val_tune, y_val_tune, X_val_final, y_val_final = controlled_split(X, y, groups)

    probas_tune, probas_final = fit_predict(kind, X_fit, y_fit, X_val_tune, X_val_final, cat_cols)
    thr, f1, prec, rec, preds_final = threshold_and_score(
        y_val_tune, probas_tune, y_val_final, probas_final
    )
    log(f"[{name}] threshold={thr:.4f}  F1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}")

    result = {"name": name, "model": kind, "threshold": float(thr), "f1": float(f1),
              "precision": float(prec), "recall": float(rec)}

    if do_error_analysis:
        result["error_analysis"] = deep_error_analysis(
            name, X_val_final, y_val_final, probas_final, preds_final, num_cols, thr
        )
    return result


# ----------------------------------------------------------------------
def main(args):
    s3 = get_s3fs()

    log(f"Loading original dataset: {args.original_path}")
    df_orig = load_dataset(args.original_path, s3)
    log(f"  shape={df_orig.shape}")

    log(f"Loading cleaned dataset: {args.clean_path}")
    df_clean = load_dataset(args.clean_path, s3)
    log(f"  shape={df_clean.shape}")

    results = []

    # FIX 2: run the SAME deep error analysis, SAME minimal feature set,
    # on BOTH datasets -- this is the actual controlled comparison that
    # was missing before. (The already-trustworthy F1 numbers from the
    # first run -- xgboost +0.0005, ensemble 0.2461 -- are not re-run here;
    # only the diagnostic that was previously invalid gets redone properly.)
    log("\n=== Deep error analysis, same feature set, ORIGINAL dataset ===")
    results.append(run_one("lightgbm_original_diagnostic", "lightgbm", df_orig,
                            args.label_col, do_error_analysis=True))

    log("\n=== Deep error analysis, same feature set, CLEANED dataset ===")
    results.append(run_one("lightgbm_cleaned_diagnostic", "lightgbm", df_clean,
                            args.label_col, do_error_analysis=True))

    log("\n=== VERDICT (feature-level, now a real apples-to-apples comparison) ===")
    diag_orig = next(r["error_analysis"] for r in results if r["name"] == "lightgbm_original_diagnostic")
    diag_clean = next(r["error_analysis"] for r in results if r["name"] == "lightgbm_cleaned_diagnostic")
    log(f"Indistinguishable features -- original: {diag_orig['n_features_indistinguishable']}/"
        f"{diag_orig['n_features_total']}   cleaned: {diag_clean['n_features_indistinguishable']}/"
        f"{diag_clean['n_features_total']}")
    gap = diag_clean["n_features_indistinguishable"] - diag_orig["n_features_indistinguishable"]
    if gap < 0:
        log(f"  -> Cleaning reduced indistinguishable features by {abs(gap)}, on the SAME "
            f"feature set. This is now a controlled result, not an artifact of comparing "
            f"different column counts. Still doesn't move F1 (per the earlier XGBoost/LightGBM "
            f"run), which is consistent with your report's own finding that reducing noise "
            f"without adding new information doesn't raise the ceiling -- but it IS evidence "
            f"the cleaning was a real, if insufficient, improvement.")
    elif gap == 0:
        log("  -> No change on the same feature set. Cleaning doesn't affect feature-level "
            "separability either, consistent with the F1 result.")
    else:
        log(f"  -> Cleaning INCREASED indistinguishable features by {gap} on the same feature "
            f"set -- unexpected, worth double-checking the cleaning script's sentinel/NaN "
            f"logic didn't introduce new noise.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nFull results written to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_path", required=True)
    parser.add_argument("--clean_path", required=True)
    parser.add_argument("--label_col", default="label_eligibilite")
    parser.add_argument("--out_json", default="battery_results_v2.json")
    args = parser.parse_args()
    main(args)
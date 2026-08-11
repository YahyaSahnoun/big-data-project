"""
train_best_leaksafe.py
========================
Goal: the strongest legitimate F1 on dataset_eligibilite_features_v3, with every
leakage guard your report identified actually enforced in code (not just described):

  0) POPULATION INTEGRITY CHECK (new -- your earlier pipeline never verified this):
     dataset_eligibilite_features_v3 has no RADICAL/client-id column, so we can't
     group-split it directly. Before trusting a plain random split on it, this
     script fetches dataset_eligibilite_final's row count (already verified:
     zero duplicate RADICAL) and confirms features_v3 has the SAME row count.
     If they match, one row == one client, and a plain random split is safe.
     If they DON'T match, the script aborts rather than silently risking leakage
     -- go find out why (a join upstream may have duplicated rows).

  1) label_code / label_nom explicitly dropped (target-derived, would leak).

  2) sqrt-scaled scale_pos_weight (your validated fix for the raw-formula
     over-correction).

  3) Optuna HPO for XGBoost + LightGBM (real search, not manual grid).

  4) Threshold selected ONLY on a disjoint tuning slice; F1 reported ONLY on an
     untouched final slice -- this is the exact protocol from your report's
     section 7.4.2, enforced here rather than just documented.

  5) Lean 2-model calibrated stack (XGB + LGBM) -- your report showed a 3rd/4th
     view only added +0.002 for real added complexity, so it's cut here.

REALISTIC EXPECTATION: your report's own 20-attempt study, with full engineered
features and extensive tuning, converged to 0.22-0.25 (best single model 0.225,
best ensemble 0.2239). This script gives that same ceiling its best legitimate
shot. It is not expected to reliably clear it, per your own Cohen's d / PCA
evidence -- if it lands in that band, that CONFIRMS the ceiling, it doesn't
mean the search failed.

USAGE (WSL / bash):
  pip install s3fs pyarrow pandas optuna xgboost lightgbm scikit-learn joblib --break-system-packages

  export MINIO_ENDPOINT="http://localhost:9000"
  export MINIO_ACCESS_KEY="minioadmin"
  export MINIO_SECRET_KEY="minioadmin123"

  python3 train_best_leaksafe.py \
      --features_path s3a://processed-data/dataset_eligibilite_features_v3 \
      --final_path s3a://processed-data/dataset_eligibilite_final \
      --outdir ./run_best_leaksafe \
      --n_trials 40 \
      --sample_frac 1.0
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve,
    roc_auc_score, average_precision_score
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_LEAK_COLS = {"label_code", "label_nom"}
LABEL_COL = "label_eligibilite"


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


def load_dataset(path, s3, columns=None):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    dataset = ds.dataset(bucket_path, filesystem=s3, format="parquet")
    table = dataset.to_table(columns=columns)
    return table.to_pandas()


def count_rows(path, s3):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    dataset = ds.dataset(bucket_path, filesystem=s3, format="parquet")
    return dataset.count_rows()


def best_threshold(probas, y_true):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    idx = np.argmax(f1s[:-1])
    return thresholds[idx], f1s[idx]


def report_at_threshold(name, probas, y_true, threshold, t0=None):
    preds = (probas >= threshold).astype(int)
    f1 = f1_score(y_true, preds, pos_label=1)
    prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_true, probas)
    roc_auc = roc_auc_score(y_true, probas)
    elapsed = f" | {time.time()-t0:.1f}s" if t0 else ""
    log(f"{name:>18s} -- thr={threshold:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f} "
        f"PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f}{elapsed}")
    return {"f1": f1, "precision": prec, "recall": rec, "pr_auc": pr_auc,
            "roc_auc": roc_auc, "threshold": float(threshold)}


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    s3 = get_s3fs()

    # ------------------------------------------------------------------
    # 0. POPULATION INTEGRITY CHECK -- do not skip this
    # ------------------------------------------------------------------
    log("0. Population integrity check (features_v3 vs. final row counts)...")
    n_final = count_rows(args.final_path, s3)
    n_features = count_rows(args.features_path, s3)
    log(f"  dataset_eligibilite_final rows   = {n_final:,}")
    log(f"  dataset_eligibilite_features_v3 rows = {n_features:,}")
    if n_final != n_features:
        raise SystemExit(
            f"ABORTING: row count mismatch ({n_final:,} vs {n_features:,}). "
            f"dataset_eligibilite_final was verified to have zero duplicate RADICAL, "
            f"i.e. one row per client. features_v3 has a DIFFERENT row count, which "
            f"means it is not a simple 1:1 column-enrichment of the same population -- "
            f"a plain random split would risk leakage. Investigate the script that "
            f"built features_v3 before proceeding; do not just widen this check."
        )
    log("  OK: row counts match -- one row per client in features_v3, plain split is safe.")

    # ------------------------------------------------------------------
    log("\n1. Loading dataset_eligibilite_features_v3...")
    df = load_dataset(args.features_path, s3)
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=RANDOM_SEED).reset_index(drop=True)
    log(f"  shape={df.shape}  pos_rate={df[LABEL_COL].mean():.4%}")

    leak_present = TARGET_LEAK_COLS.intersection(df.columns)
    if leak_present:
        log(f"  Dropping target-derived columns: {leak_present}")
        df = df.drop(columns=list(leak_present))

    y = df.pop(LABEL_COL).astype(int).values
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    X = df.values.astype(np.float32)
    feature_names = df.columns.tolist()
    log(f"  Final feature matrix: {X.shape} ({len(cat_cols)} were categorical, label-encoded)")

    with open(outdir / "feature_names_used.json", "w") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Split: fit / tune / final -- strict separation, threshold NEVER touches final
    # ------------------------------------------------------------------
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_tune, X_final, y_tune, y_final = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=RANDOM_SEED, stratify=y_rest
    )
    log(f"  fit={X_fit.shape} tune={X_tune.shape} final={X_final.shape}")

    scale_pos_sqrt = np.sqrt((len(y_fit) - y_fit.sum()) / y_fit.sum())
    log(f"  scale_pos_weight (sqrt) = {scale_pos_sqrt:.4f}")

    results = {}

    # ------------------------------------------------------------------
    # 2. Optuna -- XGBoost
    # ------------------------------------------------------------------
    from xgboost import XGBClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log("\n2. Optuna search -- XGBoost...")
    t0 = time.time()

    def xgb_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            scale_pos_weight=scale_pos_sqrt,
            random_state=RANDOM_SEED, n_jobs=args.n_jobs,
            eval_metric="logloss", tree_method="hist",
            early_stopping_rounds=40,
        )
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
        probas = model.predict_proba(X_tune)[:, 1]
        _, f1 = best_threshold(probas, y_tune)
        return f1

    xgb_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    xgb_study.optimize(xgb_objective, n_trials=args.n_trials, show_progress_bar=False)
    log(f"  Best XGB tune-F1={xgb_study.best_value:.4f} ({time.time()-t0:.1f}s)")

    xgb_model = XGBClassifier(
        **xgb_study.best_params, scale_pos_weight=scale_pos_sqrt,
        random_state=RANDOM_SEED, n_jobs=args.n_jobs, eval_metric="logloss",
        tree_method="hist", early_stopping_rounds=40,
    )
    xgb_model.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
    probas_xgb_tune = xgb_model.predict_proba(X_tune)[:, 1]
    probas_xgb_final = xgb_model.predict_proba(X_final)[:, 1]
    thr_xgb, _ = best_threshold(probas_xgb_tune, y_tune)
    results["xgboost"] = report_at_threshold("XGBoost (tuned)", probas_xgb_final, y_final, thr_xgb, t0)
    joblib.dump(xgb_model, outdir / "xgb_model.joblib")

    # ------------------------------------------------------------------
    # 3. Optuna -- LightGBM
    # ------------------------------------------------------------------
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    log("\n3. Optuna search -- LightGBM...")
    t0 = time.time()

    def lgbm_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            num_leaves=trial.suggest_int("num_leaves", 31, 200),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            scale_pos_weight=scale_pos_sqrt,
            random_state=RANDOM_SEED, n_jobs=args.n_jobs, verbose=-1,
        )
        model = LGBMClassifier(**params)
        model.fit(
            X_fit, y_fit, eval_set=[(X_tune, y_tune)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        probas = model.predict_proba(X_tune)[:, 1]
        _, f1 = best_threshold(probas, y_tune)
        return f1

    lgbm_study = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    lgbm_study.optimize(lgbm_objective, n_trials=args.n_trials, show_progress_bar=False)
    log(f"  Best LGBM tune-F1={lgbm_study.best_value:.4f} ({time.time()-t0:.1f}s)")

    lgbm_model = LGBMClassifier(
        **lgbm_study.best_params, scale_pos_weight=scale_pos_sqrt,
        random_state=RANDOM_SEED, n_jobs=args.n_jobs, verbose=-1,
    )
    lgbm_model.fit(
        X_fit, y_fit, eval_set=[(X_tune, y_tune)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    probas_lgbm_tune = lgbm_model.predict_proba(X_tune)[:, 1]
    probas_lgbm_final = lgbm_model.predict_proba(X_final)[:, 1]
    thr_lgbm, _ = best_threshold(probas_lgbm_tune, y_tune)
    results["lightgbm"] = report_at_threshold("LightGBM (tuned)", probas_lgbm_final, y_final, thr_lgbm, t0)
    joblib.dump(lgbm_model, outdir / "lgbm_model.joblib")

    # ------------------------------------------------------------------
    # 4. Calibrated stack (XGB + LGBM)
    # ------------------------------------------------------------------
    log("\n4. Calibrated stack (XGBoost + LightGBM)...")
    t0 = time.time()
    meta_X_tune = np.column_stack([probas_xgb_tune, probas_lgbm_tune])
    meta_X_final = np.column_stack([probas_xgb_final, probas_lgbm_final])

    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    meta_model.fit(meta_X_tune, y_tune)
    stack_probas_tune = meta_model.predict_proba(meta_X_tune)[:, 1]
    stack_probas_final = meta_model.predict_proba(meta_X_final)[:, 1]

    thr_stack, _ = best_threshold(stack_probas_tune, y_tune)
    results["stack_xgb_lgbm"] = report_at_threshold(
        "Stack(XGB+LGBM)", stack_probas_final, y_final, thr_stack, t0
    )
    joblib.dump(meta_model, outdir / "meta_model.joblib")

    # ------------------------------------------------------------------
    log("\n=== SUMMARY (F1, positive class, measured on the untouched final slice) ===")
    for name, r in results.items():
        log(f"  {name:>18s}: F1={r['f1']:.4f}  (threshold={r['threshold']:.4f})")

    best_name = max(results, key=lambda k: results[k]["f1"])
    log(f"\nBest configuration: {best_name} -- F1={results[best_name]['f1']:.4f}")

    with open(outdir / "results.json", "w") as f:
        json.dump({
            "results": results,
            "n_rows_used": len(y),
            "xgb_best_params": xgb_study.best_params,
            "lgbm_best_params": lgbm_study.best_params,
            "scale_pos_weight_sqrt": scale_pos_sqrt,
        }, f, indent=2)
    log(f"Saved models + results.json to {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True,
                         help="s3a://processed-data/dataset_eligibilite_features_v3")
    parser.add_argument("--final_path", required=True,
                         help="s3a://processed-data/dataset_eligibilite_final (used only for the row-count integrity check)")
    parser.add_argument("--outdir", default="./run_best_leaksafe")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()
    main(args)

"""
train_optimized_eligibilite.py
================================
Consolidates every technique that your 20-attempt study (rapport_de_stage_bcp_pfa)
actually validated as helping:
  - sqrt-scaled scale_pos_weight (raw inverse-freq over-corrected -> precision 8-9%)
  - decision threshold optimized on a DISJOINT tuning slice, never on the eval slice
  - early stopping (avoids paying for 800 fixed rounds, converges long before that)
  - Top-K feature selection by importance (tentative 13: faster, no F1 loss)
  - NEW vs. your 20 attempts: Optuna Bayesian hyperparameter search (you only did
    manual/grid tuning in tentative 5) + a lean 2-model calibrated stack (XGB+LGBM
    only -- your report showed the 4th view in the 4-view ensemble only added +0.002
    for a lot of extra complexity, so we cut it here).

REALISTIC EXPECTATION (per your own error analysis, section 7.9-7.11):
  63/66 features were statistically indiscernible (Cohen's d) between true positives
  and false negatives, and PCA showed near-total class overlap. That is a data-scope
  ceiling, not a modeling gap. This script is a strong attempt at the TOP of your
  0.22-0.25 observed range -- it is not expected to reliably blow past it, because
  your report already produced strong evidence of *why* it won't.

USAGE (WSL / bash):
  pip install optuna xgboost lightgbm catboost scikit-learn joblib numpy --break-system-packages
  python3 train_optimized_eligibilite.py \
      --X_fit /path/X_fit.npy --y_fit /path/y_fit.npy \
      --X_val /path/X_val.npy --y_val /path/y_val.npy \
      --outdir ./run_optimized \
      --n_trials 40 \
      --top_k 200

  # optional: if you have a JSON list of column names (in the same order as X columns),
  # pass --feature_names to keep the pack_actuel_x_CUSTOMER_RATING interaction feature
  # pinned in the top-K set even if raw importance ranking would drop it.
  #   --feature_names /path/feature_names.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve,
    classification_report, roc_auc_score, average_precision_score
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return {"f1": f1, "precision": prec, "recall": rec, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "threshold": float(threshold)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("Loading data...")
    X_fit = np.load(args.X_fit)
    y_fit = np.load(args.y_fit)
    X_val = np.load(args.X_val)
    y_val = np.load(args.y_val)
    log(f"  X_fit={X_fit.shape} X_val={X_val.shape} "
        f"pos_rate_fit={y_fit.mean():.4f} pos_rate_val={y_val.mean():.4f}")

    feature_names = None
    if args.feature_names:
        feature_names = json.loads(Path(args.feature_names).read_text())
        assert len(feature_names) == X_fit.shape[1], "feature_names length mismatch with X columns"

    scale_pos_sqrt = np.sqrt((len(y_fit) - y_fit.sum()) / y_fit.sum())
    log(f"  scale_pos_weight (sqrt-adjusted) = {scale_pos_sqrt:.4f}")

    # ------------------------------------------------------------------
    # 0. Feature selection (Top-K by LightGBM importance, pin known-strong interaction)
    # ------------------------------------------------------------------
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    t0 = time.time()
    log(f"\n0. Feature selection (Top {args.top_k})...")
    fs_model = LGBMClassifier(
        n_estimators=150, learning_rate=0.1, random_state=RANDOM_SEED,
        n_jobs=args.n_jobs, verbose=-1
    )
    fs_model.fit(X_fit, y_fit)
    importances = fs_model.feature_importances_
    ranked = np.argsort(importances)[::-1]
    top_idx = list(ranked[:args.top_k])

    # Pin the known strongest cross-feature if it exists and got dropped
    pinned = []
    if feature_names:
        for wanted in ("pack_actuel_x_CUSTOMER_RATING", "pack_etat_x_CUSTOMER_RATING",
                        "flux_cred_total"):
            if wanted in feature_names:
                idx = feature_names.index(wanted)
                if idx not in top_idx:
                    pinned.append(idx)
        if pinned:
            log(f"  Pinning {len(pinned)} known-strong feature(s) not in raw top-{args.top_k}: "
                f"{[feature_names[i] for i in pinned]}")
            top_idx = top_idx[: args.top_k - len(pinned)] + pinned

    top_idx = np.array(sorted(set(top_idx)))
    X_fit_fs = X_fit[:, top_idx]
    X_val_fs = X_val[:, top_idx]
    log(f"  Shape after FS: Train {X_fit_fs.shape}, Val {X_val_fs.shape} "
        f"({time.time()-t0:.1f}s)")

    # Split validation -> tune (threshold selection / Optuna eval) / final (untouched)
    X_val_tune, X_val_final, y_val_tune, y_val_final = train_test_split(
        X_val_fs, y_val, test_size=0.5, random_state=RANDOM_SEED, stratify=y_val
    )
    log(f"  Validation Tune {X_val_tune.shape}, Validation Final {X_val_final.shape}")

    results = {}

    # ------------------------------------------------------------------
    # 1. Optuna HPO for XGBoost (your best single algorithm, ~0.225)
    # ------------------------------------------------------------------
    from xgboost import XGBClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log("\n1. Optuna search -- XGBoost...")
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
        model.fit(X_fit_fs, y_fit, eval_set=[(X_val_tune, y_val_tune)], verbose=False)
        probas = model.predict_proba(X_val_tune)[:, 1]
        _, f1 = best_threshold(probas, y_val_tune)
        return f1

    xgb_study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    xgb_study.optimize(xgb_objective, n_trials=args.n_trials, show_progress_bar=False)
    log(f"  Best XGB tune-F1={xgb_study.best_value:.4f} params={xgb_study.best_params} "
        f"({time.time()-t0:.1f}s)")

    xgb_best_params = dict(
        xgb_study.best_params,
        scale_pos_weight=scale_pos_sqrt, random_state=RANDOM_SEED, n_jobs=args.n_jobs,
        eval_metric="logloss", tree_method="hist", early_stopping_rounds=40,
    )
    xgb_model = XGBClassifier(**xgb_best_params)
    xgb_model.fit(X_fit_fs, y_fit, eval_set=[(X_val_tune, y_val_tune)], verbose=False)
    probas_xgb_tune = xgb_model.predict_proba(X_val_tune)[:, 1]
    probas_xgb_final = xgb_model.predict_proba(X_val_final)[:, 1]
    thr_xgb, _ = best_threshold(probas_xgb_tune, y_val_tune)
    results["xgboost"] = report_at_threshold("XGBoost (tuned)", probas_xgb_final, y_val_final, thr_xgb, t0)
    joblib.dump(xgb_model, outdir / "xgb_model.joblib")

    # ------------------------------------------------------------------
    # 2. Optuna HPO for LightGBM
    # ------------------------------------------------------------------
    log("\n2. Optuna search -- LightGBM...")
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
            X_fit_fs, y_fit,
            eval_set=[(X_val_tune, y_val_tune)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        probas = model.predict_proba(X_val_tune)[:, 1]
        _, f1 = best_threshold(probas, y_val_tune)
        return f1

    lgbm_study = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    lgbm_study.optimize(lgbm_objective, n_trials=args.n_trials, show_progress_bar=False)
    log(f"  Best LGBM tune-F1={lgbm_study.best_value:.4f} params={lgbm_study.best_params} "
        f"({time.time()-t0:.1f}s)")

    lgbm_best_params = dict(
        lgbm_study.best_params,
        scale_pos_weight=scale_pos_sqrt, random_state=RANDOM_SEED, n_jobs=args.n_jobs, verbose=-1,
    )
    lgbm_model = LGBMClassifier(**lgbm_best_params)
    lgbm_model.fit(
        X_fit_fs, y_fit,
        eval_set=[(X_val_tune, y_val_tune)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    probas_lgbm_tune = lgbm_model.predict_proba(X_val_tune)[:, 1]
    probas_lgbm_final = lgbm_model.predict_proba(X_val_final)[:, 1]
    thr_lgbm, _ = best_threshold(probas_lgbm_tune, y_val_tune)
    results["lightgbm"] = report_at_threshold("LightGBM (tuned)", probas_lgbm_final, y_val_final, thr_lgbm, t0)
    joblib.dump(lgbm_model, outdir / "lgbm_model.joblib")

    # ------------------------------------------------------------------
    # 3. Lean 2-model calibrated stack (XGB + LGBM only -- your report showed
    #    a 3rd/4th view added only +0.002 for a lot of extra complexity)
    # ------------------------------------------------------------------
    log("\n3. Calibrated stack (XGBoost + LightGBM)...")
    t0 = time.time()
    meta_X_tune = np.column_stack([probas_xgb_tune, probas_lgbm_tune])
    meta_X_final = np.column_stack([probas_xgb_final, probas_lgbm_final])

    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    meta_model.fit(meta_X_tune, y_val_tune)
    stack_probas_tune = meta_model.predict_proba(meta_X_tune)[:, 1]
    stack_probas_final = meta_model.predict_proba(meta_X_final)[:, 1]

    thr_stack, _ = best_threshold(stack_probas_tune, y_val_tune)
    results["stack_xgb_lgbm"] = report_at_threshold(
        "Stack(XGB+LGBM)", stack_probas_final, y_val_final, thr_stack, t0
    )
    joblib.dump(meta_model, outdir / "meta_model.joblib")

    # ------------------------------------------------------------------
    # Summary + honest framing
    # ------------------------------------------------------------------
    log("\n=== SUMMARY (F1, positive class, measured on the untouched final slice) ===")
    for name, r in results.items():
        log(f"  {name:>18s}: F1={r['f1']:.4f}  (threshold={r['threshold']:.4f})")

    best_name = max(results, key=lambda k: results[k]["f1"])
    log(f"\nBest configuration: {best_name} -- F1={results[best_name]['f1']:.4f}")
    log("Reminder (per your own error analysis, ch.7 sec.9-11): if this lands in the "
        "0.22-0.25 band like the rest of the study, that is consistent with the data-scope "
        "ceiling you already demonstrated (Cohen's d, PCA overlap) -- not a sign the search "
        "was insufficient.")

    with open(outdir / "results.json", "w") as f:
        json.dump({
            "results": results,
            "top_k_indices": top_idx.tolist(),
            "xgb_best_params": xgb_study.best_params,
            "lgbm_best_params": lgbm_study.best_params,
            "scale_pos_weight_sqrt": scale_pos_sqrt,
        }, f, indent=2)
    log(f"\nSaved models + results.json to {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_fit", required=True)
    parser.add_argument("--y_fit", required=True)
    parser.add_argument("--X_val", required=True)
    parser.add_argument("--y_val", required=True)
    parser.add_argument("--feature_names", default=None,
                         help="Optional JSON list of column names, same order as X columns")
    parser.add_argument("--outdir", default="./run_optimized")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--n_trials", type=int, default=40,
                         help="Optuna trials per model. 40 is a reasonable budget on 2 cores; "
                              "raise to 80-100 overnight for a slightly wider search.")
    parser.add_argument("--n_jobs", type=int, default=2,
                         help="Set to your allocated core count (you showed Spark app with Cores=2)")
    args = parser.parse_args()
    main(args)

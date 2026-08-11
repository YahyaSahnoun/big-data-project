"""
break_ceiling_battery.py
=========================
Goal: give the F1 ceiling (0.22-0.24, chapter 7) every reasonable chance to
break, using approaches NOT already exhausted in the original 20 attempts,
and — critically — report every result with a confidence interval instead
of a single point estimate. A single-split F1 of 0.229 vs 0.224 tells you
nothing; this script tells you whether that gap is real.

What's new vs. the original battery (so this isn't just re-running the
same 20 tests):
  21. Repeated stratified k-fold CV (5x5) on the two best-known configs
      (LightGBM sqrt-weighted, XGBoost sqrt-weighted) -> mean +/- 95% CI.
      This directly answers "is 0.22-0.24 stable or noise?"
  22. Bayes-error proxy via k-NN oracle: if a k-NN classifier (which makes
      no parametric assumptions at all) also caps near the same F1, that's
      strong evidence the ceiling is in the *data*, not in any one model
      family's inductive bias.
  23. Wide randomized hyperparameter search (LightGBM + XGBoost), larger
      budget than the "optimisation des hyperparametres" attempt (#5),
      to make sure the ceiling isn't an under-tuned baseline.
  24. Pseudo-labeling / self-training loop: high-confidence unlabeled-style
      predictions folded back into training. Different failure mode than
      SMOTE (#15) — tests whether the model can bootstrap signal from its
      own confident predictions rather than synthesizing points.
  25. Monotonic-constrained LightGBM on the features with signed WOE trend
      (regularizes toward the *true* relationship instead of overfitting
      noise) — cheap to try, sometimes recovers a point or two of F1 that
      pure flexibility loses to noise on indiscernible features.
  26. Two-stage cascade re-check: eligibility-model probability as an
      extra feature fed into a second-pass classifier trained only on the
      first pass's medium-confidence band (0.3-0.7) — a more surgical
      version of the segmented-model idea (#17) that only cost you 0.22
      last time.

Everything is logged to battery_results_v2.json in the same style as your
existing battery_results.json so it's a straight diff, not a new format.

USAGE
-----
    python break_ceiling_battery.py --data /path/to/dataset_eligibilite_features_v3.parquet

If you don't have the real data handy, run with --demo to generate a
synthetic dataset with the same "indiscernible features" property (a
positive class embedded inside the negative cloud) so you can sanity
check the script mechanics before pointing it at production data.
"""

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LABEL_COL = "label_eligibilite"
RANDOM_STATE = 42
LOG = []


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.append(line)


def load_data(path=None, demo=False, n=200_000, n_features=40, pos_rate=0.0425):
    """Load the real dataset, or synthesize a 'ceiling-shaped' one for
    dry-running the script. The synthetic version deliberately buries the
    positive class inside the negative cloud (small mean shift + heavy
    overlap) so that a real ceiling should reproduce here too — useful to
    confirm the diagnostics behave sanely before touching production data.
    """
    if demo or path is None:
        log(f"DEMO MODE: synthesizing {n:,} rows / {n_features} features "
            f"with a buried positive class (pos_rate={pos_rate})")
        rng = np.random.default_rng(RANDOM_STATE)
        n_pos = int(n * pos_rate)
        n_neg = n - n_pos
        # negatives: standard normal cloud
        X_neg = rng.normal(0, 1, size=(n_neg, n_features))
        # positives: tiny mean shift on a handful of dims + same variance
        # (mirrors Cohen's d ~0.01-0.05 on most features from ch.7.9.2)
        shift = np.zeros(n_features)
        shift[:3] = [0.15, 0.10, 0.08]  # only 3/40 features carry any signal
        X_pos = rng.normal(shift, 1, size=(n_pos, n_features))
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * n_neg + [1] * n_pos)
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df[LABEL_COL] = y
        return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    log(f"Loading real dataset from {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    log(f"Loaded {len(df):,} rows / {df.shape[1]} columns")
    return df


def prep_xy(df):
    y = df[LABEL_COL].values
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).fillna(0)
    return X, y


def best_threshold_f1(y_true, proba):
    """Same procedure as tentative #6: sweep thresholds, pick argmax F1 on
    THIS split. Caller is responsible for making sure this split is a
    validation split, not the final test split (see run_repeated_cv)."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Attempt 21: repeated stratified CV with confidence intervals
# ---------------------------------------------------------------------------
def attempt_21_repeated_cv(X, y, n_splits=5, n_repeats=5):
    log("=== Attempt 21: Repeated 5x5 stratified CV (LightGBM & XGBoost) ===")
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError:
        log("  lightgbm/xgboost not installed — skipping. "
            "pip install lightgbm xgboost --break-system-packages")
        return {}

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=RANDOM_STATE)
    results = {"lightgbm": [], "xgboost": []}
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    sqrt_weight = np.sqrt(pos_weight)
    total_folds = n_splits * n_repeats
    fold_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # inner split of train for threshold selection -> avoids the
        # "threshold picked on the same data it's scored on" leak (7.4.1)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)

        for name, model in [
            ("lightgbm", lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                scale_pos_weight=sqrt_weight, random_state=RANDOM_STATE,
                verbosity=-1)),
            ("xgboost", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                scale_pos_weight=sqrt_weight, random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0)),
        ]:
            model.fit(X_fit, y_fit)
            val_proba = model.predict_proba(X_val)[:, 1]
            thresh, _ = best_threshold_f1(y_val, val_proba)
            test_proba = model.predict_proba(X_te)[:, 1]
            test_pred = (test_proba >= thresh).astype(int)
            f1 = f1_score(y_te, test_pred, zero_division=0)
            results[name].append(f1)

        elapsed = time.time() - fold_start
        avg_per_fold = elapsed / (fold_i + 1)
        remaining = avg_per_fold * (total_folds - fold_i - 1)
        log(f"  fold {fold_i+1}/{total_folds} done "
            f"(lgb={results['lightgbm'][-1]:.4f}, "
            f"xgb={results['xgboost'][-1]:.4f}) "
            f"| elapsed={elapsed/60:.1f}min | "
            f"est. remaining={remaining/60:.1f}min")

    summary = {}
    for name, scores in results.items():
        scores = np.array(scores)
        mean, sem = scores.mean(), stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
        summary[name] = {
            "n_folds": len(scores), "mean_f1": float(mean),
            "std_f1": float(scores.std()), "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]), "scores": scores.tolist(),
        }
        log(f"  {name}: F1 = {mean:.4f} +/- {scores.std():.4f} "
            f"(95% CI [{ci[0]:.4f}, {ci[1]:.4f}], n={len(scores)} folds)")
    return summary


# ---------------------------------------------------------------------------
# Attempt 22: k-NN Bayes-error proxy (model-free ceiling check)
# ---------------------------------------------------------------------------
def attempt_22_knn_oracle(X, y, sample_size=50_000, k_values=(5, 15, 35)):
    log("=== Attempt 22: k-NN Bayes-error proxy (model-agnostic check) ===")
    log("  Rationale: k-NN makes no assumption about decision-boundary "
        "shape. If it ALSO caps near 0.22-0.24, no algorithm choice will "
        "fix this — the classes truly overlap in feature space.")
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        Xs, ys = X.iloc[idx], y[idx]
    else:
        Xs, ys = X, y

    Xs_scaled = StandardScaler().fit_transform(Xs.fillna(0))
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs_scaled, ys, test_size=0.3, stratify=ys, random_state=RANDOM_STATE)

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_tr, y_tr)
        proba = knn.predict_proba(X_te)[:, 1]
        thresh, _ = best_threshold_f1(y_te, proba)  # optimistic (no held-out val) on purpose: upper bound
        pred = (proba >= thresh).astype(int)
        f1 = f1_score(y_te, pred, zero_division=0)
        results[f"knn_k{k}"] = float(f1)
        log(f"  k={k}: F1 (optimistic upper bound) = {f1:.4f}")
    return results


# ---------------------------------------------------------------------------
# Attempt 23: wider randomized hyperparameter search
# ---------------------------------------------------------------------------
def attempt_23_wide_hpo(X, y, n_iter=40):
    log(f"=== Attempt 23: Randomized HPO, budget={n_iter} (wider than #5) ===")
    try:
        import lightgbm as lgb
    except ImportError:
        log("  lightgbm not installed — skipping.")
        return {}

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pos_weight = np.sqrt((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    rng = np.random.RandomState(RANDOM_STATE)
    param_space = {
        "num_leaves": [15, 31, 63, 127, 255],
        "max_depth": [4, 6, 8, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "min_child_samples": [5, 20, 50, 100],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    best = {"f1": -1, "params": None}
    for i in range(n_iter):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        model = lgb.LGBMClassifier(**params, scale_pos_weight=pos_weight,
                                    random_state=RANDOM_STATE, verbosity=-1)
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_val)[:, 1]
        _, f1 = best_threshold_f1(y_val, proba)
        if f1 > best["f1"]:
            best = {"f1": f1, "params": {k: (int(v) if isinstance(v, np.integer) else float(v))
                                          for k, v in params.items()}}
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{n_iter} configs tried, best F1 so far = {best['f1']:.4f}")
    log(f"  Best config: F1={best['f1']:.4f}, params={best['params']}")
    return best


# ---------------------------------------------------------------------------
# Attempt 24: pseudo-labeling / self-training
# ---------------------------------------------------------------------------
def attempt_24_pseudo_labeling(X, y, rounds=3, confidence=0.9):
    log(f"=== Attempt 24: Pseudo-labeling self-training ({rounds} rounds, "
        f"conf>={confidence}) ===")
    try:
        import lightgbm as lgb
    except ImportError:
        log("  lightgbm not installed — skipping.")
        return {}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    # simulate an unlabeled pool by hiding labels on part of train (in
    # production you'd use genuinely unlabeled/未来 rows if you have any)
    X_lab, X_unlab, y_lab, _ = train_test_split(
        X_tr, y_tr, test_size=0.5, stratify=y_tr, random_state=RANDOM_STATE)

    X_lab_cur, y_lab_cur = X_lab.copy(), y_lab.copy()
    X_pool = X_unlab.copy()
    pos_weight = np.sqrt((y_lab_cur == 0).sum() / max((y_lab_cur == 1).sum(), 1))

    for r in range(rounds):
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    scale_pos_weight=pos_weight,
                                    random_state=RANDOM_STATE, verbosity=-1)
        model.fit(X_lab_cur, y_lab_cur)
        if len(X_pool) == 0:
            break
        proba = model.predict_proba(X_pool)[:, 1]
        confident_mask = (proba >= confidence) | (proba <= 1 - confidence)
        n_confident = confident_mask.sum()
        log(f"  round {r+1}: {n_confident}/{len(X_pool)} pool rows above "
            f"confidence threshold, folding them in")
        if n_confident == 0:
            break
        new_labels = (proba[confident_mask] >= 0.5).astype(int)
        X_lab_cur = pd.concat([X_lab_cur, X_pool[confident_mask]])
        y_lab_cur = np.concatenate([y_lab_cur, new_labels])
        X_pool = X_pool[~confident_mask]
        pos_weight = np.sqrt((y_lab_cur == 0).sum() / max((y_lab_cur == 1).sum(), 1))

    final_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                      scale_pos_weight=pos_weight,
                                      random_state=RANDOM_STATE, verbosity=-1)
    final_model.fit(X_lab_cur, y_lab_cur)
    X_fit2, X_val2, y_fit2, y_val2 = train_test_split(
        X_lab_cur, y_lab_cur, test_size=0.2, random_state=RANDOM_STATE)
    proba_val = final_model.predict_proba(X_val2)[:, 1]
    thresh, _ = best_threshold_f1(y_val2, proba_val)
    proba_te = final_model.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= thresh).astype(int)
    f1 = f1_score(y_te, pred_te, zero_division=0)
    log(f"  Final F1 on held-out test: {f1:.4f}")
    return {"f1": float(f1), "rounds_run": r + 1}


# ---------------------------------------------------------------------------
# Attempt 25: monotonic-constrained LightGBM
# ---------------------------------------------------------------------------
def attempt_25_monotonic(X, y):
    log("=== Attempt 25: Monotonic-constrained LightGBM ===")
    try:
        import lightgbm as lgb
    except ImportError:
        log("  lightgbm not installed — skipping.")
        return {}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)

    # Infer monotonic direction per feature from correlation sign on a
    # cheap sample rather than requiring you to hand-list WOE trends here.
    corr = pd.Series(
        {c: np.corrcoef(X_fit[c].fillna(0), y_fit)[0, 1] for c in X_fit.columns}
    ).fillna(0)
    constraints = [1 if v > 0.02 else (-1 if v < -0.02 else 0) for v in corr]
    n_constrained = sum(c != 0 for c in constraints)
    log(f"  {n_constrained}/{len(constraints)} features given a monotonic "
        f"constraint (|corr| > 0.02)")

    pos_weight = np.sqrt((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        monotone_constraints=constraints, monotone_constraints_method="advanced",
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(X_fit, y_fit)
    proba_val = model.predict_proba(X_val)[:, 1]
    thresh, _ = best_threshold_f1(y_val, proba_val)
    proba_te = model.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= thresh).astype(int)
    f1 = f1_score(y_te, pred_te, zero_division=0)
    log(f"  F1 with monotonic constraints: {f1:.4f}")
    return {"f1": float(f1), "n_constrained_features": n_constrained}


# ---------------------------------------------------------------------------
# Attempt 26: two-pass cascade on the uncertain (0.3-0.7) band
# ---------------------------------------------------------------------------
def attempt_26_uncertain_band_cascade(X, y):
    log("=== Attempt 26: Second-pass model on the uncertain (0.3-0.7) band ===")
    log("  Rationale: chapter 7.9.1 showed 63%% of false negatives already "
        "sit at proba 0.5-0.75 -- close but not over the line. Instead of "
        "re-splitting by a static segment (which failed, attempt #17), "
        "train a dedicated second-pass model ONLY on cases the first pass "
        "is genuinely unsure about.")
    try:
        import lightgbm as lgb
    except ImportError:
        log("  lightgbm not installed — skipping.")
        return {}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)
    pos_weight = np.sqrt((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    stage1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 scale_pos_weight=pos_weight,
                                 random_state=RANDOM_STATE, verbosity=-1)
    stage1.fit(X_fit, y_fit)
    proba_val1 = stage1.predict_proba(X_val)[:, 1]
    thresh1, base_f1 = best_threshold_f1(y_val, proba_val1)
    log(f"  Stage-1 alone: F1={base_f1:.4f} at threshold {thresh1:.3f}")

    # confident predictions pass straight through; uncertain band goes to stage 2
    proba_te1 = stage1.predict_proba(X_te)[:, 1]
    band_mask = (proba_te1 >= 0.3) & (proba_te1 <= 0.7)
    log(f"  {band_mask.sum()}/{len(X_te)} test rows fall in the uncertain "
        f"band and get a second look")

    fit_proba = stage1.predict_proba(X_fit)[:, 1]
    fit_band_mask = (fit_proba >= 0.3) & (fit_proba <= 0.7)
    X_fit_band = X_fit[fit_band_mask].copy()
    X_fit_band["stage1_proba"] = fit_proba[fit_band_mask]
    y_fit_band = y_fit[fit_band_mask]

    if fit_band_mask.sum() < 200 or y_fit_band.sum() < 20:
        log("  Not enough band rows to train a meaningful stage-2 model — "
            "this itself is informative (the uncertain band is too thin "
            "or too pure to learn from).")
        return {"stage1_f1": float(base_f1), "stage2_trained": False}

    pos_weight2 = np.sqrt((y_fit_band == 0).sum() / max((y_fit_band == 1).sum(), 1))
    stage2 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                 scale_pos_weight=pos_weight2,
                                 random_state=RANDOM_STATE, verbosity=-1)
    stage2.fit(X_fit_band, y_fit_band)

    final_pred = (proba_te1 >= thresh1).astype(int)
    if band_mask.sum() > 0:
        X_te_band = X_te[band_mask].copy()
        X_te_band["stage1_proba"] = proba_te1[band_mask]
        stage2_proba = stage2.predict_proba(X_te_band)[:, 1]
        final_pred[band_mask] = (stage2_proba >= 0.5).astype(int)

    f1_cascade = f1_score(y_te, final_pred, zero_division=0)
    log(f"  Cascade (stage1 + stage2 on uncertain band): F1={f1_cascade:.4f} "
        f"(stage1 alone was {base_f1:.4f})")
    return {"stage1_f1": float(base_f1), "cascade_f1": float(f1_cascade),
            "stage2_trained": True, "band_size": int(band_mask.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to dataset_eligibilite_features_v3 (parquet/csv)")
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic data to sanity-check the script")
    parser.add_argument("--out", type=str, default="battery_results_v2.json")
    args = parser.parse_args()

    t0 = time.time()
    df = load_data(args.data, demo=args.demo or args.data is None)
    X, y = prep_xy(df)
    log(f"Prepared X={X.shape}, positives={int(y.sum())} ({y.mean()*100:.2f}%)")

    all_results = {"generated_at": datetime.now().isoformat(), "attempts": {}}
    all_results["attempts"]["21_repeated_cv"] = attempt_21_repeated_cv(X, y)
    all_results["attempts"]["22_knn_oracle"] = attempt_22_knn_oracle(X, y)
    all_results["attempts"]["23_wide_hpo"] = attempt_23_wide_hpo(X, y)
    all_results["attempts"]["24_pseudo_labeling"] = attempt_24_pseudo_labeling(X, y)
    all_results["attempts"]["25_monotonic"] = attempt_25_monotonic(X, y)
    all_results["attempts"]["26_uncertain_cascade"] = attempt_26_uncertain_band_cascade(X, y)

    log("=== VERDICT ===")
    cv = all_results["attempts"].get("21_repeated_cv", {})
    if cv:
        for name, s in cv.items():
            log(f"  {name}: {s['mean_f1']:.4f} +/- {s['std_f1']:.4f} "
                f"over {s['n_folds']} folds — this IS your confidence "
                f"interval on the ceiling, not a single lucky/unlucky split.")
    knn = all_results["attempts"].get("22_knn_oracle", {})
    if knn:
        knn_max = max(knn.values())
        log(f"  Model-free k-NN best F1: {knn_max:.4f}. If this is in the "
            f"same 0.20-0.26 band as the gradient boosters, no algorithm "
            f"family will break the ceiling on these features.")
    log(f"Total runtime: {time.time()-t0:.1f}s")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
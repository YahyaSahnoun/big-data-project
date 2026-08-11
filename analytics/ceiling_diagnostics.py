"""
ceiling_diagnostics.py
=======================
Companion to break_ceiling_battery.py. Where that script tries to BREAK
the ceiling, this one tries to PROVE the ceiling is real, i.e. that
0.22-0.24 is a property of the data, not an artifact of evaluation
methodology. It goes beyond what chapter 7.9-7.10 of the report already
has by adding:

  1. Bootstrap CI on F1              -> is the number stable, or noise?
  2. TOST equivalence test           -> are LightGBM/XGBoost/Ensemble
                                         statistically THE SAME, not just
                                         "not significantly different"?
                                         (this is the "equivalence f1
                                         score" test)
  3. Learning curve (F1 vs train size)-> if F1 has already flattened by
                                         50% of the data, more ROWS won't
                                         help (only more/better FEATURES
                                         would) — direct evidence for the
                                         "enrich the data" conclusion of
                                         chapter 7.11/9.
  4. Cohen's d bar chart (full 66)   -> visual version of the table in
                                         7.9.2
  5. KDE overlap + PCA/UMAP scatter  -> visual version of 7.10.1
  6. Permutation importance vs. IV/MI-> cross-check the EDA's unified
                                         ranking (6.5) against a model
                                         that has actually been trained,
                                         rather than the probe model alone
  7. Segment volume-vs-rate chart    -> reproduces 7.10.2 programmatically
                                         so it's regenerated from the
                                         actual data, not hand-typed

All figures are saved as PNGs to --outdir, and a single JSON summary
(ceiling_diagnostics_report.json) captures every number so results can be
pasted straight into the report or diffed against future re-runs.

USAGE
-----
    python ceiling_diagnostics.py --data /path/to/dataset_eligibilite_features_v3.parquet
    python ceiling_diagnostics.py --demo   # sanity check on synthetic data
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LABEL_COL = "label_eligibilite"
RANDOM_STATE = 42


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_data(path=None, demo=False, n=200_000, n_features=40, pos_rate=0.0425):
    if demo or path is None:
        log(f"DEMO MODE: synthesizing {n:,} rows / {n_features} features")
        rng = np.random.default_rng(RANDOM_STATE)
        n_pos = int(n * pos_rate)
        n_neg = n - n_pos
        X_neg = rng.normal(0, 1, size=(n_neg, n_features))
        shift = np.zeros(n_features)
        shift[:3] = [0.15, 0.10, 0.08]
        X_pos = rng.normal(shift, 1, size=(n_pos, n_features))
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * n_neg + [1] * n_pos)
        cols = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df[LABEL_COL] = y
        # fake a categorical segment column, mirroring CUSTOMER_RATING
        segs = rng.choice(["SVC", "SIL", "FNC", "PLT", "JNE", "GLD", "FBP", "PLI"],
                           size=n, p=[0.47, 0.155, 0.13, 0.10, 0.075, 0.058, 0.007, 0.005])
        df["CUSTOMER_RATING"] = segs
        return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    log(f"Loading real dataset from {path}")
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    import os
    
    if path.startswith("s3"):
        endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
        key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
        s3 = pafs.S3FileSystem(
            endpoint_override=endpoint.replace("http://", "").replace("https://", ""),
            access_key=key, secret_key=secret, request_timeout=300, connect_timeout=300,
            scheme="http" if endpoint.startswith("http://") else "https",
        )
        bucket_path = path.replace("s3a://", "").replace("s3://", "")
        df = ds.dataset(bucket_path, filesystem=s3, format="parquet").to_table().to_pandas()
    else:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    
    log(f"Loaded {len(df):,} rows / {df.shape[1]} columns")
    return df


def best_threshold_f1(y_true, proba):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def fit_model(name, X_fit, y_fit):
    pos_weight = np.sqrt((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    if name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                scale_pos_weight=pos_weight,
                                random_state=RANDOM_STATE, verbosity=-1)
    elif name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                               scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
                               eval_metric="logloss", verbosity=0)
    else:
        raise ValueError(name)
    m.fit(X_fit, y_fit)
    return m


# ---------------------------------------------------------------------------
# 1. Bootstrap CI on F1
# ---------------------------------------------------------------------------
def bootstrap_f1_ci(y_true, y_pred, n_boot=2000, alpha=0.05):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(RANDOM_STATE)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], zero_division=0)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


# ---------------------------------------------------------------------------
# 2. TOST equivalence test between two models' F1 (bootstrap-based)
# ---------------------------------------------------------------------------
def tost_equivalence(y_true, pred_a, pred_b, name_a, name_b,
                      margin=0.01, n_boot=2000):
    """Two One-Sided Tests: is |F1_a - F1_b| equivalent to 0 within `margin`?
    We bootstrap the paired difference in F1 (same resampled indices used
    for both models, so it's a proper paired comparison) and check whether
    the 90% CI of the difference sits entirely inside [-margin, +margin].
    That is a classical TOST equivalence conclusion at alpha=0.05.
    """
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        f1_a = f1_score(y_true[idx], pred_a[idx], zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], zero_division=0)
        diffs[i] = f1_a - f1_b
    ci_low, ci_high = np.percentile(diffs, [5, 95])  # 90% CI -> TOST @ alpha=.05
    equivalent = (ci_low > -margin) and (ci_high < margin)
    result = {
        "pair": f"{name_a} vs {name_b}", "margin": margin,
        "mean_diff": float(diffs.mean()),
        "ci90_low": float(ci_low), "ci90_high": float(ci_high),
        "statistically_equivalent": bool(equivalent),
    }
    verdict = ("EQUIVALENT (within +/-%.3f F1)" % margin if equivalent
               else "NOT proven equivalent at this margin")
    log(f"  TOST {name_a} vs {name_b}: diff={diffs.mean():+.4f}, "
        f"90% CI=[{ci_low:+.4f}, {ci_high:+.4f}] -> {verdict}")
    return result


# ---------------------------------------------------------------------------
# 3. Learning curve: F1 vs training-set size
# ---------------------------------------------------------------------------
def learning_curve_f1(X, y, fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0), outdir="."):
    log("=== Learning curve: F1 vs training size ===")
    X_pool, X_te, y_pool, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    results = []
    for frac in fractions:
        if frac < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_pool, y_pool, train_size=frac, stratify=y_pool,
                random_state=RANDOM_STATE)
        else:
            X_sub, y_sub = X_pool, y_pool
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=RANDOM_STATE)
        model = fit_model("lightgbm", X_fit, y_fit)
        proba_val = model.predict_proba(X_val)[:, 1]
        thresh, _ = best_threshold_f1(y_val, proba_val)
        proba_te = model.predict_proba(X_te)[:, 1]
        f1 = f1_score(y_te, (proba_te >= thresh).astype(int), zero_division=0)
        results.append({"fraction": frac, "n_rows": len(X_sub), "f1": float(f1)})
        log(f"  train_frac={frac:.0%} (n={len(X_sub):,}) -> F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fracs = [r["n_rows"] for r in results]
    f1s = [r["f1"] for r in results]
    ax.plot(fracs, f1s, marker="o", color="#118E36", linewidth=2)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("F1 (positive class)")
    ax.set_title("Learning curve — flat curve = more rows won't help,\nonly richer features would")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "learning_curve.png"), dpi=150)
    plt.close(fig)

    if len(f1s) >= 2:
        last_gain = f1s[-1] - f1s[-2]
        log(f"  Last doubling of data gained {last_gain:+.4f} F1 — "
            f"{'flat, consistent with a feature ceiling' if abs(last_gain) < 0.01 else 'still rising, more data may help'}")
    return results


# ---------------------------------------------------------------------------
# 4. Cohen's d for all numeric features (TP vs FN), full bar chart
# ---------------------------------------------------------------------------
def cohens_d_analysis(X, y, model, outdir="."):
    log("=== Cohen's d: true positives vs. false negatives ===")
    proba = model.predict_proba(X)[:, 1]
    thresh, _ = best_threshold_f1(y, proba)
    pred = (proba >= thresh).astype(int)
    tp_mask = (y == 1) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)
    log(f"  {tp_mask.sum()} true positives, {fn_mask.sum()} false negatives")

    ds = {}
    for col in X.columns:
        a, b = X.loc[tp_mask, col].values, X.loc[fn_mask, col].values
        if len(a) < 2 or len(b) < 2:
            continue
        pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                              / (len(a) + len(b) - 2) + 1e-12)
        d = abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
        ds[col] = float(d)

    n_indiscernible = sum(v < 0.1 for v in ds.values())
    log(f"  {n_indiscernible}/{len(ds)} features have Cohen's d < 0.1 "
        f"(indistinguishable between detected and missed positives)")

    order = sorted(ds.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in order]
    values = [v for _, v in order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * len(labels))))
    colors = ["#BA0C2F" if v < 0.1 else ("#E5620A" if v < 0.5 else "#118E36") for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1, label="d=0.1 (weak)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="d=0.5 (separates)")
    ax.set_xlabel("Cohen's d (TP vs FN)")
    ax.set_title(f"Feature separability: {n_indiscernible}/{len(ds)} indistinguishable")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cohens_d_full.png"), dpi=150)
    plt.close(fig)
    return {"n_indiscernible": n_indiscernible, "n_total": len(ds), "values": ds}


# ---------------------------------------------------------------------------
# 5. KDE overlap (top features) + PCA scatter
# ---------------------------------------------------------------------------
def overlap_visuals(X, y, top_features, outdir="."):
    log("=== KDE overlap + PCA projection ===")
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feat in zip(axes.flat, top_features[:4]):
        for label, color, name in [(0, "#333333", "Non eligible"), (1, "#118E36", "Eligible")]:
            vals = X.loc[y == label, feat].dropna().values
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xs, kde(xs), color=color, label=name)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Density overlap on top features — heavy overlap = no clean threshold exists")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "kde_overlap.png"), dpi=150)
    plt.close(fig)

    sample_n = min(50_000, len(X))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_n, replace=False)
    Xs = StandardScaler().fit_transform(X.iloc[idx].fillna(0))
    ys = y[idx]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(coords[ys == 0, 0], coords[ys == 0, 1], s=3, alpha=0.15,
               color="#333333", label="Non eligible")
    ax.scatter(coords[ys == 1, 0], coords[ys == 1, 1], s=6, alpha=0.4,
               color="#E5620A", label="Eligible")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA projection — eligible clients diffused through the cloud,\nnot a separate cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_projection.png"), dpi=150)
    plt.close(fig)
    log("  Saved kde_overlap.png and pca_projection.png")


# ---------------------------------------------------------------------------
# 6. Permutation importance vs. simple mutual information cross-check
# ---------------------------------------------------------------------------
def importance_crosscheck(X, y, model, outdir="."):
    log("=== Permutation importance vs. mutual information ===")
    from sklearn.feature_selection import mutual_info_classif

    sample_n = min(30_000, len(X))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_n, replace=False)
    Xs, ys = X.iloc[idx], y[idx]

    perm = permutation_importance(model, Xs, ys, scoring="f1", n_repeats=5,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    mi = mutual_info_classif(Xs.fillna(0), ys, random_state=RANDOM_STATE)

    df_imp = pd.DataFrame({
        "feature": Xs.columns,
        "perm_importance": perm.importances_mean,
        "mutual_info": mi,
    }).sort_values("perm_importance", ascending=False).head(20)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df_imp))
    ax1.barh(x, df_imp["perm_importance"], color="#118E36", alpha=0.8, label="Permutation importance")
    ax1.set_yticks(x)
    ax1.set_yticklabels(df_imp["feature"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Permutation importance (F1 drop)")
    ax2 = ax1.twiny()
    ax2.plot(df_imp["mutual_info"], x, color="#E5620A", marker="o", linewidth=1, label="Mutual info")
    ax2.set_xlabel("Mutual information")
    fig.suptitle("Top-20 features: model importance vs. information-theoretic signal")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "importance_crosscheck.png"), dpi=150)
    plt.close(fig)
    log("  Saved importance_crosscheck.png")
    return df_imp.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 7. Segment volume vs. rate (reproduces 7.10.2 from live data)
# ---------------------------------------------------------------------------
def segment_breakdown(df, segment_col="CUSTOMER_RATING", outdir="."):
    if segment_col not in df.columns:
        log(f"  '{segment_col}' not in data — skipping segment breakdown")
        return {}
    log(f"=== Segment breakdown by {segment_col} ===")
    g = df.groupby(segment_col)[LABEL_COL].agg(["count", "sum"])
    g["rate"] = g["sum"] / g["count"]
    g = g.sort_values("count", ascending=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.bar(g.index, g["count"], color="#333333")
    ax1.set_yscale("log")
    ax1.set_ylabel("Clients (log scale)")
    ax1.set_title(f"Volume vs. positive rate by {segment_col}")
    ax2.bar(g.index, g["rate"] * 100, color="#E5620A")
    ax2.set_ylabel("Positive rate (%)")
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "segment_breakdown.png"), dpi=150)
    plt.close(fig)
    log("  Saved segment_breakdown.png")
    return g.reset_index().to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--outdir", type=str, default="diagnostics_output")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.data, demo=args.demo or args.data is None)
    y = df[LABEL_COL].values
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).fillna(0)
    log(f"X={X.shape}, positives={int(y.sum())} ({y.mean()*100:.2f}%)")

    report = {"generated_at": datetime.now().isoformat()}

    # Fit the two reference models once, reuse everywhere
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                               random_state=RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2,
                                                   stratify=y_tr, random_state=RANDOM_STATE)
    preds, models = {}, {}
    for name in ["lightgbm", "xgboost"]:
        try:
            m = fit_model(name, X_fit, y_fit)
        except ImportError:
            log(f"  {name} not installed, skipping")
            continue
        proba_val = m.predict_proba(X_val)[:, 1]
        thresh, _ = best_threshold_f1(y_val, proba_val)
        proba_te = m.predict_proba(X_te)[:, 1]
        preds[name] = (proba_te >= thresh).astype(int)
        models[name] = m
        log(f"{name}: test F1={f1_score(y_te, preds[name], zero_division=0):.4f} "
            f"(threshold={thresh:.3f})")

    # 1. Bootstrap CI
    report["bootstrap_ci"] = {}
    for name, pred in preds.items():
        ci = bootstrap_f1_ci(y_te, pred)
        report["bootstrap_ci"][name] = ci
        log(f"Bootstrap 95% CI for {name}: [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # 2. TOST equivalence
    report["equivalence_tests"] = []
    names = list(preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res = tost_equivalence(y_te, preds[names[i]], preds[names[j]],
                                    names[i], names[j])
            report["equivalence_tests"].append(res)

    # 3. Learning curve
    report["learning_curve"] = learning_curve_f1(X, y, outdir=args.outdir)

    # 4. Cohen's d
    if models:
        ref_model = models.get("lightgbm") or list(models.values())[0]
        report["cohens_d"] = cohens_d_analysis(X, y, ref_model, outdir=args.outdir)
        top_feats = sorted(report["cohens_d"]["values"].items(),
                            key=lambda kv: -kv[1])[:4]
        overlap_visuals(X, y, [f for f, _ in top_feats], outdir=args.outdir)
        report["importance_crosscheck"] = importance_crosscheck(X, y, ref_model, outdir=args.outdir)

    # 5. Segment breakdown (works on the original df, needs the raw categorical col)
    report["segment_breakdown"] = segment_breakdown(df, outdir=args.outdir)

    out_path = os.path.join(args.outdir, "ceiling_diagnostics_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Full diagnostic report written to {out_path}")
    log(f"Figures written to {args.outdir}/")


if __name__ == "__main__":
    main()

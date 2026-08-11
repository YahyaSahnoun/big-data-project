"""
close_f1_chapter.py
===================
Definitive comparison of two eligibility datasets.

Primary protocol: paired, apples-to-apples comparison on the same RADICAL
customers when both datasets contain the group key.

Fallback protocol: independent comparison when an older derived bucket has no
RADICAL and cannot be row-paired. This is still useful for closing the chapter,
but the JSON and PNG explicitly mark it as an unpaired population comparison.

Outputs:
  - f1_bucket_comparison.json : folds, schema check, CI and verdict
  - f1_bucket_comparison.png  : report-ready illustration
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


RANDOM_SEED = 42
TARGET_LEAK_COLS = {"label_eligibilite", "label_code", "label_nom"}
ID_COLS = {
    "RADICAL", "BANQUE", "AGENCE", "GENERIC", "PLURAL", "CCLE",
    "DATE_OF_BIRTH", "LIBELLE_VILLE", "digital_date_activation",
    "derniere_operation_gab",
}


def log(message):
    print(message, flush=True)


def s3_filesystem():
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    return pafs.S3FileSystem(
        endpoint_override=endpoint.replace("http://", "").replace("https://", ""),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin123"),
        request_timeout=300, connect_timeout=300,
        scheme="http" if endpoint.startswith("http://") else "https",
    )


def load_dataset(path, filesystem):
    bucket_path = path.replace("s3a://", "").replace("s3://", "")
    return ds.dataset(bucket_path, filesystem=filesystem, format="parquet").to_table().to_pandas()


def fingerprint(values, labels):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(pd.Series(values), index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(labels), index=False).values.tobytes())
    return h.hexdigest()


def validate_group_column(frame, name, group_col):
    if group_col not in frame.columns:
        return None
    if frame[group_col].isna().any():
        raise ValueError(f"{name} dataset has null {group_col}; cannot make a group-safe split.")
    if frame[group_col].duplicated().any():
        raise ValueError(
            f"{name} dataset has duplicate {group_col} values. This script requires one row per "
            f"{group_col} so the pairing/split is unambiguous."
        )
    return frame[group_col].to_numpy()


def choose_protocol(previous, candidate, label_col, group_col, pairing_mode):
    for name, frame in (("previous", previous), ("candidate", candidate)):
        if label_col not in frame.columns:
            raise ValueError(f"{name} dataset lacks required label column: {label_col}")

    previous_has_group = group_col in previous.columns
    candidate_has_group = group_col in candidate.columns

    if previous_has_group and candidate_has_group:
        validate_group_column(previous, "previous", group_col)
        validate_group_column(candidate, "candidate", group_col)
        prev = previous.set_index(group_col, drop=False)
        cand = candidate.set_index(group_col, drop=False)
        common_groups = prev.index.intersection(cand.index).sort_values()
        if len(common_groups) < 1000:
            raise ValueError(f"Only {len(common_groups):,} common {group_col} values; refusing an unreliable comparison.")
        prev = prev.loc[common_groups].reset_index(drop=True)
        cand = cand.loc[common_groups].reset_index(drop=True)
        if not np.array_equal(prev[label_col].to_numpy(), cand[label_col].to_numpy()):
            mismatch = int((prev[label_col].to_numpy() != cand[label_col].to_numpy()).sum())
            raise ValueError(
                f"{mismatch:,} aligned customers have different labels. This is a label-definition change, "
                "not a feature/bucket comparison."
            )
        return prev, cand, prev[group_col].to_numpy(), cand[group_col].to_numpy(), {
            "mode": "radical_intersection",
            "paired_indices": True,
            "comparison_strength": "strong",
            "limitation": None,
        }

    if pairing_mode == "strict":
        missing = []
        if not previous_has_group:
            missing.append(f"previous.{group_col}")
        if not candidate_has_group:
            missing.append(f"candidate.{group_col}")
        raise ValueError(f"Strict mode failed; missing group column(s): {missing}")

    if pairing_mode in {"auto", "row_order"} and (previous_has_group or candidate_has_group) and len(previous) == len(candidate):
        if np.array_equal(previous[label_col].to_numpy(), candidate[label_col].to_numpy()):
            group_source_name = "previous" if previous_has_group else "candidate"
            group_source = previous if previous_has_group else candidate
            groups = validate_group_column(group_source, group_source_name, group_col)
            return previous.reset_index(drop=True), candidate.reset_index(drop=True), groups, groups, {
                "mode": "row_order_proxy",
                "paired_indices": True,
                "comparison_strength": "medium",
                "limitation": (
                    f"{group_col} is missing from one dataset, so customers are paired by row order "
                    f"after equal row-count and row-level label checks. Splits use {group_source_name}.{group_col}."
                ),
            }
        if pairing_mode == "row_order":
            mismatch = int((previous[label_col].to_numpy() != candidate[label_col].to_numpy()).sum())
            raise ValueError(f"Row-order mode failed; {mismatch:,} row-aligned labels differ.")

    if pairing_mode in {"auto", "independent"}:
        previous_groups = validate_group_column(previous, "previous", group_col)
        candidate_groups = validate_group_column(candidate, "candidate", group_col)
        return previous.reset_index(drop=True), candidate.reset_index(drop=True), previous_groups, candidate_groups, {
            "mode": "independent_population",
            "paired_indices": False,
            "comparison_strength": "limited",
            "limitation": (
                f"The datasets cannot be paired: previous rows={len(previous):,}, candidate rows={len(candidate):,}, "
                f"previous_has_{group_col}={previous_has_group}, candidate_has_{group_col}={candidate_has_group}. "
                "Each bucket is evaluated on its own repeated holdout splits with the same common feature protocol. "
                "Use this to discuss whether the F1 wall persists, not as a same-customer A/B claim."
            ),
        }

    raise ValueError("No valid comparison protocol could be selected.")


def common_features(previous, candidate):
    features = sorted((set(previous.columns) & set(candidate.columns)) - (TARGET_LEAK_COLS | ID_COLS))
    if not features:
        raise ValueError("No common non-leaking features remain after exclusions.")
    return features


def make_numeric_pair(previous, candidate, features):
    a, b = previous[features].copy(), candidate[features].copy()
    categorical = []
    for col in features:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            a[col] = pd.to_numeric(a[col], errors="coerce").fillna(-999.0).astype("float32")
            b[col] = pd.to_numeric(b[col], errors="coerce").fillna(-999.0).astype("float32")
        else:
            categorical.append(col)
            levels = pd.Index(pd.concat([a[col], b[col]], ignore_index=True).dropna().astype(str).unique())
            lookup = {value: i for i, value in enumerate(levels)}
            a[col] = a[col].astype(str).map(lookup).fillna(-1).astype("int32")
            b[col] = b[col].astype(str).map(lookup).fillna(-1).astype("int32")
    return a, b, categorical


def best_threshold(y_true, probabilities):
    thresholds = np.linspace(0.02, 0.80, 157)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def make_model(y_train, seed):
    from lightgbm import LGBMClassifier
    positives = max(int(np.sum(y_train)), 1)
    scale_pos_weight = np.sqrt((len(y_train) - positives) / positives)
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def split_indices(y, groups, test_size, tune_size, seed):
    rows = np.arange(len(y))
    if groups is None:
        outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        fit_pool, test_idx = next(outer.split(rows, y))
        inner = StratifiedShuffleSplit(n_splits=1, test_size=tune_size, random_state=seed + 10000)
        fit_rel, tune_rel = next(inner.split(rows[fit_pool], y[fit_pool]))
    else:
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        fit_pool, test_idx = next(outer.split(rows, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=tune_size, random_state=seed + 10000)
        fit_rel, tune_rel = next(inner.split(rows[fit_pool], y[fit_pool], groups[fit_pool]))
    return fit_pool[fit_rel], fit_pool[tune_rel], test_idx


def score_one(X, y, fit_idx, tune_idx, test_idx, seed):
    model = make_model(y[fit_idx], seed)
    model.fit(X.iloc[fit_idx], y[fit_idx])
    threshold = best_threshold(y[tune_idx], model.predict_proba(X.iloc[tune_idx])[:, 1])
    prediction = model.predict_proba(X.iloc[test_idx])[:, 1] >= threshold
    y_test = y[test_idx]
    return {
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "threshold": threshold,
    }


def difference_summary(previous_scores, candidate_scores, practical_margin, paired):
    differences = np.asarray(candidate_scores) - np.asarray(previous_scores)
    n = len(differences)
    mean = float(differences.mean())
    if n < 2 or np.isclose(differences.std(ddof=1), 0):
        low = high = mean
        p_value = None
    else:
        sem = stats.sem(differences)
        low, high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)

    if low > practical_margin:
        verdict = "candidate materially better"
    elif high < -practical_margin:
        verdict = "candidate materially worse"
    elif low >= -practical_margin and high <= practical_margin:
        verdict = "practically equivalent within the pre-registered F1 margin"
    else:
        verdict = "inconclusive: more repeats or a narrower question is needed"
    return {
        "n_repeats": n,
        "paired": paired,
        "mean_difference_candidate_minus_previous": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_value_two_sided": p_value,
        "practical_margin": practical_margin,
        "verdict": verdict,
    }


def make_figure(previous_name, candidate_name, folds, summary, out_path, protocol_note):
    prev = np.array([fold["previous"]["f1"] for fold in folds])
    cand = np.array([fold["candidate"]["f1"] for fold in folds])
    diffs = cand - prev
    margin = summary["practical_margin"]
    paired_word = "Paired" if summary["paired"] else "Independent"

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 0.85], hspace=0.40, wspace=0.30)

    ax = fig.add_subplot(grid[0, 0])
    for a, b in zip(prev, cand):
        ax.plot([0, 1], [a, b], color="#9aa5b1", linewidth=1, alpha=0.75)
        ax.scatter([0, 1], [a, b], c=["#64748b", "#118e36"], s=35, zorder=3)
    ax.set_xticks([0, 1], [previous_name, candidate_name])
    ax.set_ylabel("Held-out F1")
    ax.set_title(f"{paired_word} repeated holdouts")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[0, 1])
    ax.axhspan(-margin, margin, color="#dbeafe", alpha=0.8, label=f"Practical band +/-{margin:.3f}")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.scatter(np.arange(1, len(diffs) + 1), diffs, color="#e5620a", s=42, zorder=3)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Delta F1: candidate - previous")
    ax.set_title("F1 differences")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    delta = summary["mean_difference_candidate_minus_previous"]
    ci = f"[{summary['ci95_low']:+.4f}, {summary['ci95_high']:+.4f}]"
    text = (
        f"Decision: {summary['verdict']}\n"
        f"Mean delta F1 = {delta:+.4f}; 95% CI = {ci}; practical margin = +/-{margin:.3f}; repeats = {summary['n_repeats']}\n"
        f"Protocol: {protocol_note}"
    )
    color = "#118e36" if "equivalent" in summary["verdict"] or "better" in summary["verdict"] else "#ba0c2f"
    ax.text(0.02, 0.55, text, va="center", ha="left", fontsize=11, color="#0f172a",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f8fafc", "edgecolor": color, "linewidth": 2})
    fig.suptitle("Dataset-change F1 decision", x=0.07, ha="left", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def protocol_note(pairing):
    if pairing["mode"] == "radical_intersection":
        return "same RADICAL customers; common non-leaking features; threshold tuned on a separate group-held-out validation split."
    if pairing["mode"] == "row_order_proxy":
        return "row-order paired labels; group-held-out split uses the available RADICAL column; common non-leaking features."
    return "unpaired populations; each bucket evaluated on its own valid holdout split; common non-leaking features."


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    filesystem = s3_filesystem()
    log(f"Loading previous dataset: {args.previous_path}")
    previous_raw = load_dataset(args.previous_path, filesystem)
    log(f"Loading candidate dataset: {args.candidate_path}")
    candidate_raw = load_dataset(args.candidate_path, filesystem)

    previous, candidate, previous_groups, candidate_groups, pairing = choose_protocol(
        previous_raw, candidate_raw, args.label_col, args.group_col, args.pairing_mode
    )
    features = common_features(previous, candidate)
    X_previous, X_candidate, categorical = make_numeric_pair(previous, candidate, features)
    y_previous = previous[args.label_col].astype(int).to_numpy()
    y_candidate = candidate[args.label_col].astype(int).to_numpy()

    log(f"Protocol: {pairing['mode']} ({pairing['comparison_strength']})")
    log(f"Previous rows={len(y_previous):,}, candidate rows={len(y_candidate):,}; common features={len(features)} ({len(categorical)} categorical).")
    if pairing["limitation"]:
        log(f"LIMITATION: {pairing['limitation']}")
    if args.dry_run or args.n_repeats <= 0:
        log("DRY RUN OK: loading, schema validation, protocol selection, and feature encoding succeeded.")
        return

    folds = []
    for repeat in range(args.n_repeats):
        seed = args.seed + repeat
        if pairing["paired_indices"]:
            fit_idx, tune_idx, test_idx = split_indices(y_previous, previous_groups, args.test_size, args.tune_size, seed)
            prev_fit, prev_tune, prev_test = fit_idx, tune_idx, test_idx
            cand_fit, cand_tune, cand_test = fit_idx, tune_idx, test_idx
        else:
            prev_fit, prev_tune, prev_test = split_indices(y_previous, previous_groups, args.test_size, args.tune_size, seed)
            cand_fit, cand_tune, cand_test = split_indices(y_candidate, candidate_groups, args.test_size, args.tune_size, seed)

        previous_result = score_one(X_previous, y_previous, prev_fit, prev_tune, prev_test, seed)
        candidate_result = score_one(X_candidate, y_candidate, cand_fit, cand_tune, cand_test, seed)
        folds.append({
            "repeat": repeat + 1,
            "previous_n_fit": int(len(prev_fit)), "previous_n_tune": int(len(prev_tune)), "previous_n_test": int(len(prev_test)),
            "candidate_n_fit": int(len(cand_fit)), "candidate_n_tune": int(len(cand_tune)), "candidate_n_test": int(len(cand_test)),
            "previous": previous_result, "candidate": candidate_result,
        })
        log(f"Repeat {repeat + 1:02d}/{args.n_repeats}: previous={previous_result['f1']:.4f}, candidate={candidate_result['f1']:.4f}, diff={candidate_result['f1']-previous_result['f1']:+.4f}")

    summary = difference_summary(
        [x["previous"]["f1"] for x in folds],
        [x["candidate"]["f1"] for x in folds],
        args.practical_margin,
        pairing["paired_indices"],
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "label_col": args.label_col,
            "group_col": args.group_col,
            "common_features_only": True,
            "threshold_tuned_on_disjoint_validation": True,
            "model": "LightGBM",
            "seed": args.seed,
            "pairing": pairing,
        },
        "datasets": {
            "previous_path": args.previous_path,
            "candidate_path": args.candidate_path,
            "previous_rows": int(len(y_previous)),
            "candidate_rows": int(len(y_candidate)),
            "previous_positive_rate": float(y_previous.mean()),
            "candidate_positive_rate": float(y_candidate.mean()),
            "previous_identity_label_fingerprint": fingerprint(previous_groups if previous_groups is not None else np.arange(len(y_previous)), y_previous),
            "candidate_identity_label_fingerprint": fingerprint(candidate_groups if candidate_groups is not None else np.arange(len(y_candidate)), y_candidate),
        },
        "schema": {"common_feature_count": len(features), "categorical_feature_count": len(categorical), "common_features": features},
        "folds": folds,
        "f1_difference_summary": summary,
    }
    json_path = os.path.join(args.outdir, "f1_bucket_comparison.json")
    png_path = os.path.join(args.outdir, "f1_bucket_comparison.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    make_figure("Previous bucket", "Candidate bucket", folds, summary, png_path, protocol_note(pairing))
    log(f"\nVERDICT: {summary['verdict']}")
    log(f"JSON: {json_path}\nPNG:  {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 bucket comparison with explicit paired/independent protocols")
    parser.add_argument("--previous-path", required=True, help="Former S3 dataset path")
    parser.add_argument("--candidate-path", required=True, help="Intended S3 dataset path")
    parser.add_argument("--outdir", default="f1_bucket_comparison")
    parser.add_argument("--label-col", default="label_eligibilite")
    parser.add_argument("--group-col", default="RADICAL")
    parser.add_argument("--pairing-mode", choices=["auto", "strict", "row_order", "independent"], default="auto")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--tune-size", type=float, default=0.20)
    parser.add_argument("--practical-margin", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Validate loading/schema/protocol and exit before training.")
    main(parser.parse_args())

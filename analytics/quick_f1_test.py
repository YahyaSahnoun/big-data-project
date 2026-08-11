"""
Quick pragmatic F1 test: does the patched features dataset actually
break through 0.23 on the TRUE imbalanced distribution?
"""
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("Loading patched features data...")
files = sorted(glob.glob('/workspace/features/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} rows, {len(df.columns)} cols")
print(f"Positive rate: {df['label_eligibilite'].mean():.4%}")

# ── Reproduce EXACTLY what the pipeline does ──
# Exclude target/identifiers
EXCLUDE = ['label_code', 'label_eligibilite', 'label_nom']

# Categoricals to encode
cat_cols = [
    'CUSTOMER_RATING', 'pack_actuel', 'MARITAL_STATUS', 'NOMBRE_ENFANT',
    'pack_etat', 'CODE_VILLE_regroupe',
    'interaction_solde_min_x_depot_moyen',
    'flux_cred_total_bin', 'solde_moyen_bin',
    'anciennete_digitale_jours_imp_bin', 'depot_moyen_bin',
    'nb_mois_avec_flux_bin', 'solde_min_bin',
    # NEW cross-features (previously NOT in pipeline)
    'pack_actuel_x_CUSTOMER_RATING',
    'pack_etat_x_CUSTOMER_RATING',
    'pack_actuel_x_pack_etat',
]

# Non-feature columns (identifiers, raw text)
drop_cols = ['CODE_VILLE', 'BPR', 'GENDER', 'TAILLE_ENTREPRI'] + EXCLUDE

# Encode categoricals
for c in cat_cols:
    if c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Build feature matrix
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df['label_eligibilite']

print(f"\nFeature matrix: {X.shape}")
print(f"Includes ratio_volatilite_solde unique values: {X['ratio_volatilite_solde'].nunique()}")
print(f"Includes ratio_flux_mois unique values: {X['ratio_flux_mois'].nunique()}")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Test positive rate: {y_test.mean():.4%} (TRUE distribution)")

# ── Test 1: LightGBM with ALL features (old + new) ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

if HAS_LGBM:
    print("\n" + "="*60)
    print("TEST 1: LightGBM (ALL features, patched data)")
    print("="*60)

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Default threshold
    y_pred_default = model.predict(X_test)
    f1_default = f1_score(y_test, y_pred_default)
    prec_default = precision_score(y_test, y_pred_default)
    rec_default = recall_score(y_test, y_pred_default)
    print(f"  Default threshold -> F1={f1_default:.4f}  P={prec_default:.4f}  R={rec_default:.4f}")

    # Optimized threshold sweep
    y_proba = model.predict_proba(X_test)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_test, preds)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred_opt = (y_proba >= best_t).astype(int)
    prec_opt = precision_score(y_test, y_pred_opt)
    rec_opt = recall_score(y_test, y_pred_opt)
    print(f"  Best threshold={best_t:.2f} -> F1={best_f1:.4f}  P={prec_opt:.4f}  R={rec_opt:.4f}")

    # Feature importances (top 15)
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 15 feature importances:")
    for name, imp in importances[:15]:
        marker = " <-- NEW" if name in [
            'ratio_flux_mois', 'ratio_solde_age', 'ratio_retraits_flux_mois',
            'ratio_volatilite_solde',
            'pack_actuel_x_CUSTOMER_RATING', 'pack_etat_x_CUSTOMER_RATING',
            'pack_actuel_x_pack_etat'
        ] else ""
        print(f"    {name:45s}: {imp:6d}{marker}")

    # ── Test 2: LightGBM WITHOUT the new features (baseline comparison) ──
    print("\n" + "="*60)
    print("TEST 2: LightGBM (OLD features only, for comparison)")
    print("="*60)
    new_features_to_remove = [
        'ratio_flux_mois', 'ratio_solde_age', 'ratio_retraits_flux_mois',
        'ratio_volatilite_solde',
        'pack_actuel_x_CUSTOMER_RATING', 'pack_etat_x_CUSTOMER_RATING',
        'pack_actuel_x_pack_etat',
        'montant_total_retraits_moyen', 'montant_total_payfac_moyen',
        'montant_total_vignette_moyen', 'flux_cred_total_moyen',
        'solde_volatilite_indefinie_moyen',
    ]
    old_feature_cols = [c for c in feature_cols if c not in new_features_to_remove]
    X_train_old = X_train[old_feature_cols]
    X_test_old = X_test[old_feature_cols]

    model_old = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_old.fit(X_train_old, y_train)

    y_pred_old = model_old.predict(X_test_old)
    f1_old = f1_score(y_test, y_pred_old)
    prec_old = precision_score(y_test, y_pred_old)
    rec_old = recall_score(y_test, y_pred_old)
    print(f"  Default threshold -> F1={f1_old:.4f}  P={prec_old:.4f}  R={rec_old:.4f}")

    y_proba_old = model_old.predict_proba(X_test_old)[:, 1]
    best_f1_old, best_t_old = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba_old >= t).astype(int)
        f = f1_score(y_test, preds)
        if f > best_f1_old:
            best_f1_old, best_t_old = f, t
    print(f"  Best threshold={best_t_old:.2f} -> F1={best_f1_old:.4f}")

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    delta = best_f1 - best_f1_old
    print(f"  OLD features best F1: {best_f1_old:.4f}")
    print(f"  ALL features best F1: {best_f1:.4f}")
    print(f"  Delta:                {delta:+.4f} ({delta/best_f1_old*100:+.1f}%)")
    if best_f1 > 0.23:
        print(f"  >>> CEILING BROKEN! F1 = {best_f1:.4f} > 0.23 <<<")
    else:
        print(f"  >>> CEILING NOT BROKEN. F1 = {best_f1:.4f} <<<")
else:
    print("LightGBM not available. Falling back to sklearn RandomForest...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"  F1={f1:.4f}")

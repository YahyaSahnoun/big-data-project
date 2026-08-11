"""
Script de scoring batch (Production)
====================================
Ce script enchaîne les deux modèles comme spécifié dans la section 7.8 :
1. Chargement des données pré-encodées (Spark pipeline → numpy, cache XY).
2. Application du modèle d'éligibilité (LGBMClassifier, 1134 features, seuil optimisé).
3. Filtrage des clients prédits "éligibles" (via le seuil optimisé).
4. Application du modèle produit (One-vs-Rest) sur les éligibles uniquement.
5. Sauvegarde des résultats + export JSON pour le dashboard interactif.

Usage : docker exec jupyter python3 scoring_batch.py
"""
import numpy as np
import pandas as pd
import json
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("   LANCEMENT DU SCORING DE PRODUCTION")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES (pré-encodées par le pipeline Spark)
# ──────────────────────────────────────────────────────────────
print("\n[1/6] Chargement des données pré-encodées (X/y cache)...")

XY_CACHE = "/workspace/xy_cache"
X_fit = np.load(f"{XY_CACHE}/X_fit_f32.npy", mmap_mode="r")
X_val = np.load(f"{XY_CACHE}/X_val_f32.npy", mmap_mode="r")
y_fit = np.load(f"{XY_CACHE}/y_fit.npy", mmap_mode="r")
y_val = np.load(f"{XY_CACHE}/y_val.npy", mmap_mode="r")

# Reconstruct full population (fit + val)
n_total = X_fit.shape[0] + X_val.shape[0]
n_features = X_fit.shape[1]

print(f"  X_fit  : {X_fit.shape}")
print(f"  X_val  : {X_val.shape}")
print(f"  Total  : {n_total:,} clients × {n_features} features")
print(f"  Pos rate fit : {y_fit.mean():.4%}")
print(f"  Pos rate val : {y_val.mean():.4%}")

# Load label_nom for product ground truth
label_nom_fit = np.load(f"{XY_CACHE}/label_nom_fit.npy", allow_pickle=True)
label_nom_val = np.load(f"{XY_CACHE}/label_nom_val.npy", allow_pickle=True)

# ──────────────────────────────────────────────────────────────
# 2. CHARGEMENT DES MODÈLES
# ──────────────────────────────────────────────────────────────
print("\n[2/6] Chargement des modèles sauvegardés...")

CKPT = "/workspace/models_checkpoint"

# Eligibility model (LGBMClassifier, trained on 1134 features)
elig_model = joblib.load(f"{CKPT}/LGBMClassifier.joblib")
print(f"  Modèle éligibilité : LGBMClassifier ({elig_model.n_features_} features)")

# Load threshold metadata
with open(f"{CKPT}/final_model_meta.json", "r") as f:
    meta = json.load(f)

seuil_lgbm = meta["seuil_lgbm"]
seuil_xgb  = meta["seuil_xgb"]
seuil_ovr  = meta["seuil_ovr"]
seuil_decision = meta["seuil_decision"]

print(f"  Seuil LightGBM  : {seuil_lgbm:.4f}")
print(f"  Seuil XGBoost   : {seuil_xgb:.4f}")
print(f"  Seuil OvR       : {seuil_ovr:.4f}")
print(f"  Seuil Décision  : {seuil_decision:.4f}")

# ──────────────────────────────────────────────────────────────
# 3. SCORING ÉLIGIBILITÉ (sur val set = population à scorer)
# ──────────────────────────────────────────────────────────────
print("\n[3/6] Scoring éligibilité sur le jeu de validation...")

# We score the validation set (true holdout, never seen during training)
probas_elig = elig_model.predict_proba(X_val)[:, 1]

# Apply optimized threshold
est_eligible = (probas_elig >= seuil_lgbm).astype(int)

nb_eligibles = est_eligible.sum()
print(f"  Probabilités calculées pour {len(X_val):,} clients.")
print(f"  Seuil appliqué : {seuil_lgbm:.4f}")
print(f"  Clients prédits éligibles : {nb_eligibles:,} ({nb_eligibles/len(X_val):.2%})")

# Performance metrics (we have ground truth on val)
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

f1  = f1_score(y_val, est_eligible)
prec = precision_score(y_val, est_eligible)
rec  = recall_score(y_val, est_eligible)
cm   = confusion_matrix(y_val, est_eligible)

print(f"\n  Performances (validation, distribution réelle) :")
print(f"    F1 Score  : {f1:.4f}")
print(f"    Précision : {prec:.4f}")
print(f"    Rappel    : {rec:.4f}")
print(f"    Matrice de confusion :")
print(f"      TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"      FN={cm[1][0]:,}  TP={cm[1][1]:,}")

# ──────────────────────────────────────────────────────────────
# 4. SCORING PRODUIT (One-vs-Rest sur les éligibles uniquement)
# ──────────────────────────────────────────────────────────────
print("\n[4/6] Modèle Produit (OvR) sur les clients éligibles...")

# Identify indices of eligible clients for product model training
# Train on the fit set using actual product labels (label_nom)
mask_has_product = (label_nom_fit != None) & (label_nom_fit != 'None') & (label_nom_fit != '')
X_prod_train = X_fit[mask_has_product]
y_prod_train = label_nom_fit[mask_has_product]

unique_products = np.unique(y_prod_train)
print(f"  Produits uniques dans le jeu d'entraînement : {list(unique_products)}")
print(f"  Échantillon d'entraînement produit : {len(X_prod_train):,} clients")

from lightgbm import LGBMClassifier

prod_model = LGBMClassifier(
    objective='multiclass',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1
)
prod_model.fit(X_prod_train, y_prod_train)
print(f"  Modèle produit OvR entraîné ({prod_model.n_classes_} classes).")

# Apply to eligible clients ONLY
idx_eligible = np.where(est_eligible == 1)[0]
X_eligible = X_val[idx_eligible]

produit_recommande = np.full(len(X_val), None, dtype=object)
if len(X_eligible) > 0:
    preds_product = prod_model.predict(X_eligible)
    produit_recommande[idx_eligible] = preds_product
    print(f"  Produit recommandé attribué à {len(X_eligible):,} clients éligibles.")

# Product breakdown
from collections import Counter
prod_counts = Counter(preds_product)
print(f"\n  Répartition des produits recommandés :")
for prod, count in prod_counts.most_common():
    print(f"    {prod}: {count:,} ({count/len(preds_product):.1%})")

# ──────────────────────────────────────────────────────────────
# 5. CONSTRUCTION ET SAUVEGARDE DU RÉSULTAT
# ──────────────────────────────────────────────────────────────
print("\n[5/6] Construction du résultat final...")

df_result = pd.DataFrame({
    'client_id': range(len(X_val)),
    'probabilite_eligibilite': probas_elig.round(6),
    'est_eligible': est_eligible,
    'produit_recommande': produit_recommande
})

out_csv = '/workspace/scoring_results.csv'
df_result.to_csv(out_csv, index=False)
print(f"  Résultat CSV sauvegardé : {out_csv}")
print(f"  Lignes : {len(df_result):,}")

# ──────────────────────────────────────────────────────────────
# 6. EXPORT JSON POUR LE DASHBOARD INTERACTIF
# ──────────────────────────────────────────────────────────────
print("\n[6/6] Export JSON pour le dashboard...")

out_dir = '/workspace/error_analysis_results/visuals'
os.makedirs(out_dir, exist_ok=True)

# Probability histogram
hist_all, bins = np.histogram(probas_elig, bins=50)
hist_elig, _ = np.histogram(probas_elig[est_eligible == 1], bins=bins)
hist_not, _  = np.histogram(probas_elig[est_eligible == 0], bins=bins)

# Funnel data
funnel_labels = ['Population Totale', 'Prédits Éligibles']
funnel_values = [int(len(X_val)), int(nb_eligibles)]
for prod, count in prod_counts.most_common():
    funnel_labels.append(f'  → {prod}')
    funnel_values.append(int(count))

dashboard_data = {
    "summary": {
        "total_population": int(len(X_val)),
        "predicted_eligible": int(nb_eligibles),
        "predicted_not_eligible": int(len(X_val) - nb_eligibles),
        "eligibility_rate": float(nb_eligibles / len(X_val)),
        "threshold": float(seuil_lgbm),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec)
    },
    "probability_histogram": {
        "bins": bins.tolist(),
        "counts_all": hist_all.tolist(),
        "counts_eligible": hist_elig.tolist(),
        "counts_not_eligible": hist_not.tolist()
    },
    "product_breakdown": {str(k): int(v) for k, v in prod_counts.most_common()},
    "confusion_matrix": {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1])
    },
    "funnel": {
        "labels": funnel_labels,
        "values": funnel_values
    }
}

json_path = f'{out_dir}/dashboard_data.json'
with open(json_path, 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print(f"  Dashboard JSON sauvegardé : {json_path}")

print("\n" + "=" * 60)
print("   ✅ SCORING TERMINÉ AVEC SUCCÈS")
print("=" * 60)
print(f"\n  Population  : {len(X_val):,}")
print(f"  Éligibles   : {nb_eligibles:,} ({nb_eligibles/len(X_val):.2%})")
print(f"  F1 Score    : {f1:.4f}")
print(f"  Précision   : {prec:.4f}")
print(f"  Rappel      : {rec:.4f}")
print(f"  Produits    : {len(prod_counts)} catégories")
print(f"\n  Fichiers générés :")
print(f"    {out_csv}")
print(f"    {json_path}")

"""
clean_dataset.py — Nettoyage complet Silver -> Gold (scoring épargne)
======================================================================

Remplace l'ancienne version de clean_dataset.py (qui pointait vers des
chemins obsolètes dataset_train_produits/dataset_a_scorer). Cette version
correspond à la Partie 1 (pipeline Spark) de EDA_ultimate_eligibilite.ipynb,
extraite du notebook et rendue non-interactive pour tourner en spark-submit
/ Airflow (suppression de !pip install, matplotlib/seaborn optionnels et
désactivés par défaut, pas de display()).

Étapes appliquées identiquement au train (fit) et au scoring (reload),
via le paramètre is_train de traiter_dataset() :
  1. Doublons (dropDuplicates)
  2. Nulls (règles métier -- cf. section 6.5bis du GUIDE_MAITRE)
  3. Imputation médiane (anciennete_digitale_jours, recence_gab_jours)
  4. Valeurs impossibles (compteurs/montants négatifs, âges absurdes)
  5. Plafonnement statistique (winsorisation IQR, zero-inflated géré à part)
  5bis. Suppression des flags _etait_extreme à faible variance ET faible lien
        avec la cible
  5ter. Suppression des colonnes catégorielles constantes
  6. Réduction de dimensions (colonnes techniques, redondances, age_client,
     solde_volatilite_relative)
  6bis. Imputation médiane de solde_volatilite_relative

Bornes/imputers/listes appris UNIQUEMENT sur le jeu "train" de chaque
dataset, sauvegardés, puis rechargés tels quels si is_train=False (aucune
fuite train -> scoring).

⚠️ IMPORTANT — deux populations, deux jeux d'artefacts :
dataset_eligibilite et dataset_produit sont deux populations différentes
(le second est un sous-ensemble du premier, filtré sur label_eligibilite=1).
Chacun a donc son propre jeu d'imputers/bornes/flags (suffixe _eligibilite /
_produit dans les chemins d'artefacts ci-dessous) pour ne pas se marcher
dessus au moment de l'écriture.

Usage (identique au pattern déjà utilisé pour build_dataset_final.py) :
    docker cp clean_dataset.py spark-master:/opt/spark/work-dir/
    docker exec spark-master /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --conf spark.sql.shuffle.partitions=8 \\
        /opt/spark/work-dir/clean_dataset.py
"""

import json
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import Imputer, ImputerModel

# ============================================================
# Configuration
# ============================================================

spark = (
    SparkSession.builder.appName("clean_dataset_silver_to_gold")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.shuffle.partitions", 8)

# Mettre à True pour générer les boxplots (nécessite matplotlib/seaborn dans
# l'image Spark -- désactivé par défaut car inutile/coûteux en exécution
# batch/Airflow non surveillée).
GENERER_PLOTS = False

IQR_K = 1.5
SEUIL_TAUX_ACTIVATION_FLAG = 0.03
SEUIL_MAX_MOIS_OBSERVES = 36
SEUIL_MAX_ENFANTS = 12
SEUIL_ZERO_INFLATED = 0.5
SEUIL_LIFT_MINIMUM = 1.5

COLS_COMPTAGE_NON_NEGATIVES = [
    "nb_mois_observes_solde", "nb_mois_avec_flux", "nb_operations_gab",
    "nb_retraits", "nb_paiements_digitaux", "nb_vignettes_payees",
    "NOMBRE_ENFANT",
]

COLS_MONTANT_NON_NEGATIFS = [
    "depot_moyen", "flux_cred_moyen", "flux_cred_total",
    "montant_total_gab", "montant_moyen_gab",
    "montant_total_retraits", "montant_total_payfac",
    "montant_total_vignette",
]

COLS_A_PLAFONNER = [
    "solde_moyen", "solde_min", "solde_max", "depot_moyen",
    "flux_cred_moyen", "flux_cred_total",
    "montant_total_gab", "montant_moyen_gab",
    "montant_total_retraits", "montant_total_payfac",
    "montant_total_vignette",
]

IMPUTER_INPUT_COLS = ["anciennete_digitale_jours", "recence_gab_jours"]
IMPUTER_OUTPUT_COLS = ["anciennete_digitale_jours_imp", "recence_gab_jours_imp"]

COLS_CATEGORIELLES_BASSE_CARDINALITE = [
    "GENDER", "TAILLE_ENTREPRI", "pack_actuel", "pack_etat",
    "CUSTOMER_RATING", "MARITAL_STATUS", "BPR",
]
COLS_CATEGORIELLES_HAUTE_CARDINALITE = ["CODE_VILLE"]

DATE_REFERENCE_STR = "31/12/2025"  # année du fichier le plus récent (OPK2025/SOLDE_2025)

SCHEMA_LISTE_COLONNES = StructType([StructField("colonne", StringType(), True)])

# ⚠️ Chemins alignés sur ce qui existe RÉELLEMENT dans le bucket processed-data
# aujourd'hui (build_dataset_final.py écrit dataset_eligibilite/ et
# dataset_produit/, pas les anciens dataset_train_produits/dataset_a_scorer).
DATASETS = {
    "eligibilite": {
        "path_in": "s3a://processed-data/dataset_eligibilite/",
        "path_out": "s3a://processed-data/dataset_eligibilite_final/",
    },
    "produit": {
        "path_in": "s3a://processed-data/dataset_produit/",
        "path_out": "s3a://processed-data/dataset_produit_final/",
    },
}


def artefact_paths(suffixe: str) -> dict:
    """Chemins d'artefacts (imputers/bornes/flags/constantes) propres à chaque
    dataset -- évite que le nettoyage de dataset_produit n'écrase les artefacts
    appris sur dataset_eligibilite (et vice-versa)."""
    base = f"s3a://ml-scoring/models/clean_{suffixe}"
    return {
        "imputer": f"{base}/imputer_anciennete_recence",
        "imputer_volatilite": f"{base}/imputer_solde_volatilite",
        "bornes": f"{base}/outlier_bounds/",
        "flags_a_dropper": f"{base}/flags_extreme_a_dropper/",
        "colonnes_constantes": f"{base}/colonnes_constantes/",
    }


# ============================================================
# 1. Doublons
# ============================================================

def verifier_et_dedupliquer(df: DataFrame, label: str) -> DataFrame:
    """Unicité de RADICAL (normalement déjà garantie par build_dataset_final.py
    -- ce check confirme juste l'absence de régression) + suppression des
    doublons stricts."""
    print(f"\n{'=' * 20} DOUBLONS : {label} {'=' * 20}")
    total = df.count()
    radical_distincts = df.select("RADICAL").distinct().count()
    print(f"Lignes : {total} | RADICAL distincts : {radical_distincts}")
    if total != radical_distincts:
        print(f"ATTENTION : {total - radical_distincts} RADICAL en double -- "
              f"ne devrait pas arriver ici, vérifier build_dataset_final.py")
    else:
        print("OK : RADICAL unique.")

    sans_doublons = df.dropDuplicates().count()
    if total != sans_doublons:
        print(f"ATTENTION : {total - sans_doublons} doublon(s) strict(s) -> suppression")
        df = df.dropDuplicates()
    else:
        print("OK : aucun doublon strict.")
    return df


# ============================================================
# 2. Nulls
# ============================================================

def clean_dataset(df: DataFrame, label: str) -> DataFrame:
    n_avant = df.count()

    if "GENDER" in df.columns:
        df = df.withColumn(
            "GENDER",
            F.when(F.col("GENDER").isin("FÃ©minin", "Féminin"), "F")
             .when(F.col("GENDER") == "Masculin", "M")
             .otherwise(None),
        )

    if "LIBELLE_VILLE" in df.columns:
        df = df.drop("LIBELLE_VILLE")

    subset_dropna = [c for c in ["BPR", "GENDER"] if c in df.columns]
    if subset_dropna:
        df = df.dropna(subset=subset_dropna)

    if "NOMBRE_ENFANT" in df.columns:
        df = df.fillna({"NOMBRE_ENFANT": 0})

    if "TAILLE_ENTREPRI" in df.columns:
        df = df.fillna({"TAILLE_ENTREPRI": "PARTICULIER"})

    pack_cols = {}
    if "pack_actuel" in df.columns:
        pack_cols["pack_actuel"] = "SANS_PACK"
    if "pack_etat" in df.columns:
        pack_cols["pack_etat"] = "SANS_ETAT"
    if pack_cols:
        df = df.fillna(pack_cols)

    cat_non_renseigne = [c for c in ["MARITAL_STATUS", "CUSTOMER_RATING"] if c in df.columns]
    if cat_non_renseigne:
        df = df.fillna({c: "NON_RENSEIGNE" for c in cat_non_renseigne})

    montants_zero = [c for c in ["depot_moyen", "montant_moyen_gab"] if c in df.columns]
    if montants_zero:
        df = df.fillna({c: 0.0 for c in montants_zero})

    if "digital_date_activation" in df.columns:
        df = (
            df.withColumn(
                "jamais_active_digital",
                F.when(F.col("digital_date_activation").isNull(), 1).otherwise(0),
            )
            .withColumn(
                "anciennete_digitale_jours",
                F.when(F.col("digital_date_activation").isNull(), F.lit(None))
                .otherwise(F.datediff(F.current_date(), F.to_date("digital_date_activation", "dd/MM/yyyy"))),
            )
            .drop("digital_date_activation")
        )

    if "derniere_operation_gab" in df.columns:
        df = (
            df.withColumn(
                "jamais_utilise_gab",
                F.when(F.col("derniere_operation_gab").isNull(), 1).otherwise(0),
            )
            .withColumn(
                "recence_gab_jours",
                F.when(F.col("derniere_operation_gab").isNull(), F.lit(None))
                .otherwise(F.datediff(F.current_date(), F.to_date(F.col("derniere_operation_gab"), "dd/MM/yyyy HH:mm:ss"))),
            )
            .drop("derniere_operation_gab")
        )

    n_apres = df.count()
    print(f"    [{label}] Lignes avant : {n_avant} | après nettoyage nulls : {n_apres}")
    return df


def fit_and_apply_imputer_on_train(df_train: DataFrame, imputer_path: str) -> DataFrame:
    cols_present = [c for c in IMPUTER_INPUT_COLS if c in df_train.columns]
    if not cols_present:
        return df_train
    out_cols = [IMPUTER_OUTPUT_COLS[IMPUTER_INPUT_COLS.index(c)] for c in cols_present]

    imputer = Imputer(inputCols=cols_present, outputCols=out_cols, strategy="median")
    imputer_model = imputer.fit(df_train)

    medianes = {c: df_train.approxQuantile(c, [0.5], 0.01)[0] for c in cols_present}
    print(f"Médianes apprises sur le train : {medianes}")

    df_train_imp = imputer_model.transform(df_train).drop(*cols_present)
    imputer_model.write().overwrite().save(imputer_path)
    print(f"Modèle d'imputation sauvegardé : {imputer_path}")
    return df_train_imp


def apply_saved_imputer(df_scorer: DataFrame, imputer_path: str) -> DataFrame:
    cols_present = [c for c in IMPUTER_INPUT_COLS if c in df_scorer.columns]
    if not cols_present:
        return df_scorer
    imputer_model = ImputerModel.load(imputer_path)
    return imputer_model.transform(df_scorer).drop(*cols_present)


# ============================================================
# 3. Diagnostic (facultatif, imprimé mais jamais bloquant)
# ============================================================

def rapport_diagnostic(df: DataFrame, cols: list = None) -> None:
    if cols is None:
        cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    print(f"\n{'=' * 20} DIAGNOSTIC VALEURS ABERRANTES {'=' * 20}")
    for c in cols:
        if c not in df.columns:
            continue
        stats = df.select(F.min(c).alias("min"), F.max(c).alias("max"), F.mean(c).alias("mean")).collect()[0]
        q1, med, q3, p01, p99 = df.approxQuantile(c, [0.25, 0.5, 0.75, 0.01, 0.99], 0.01)
        iqr = q3 - q1
        borne_basse, borne_haute = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        n_negatifs = df.filter(F.col(c) < 0).count()
        n_hors_bornes = df.filter((F.col(c) < borne_basse) | (F.col(c) > borne_haute)).count()
        print(f"{c:28s} min={stats['min']!s:>12} max={stats['max']!s:>14} mean={stats['mean']:.1f} "
              f"médiane={med:.1f} p1={p01:.1f} p99={p99:.1f} "
              f"bornes_IQR=[{borne_basse:.1f},{borne_haute:.1f}] négatifs={n_negatifs} hors_bornes={n_hors_bornes}")


def rapport_dates_naissance(df: DataFrame, col: str = "DATE_OF_BIRTH") -> None:
    if col not in df.columns:
        return
    df_age = df.withColumn("_age_tmp", F.floor(F.datediff(F.current_date(), F.to_date(F.col(col), "dd/MM/yyyy")) / 365.25))
    print(f"\n{'=' * 20} DIAGNOSTIC {col} {'=' * 20}")
    df_age.select(F.min("_age_tmp").alias("age_min"), F.max("_age_tmp").alias("age_max")).show()
    print(f"Dates dans le futur : {df_age.filter(F.col('_age_tmp') < 0).count()}")
    print(f"Âge < 16 ans : {df_age.filter((F.col('_age_tmp') >= 0) & (F.col('_age_tmp') < 16)).count()}")
    print(f"Âge > 100 ans : {df_age.filter(F.col('_age_tmp') > 100).count()}")


def rapport_cardinalite(df: DataFrame, cols: list) -> None:
    print(f"\n{'=' * 20} CARDINALITE (colonnes a surveiller) {'=' * 20}")
    n_total = df.count()
    for c in cols:
        if c not in df.columns:
            continue
        n_uniq = df.select(c).distinct().count()
        print(f"  {c:20s} {n_uniq:5d} modalités sur {n_total} lignes ({n_uniq / n_total:.1%})")


# ============================================================
# 4. Valeurs impossibles
# ============================================================

def corriger_valeurs_impossibles(df: DataFrame, is_train: bool) -> DataFrame:
    n_avant = df.count()

    for c in COLS_COMPTAGE_NON_NEGATIVES:
        if c in df.columns:
            n_neg = df.filter(F.col(c) < 0).count()
            if n_neg > 0:
                print(f"  {c} : {n_neg} valeur(s) négative(s) -> 0")
                df = df.withColumn(c, F.when(F.col(c) < 0, 0).otherwise(F.col(c)))

    c_mois = "nb_mois_observes_solde"
    if c_mois in df.columns:
        df = df.withColumn(f"{c_mois}_etait_extreme", F.when(F.col(c_mois) > SEUIL_MAX_MOIS_OBSERVES, 1).otherwise(0))
        df = df.withColumn(c_mois, F.when(F.col(c_mois) > SEUIL_MAX_MOIS_OBSERVES, F.lit(SEUIL_MAX_MOIS_OBSERVES)).otherwise(F.col(c_mois)))

    c_enfants = "NOMBRE_ENFANT"
    if c_enfants in df.columns:
        df = df.withColumn(f"{c_enfants}_etait_extreme", F.when(F.col(c_enfants) > SEUIL_MAX_ENFANTS, 1).otherwise(0))
        df = df.withColumn(c_enfants, F.when(F.col(c_enfants) > SEUIL_MAX_ENFANTS, F.lit(SEUIL_MAX_ENFANTS)).otherwise(F.col(c_enfants)))

    for c in COLS_MONTANT_NON_NEGATIFS:
        if c in df.columns:
            n_neg = df.filter(F.col(c) < 0).count()
            if n_neg > 0:
                print(f"  {c} : {n_neg} valeur(s) négative(s) -> 0")
                df = df.withColumn(c, F.when(F.col(c) < 0, 0.0).otherwise(F.col(c)))

    if "DATE_OF_BIRTH" in df.columns:
        df = df.withColumn("_age_tmp", F.floor(F.datediff(F.current_date(), F.to_date(F.col("DATE_OF_BIRTH"), "dd/MM/yyyy")) / 365.25))
        n_suspect = df.filter((F.col("_age_tmp") < 16) | (F.col("_age_tmp") > 100)).count()
        if is_train:
            if n_suspect > 0:
                print(f"  DATE_OF_BIRTH : {n_suspect} âge(s) impossible(s) -> lignes supprimées (train)")
            df = df.filter((F.col("_age_tmp") >= 16) & (F.col("_age_tmp") <= 100))
        elif n_suspect > 0:
            print(f"  DATE_OF_BIRTH : {n_suspect} âge(s) impossible(s) (scoring, non supprimés, plafonnés plus loin)")
        df = df.drop("_age_tmp")

    n_apres = df.count()
    if is_train:
        print(f"  Lignes avant/après (valeurs impossibles, train) : {n_avant} -> {n_apres}")
    return df


# ============================================================
# 5. Plafonnement statistique (winsorisation IQR)
# ============================================================

def detecter_colonnes_zero_inflated(df_train: DataFrame, cols: list) -> set:
    n_total = df_train.count()
    zero_inflated = set()
    for c in cols:
        if c not in df_train.columns:
            continue
        part_zero = (df_train.filter(F.col(c) == 0).count() / n_total) if n_total else 0
        if part_zero >= SEUIL_ZERO_INFLATED:
            zero_inflated.add(c)
            print(f"  {c:28s} -> zero-inflated ({part_zero:.0%} de zéros)")
    return zero_inflated


def apprendre_bornes_plafonnement(df_train: DataFrame, bornes_path: str, cols: list = None) -> dict:
    if cols is None:
        cols = COLS_A_PLAFONNER
    print("\nDétection des colonnes zero-inflated (train) :")
    cols_zero_inflated = detecter_colonnes_zero_inflated(df_train, cols)

    bornes = {}
    for c in cols:
        if c not in df_train.columns:
            continue
        est_zi = c in cols_zero_inflated
        base_df = df_train.filter(F.col(c) > 0) if est_zi else df_train
        if base_df.count() < 10:
            continue
        q1, q3 = base_df.approxQuantile(c, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        if est_zi:
            lo = min(lo, 0.0)
        bornes[c] = {"min": lo, "max": hi}

    print(f"\nBornes apprises sur le train ({len(bornes)} colonnes) :")
    for c, b in bornes.items():
        tag = " [zero-inflated]" if c in cols_zero_inflated else ""
        print(f"  {c:28s} -> [{b['min']:.1f}, {b['max']:.1f}]{tag}")

    sauvegarder_bornes(bornes, bornes_path)
    return bornes


def sauvegarder_bornes(bornes: dict, bornes_path: str) -> None:
    rows = [(c, float(b["min"]), float(b["max"])) for c, b in bornes.items()]
    spark.createDataFrame(rows, ["colonne", "borne_min", "borne_max"]) \
        .write.mode("overwrite").json(bornes_path)
    print(f"Bornes sauvegardées : {bornes_path}")


def charger_bornes_plafonnement(bornes_path: str) -> dict:
    df_bornes = spark.read.json(bornes_path)
    return {r["colonne"]: {"min": r["borne_min"], "max": r["borne_max"]} for r in df_bornes.collect()}


def appliquer_plafonnement(df: DataFrame, bornes: dict) -> DataFrame:
    n_plafonnes_total = 0
    for c, b in bornes.items():
        if c not in df.columns:
            continue
        lo, hi = b["min"], b["max"]
        df = df.withColumn(f"{c}_etait_extreme", F.when((F.col(c) < lo) | (F.col(c) > hi), 1).otherwise(0))
        n_hors = df.filter((F.col(c) < lo) | (F.col(c) > hi)).count()
        n_plafonnes_total += n_hors
        if n_hors > 0:
            print(f"  {c} : {n_hors} valeur(s) plafonnée(s) vers [{lo:.1f}, {hi:.1f}]")
        df = df.withColumn(c, F.when(F.col(c) < lo, F.lit(lo)).when(F.col(c) > hi, F.lit(hi)).otherwise(F.col(c)))
    print(f"  Total plafonné (toutes colonnes) : {n_plafonnes_total}")
    return df


# ============================================================
# 5bis. Flags _etait_extreme à faible variance ET faible lien avec la cible
# ============================================================

def identifier_flags_a_dropper(df_train: DataFrame, seuil: float = SEUIL_TAUX_ACTIVATION_FLAG,
                                 seuil_lift: float = SEUIL_LIFT_MINIMUM) -> list:
    flags = [c for c in df_train.columns if c.endswith("_etait_extreme")]
    if not flags:
        return []

    n_total = df_train.count()
    a_cible = "label_eligibilite" in df_train.columns
    taux_eligible_global = (
        df_train.filter(F.col("label_eligibilite") == 1).count() / n_total
        if a_cible and n_total else None
    )

    print(f"\nTaux d'activation des flags _etait_extreme (seuil population = {seuil:.0%}"
          f"{f', seuil lift = {seuil_lift:.1f}x' if a_cible else ' -- label_eligibilite absent, ancienne regle utilisee'}) :")

    flags_a_dropper = []
    for c in sorted(flags):
        n_actifs = df_train.filter(F.col(c) == 1).count()
        taux = n_actifs / n_total if n_total else 0.0

        lift = None
        if a_cible and n_actifs > 0 and taux_eligible_global:
            taux_eligible_si_actif = (
                df_train.filter((F.col(c) == 1) & (F.col("label_eligibilite") == 1)).count() / n_actifs
            )
            lift = taux_eligible_si_actif / taux_eligible_global if taux_eligible_global else None

        rare = taux < seuil
        lie_a_la_cible = (lift is not None) and (lift >= seuil_lift or lift <= 1 / seuil_lift)
        drop = rare and not lie_a_la_cible

        lift_str = f"lift={lift:.2f}x" if lift is not None else "lift=n/a"
        decision = "DROP" if drop else ("garde (rare mais lie a la cible)" if (rare and lie_a_la_cible) else "garde")
        print(f"  {c:45s} {taux:6.2%}  {lift_str:14s} [{decision}]")

        if drop:
            flags_a_dropper.append(c)

    print(f"\n{len(flags_a_dropper)}/{len(flags)} flag(s) a dropper : {flags_a_dropper}")
    return flags_a_dropper


def sauvegarder_flags_a_dropper(flags_a_dropper: list, path: str) -> None:
    spark.createDataFrame([(c,) for c in flags_a_dropper], SCHEMA_LISTE_COLONNES) \
        .write.mode("overwrite").json(path)
    print(f"Liste des flags a dropper sauvegardee ({len(flags_a_dropper)}) : {path}")


def charger_flags_a_dropper(path: str) -> list:
    df_flags = spark.read.schema(SCHEMA_LISTE_COLONNES).json(path)
    return [r["colonne"] for r in df_flags.collect()]


def dropper_flags_extreme(df: DataFrame, flags_a_dropper: list) -> DataFrame:
    flags_presents = [c for c in flags_a_dropper if c in df.columns]
    if flags_presents:
        df = df.drop(*flags_presents)
        print(f"Flags _etait_extreme supprimes ({len(flags_presents)}) : {flags_presents}")
    return df


# ============================================================
# 5ter. Colonnes catégorielles constantes
# ============================================================

def identifier_colonnes_constantes(df_train: DataFrame, cols: list = None) -> list:
    if cols is None:
        cols = COLS_CATEGORIELLES_BASSE_CARDINALITE
    constantes = []
    print("\nCardinalité des colonnes catégorielles basse-cardinalité (train) :")
    for c in cols:
        if c not in df_train.columns:
            continue
        n_uniq = df_train.select(c).distinct().count()
        statut = "-> CONSTANTE, DROP" if n_uniq <= 1 else "OK"
        print(f"  {c:20s} {n_uniq:3d} modalité(s)  {statut}")
        if n_uniq <= 1:
            constantes.append(c)
    return constantes


def sauvegarder_colonnes_constantes(colonnes: list, path: str) -> None:
    spark.createDataFrame([(c,) for c in colonnes], SCHEMA_LISTE_COLONNES) \
        .write.mode("overwrite").json(path)
    print(f"Colonnes constantes sauvegardées ({len(colonnes)}) : {path}")


def charger_colonnes_constantes(path: str) -> list:
    df_cst = spark.read.schema(SCHEMA_LISTE_COLONNES).json(path)
    return [r["colonne"] for r in df_cst.collect()]


def dropper_colonnes_constantes(df: DataFrame, colonnes: list) -> DataFrame:
    presentes = [c for c in colonnes if c in df.columns]
    if presentes:
        df = df.drop(*presentes)
        print(f"Colonnes constantes supprimées ({len(presentes)}) : {presentes}")
    return df


# ============================================================
# 6. Réduction de dimensions
# ============================================================

def reduire_dimensions_et_deriver_features(df: DataFrame) -> DataFrame:
    cols_techniques = ["RADICAL", "BANQUE", "AGENCE", "GENERIC", "PLURAL", "CCLE"]
    df = df.drop(*[c for c in cols_techniques if c in df.columns])

    if "digital_toujours_abonne" in df.columns:
        df = df.drop("digital_toujours_abonne")  # r=-0.999 avec jamais_active_digital

    if all(c in df.columns for c in ["solde_max", "solde_min", "solde_moyen"]):
        df = (
            df.withColumn(
                "solde_volatilite_indefinie",
                F.when(F.col("solde_moyen") <= 0, 1).otherwise(0),
            )
            .withColumn(
                "solde_volatilite_relative",
                F.when(F.col("solde_moyen") > 0, (F.col("solde_max") - F.col("solde_min")) / F.col("solde_moyen"))
                .otherwise(F.lit(None)),
            )
            .drop("solde_max")
        )

    if "flux_cred_moyen" in df.columns:
        df = df.drop("flux_cred_moyen")  # r=0.956 avec flux_cred_total

    if "DATE_OF_BIRTH" in df.columns:
        date_ref = F.to_date(F.lit(DATE_REFERENCE_STR), "dd/MM/yyyy")
        df = (
            df.withColumn("age_client", F.floor(F.datediff(date_ref, F.to_date("DATE_OF_BIRTH", "dd/MM/yyyy")) / 365.25))
            .drop("DATE_OF_BIRTH")
        )

    return df


def fit_and_apply_imputer_volatilite(df_train: DataFrame, imputer_path: str) -> DataFrame:
    if "solde_volatilite_relative" not in df_train.columns:
        return df_train

    imputer = Imputer(
        inputCols=["solde_volatilite_relative"],
        outputCols=["solde_volatilite_relative_imp"],
        strategy="median",
    )
    imputer_model = imputer.fit(df_train)

    mediane = df_train.approxQuantile("solde_volatilite_relative", [0.5], 0.01)[0]
    print(f"Mediane solde_volatilite_relative apprise sur le train : {mediane}")

    df_train_imp = imputer_model.transform(df_train).drop("solde_volatilite_relative")
    imputer_model.write().overwrite().save(imputer_path)
    print(f"Modele d'imputation sauvegarde : {imputer_path}")
    return df_train_imp


def apply_saved_imputer_volatilite(df_scorer: DataFrame, imputer_path: str) -> DataFrame:
    if "solde_volatilite_relative" not in df_scorer.columns:
        return df_scorer
    imputer_model = ImputerModel.load(imputer_path)
    return imputer_model.transform(df_scorer).drop("solde_volatilite_relative")


# ============================================================
# 8. Visualisation (facultative, désactivée par défaut en batch)
# ============================================================

def plot_boxplots_grid(df: DataFrame, cols: list, sample_n: int = 20000, ncols: int = 3):
    if not GENERER_PLOTS:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    n_total = df.count()
    fraction = min(1.0, sample_n / n_total) if n_total else 1.0
    cols_presentes = [c for c in cols if c in df.columns]
    if not cols_presentes:
        return
    sample_pd = df.select(cols_presentes).sample(fraction=fraction, seed=42).toPandas()

    nrows = (len(cols_presentes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    i = 0
    for i, c in enumerate(cols_presentes):
        data = pd.to_numeric(sample_pd[c], errors="coerce").dropna()
        if len(data) == 0:
            continue
        sns.boxplot(x=data, ax=axes[i])
        axes[i].set_title(c, fontsize=10)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f"/tmp/boxplots_{'_'.join(cols_presentes[:2])}.png")
    plt.close()


# ============================================================
# 9. Orchestration
# ============================================================

def traiter_dataset(path_in: str, path_out: str, is_train: bool, label: str, suffixe: str) -> DataFrame:
    print(f"\n{'#' * 25} {label.upper()} {'#' * 25}")
    artefacts = artefact_paths(suffixe)
    df = spark.read.parquet(path_in)

    df = verifier_et_dedupliquer(df, label)
    df = clean_dataset(df, label)

    if is_train:
        df = fit_and_apply_imputer_on_train(df, artefacts["imputer"])
    else:
        df = apply_saved_imputer(df, artefacts["imputer"])

    rapport_diagnostic(df, COLS_A_PLAFONNER)
    rapport_dates_naissance(df)
    rapport_cardinalite(df, COLS_CATEGORIELLES_HAUTE_CARDINALITE)
    plot_boxplots_grid(df, COLS_A_PLAFONNER)

    df = corriger_valeurs_impossibles(df, is_train=is_train)

    if is_train:
        bornes = apprendre_bornes_plafonnement(df, artefacts["bornes"])
    else:
        bornes = charger_bornes_plafonnement(artefacts["bornes"])
    df = appliquer_plafonnement(df, bornes)

    if is_train:
        flags_a_dropper = identifier_flags_a_dropper(df)
        sauvegarder_flags_a_dropper(flags_a_dropper, artefacts["flags_a_dropper"])
    else:
        flags_a_dropper = charger_flags_a_dropper(artefacts["flags_a_dropper"])
    df = dropper_flags_extreme(df, flags_a_dropper)

    if is_train:
        colonnes_constantes = identifier_colonnes_constantes(df)
        sauvegarder_colonnes_constantes(colonnes_constantes, artefacts["colonnes_constantes"])
    else:
        colonnes_constantes = charger_colonnes_constantes(artefacts["colonnes_constantes"])
    df = dropper_colonnes_constantes(df, colonnes_constantes)

    df = reduire_dimensions_et_deriver_features(df)

    if is_train:
        df = fit_and_apply_imputer_volatilite(df, artefacts["imputer_volatilite"])
    else:
        df = apply_saved_imputer_volatilite(df, artefacts["imputer_volatilite"])

    print("\nAprès traitement complet :")
    rapport_diagnostic(df, [c for c in COLS_A_PLAFONNER if c in df.columns])
    plot_boxplots_grid(df, [c for c in COLS_A_PLAFONNER if c in df.columns])

    if "label_nom" in df.columns:
        print("\nÉquilibre des classes de la cible (label_nom) :")
        df.groupBy("label_nom").count().orderBy(F.desc("count")).show(truncate=False)
    if "label_eligibilite" in df.columns:
        print("\nÉquilibre du label d'éligibilité :")
        df.groupBy("label_eligibilite").count().orderBy("label_eligibilite").show()

    print(f"\nÉcriture : {path_out}")
    df.write.mode("overwrite").parquet(path_out)
    print(f"OK : {label} -> {df.count()} lignes, {len(df.columns)} colonnes.")
    return df


if __name__ == "__main__":
    # La population d'éligibilité (totalité de PERIMETRE) est traitée comme 
    # son propre "train" -- is_train=True -- car destinée à entraîner le modèle
    # binaire d'éligibilité.
    traiter_dataset(
        DATASETS["eligibilite"]["path_in"],
        DATASETS["eligibilite"]["path_out"],
        is_train=True,
        label="dataset_eligibilite",
        suffixe="eligibilite",
    )

    print("\nTerminé. Fichiers écrits :")
    print("  - processed-data/dataset_eligibilite_final/  (modèle principal, binaire)")

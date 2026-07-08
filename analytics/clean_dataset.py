"""
clean_dataset.py — Nettoyage post-jointure du dataset scoring épargne
======================================================================

À lancer APRÈS build_dataset_final.py (qui écrit dataset_train_produits
et dataset_a_scorer dans processed-data), et AVANT l'entraînement
(section 7.5 du guide) et AVANT le scoring batch (section 8).

Étapes :
  1. Nettoyage de base (drop colonnes/lignes, fillna catégoriel/0,
     dérivation flag + jours pour les 2 colonnes de dates) -- appliqué
     identiquement aux deux datasets.
  2. Imputation médiane (Option A) de anciennete_digitale_jours et
     recence_gab_jours : l'Imputer est ENTRAÎNÉ UNIQUEMENT sur
     dataset_train_produits, puis rechargé et réappliqué tel quel sur
     dataset_a_scorer -- mêmes médianes des deux côtés, aucune fuite
     entre le jeu d'entraînement et la population à scorer.

Usage :
    docker cp clean_dataset.py spark-master:/opt/spark/work-dir/
    docker exec -it spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.sql.shuffle.partitions=8 \
        /opt/spark/work-dir/clean_dataset.py
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer

spark = (
    SparkSession.builder.appName("clean_dataset")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

IMPUTER_INPUT_COLS = ["anciennete_digitale_jours", "recence_gab_jours"]
IMPUTER_OUTPUT_COLS = ["anciennete_digitale_jours_imp", "recence_gab_jours_imp"]
IMPUTER_MODEL_PATH = "s3a://ml-scoring/models/imputer_anciennete_recence"


def clean_dataset(df: DataFrame) -> DataFrame:
    """
    Applique toutes les règles de nettoyage décidées pour ce projet.
    Chaque étape est commentée avec la décision métier qui la justifie
    (voir section 6.5 / 6.5bis du guide maître pour le détail).
    """

    n_avant = df.count()

    # --- 1. LIBELLE_VILLE : redondant avec CODE_VILLE (le code est fiable,
    #        0 null) -> on garde le code, on jette le libellé plutôt que
    #        d'imputer un texte qui n'apporte rien de plus au modèle ---
    if "LIBELLE_VILLE" in df.columns:
        df = df.drop("LIBELLE_VILLE")

    # --- 2. BPR / GENDER : nulls négligeables (2 et 1 lignes sur l'échantillon
    #        de référence) -> on supprime juste ces lignes, pas d'imputation ---
    subset_dropna = [c for c in ["BPR", "GENDER"] if c in df.columns]
    if subset_dropna:
        df = df.dropna(subset=subset_dropna)

    # --- 3. NOMBRE_ENFANT : null = pas d'enfant, PAS une valeur manquante
    #        à imputer par une médiane -> fillna(0) directement ---
    if "NOMBRE_ENFANT" in df.columns:
        df = df.fillna({"NOMBRE_ENFANT": 0})

    # --- 4. TAILLE_ENTREPRI : null = compte particulier (pas d'entreprise),
    #        pas une donnée manquante -> valeur catégorielle explicite.
    #        Devient un signal utile (particulier vs professionnel) au lieu
    #        d'être dropé ou imputé avec du bruit ---
    if "TAILLE_ENTREPRI" in df.columns:
        df = df.fillna({"TAILLE_ENTREPRI": "PARTICULIER"})

    # --- 4bis. pack_actuel / pack_etat : nulls groupés sur les mêmes clients
    #           (986 chacun) = clients sans pack digital, pas une vraie
    #           valeur manquante -> catégorie explicite, pas le mode ---
    pack_cols = {}
    if "pack_actuel" in df.columns:
        pack_cols["pack_actuel"] = "SANS_PACK"
    if "pack_etat" in df.columns:
        pack_cols["pack_etat"] = "SANS_ETAT"
    if pack_cols:
        df = df.fillna(pack_cols)

    # --- 5. depot_moyen / montant_moyen_gab : null = absence d'activité
    #        observée -> 0, cohérent avec le traitement déjà appliqué à
    #        flux_cred_moyen (NaN = 0, jamais une moyenne) ---
    montants_zero = [c for c in ["depot_moyen", "montant_moyen_gab"] if c in df.columns]
    if montants_zero:
        df = df.fillna({c: 0.0 for c in montants_zero})

    # --- 6. digital_date_activation : une date ne peut pas être mise à 0
    #        (serait confondu avec "activé aujourd'hui"). On dérive une
    #        ancienneté en jours + un flag explicite, puis on jette la
    #        date brute qui n'est de toute façon pas exploitable telle
    #        quelle par VectorAssembler ---
    if "digital_date_activation" in df.columns:
        df = (
            df.withColumn(
                "jamais_active_digital",
                F.when(F.col("digital_date_activation").isNull(), 1).otherwise(0),
            )
            .withColumn(
                "anciennete_digitale_jours",
                F.when(F.col("digital_date_activation").isNull(), F.lit(None))
                .otherwise(
                    F.datediff(
                        F.current_date(),
                        F.to_date("digital_date_activation", "dd/MM/yyyy"),
                    )
                ),
            )
            .drop("digital_date_activation")
        )

    # --- 7. derniere_operation_gab : même logique que #6, format datetime ---
    if "derniere_operation_gab" in df.columns:
        df = (
            df.withColumn(
                "jamais_utilise_gab",
                F.when(F.col("derniere_operation_gab").isNull(), 1).otherwise(0),
            )
            .withColumn(
                "recence_gab_jours",
                F.when(F.col("derniere_operation_gab").isNull(), F.lit(None))
                .otherwise(
                    F.datediff(
                        F.current_date(),
                        F.to_date(
                            F.col("derniere_operation_gab"), "dd/MM/yyyy HH:mm:ss"
                        ),
                    )
                ),
            )
            .drop("derniere_operation_gab")
        )

    n_apres = df.count()
    print(f"    Lignes avant : {n_avant} | après (dropna BPR/GENDER) : {n_apres}")

    return df


def apply_base_cleaning(path_in: str, label: str) -> DataFrame:
    print(f"\n{'=' * 20} NETTOYAGE DE BASE : {label} {'=' * 20}")
    print(f"Lecture : {path_in}")
    df = spark.read.parquet(path_in)

    print("Nulls avant nettoyage :")
    df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show(truncate=False, vertical=True)

    df_clean = clean_dataset(df)
    return df_clean


def fit_and_apply_imputer_on_train(df_train: DataFrame) -> DataFrame:
    """
    Option A : Imputer (médiane) sur anciennete_digitale_jours / recence_gab_jours.
    Fit UNIQUEMENT sur le train (clients_avec_label) -- jamais sur la population
    à scorer, pour éviter toute fuite d'information et rester cohérent avec la
    logique déjà utilisée pour le Pipeline MLlib (fit sur train, transform partout).
    Le modèle d'imputation est sauvegardé pour être réappliqué tel quel sur
    dataset_a_scorer (mêmes médianes, pas recalculées).
    """
    cols_present = [c for c in IMPUTER_INPUT_COLS if c in df_train.columns]
    if not cols_present:
        return df_train

    out_cols = [IMPUTER_OUTPUT_COLS[IMPUTER_INPUT_COLS.index(c)] for c in cols_present]

    imputer = Imputer(inputCols=cols_present, outputCols=out_cols, strategy="median")
    imputer_model = imputer.fit(df_train)

    medianes = {c: df_train.approxQuantile(c, [0.5], 0.01)[0] for c in cols_present}
    print(f"Médianes apprises sur le train : {medianes}")

    df_train_imp = imputer_model.transform(df_train)

    print(f"Sauvegarde du modèle d'imputation : {IMPUTER_MODEL_PATH}")
    imputer_model.write().overwrite().save(IMPUTER_MODEL_PATH)

    return df_train_imp


def apply_saved_imputer(df_scorer: DataFrame) -> DataFrame:
    """Recharge l'Imputer entraîné sur le train et l'applique tel quel au
    dataset à scorer -- mêmes médianes des deux côtés, aucune fuite."""
    from pyspark.ml.feature import ImputerModel

    cols_present = [c for c in IMPUTER_INPUT_COLS if c in df_scorer.columns]
    if not cols_present:
        return df_scorer

    print(f"Chargement du modèle d'imputation : {IMPUTER_MODEL_PATH}")
    imputer_model = ImputerModel.load(IMPUTER_MODEL_PATH)
    return imputer_model.transform(df_scorer)


def show_nulls_and_write(df: DataFrame, path_out: str, label: str):
    print("Nulls après nettoyage complet (base + imputation) :")
    df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show(truncate=False, vertical=True)

    print(f"Écriture : {path_out}")
    df.write.mode("overwrite").parquet(path_out)
    print(f"OK : {label} nettoyé et écrit.\n")


if __name__ == "__main__":
    # Adapter ces deux chemins si vos noms de dossiers Parquet diffèrent
    PATH_TRAIN_IN = "s3a://processed-data/dataset_train_produits/"
    PATH_TRAIN_OUT = "s3a://processed-data/dataset_train_produits_clean/"
    PATH_SCORER_IN = "s3a://processed-data/dataset_a_scorer/"
    PATH_SCORER_OUT = "s3a://processed-data/dataset_a_scorer_clean/"

    # 1. Nettoyage de base + fit de l'Imputer SUR LE TRAIN UNIQUEMENT
    df_train = apply_base_cleaning(PATH_TRAIN_IN, "dataset_train_produits (clients avec label)")
    df_train = fit_and_apply_imputer_on_train(df_train)
    show_nulls_and_write(df_train, PATH_TRAIN_OUT, "dataset_train_produits (clients avec label)")

    # 2. Nettoyage de base + application du MÊME Imputer (rechargé) sur le scoring
    df_scorer = apply_base_cleaning(PATH_SCORER_IN, "dataset_a_scorer (population complète à scorer)")
    df_scorer = apply_saved_imputer(df_scorer)
    show_nulls_and_write(df_scorer, PATH_SCORER_OUT, "dataset_a_scorer (population complète à scorer)")

    print("\nTerminé. anciennete_digitale_jours_imp / recence_gab_jours_imp sont")
    print("désormais sans null (médianes apprises sur le train, réutilisées pour")
    print("le scoring). Utiliser les colonnes *_imp (pas les brutes) dans le")
    print("VectorAssembler, aux côtés des flags jamais_active_digital / jamais_utilise_gab.")
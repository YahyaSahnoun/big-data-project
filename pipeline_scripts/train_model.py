"""
train_model.py — Entraînement production du modèle d'éligibilité (binaire)
============================================================================

Version "allégée" de pipeline_training_v1_5.ipynb pour exécution non
surveillée (Airflow) : PAS de benchmark multi-algorithmes (RandomForest vs
LogisticRegression vs DecisionTree vs NaiveBayes vs XGBoost vs LightGBM,
sections 9/9bis/9quater du notebook), PAS de CrossValidator, PAS de
!pip install, PAS de plots. Un seul algorithme MLlib est entraîné
directement, poids par classe + split 80/20 pour une évaluation honnête,
puis refit sur 100% du train avant sauvegarde -- exactement la logique des
sections 4 à 7 et 10 du notebook, mais figée sur un seul algo.

⚠️ CHOIX DE L'ALGORITHME (ALGO_RETENU ci-dessous) :
Le notebook sélectionne l'algo gagnant dynamiquement via
`comparaison_finale.sort_values("f1_classe1_val", ...)`, qui inclut aussi
XGBoost/LightGBM (sklearn, section 9quater) -- ce tableau complet n'est
disponible qu'en relançant le notebook interactivement, donc pas repris
tel quel ici. Ce script ne couvre QUE la famille MLlib (RandomForest,
LogisticRegression, DecisionTree, NaiveBayes gaussien) -- tous entraînables
directement en spark-submit. Si le benchmark désigne XGBoost/LightGBM comme
vainqueur, ce sont des modèles sklearn : ils ne peuvent PAS tourner dans ce
spark-submit et nécessiteraient une tâche Airflow séparée (conteneur Python
avec sklearn/xgboost/lightgbm/joblib, pas Spark). À adapter le jour où ce
cas se présente.
-> Mets ici le nom du vainqueur réel de ta dernière exécution du notebook
   (parmi "RandomForest", "LogisticRegression", "DecisionTree",
   "NaiveBayes_Gaussian") :
ALGO_RETENU = "RandomForest"

⚠️ BUG CORRIGÉ vs le notebook : MODEL_PATH et PATH_PREDICTIONS_OUT n'étaient
jamais définis dans pipeline_training_v1_5.ipynb (référencés en section 10/11
sans assignation visible dans les cellules fournies) -- définis explicitement
ci-dessous, alignés sur la convention de nommage du GUIDE_MAITRE (section 8).

Usage :
    docker cp train_model.py spark-master:/opt/spark/work-dir/
    docker exec spark-master /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --conf spark.sql.shuffle.partitions=64 \\
        /opt/spark/work-dir/train_model.py
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (
    RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes,
)

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# Entrée : sortie du clean_dataset.py de production (dataset_eligibilite_final)
PATH_TRAIN_IN = "s3a://processed-data/dataset_eligibilite_final/"

# Sortie : modèle sauvegardé (chemin utilisé aussi par score_batch.py)
MODEL_PATH = "s3a://ml-scoring/models/pipeline_eligibilite_v1"

COLS_CATEGORIELLES_BASSE_CARDINALITE = [
    "GENDER", "TAILLE_ENTREPRI", "pack_actuel", "pack_etat",
    "CUSTOMER_RATING", "MARITAL_STATUS", "BPR",
]
COL_HAUTE_CARDINALITE = "CODE_VILLE"
COL_LABEL = "label_eligibilite"
COLS_A_EXCLURE_DES_FEATURES = ["label_code", "label_eligibilite"]


def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("spark://spark-master:7077")
        .appName("train_model_eligibilite")
        .config(
            "spark.jars",
            "/home/jovyan/jars/hadoop-aws-3.3.4.jar,"
            "/home/jovyan/jars/aws-java-sdk-bundle-1.12.262.jar"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.memoryOverhead", "512m")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def colonnes_features_numeriques(df: DataFrame) -> list:
    cols_categorielles_brutes = set(COLS_CATEGORIELLES_BASSE_CARDINALITE + [COL_HAUTE_CARDINALITE])
    return [
        c for c, t in df.dtypes
        if t in ("int", "bigint", "double", "float")
        and c not in COLS_A_EXCLURE_DES_FEATURES
        and c not in cols_categorielles_brutes
    ]


def calculer_poids_classe(df: DataFrame, col_label: str = COL_LABEL) -> DataFrame:
    """Pondération inverse-fréquence adoucie (racine carrée) -- cf. guide
    7.6quater : la formule brute total/(nb_classes*effectif) sur-corrige un
    déséquilibre aussi marqué (rappel correct mais précision ~8-9%)."""
    effectifs = df.groupBy(col_label).count()
    total = df.count()
    nb_classes = effectifs.count()
    poids = effectifs.withColumn(
        "poids_classe", F.sqrt(total / (nb_classes * F.col("count")))
    ).select(col_label, "poids_classe")
    print(f"\nPoids par classe adouci -- sqrt (total={total}, nb_classes={nb_classes}) :")
    poids.orderBy(col_label).show()
    return poids


def construire_stages_encodage(cols_basse_cardinalite: list, col_haute_cardinalite: str):
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cols_basse_cardinalite
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in cols_basse_cardinalite
    ]
    indexer_haute_card = StringIndexer(
        inputCol=col_haute_cardinalite, outputCol=f"{col_haute_cardinalite}_idx", handleInvalid="keep"
    )
    return indexers + encoders + [indexer_haute_card]


def construire_classifieur(nom_algo: str, max_bins: int):
    """Un seul classifieur, choisi par ALGO_RETENU -- pas de grille
    d'hyperparamètres (pas de CrossValidator) : valeurs par défaut
    raisonnables, à ajuster ici une fois qu'un vrai résultat de benchmark
    est disponible."""
    if nom_algo == "RandomForest":
        return RandomForestClassifier(
            labelCol="label_idx", featuresCol="features", predictionCol="prediction",
            probabilityCol="probability", weightCol="poids_classe",
            numTrees=30, maxDepth=5, maxBins=max_bins, seed=RANDOM_SEED,
        )
    if nom_algo == "LogisticRegression":
        return LogisticRegression(
            labelCol="label_idx", featuresCol="features", predictionCol="prediction",
            probabilityCol="probability", weightCol="poids_classe", family="binomial",
        )
    if nom_algo == "DecisionTree":
        return DecisionTreeClassifier(
            labelCol="label_idx", featuresCol="features", predictionCol="prediction",
            probabilityCol="probability", weightCol="poids_classe",
            maxDepth=8, maxBins=max_bins, seed=RANDOM_SEED,
        )
    if nom_algo == "NaiveBayes_Gaussian":
        return NaiveBayes(
            labelCol="label_idx", featuresCol="features", predictionCol="prediction",
            probabilityCol="probability", weightCol="poids_classe", modelType="gaussian",
        )
    raise ValueError(
        f"ALGO_RETENU={nom_algo!r} non supporté par ce script MLlib -- "
        f"attendu parmi RandomForest/LogisticRegression/DecisionTree/NaiveBayes_Gaussian "
        f"(XGBoost/LightGBM nécessitent une tâche Airflow séparée, cf. docstring)."
    )


if __name__ == "__main__":
    spark = get_spark()

    df_train_full = spark.read.parquet(PATH_TRAIN_IN)
    print(f"Lignes chargées : {df_train_full.count()}")
    df_train_full.groupBy(COL_LABEL).count().show()

    feature_cols_numeriques = colonnes_features_numeriques(df_train_full)
    print(f"Features numériques ({len(feature_cols_numeriques)}) : {feature_cols_numeriques}")

    # --- Split 80/20 : sert UNIQUEMENT à mesurer un F1 honnête avant le
    # refit final sur 100% -- reprend la logique du notebook (section 4/8),
    # sans le benchmark multi-algo (sections 9/9bis/9quater).
    df_fit, df_val = df_train_full.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    df_fit.cache()
    df_val.cache()
    print(f"Fit : {df_fit.count()} lignes | Val : {df_val.count()} lignes")

    poids_par_classe_fit = calculer_poids_classe(df_fit)
    df_fit = df_fit.join(poids_par_classe_fit, on=COL_LABEL)

    encodage_stages = construire_stages_encodage(COLS_CATEGORIELLES_BASSE_CARDINALITE, COL_HAUTE_CARDINALITE)
    label_indexer = StringIndexer(inputCol=COL_LABEL, outputCol="label_idx", handleInvalid="error")

    feature_cols_encodees = [f"{c}_ohe" for c in COLS_CATEGORIELLES_BASSE_CARDINALITE] + [f"{COL_HAUTE_CARDINALITE}_idx"]
    feature_cols = feature_cols_numeriques + feature_cols_encodees
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    # maxBins dynamique >= nb de modalités de CODE_VILLE (défaut Spark=32
    # bien trop bas -- cf. notebook section 6, IllegalArgumentException sinon)
    nb_modalites_ville = df_train_full.select(COL_HAUTE_CARDINALITE).distinct().count()
    max_bins = max(32, nb_modalites_ville + 1)
    print(f"CODE_VILLE : {nb_modalites_ville} modalités observées -> maxBins={max_bins}")

    clf = construire_classifieur(ALGO_RETENU, max_bins)
    pipeline = Pipeline(stages=encodage_stages + [label_indexer, assembler, clf])

    print(f"\nEntraînement (fit 80%) -- algo={ALGO_RETENU}...")
    pipeline_model = pipeline.fit(df_fit)

    print("\nÉvaluation sur le set de validation (20%) :")
    predictions_val = pipeline_model.transform(df_val)
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    evaluator_f1_classe1 = MulticlassClassificationEvaluator(
        labelCol="label_idx", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0
    )
    f1_classe1_val = evaluator_f1_classe1.evaluate(predictions_val)
    print(f"F1 classe 1 (validation) : {f1_classe1_val:.4f}")

    # --- Refit sur 100% du train (section 10 du notebook) : le split 80/20
    # ne sert qu'à évaluer honnêtement, pas à priver le modèle final de 20%
    # des données.
    print("\nRefit sur 100% du train...")
    poids_par_classe_full = calculer_poids_classe(df_train_full)
    df_train_full_pondere = df_train_full.join(poids_par_classe_full, on=COL_LABEL)

    pipeline_final = Pipeline(stages=encodage_stages + [label_indexer, assembler, construire_classifieur(ALGO_RETENU, max_bins)])
    pipeline_model_final = pipeline_final.fit(df_train_full_pondere)

    pipeline_model_final.write().overwrite().save(MODEL_PATH)
    print(f"\nPipelineModel sauvegardé : {MODEL_PATH}")
    print(f"F1 classe 1 (validation, avant refit 100%) : {f1_classe1_val:.4f}")

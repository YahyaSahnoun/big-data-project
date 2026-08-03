"""
score_batch.py — Scoring batch en cascade (éligibilité, puis produit)
========================================================================

Reprend la logique décrite au GUIDE_MAITRE section 8 : le scoring batch
enchaîne DEUX modèles, le second (produit, multiclasse) ne s'appliquant
qu'aux clients que le premier (éligibilité, binaire) juge éligibles.

⚠️ ÉTAT ACTUEL DU PROJET (à corriger dans ce script au fur et à mesure) :

1. PAS DE POPULATION "À SCORER" SÉPARÉE POUR L'INSTANT.
   Contrairement à l'ancien cadrage (dataset_a_scorer), le recadrage en 2
   modèles (build_dataset_final.py) fait que dataset_eligibilite couvre déjà
   TOUT PERIMETRE avec un vrai label -- il n'y a plus de "population inconnue"
   à noter pour l'instant. PATH_A_SCORER pointe donc vers un chemin qui
   n'existe pas encore (à créer le jour où de nouveaux clients hors
   PERIMETRE doivent être notés). Le script s'arrête proprement (et fait
   échouer la tâche Airflow, volontairement) tant que ce chemin n'existe pas
   -- pour ne pas écrire silencieusement un résultat vide/trompeur.

2. PAS ENCORE DE MODÈLE PRODUIT (multiclasse) ENTRAÎNÉ.
   pipeline_training_v1_5.ipynb ne couvre QUE l'entraînement du modèle
   d'éligibilité (binaire) -- le modèle produit (section 7.5-7.9 du guide,
   Tomek Links + multiclasse) n'a pas encore de script de production. Tant
   que MODEL_PATH_PRODUIT n'existe pas sur MinIO, l'étape 2 (cascade) est
   sautée automatiquement (AVEC_MODELE_PRODUIT ci-dessous) et seul le score
   d'éligibilité est écrit.

Usage :
    docker cp score_batch.py spark-master:/opt/spark/work-dir/
    docker exec spark-master /opt/spark/bin/spark-submit \\
        --master spark://spark-master:7077 \\
        --conf spark.sql.shuffle.partitions=8 \\
        /opt/spark/work-dir/score_batch.py
"""

import sys

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString

# ============================================================
# Configuration
# ============================================================

# ⚠️ N'existe pas encore -- à créer/pointer vers la vraie source une fois
# qu'une population de nouveaux clients à scorer existe (cf. point 1 ci-dessus).
PATH_A_SCORER = "s3a://processed-data/clients_a_scorer_final/"

MODEL_PATH_ELIGIBILITE = "s3a://ml-scoring/models/pipeline_eligibilite_v1"
MODEL_PATH_PRODUIT = "s3a://ml-scoring/models/pipeline_produit_v1"

PATH_SCORES_OUT = "s3a://ml-scoring/scores_clients/"

# Passera automatiquement à True dès que MODEL_PATH_PRODUIT existera --
# détecté au runtime (cf. def modele_produit_disponible ci-dessous), pas
# besoin de modifier cette constante à la main.
AVEC_MODELE_PRODUIT = None  # calculé au runtime, ne pas modifier ici


def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("score_batch_cascade")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def chemin_existe(spark: SparkSession, path: str) -> bool:
    """Vérifie l'existence d'un chemin s3a:// via l'API Hadoop FS (pas de
    try/except autour d'un .load() qui masquerait d'autres erreurs)."""
    hadoop_conf = spark._jsc.hadoopConfiguration()
    jvm = spark._jvm
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


if __name__ == "__main__":
    spark = get_spark()

    if not chemin_existe(spark, PATH_A_SCORER):
        print(f"ERREUR : {PATH_A_SCORER} n'existe pas encore.")
        print("Ce script suppose une population de clients à scorer distincte de")
        print("dataset_eligibilite (déjà entièrement labellisé, cf. build_dataset_final.py).")
        print("Créer/pointer PATH_A_SCORER vers la vraie source avant de relancer.")
        spark.stop()
        sys.exit(1)  # échec net et explicite -- pas d'écriture silencieuse d'un résultat vide

    AVEC_MODELE_PRODUIT = chemin_existe(spark, MODEL_PATH_PRODUIT)
    if not AVEC_MODELE_PRODUIT:
        print(f"INFO : {MODEL_PATH_PRODUIT} introuvable -- modèle produit pas encore "
              f"entraîné, seul le score d'éligibilité sera calculé.")

    all_clients = spark.read.parquet(PATH_A_SCORER)

    # 1) Modèle principal : éligible ou non, sur TOUTE la population
    pipeline_eligibilite = PipelineModel.load(MODEL_PATH_ELIGIBILITE)
    predictions_eligibilite = pipeline_eligibilite.transform(all_clients)

    # Ré-encodage 0/1 -> label lisible via le StringIndexer du pipeline (même
    # logique que le notebook, section 11 : IndexToString plutôt qu'un UDF manuel)
    label_indexer_model = next(
        s for s in pipeline_eligibilite.stages if s.__class__.__name__ == "StringIndexerModel"
        and s.getOutputCol() == "label_idx"
    )
    converter_eligibilite = IndexToString(
        inputCol="prediction", outputCol="eligible_predit", labels=label_indexer_model.labels
    )
    predictions_eligibilite = converter_eligibilite.transform(predictions_eligibilite)

    cols_id = [c for c in ["RADICAL", "CODE_VILLE"] if c in predictions_eligibilite.columns]
    scores_eligibilite = (
        predictions_eligibilite
        .select(*cols_id, "eligible_predit", "probability")
        .withColumnRenamed("probability", "probabilite_eligibilite")
    )

    if AVEC_MODELE_PRODUIT:
        # 2) Modèle bonus : uniquement sur les clients prédits éligibles
        clients_eligibles = all_clients.join(
            scores_eligibilite.filter("eligible_predit = '1'").select(*cols_id), cols_id
        )
        pipeline_produit = PipelineModel.load(MODEL_PATH_PRODUIT)
        predictions_produit = pipeline_produit.transform(clients_eligibles)

        label_indexer_produit = next(
            s for s in pipeline_produit.stages if s.__class__.__name__ == "StringIndexerModel"
            and s.getOutputCol() == "label_idx"
        )
        converter_produit = IndexToString(
            inputCol="prediction", outputCol="produit_predit", labels=label_indexer_produit.labels
        )
        predictions_produit = converter_produit.transform(predictions_produit)

        scores_produit = (
            predictions_produit
            .select(*cols_id, "produit_predit", "probability")
            .withColumnRenamed("probability", "probabilite_produit")
        )

        # 3) Assemblage : un client non éligible n'a pas de produit_predit (NULL, normal)
        scores = scores_eligibilite.join(scores_produit, cols_id, "left")
    else:
        scores = scores_eligibilite

    scores.coalesce(8).write.mode("overwrite").parquet(PATH_SCORES_OUT)
    print(f"OK : scores écrits dans {PATH_SCORES_OUT} ({scores.count()} clients).")

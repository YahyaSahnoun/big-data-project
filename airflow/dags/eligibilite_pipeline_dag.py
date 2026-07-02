"""
DAG de référence illustrant le flux décrit dans l'architecture :

    Nifi (ingestion)  ->  Airflow (orchestration)  ->  Spark (calcul distribué + entraînement)
                                                            |
                                                            v
                                                  MLflow (registre de modèles)

Le SparkSubmitOperator s'appuie sur pyspark installé dans l'image Airflow
(_PIP_ADDITIONAL_REQUIREMENTS dans docker-compose.yml) pour exécuter la
commande spark-submit vers spark://spark-master:7077.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.base import PokeReturnValue
from airflow.decorators import task

with DAG(
    dag_id="pipeline_eligibilite_epargne",
    description="Ingestion Nifi -> Feature engineering + entrainement Spark -> Registre MLflow",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "eligibilite", "epargne"],
) as dag:

    @task.sensor(poke_interval=30, timeout=1800, mode="reschedule")
    def attendre_fin_flux_nifi() -> PokeReturnValue:
        """
        Vérifie (via l'API REST Nifi) que le flow group d'ingestion
        a bien déposé le dernier lot de données clients avant de lancer
        le calcul Spark. Remplacer par un vrai appel à l'API Nifi
        (http://nifi:8443/nifi-api/...) en production.
        """
        fichier_pret = True  # placeholder : vérifier la présence/fraîcheur du fichier
        return PokeReturnValue(is_done=fichier_pret)

    entrainer_modele = SparkSubmitOperator(
        task_id="entrainer_modele_eligibilite",
        application="/opt/airflow/spark-jobs/train_model.py",
        conn_id="spark_default",  # à configurer dans Airflow UI : spark://spark-master:7077
        executor_memory="2g",
        total_executor_cores=4,
        verbose=True,
    )

    attendre_fin_flux_nifi() >> entrainer_modele

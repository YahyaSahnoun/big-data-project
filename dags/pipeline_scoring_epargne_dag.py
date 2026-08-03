"""
pipeline_scoring_epargne_dag.py
================================

DAG Airflow orchestrant le pipeline de scoring épargne de bout en bout :

    build_dataset  →  clean_dataset  →  train_model  →  score_batch

Chaque étape est un job Spark, exécuté dans le conteneur `spark-master`
via le même pattern que celui déjà utilisé manuellement :

    docker cp <script>.py spark-master:/opt/spark/work-dir/
    docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 ...

PRÉ-REQUIS CÔTÉ INFRA (à faire une seule fois) :
  1. Le conteneur qui exécute Airflow (webserver + scheduler) doit avoir
     accès au démon Docker de l'hôte : monter le socket dans le service
     `airflow` du docker-compose :
         volumes:
           - /var/run/docker.sock:/var/run/docker.sock
     et installer le client `docker` dans l'image Airflow (ou utiliser une
     image type `docker.io/apache/airflow` + `apt-get install docker.io`,
     ou passer par le `docker-cli` binaire statique).
  2. Les scripts .py doivent être présents sur l'hôte (ou dans le volume
     partagé avec le conteneur Airflow) au chemin défini par SCRIPTS_DIR
     ci-dessous.
  3. Airflow et spark-master doivent être sur le même réseau Docker
     (`pipeline-net`) pour que `docker exec spark-master ...` fonctionne
     -- en réalité `docker exec` se fait depuis l'hôte Docker, donc c'est
     surtout `docker cp`/`docker exec` qui doivent être exécutables par
     l'utilisateur du conteneur Airflow (accès au socket suffit, le réseau
     n'a pas besoin d'inclure Airflow lui-même).

⚠️ À ADAPTER avant utilisation :
  - TRAIN_SCRIPT / SCORE_SCRIPT : noms de fichiers réels de vos scripts
    d'entraînement et de scoring batch (je n'ai pas eu accès à
    pipeline_training_v1_5.ipynb / build_dataset_final.py pour connaître
    les noms exacts -- remplace les placeholders ci-dessous).
  - SCRIPTS_DIR : chemin, sur l'hôte (ou volume monté), où se trouvent
    tous les scripts .py à copier dans spark-master.
  - Les chemins S3A (processed-data/..., models/...) doivent correspondre
    à ceux réellement utilisés dans build_dataset_final.py / clean_dataset.py.

TOGGLE MANUEL / PÉRIODIQUE :
  Le planning se pilote via une Airflow Variable nommée `pipeline_schedule`,
  modifiable dans l'UI (Admin > Variables) sans toucher au code :
      - "manual"  -> DAG déclenché uniquement à la main (schedule=None)
      - "daily"   -> tous les jours à 02:00
      - "weekly"  -> tous les lundis à 02:00
  Valeur par défaut si la Variable n'existe pas encore : "manual".
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Nom du conteneur Spark cible
SPARK_CONTAINER = "spark-master"
SPARK_MASTER_URL = "spark://spark-master:7077"
SPARK_WORKDIR_IN_CONTAINER = "/opt/spark/work-dir"
SPARK_SUBMIT_BIN = "/opt/spark/bin/spark-submit"

# Répertoire hôte (ou volume partagé avec Airflow) contenant les scripts .py
# À ADAPTER si vos scripts vivent ailleurs.
SCRIPTS_DIR = "/opt/pipeline_scripts"

# Les 4 scripts existent maintenant réellement dans SCRIPTS_DIR (convertis
# depuis build_dataset_final.py + EDA_ultimate_eligibilite.ipynb Partie 1 +
# pipeline_training_v1_5.ipynb, versions "production" sans benchmark/plots).
BUILD_SCRIPT = "build_dataset_final.py"
CLEAN_SCRIPT = "clean_dataset.py"
TRAIN_SCRIPT = "train_model.py"
SCORE_SCRIPT = "score_batch.py"

# Options spark-submit communes (ajuste selon les besoins mémoire/CPU réels)
SPARK_SUBMIT_CONF = "--conf spark.sql.shuffle.partitions=8"

# ---------------------------------------------------------------------------
# Toggle manuel / périodique piloté par Airflow Variable
# ---------------------------------------------------------------------------

_SCHEDULE_MAP = {
    "manual": None,
    "daily": "0 2 * * *",     # tous les jours à 02h00
    "weekly": "0 2 * * 1",    # tous les lundis à 02h00
}

_schedule_choice = Variable.get("pipeline_schedule", default_var="manual").strip().lower()
SCHEDULE_INTERVAL = _SCHEDULE_MAP.get(_schedule_choice, None)

# ---------------------------------------------------------------------------
# Helper : construit la commande bash "docker cp + docker exec spark-submit"
# ---------------------------------------------------------------------------

def spark_submit_command(script_name: str, extra_conf: str = "") -> str:
    """
    Copie le script dans le conteneur spark-master puis lance spark-submit.
    `set -euo pipefail` garantit que la tâche Airflow échoue si l'une des
    deux commandes échoue (copie ou exécution).
    """
    return f"""
set -euo pipefail
echo ">>> Copie de {script_name} vers {SPARK_CONTAINER}:{SPARK_WORKDIR_IN_CONTAINER}/"
docker cp "{SCRIPTS_DIR}/{script_name}" "{SPARK_CONTAINER}:{SPARK_WORKDIR_IN_CONTAINER}/"

echo ">>> Lancement de spark-submit pour {script_name}"
docker exec "{SPARK_CONTAINER}" {SPARK_SUBMIT_BIN} \\
    --master {SPARK_MASTER_URL} \\
    {SPARK_SUBMIT_CONF} {extra_conf} \\
    "{SPARK_WORKDIR_IN_CONTAINER}/{script_name}"
""".strip()


# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "data-engineering-bp",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="pipeline_scoring_epargne",
    description="Pipeline complet de scoring d'éligibilité épargne : build -> clean -> train -> score",
    default_args=default_args,
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["banque-populaire", "pfa", "scoring-epargne", "spark"],
) as dag:

    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=spark_submit_command(BUILD_SCRIPT),
    )

    clean_dataset = BashOperator(
        task_id="clean_dataset",
        bash_command=spark_submit_command(CLEAN_SCRIPT),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=spark_submit_command(TRAIN_SCRIPT),
    )

    # ⚠️ Échouera tant que PATH_A_SCORER (dans score_batch.py) ne pointe pas
    # vers une vraie population à scorer -- comportement voulu (sys.exit(1)
    # explicite côté script plutôt qu'une écriture silencieuse). Retire cette
    # tâche du DAG tant que ce n'est pas résolu, sinon chaque run se termine
    # en échec ici.
    score_batch = BashOperator(
        task_id="score_batch",
        bash_command=spark_submit_command(SCORE_SCRIPT),
    )

    build_dataset >> clean_dataset >> train_model >> score_batch

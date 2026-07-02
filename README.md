# Cluster Big Data - Éligibilité épargne (Docker Compose)

Transposition en `docker-compose` de l'architecture Master/Workers décrite précédemment.

## ⚠️ Ce que change Docker Compose par rapport à Docker Swarm

Ton précédent cluster Hadoop tournait sur 3 **nœuds physiques/VMs distincts** orchestrés
par Docker Swarm. `docker-compose` seul déploie tous les conteneurs sur **un seul hôte
Docker** — il n'y a pas de vraie répartition multi-machines.

Ici, la séparation Master/Workers est donc **logique** (un service = un rôle), pas
physique. C'est l'usage normal de Compose : environnement de développement/test local,
avant un déploiement multi-nœuds réel.

Deux chemins pour passer en vrai multi-nœuds ensuite, sans réécrire les fichiers :
- **`docker stack deploy -c docker-compose.yml eligibilite`** en mode Swarm (le format
  Compose reste compatible), en ajoutant des contraintes de placement
  (`deploy.placement.constraints`) pour épingler `spark-master`/`nifi`/`airflow` sur le
  nœud manager et `spark-worker-*` sur les nœuds workers — exactement ta topologie
  précédente.
- Kubernetes (Helm charts Airflow/Spark/Nifi officiels) si le volume ou la charge
  dépasse ce que Swarm gère confortablement.

## Structure

```
bigdata-cluster/
├── docker-compose.yml
├── spark/
│   ├── Dockerfile          # image Spark 3.5 + stack ML (même image master/workers)
│   ├── requirements.txt    # pandas, numpy, scikit-learn, xgboost, lightgbm, mlflow
│   └── jobs/
│       └── train_model.py  # job spark-submit : feature engineering + entraînement
├── mlflow/
│   └── Dockerfile
├── airflow/
│   └── dags/
│       └── eligibilite_pipeline_dag.py
└── README.md
```

## Lancement

```bash
cd bigdata-cluster
docker compose up -d --build
```

Le premier démarrage prend quelques minutes (build de l'image Spark + migrations Airflow).

## Accès aux UI

| Service       | URL                     | Identifiants        |
|---------------|--------------------------|----------------------|
| Airflow       | http://localhost:8080    | admin / admin        |
| Nifi          | https://localhost:8443   | admin / ChangeMeNow123! |
| Spark Master UI | http://localhost:8081  | -                     |
| MLflow        | http://localhost:5000    | -                     |

## Configuration à faire une fois l'Airflow UI démarrée

Dans **Admin > Connections**, créer/éditer la connexion `spark_default` :
- Conn Type : `Spark`
- Host : `spark://spark-master`
- Port : `7077`

## Points d'attention avant un usage production

- `_PIP_ADDITIONAL_REQUIREMENTS` dans le service Airflow est pratique pour du dev
  rapide mais réinstalle les paquets à chaque démarrage de conteneur : à remplacer par
  une image Airflow custom (`FROM apache/airflow:2.9.3` + `pip install` figé) en prod.
- Une seule instance Nifi ici : un vrai cluster Nifi multi-nœuds nécessite Zookeeper et
  les variables `NIFI_CLUSTER_IS_NODE`, `NIFI_ZK_CONNECT_STRING`, etc.
- `mlflow.db` en SQLite convient pour une démo ; en production, pointer
  `--backend-store-uri` vers une base Postgres partagée.
- Aucun mot de passe de ce fichier n'est destiné à être utilisé tel quel — à externaliser
  via un `.env` ou un gestionnaire de secrets.

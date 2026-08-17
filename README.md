# Big Data Cluster - Éligibilité Épargne (Savings Eligibility Prediction)

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-F05032?style=for-the-badge&logo=apache-spark&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Apache NiFi](https://img.shields.io/badge/Apache%20NiFi-72B961?style=for-the-badge&logo=Apache&logoColor=white)

## 🎯 Project Overview

This project implements an end-to-end Big Data & Machine Learning pipeline to predict **Savings Eligibility (Éligibilité Épargne)**. It is designed to process large volumes of client data, extract meaningful features, train distributed machine learning models, and score new clients in batch mode. 

The entire infrastructure is containerized using `docker-compose`, providing a robust, logical separation of Master/Worker roles suitable for local development and easily transposable to Docker Swarm or Kubernetes for production.

---

## 📊 Analytics & Machine Learning Pipeline (Core Focus)

The true value of this cluster lies in its robust analytics capabilities, powered by Apache Spark and Python's data science ecosystem. The machine learning workflow is designed to handle imbalanced datasets and high-cardinality categorical features natively in a distributed environment.

### 1. Exploratory Data Analysis (EDA) & Prototyping
The `analytics/` folder contains Jupyter notebooks used by data scientists to explore the data and prototype models before productionizing them:
*   **EDA_advanced_eligibilite_v6.ipynb** & **EDA_visualisation.ipynb**: Deep dive into variable distributions, correlations, and business insights.
*   **diagnostic_erreurs_eligibilite.ipynb**: Analysis of model misclassifications to iteratively improve feature engineering.
*   **pipeline_v1_11_V2.ipynb**: Full interactive pipeline development, including multi-algorithm benchmarking (RandomForest, LightGBM, XGBoost, etc.).

### 2. Distributed Data Preparation (`clean_dataset.py`)
Data quality is handled at scale using PySpark. The pipeline cleans the raw data ingested by NiFi, handles missing values, and prepares a standardized `dataset_eligibilite_final` parquet file stored in MinIO (`processed-data` bucket).

### 3. Model Training & Evaluation (`train_model.py`)
The automated training script leverages **Spark MLlib** for distributed model training:
*   **Feature Engineering**: Uses `StringIndexer`, `OneHotEncoder`, and `VectorAssembler` to handle both low and high-cardinality categorical variables (e.g., `CODE_VILLE`).
*   **Class Imbalance Handling**: Automatically calculates and applies smoothed inverse-frequency class weights (square root strategy) to optimize for recall without sacrificing precision on the minority class.
*   **Evaluation**: The data is split 80/20 to evaluate the **F1-Score** (focused on class 1 - eligible) before refitting on 100% of the dataset for production use.
*   **Algorithms Supported**: Extensible to multiple classifiers like `RandomForest`, `LogisticRegression`, `DecisionTree`, and `NaiveBayes`.

### 4. Batch Scoring (`score_batch.py`)
A dedicated Spark job loads the serialized `PipelineModel` from MinIO (`ml-scoring` bucket) to score new, unseen client records in bulk, outputting the eligibility probabilities and final predictions.

---

## 🏗️ Architecture & Tech Stack

The cluster simulates a modern Big Data Lakehouse architecture:

*   **Ingestion**: **Apache NiFi** reads raw client data and pushes it to object storage.
*   **Object Storage**: **MinIO** acts as the data lake (replacing HDFS), providing S3-compatible storage with buckets for `raw-data`, `processed-data`, and `ml-scoring`.
*   **Data Catalog**: **Hive Metastore** (backed by Postgres) catalogs the structured data, allowing tools to query it seamlessly.
*   **Compute Engine**: **Apache Spark** (Master + Worker) executes the heavy data processing and ML tasks.
*   **Orchestration**: **Apache Airflow** (Webserver + Scheduler) triggers the data pipelines via DAGs (`pipeline_scoring_epargne_dag.py`).
*   **BI / Querying**: **Spark Thrift Server** is exposed on port `10000`, enabling BI tools like Power BI or Tableau to query the processed Hive tables directly.

---

## 📁 Repository Structure

```text
bigdata-cluster/
├── analytics/                 # Jupyter notebooks for EDA, ML prototyping, and sample datasets
├── pipeline_scripts/          # Production PySpark scripts (clean, train, score)
├── dags/                      # Airflow DAGs for pipeline orchestration
├── spark/                     # Spark 3.5 + ML Stack Docker build files
├── airflow/                   # Custom Airflow Dockerfile (with extra dependencies)
├── hive/                      # Hive Metastore build files
├── minio-init/                # Scripts to auto-create S3 buckets
├── pg-init-scripts/           # Postgres multi-database initialization scripts
├── docker-compose.yml         # Main infrastructure definition
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose installed.
*   At least 16GB of RAM available (the cluster is optimized for this budget).

### Launching the Cluster

1.  Clone the repository and navigate to the directory:
    ```bash
    cd bigdata-cluster
    ```
2.  Start the cluster in detached mode (this will build the custom Spark and Airflow images on first run):
    ```bash
    docker compose up -d --build
    ```
    *Note: The first startup takes a few minutes as it builds images and runs database migrations.*

---

## 🌐 Accessing the Services

Once the cluster is up and running, you can access the various user interfaces:

| Service | URL | Default Credentials |
| :--- | :--- | :--- |
| **Apache Airflow** (Orchestration) | [http://localhost:8081](http://localhost:8081) | `admin` / `admin` |
| **Jupyter Lab** (Analytics & EDA) | [http://localhost:8888](http://localhost:8888) | *(Check Docker logs for token if required)* |
| **MinIO Console** (Storage Lake) | [http://localhost:9001](http://localhost:9001) | `minioadmin` / `minioadmin123` |
| **Apache NiFi** (Ingestion) | [https://localhost:8443](https://localhost:8443) | `admin` / `admin12345678` |
| **Spark Master UI** | [http://localhost:8080](http://localhost:8080) | - |
| **Spark Thrift Server** (BI) | `jdbc:hive2://localhost:10000` | - |

---

## ⚙️ Configuration & Post-Launch Steps

### 1. Airflow Spark Connection
Once Airflow is started, configure the Spark connection to allow Airflow to trigger Spark jobs:
1. Go to **Admin > Connections** in the Airflow UI.
2. Edit or create the `spark_default` connection:
   *   **Conn Type**: `Spark`
   *   **Host**: `spark://spark-master`
   *   **Port**: `7077`

### 2. Path Configurations
Ensure that the volume mounts in `docker-compose.yml` point to the correct local paths, especially for NiFi data ingestion (`~/Desktop/data_clients`).

### 3. Production Readiness Warnings
*   **Security**: Do not use the default passwords (`admin`, `minioadmin123`) in a production environment. Use a secrets manager or `.env` files.
*   **Airflow Image**: The current Airflow setup installs packages dynamically. For production, bake these into the Docker image directly.
*   **NiFi Cluster**: This is a single-node NiFi setup. True production clustering requires ZooKeeper.

"""
Job soumis via spark-submit depuis Airflow.

Illustre le pattern décrit dans l'architecture :
1. Spark (executors sur les workers) fait le feature engineering distribué.
2. Le jeu de données réduit est ramené en Pandas sur le driver.
3. XGBoost entraîne le modèle avec la stack ML locale du worker/driver.
4. Le modèle et ses métriques sont loggés dans MLflow (registre côté master).

Pour une vraie distribution de l'entraînement des arbres (gros volumes),
remplacer l'étape 3 par XGBoost4J-Spark ou SynapseML (LightGBM on Spark).
"""

import mlflow
import mlflow.xgboost
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    spark = SparkSession.builder.appName("eligibilite-epargne-training").getOrCreate()

    # 1. Lecture + feature engineering distribué sur les workers
    df = spark.read.parquet("/opt/spark-jobs/data/clients.parquet")
    df_features = (
        df.withColumn("anciennete_mois", F.months_between(F.current_date(), F.col("date_ouverture_compte")))
        .withColumn("ratio_epargne_revenu", F.col("solde_moyen") / F.col("revenu_mensuel"))
        .select(
            "anciennete_mois",
            "ratio_epargne_revenu",
            "nb_produits_detenus",
            "score_risque",
            "eligible",
        )
        .na.drop()
    )

    # 2. Réduction vers Pandas sur le driver (jeu déjà agrégé/filtré par Spark)
    pdf = df_features.toPandas()
    X = pdf.drop(columns=["eligible"])
    y = pdf["eligible"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entraînement XGBoost (stack ML locale du nœud)
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("eligibilite-epargne")

    with mlflow.start_run():
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="auc",
        )
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("auc", auc)
        mlflow.log_params(model.get_params())
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"Modèle entraîné - AUC test : {auc:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()

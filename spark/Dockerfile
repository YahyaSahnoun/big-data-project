# Image Spark 3.5 (Bitnami) enrichie avec la stack ML.
# Le MEME Dockerfile sert au spark-master ET aux spark-worker-*,
# seule la variable d'environnement SPARK_MODE change au runtime
# (voir docker-compose.yml). C'est cohérent avec Hadoop : une seule
# image "worker" pouvait tourner en DataNode ou NodeManager selon
# le daemon lancé au démarrage.

FROM bitnami/spark:3.5

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER 1001

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("discovery")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

# Réduire la verbosité des logs Spark pour ne voir que les erreurs
spark.sparkContext.setLogLevel("WARN")

sc = spark.sparkContext
path = sc._jvm.org.apache.hadoop.fs.Path("s3a://raw-data/")
fs = path.getFileSystem(sc._jsc.hadoopConfiguration())
files = [f.getPath().toString() for f in fs.listStatus(path) if not f.isDirectory()]

print(f"\n>>> {len(files)} fichiers détectés dans S3\n")
for f in files:  # la liste déjà utilisée dans discover_files.py
    df_tmp = spark.read.option("header", "true").option("sep", ";").csv(f)
    cols_matching = [c for c in df_tmp.columns if "CREATE" in c.upper()]
    if cols_matching:
        print(f"{f.split('/')[-1]} : {cols_matching}")
for f in sorted(files):
    nom = f.split("/")[-1]
    print(f"\n{'='*20} ANALYSE : {nom} {'='*20}")
    
    # Lecture en tant que CSV avec délimiteur ';'
    # header=True suppose que votre première ligne contient les noms de colonnes
    df = spark.read.option("header", "true").option("sep", ";").csv(f)
    
    print(f"Nombre total de lignes : {df.count()}")
    print("Aperçu des données :")
    df.show(3, truncate=False)

    
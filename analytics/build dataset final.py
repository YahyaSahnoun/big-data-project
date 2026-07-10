"""
build_dataset_final.py
=======================
Construit dataset_final (1 ligne / client) à partir des fichiers bruts,
suivant la stratégie de la section 6.4 du GUIDE_MAITRE.md :
  PERIMETRE (base) --LEFT JOIN--> cible dédupliquée + features agrégées
puis scinde en (clients_avec_label, clients_a_scorer).

À exécuter par blocs (voir les commentaires "ÉTAPE") : les étapes 2 et 5
contiennent des points d'arrêt volontaires où il faut lire la sortie
avant de continuer, pas juste lancer le script en entier d'un coup.

Lancement (depuis le conteneur spark-master, ou via spark-submit local) :
    spark-submit --master spark://spark-master:7077 build_dataset_final.py
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
spark = (
    SparkSession.builder
    .appName("scoring_epargne_construction_dataset")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

RAW = "s3a://raw-data/"


def read_csv(pattern):
    """Lit un ou plusieurs fichiers bruts (sep=';', encodage latin1)."""
    return (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("encoding", "latin1")
        .csv(f"{RAW}{pattern}")
    )


def to_num(colname):
    """Convertit une colonne montant au format français ('5126,34') en double."""
    return F.regexp_replace(F.col(colname), ",", ".").cast("double")


# ============================================================
# ÉTAPE 0 — Chargement des fichiers bruts
# ============================================================
# Adaptez les motifs (*.txt) si vos noms de fichiers réels diffèrent
# légèrement de ceux du guide — un df.show(2) rapide après chaque lecture
# permet de confirmer que le bon fichier a été chargé.
perimetre = read_csv("ATT_PROD_EPARGNE_PERIMETRE*.txt")
assi      = read_csv("ATT_PROD_EPARGNE_ASSI*.txt")

opk_2023  = read_csv("ATT_HISSAB_OPK2023*.txt")
opk_2024  = read_csv("ATT_PROD_EPARGNE_OPK2024*.txt")
opk_2025  = read_csv("ATT_HISSAB_OPK2025*.txt")

digitaux  = read_csv("ATT_HISSAB_PRODUIT_DIGITAUX*.txt")

solde_1   = read_csv("ATT_PROD_EPARGNE_SOLDE_2025*.txt")
solde_2   = read_csv("ATT_PROD_EPARGNE_SOLDE_COMPLEMENT*.txt")

depot_1   = read_csv("ATT_PROD_EPARGNE_DEPOT_BILANCEIL_2023*.txt")
depot_2   = read_csv("ATT_PROD_EPARGNE_DEPONT_BILANCIEL_COMPLEMENT*.txt")

flux_1    = read_csv("ATT_PROD_EPARGNE_FLUX_2023*.txt")
flux_2    = read_csv("ATT_PROD_EPARGNE_FLUX_2025*.txt")

gab_1     = read_csv("ATT_PROD_EPARGNE_OPERATION_GAB2023*.txt")
gab_2     = read_csv("ATT_PROD_EPARGNE_OPER_GAB_COMPLEMENT*.txt")

retrait_1 = read_csv("ATT_HISSAB_OPE_RETRAIT2023*.txt")
retrait_2 = read_csv("ATT_PROD_EPARGNE_OPER_RETRAIT_COMPLEMENT*.txt")

payfac_1  = read_csv("ATT_PROD_EPARGNE_DIGI_PAYFAC_COMPLEMENT*.txt")
payfac_2  = read_csv("ATT_PROD_EPARGNE_DIGI_PAYFAC_2024*.txt")
payfac_3  = read_csv("ATT_HISSAB_DIGI_PAYFAC2023*.txt")

vignette_1 = read_csv("ATT_PROD_EPARGNE_VIGNETTE.txt")
vignette_2 = read_csv("ATT_PROD_EPARGNE_VIGNETTE_COMPLEMENT*.txt")

print(f"PERIMETRE : {perimetre.count()} lignes, "
      f"{perimetre.select('RADICAL').distinct().count()} RADICAL distincts")


# ============================================================
# ÉTAPE 1 — Vérification + correction : unicité de RADICAL sur PERIMETRE
# ============================================================
# FIX 1 : le simple avertissement ne suffit plus. On inspecte d'abord les
# doublons pour comprendre leur nature (même client avec 2 comptes, ou
# vraies lignes dupliquées ?), puis on déduplique par un row_number() pour
# forcer l'unicité de RADICAL avant toute jointure — sinon ÉTAPE 4 duplique
# des lignes silencieusement et ÉTAPE 5 (n_final == n_perimetre) échouera.
n_total = perimetre.count()
n_distinct = perimetre.select("RADICAL").distinct().count()

if n_total != n_distinct:
    print(f"ATTENTION : RADICAL n'est pas unique seul ({n_distinct} distincts pour "
          f"{n_total} lignes).")

    print("\n>>> Aperçu des RADICAL en double (à examiner avant de trancher) :")
    dup_radicals = perimetre.groupBy("RADICAL").count().filter("count > 1").select("RADICAL")
    perimetre.join(dup_radicals, "RADICAL").orderBy("RADICAL").show(20, truncate=False)

    print(">>> Dédoublonnage appliqué : on garde une ligne par RADICAL "
          "(ordre BANQUE, AGENCE — à ajuster si l'inspection ci-dessus "
          "révèle un critère de sélection plus pertinent, ex. compte le "
          "plus récent).")
    w_perimetre = Window.partitionBy("RADICAL").orderBy(F.col("BANQUE"), F.col("AGENCE"))
    perimetre = (
        perimetre
        .withColumn("rang", F.row_number().over(w_perimetre))
        .filter("rang = 1")
        .drop("rang")
    )
    print(f"PERIMETRE dédoublonné : {perimetre.count()} lignes (devrait être {n_distinct}).")
else:
    print("OK : RADICAL est unique sur PERIMETRE, utilisable seul comme clé de jointure.")

# ============================================================
# ÉTAPE 2 — POINT D'ARRÊT : inventaire et validation de la cible (ASSI)
# ============================================================
print("\n>>> Valeurs distinctes de produit dans ASSI :")
assi.groupBy("CODE_PRODUIT", "LIBELLE_PRODUIT").count().orderBy(F.desc("count")).show(20, truncate=False)
print(">>> ARRÊTEZ-VOUS ICI. Confirmez avec l'encadrant :")
print("    1) que ces valeurs sont bien les 5 produits d'épargne attendus,")
print("    2) qu'aucune n'est un produit d'assurance/prévoyance hors scope,")
print("    avant d'exécuter la suite du script.")

# FIX 2 : sur les 6 codes présents dans ASSI, seuls 3 sont réellement des
# produits d'épargne (09 = MaRetraite, 53 = Avenir Mes Enfants,
# 18 = Epargne Evolution) ; 86, 98, 99 sont des produits d'assurance /
# d'assistance hors scope. On filtre AVANT de vérifier l'exclusivité,
# sinon un client qui a une assurance ET un produit d'épargne apparaît à
# tort comme un "doublon" alors que l'exclusivité annoncée par
# l'encadrant ne concerne que les produits d'épargne entre eux.
# À VALIDER avec l'encadrant avant la soutenance (déduction, pas certitude à 100%).
PRODUITS_EPARGNE_VALIDES = ["53", "09", "18"]  # Avenir Mes Enfants / MaRetraite / Epargne Evolution
assi = assi.filter(F.col("CODE_PRODUIT").isin(PRODUITS_EPARGNE_VALIDES))

print(f"\nASSI filtré aux 3 produits d'épargne confirmés : {assi.count()} lignes")
assi.groupBy("CODE_PRODUIT", "LIBELLE_PRODUIT").count().show()

# Vérification de l'exclusivité produit (annoncée "confirmée" par l'encadrant,
# mais à valider empiriquement quand même) — désormais sur ASSI déjà filtré,
# donc le nombre de doublons attendu doit être bien plus faible qu'avant.
doublons = assi.groupBy("RADICAL").count().filter("count > 1")
nb_doublons = doublons.count()
print(f"\n{nb_doublons} client(s) avec plus d'un produit d'épargne dans ASSI (après filtrage).")

if nb_doublons > 0:
    print("Exclusivité NON respectée -> dédoublonnage par DATE_CHARG le plus récent.")
    w = Window.partitionBy("RADICAL").orderBy(F.col("DATE_CHARG").desc())
    cible = (
        assi
        .withColumn("rang", F.row_number().over(w))
        .filter("rang = 1")
        .select("RADICAL",
                F.col("CODE_PRODUIT").alias("label_code"),
                F.col("LIBELLE_PRODUIT").alias("label_nom"))
    )
else:
    print("Exclusivité confirmée empiriquement, pas de dédoublonnage nécessaire.")
    cible = assi.select("RADICAL",
                         F.col("CODE_PRODUIT").alias("label_code"),
                         F.col("LIBELLE_PRODUIT").alias("label_nom"))

# ============================================================
# ÉTAPE 3 — Agrégation de chaque table 1:N à 1 ligne / client
# ============================================================

# --- OPK / PACK : union des 3 années, on garde le pack le plus récent ---
opk_union = (opk_2023
             .unionByName(opk_2024, allowMissingColumns=True)
             .unionByName(opk_2025, allowMissingColumns=True))
w_opk = Window.partitionBy("RADICAL").orderBy(F.col("DATE_CHARG").desc())
opk_agg = (
    opk_union
    .withColumn("rang", F.row_number().over(w_opk))
    .filter("rang = 1")
    .select("RADICAL",
            F.col("CODE_PACK").alias("pack_actuel"),
            F.col("ETATC").alias("pack_etat"))
)

# --- PRODUIT_DIGITAUX : a priori déjà 1 ligne / client ---
digitaux_clean = digitaux.select(
    "RADICAL",
    F.col("DATE_VAL_ABON").alias("digital_date_activation"),
    F.when(F.col("DATE_RES_ABON").isNull(), F.lit(1)).otherwise(F.lit(0))
     .alias("digital_toujours_abonne"),
)

# --- SOLDE : union puis agrégation temporelle ---
solde_union = solde_1.unionByName(solde_2, allowMissingColumns=True)
solde_agg = (
    solde_union
    .withColumn("SOLDEVERIF_num", to_num("SOLDEVERIF"))
    .groupBy("RADICAL")
    .agg(
        F.avg("SOLDEVERIF_num").alias("solde_moyen"),
        F.min("SOLDEVERIF_num").alias("solde_min"),
        F.max("SOLDEVERIF_num").alias("solde_max"),
        F.count("*").alias("nb_mois_observes_solde"),
    )
)

# --- DEPOT_BILANCIEL ---
depot_union = depot_1.unionByName(depot_2, allowMissingColumns=True)
depot_agg = (
    depot_union
    .withColumn("MONTANT_DEPOT_num", to_num("MONTANT_DEPOT"))
    .groupBy("RADICAL")
    .agg(F.avg("MONTANT_DEPOT_num").alias("depot_moyen"))
)

# --- FLUX : NaN = pas de mouvement ce mois-là -> 0, pas une valeur à imputer ---
flux_union = flux_1.unionByName(flux_2, allowMissingColumns=True)
flux_agg = (
    flux_union
    .withColumn("FLUX_CRED_num", to_num("FLUX_CRED"))
    .fillna(0, subset=["FLUX_CRED_num"])
    .groupBy("RADICAL")
    .agg(
        F.avg("FLUX_CRED_num").alias("flux_cred_moyen"),
        F.sum("FLUX_CRED_num").alias("flux_cred_total"),
        F.count(F.when(F.col("FLUX_CRED_num") > 0, 1)).alias("nb_mois_avec_flux"),
    )
)

# --- OPERATIONS GAB : logique RFM (récence / fréquence / montant) ---
gab_union = gab_1.unionByName(gab_2, allowMissingColumns=True)
gab_agg = (
    gab_union
    .withColumn("MONTANT_num", to_num("MONTANT"))
    .withColumn("DATE_OP_ts", F.to_timestamp("DATE_OP", "dd/MM/yyyy HH:mm:ss"))  # FIX : cast
    # avant le max, sinon comparaison alphabétique de chaînes -> "dernière
    # opération" potentiellement fausse (ex. "01/01/2024" < "31/12/2023"
    # alphabétiquement, alors que 2024 est chronologiquement postérieur)
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_operations_gab"),
        F.sum("MONTANT_num").alias("montant_total_gab"),
        F.avg("MONTANT_num").alias("montant_moyen_gab"),
        F.max("DATE_OP_ts").alias("derniere_operation_gab"),  # FIX : sur la colonne castée
    )
)
# --- RETRAITS ---
#  (investigation post-hoc) : ce fichier N'EST PAS un sous-ensemble
# garanti des opérations GAB malgré ce que suggérait initialement le
# guide (section 3.4). Vérifié empiriquement : 365 122 clients présents
# dans RETRAIT sont absents de GAB, ce qui est impossible pour un vrai
# sous-ensemble. Il s'agit de deux sources indépendantes (fichiers et
# fenêtres temporelles distincts), à traiter comme deux features
# comportementales séparées, sans relation hiérarchique imposée.
retrait_union = retrait_1.unionByName(retrait_2, allowMissingColumns=True)
retrait_agg = (
    retrait_union
    .withColumn("MONTANT_num", to_num("MONTANT"))
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_retraits"),
        F.sum("MONTANT_num").alias("montant_total_retraits"),
    )
)

# --- PAYFAC (paiements digitaux confirmés) ---
payfac_union = (payfac_1
                .unionByName(payfac_2, allowMissingColumns=True)
                .unionByName(payfac_3, allowMissingColumns=True))
payfac_agg = (
    payfac_union
    .withColumn("MONTANT_num", to_num("MONTANT_TOTAL"))
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_paiements_digitaux"),
        F.sum("MONTANT_num").alias("montant_total_payfac"),
    )
)

# --- VIGNETTE (0..N lignes / client) ---
vignette_union = vignette_1.unionByName(vignette_2, allowMissingColumns=True)
total_ttc_col = "TOTAL_TTC" if "TOTAL_TTC" in vignette_union.columns else "TOTAL_TTC/100"
vignette_agg = (
    vignette_union
    .withColumn("TOTAL_TTC_num", to_num(total_ttc_col))
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_vignettes_payees"),
        F.sum("TOTAL_TTC_num").alias("montant_total_vignette"),
    )
)

# ============================================================
# ÉTAPE 4 — Assemblage final : LEFT JOIN depuis PERIMETRE
# ============================================================
dataset_final = (
    perimetre
    .join(cible,          on="RADICAL", how="left")
    .join(opk_agg,         on="RADICAL", how="left")
    .join(digitaux_clean,  on="RADICAL", how="left")
    .join(solde_agg,       on="RADICAL", how="left")
    .join(depot_agg,       on="RADICAL", how="left")
    .join(flux_agg,        on="RADICAL", how="left")
    .join(gab_agg,         on="RADICAL", how="left")
    .join(retrait_agg,     on="RADICAL", how="left")
    .join(payfac_agg,      on="RADICAL", how="left")
    .join(vignette_agg,    on="RADICAL", how="left")
)

# Un compteur/montant absent après un LEFT JOIN = 0 (aucune activité), pas une
# valeur manquante à imputer par une moyenne.
compteurs_a_zero = [
    "nb_operations_gab", "montant_total_gab", "nb_retraits", "montant_total_retraits",
    "nb_paiements_digitaux", "montant_total_payfac", "nb_vignettes_payees",
    "montant_total_vignette", "nb_mois_observes_solde", "nb_mois_avec_flux",
    "flux_cred_total", "digital_toujours_abonne",
]
dataset_final = dataset_final.fillna(
    0, subset=[c for c in compteurs_a_zero if c in dataset_final.columns]
)

# ============================================================
# ÉTAPE 5 — POINT D'ARRÊT : vérifications de cohérence
# ============================================================
n_perimetre = perimetre.count()
n_final = dataset_final.count()
print(f"\nPERIMETRE : {n_perimetre} lignes | dataset_final : {n_final} lignes")
if n_final != n_perimetre:
    print("ATTENTION : le nombre de lignes a changé après les jointures -> une des "
          "tables agrégées en étape 3 contient encore plusieurs lignes pour un même "
          "RADICAL. Corrigez l'agrégation fautive avant de continuer (ne poursuivez "
          "pas l'étape 6 tant que ce nombre ne correspond pas).")
else:
    print("OK : aucune duplication introduite par les jointures.")

print("\nRépartition des classes (produit détenu) :")
dataset_final.groupBy("label_nom").count().orderBy(F.desc("count")).show(10, truncate=False)

# ============================================================
# ÉTAPE 6 — Split population d'entraînement / population à scorer
# ============================================================
clients_avec_label = dataset_final.filter(F.col("label_nom").isNotNull())
clients_a_scorer   = dataset_final.filter(F.col("label_nom").isNull())

print(f"\nClients avec produit connu (entraînement/évaluation) : {clients_avec_label.count()}")
print(f"Clients sans produit connu (population à scorer)      : {clients_a_scorer.count()}")

# ============================================================
# ÉTAPE 7 — Écriture dans processed-data (entrée de la section 7 du guide)
# ============================================================
#clients_avec_label.write.mode("overwrite").parquet("s3a://processed-data/dataset_train_produits/")
clients_a_scorer.write.mode("overwrite").parquet("s3a://processed-data/dataset_a_scorer/")

print("\nTerminé. Fichiers écrits dans processed-data/dataset_train_produits/ "
      "et processed-data/dataset_a_scorer/.")
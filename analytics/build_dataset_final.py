"""
build_dataset_final.py
=======================
Construit dataset_final (1 ligne / client) à partir des fichiers bruts,
suivant la stratégie de la section 6.4 du GUIDE_MAITRE.md :
  PERIMETRE (base) --LEFT JOIN--> cible dédupliquée + features agrégées

RECADRAGE DU PROBLÈME (important, cf. discussion) :
-----------------------------------------------------
Le cadrage initial traitait l'absence de correspondance dans ASSI comme
une population "à scorer" (cible inconnue, à prédire). Ce n'est pas le
cas : ne détenir AUCUN des 3 produits d'épargne EST une information —
c'est une 4e classe à part entière ("aucun produit"), au même titre que
les 3 produits eux-mêmes. On dispose donc en réalité d'un ground truth
complet sur la totalité de PERIMETRE, pas seulement sur le sous-ensemble
présent dans ASSI.

Le besoin métier se découpe donc en DEUX modèles, pas un :
  1) MODÈLE PRINCIPAL — éligibilité : le client détient-il un produit
     d'épargne (n'importe lequel des 3) ou aucun ? Binaire, entraîné sur
     la TOTALITÉ de dataset_final (plus de population "à scorer" pour
     cette tâche : tout le monde a un label 0/1).
  2) MODÈLE BONUS — lequel des 3 produits : multiclasse, entraîné
     UNIQUEMENT sur le sous-ensemble de clients éligibles (label
     d'éligibilité = 1), car "lequel" n'a de sens que si la réponse à la
     première question est "oui".

À exécuter par blocs (voir les commentaires "ÉTAPE") : l'étape 2 contient
un point d'arrêt qui est maintenant un vrai gate (assert), pas seulement
un print -- voir CONFIRME_PRODUITS_EPARGNE ci-dessous. Ne le passez à
True qu'après avoir lu la sortie de l'inventaire ASSI et validé avec
l'encadrant que les 3 codes retenus sont corrects. L'étape 5 reste elle
aussi bloquante (assert sur n_final == n_perimetre).

Lancement (depuis le conteneur spark-master, ou via spark-submit local) :
    spark-submit --master spark://spark-master:7077 build_dataset_final.py
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

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

# ============================================================
# GATE ÉTAPE 2 — remplace l'ancien "print + continuer tout seul"
# ============================================================
# L'ancienne version se contentait d'imprimer "ARRÊTEZ-VOUS ICI" puis
# continuait immédiatement -- un lancement en `spark-submit` de bout en
# bout passait donc ce point d'arrêt sans que personne ne le lise. Ce
# flag rend le point d'arrêt réel : tant qu'il n'est pas passé à True
# (après avoir lu l'inventaire ASSI produit par l'ÉTAPE 2 et confirmé
# avec l'encadrant que 53/09/18 sont bien les 3 seuls codes d'épargne),
# le script s'arrête net avec une AssertionError avant de filtrer/joindre
# quoi que ce soit sur cette base.
CONFIRME_PRODUITS_EPARGNE = True  # <-- ne passer à True qu'après relecture de l'inventaire ASSI

# ============================================================
# DATE DE RÉFÉRENCE — dérivée du nom des fichiers les plus récents
# ============================================================
# Les données couvrent plusieurs années (fichiers suffixés _2023/_2024/_2025).
# current_date() n'aurait aucun sens pour un âge/ancienneté/récence : la
# date d'exécution du script n'a aucun lien avec la période observée dans
# les données. Référence unique, dérivée de l'année la plus récente
# présente dans les noms de fichiers, réutilisée pour TOUT calcul
# d'âge/ancienneté/récence (y compris les nouvelles features
# anciennete_digitale_jours / recence_gab_jours ci-dessous).
ANNEE_REFERENCE = 2025  # année du suffixe le plus récent (OPK2025 / SOLDE_2025 / FLUX_2025)
DATE_REFERENCE = F.to_date(F.lit(f"31/12/{ANNEE_REFERENCE}"), "dd/MM/yyyy")


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


def to_date_charg(colname="DATE_CHARG"):
    """
    Convertit DATE_CHARG (chaîne 'dd/MM/yyyy') en vraie date, pour un tri
    chronologique correct. NE JAMAIS trier '.desc()' sur la colonne brute :
    un tri alphabétique classerait "01/11/2023" APRÈS "01/02/2025" (le
    caractère '1' de "11" est > au caractère '0' de "02" en 4e position),
    ce qui inverserait "le plus récent". Ce bug est particulièrement grave
    pour ASSI : le fichier est un instantané mensuel répété (~28 lignes par
    client en moyenne, cf. investigation), donc "le plus récent" détermine
    littéralement le PRODUIT ASSIGNÉ AU CLIENT (label_nom) -- un mauvais
    tri peut assigner le mauvais produit à des centaines de milliers de
    clients silencieusement (aucune erreur levée).
    """
    return F.to_date(F.col(colname), "dd/MM/yyyy")


def verifier_unicite_radical(tables: dict, arreter_si_probleme: bool = True) -> None:
    """
    Vérifie, pour chaque table agrégée candidate à une jointure sur
    RADICAL, qu'elle contient bien au maximum 1 ligne par RADICAL --
    AVANT de les joindre, pas après (c'est ce contrôle, ajouté après coup,
    qui a permis de diagnostiquer le bug produit_digitaux : 1 790 451
    lignes pour seulement 1 765 730 RADICAL distincts).

    arreter_si_probleme=True lève une AssertionError si un problème est
    détecté, au lieu de se contenter d'un avertissement -- pour ne plus
    jamais reproduire un dataset_final silencieusement corrompu (cf. guide
    maître, tableau des bugs : "remplacer le print() par un assert").
    """
    print("\n>>> Vérification unicité RADICAL, table par table (avant jointure) :")
    problemes = []
    for nom, df_feat in tables.items():
        total = df_feat.count()
        distinct = df_feat.select("RADICAL").distinct().count()
        if total == distinct:
            statut = "OK"
        else:
            statut = f"⚠ PROBLÈME (+{total - distinct} lignes en trop)"
            problemes.append(nom)
        print(f"  {nom:24s} total={total:>10} distinct={distinct:>10}  {statut}")

    if problemes:
        message = (
            f"Table(s) non uniques sur RADICAL avant jointure : {problemes}. "
            f"Corrigez leur agrégation/dédoublonnage avant de continuer."
        )
        if arreter_si_probleme:
            raise AssertionError(message)
        else:
            print(f"ATTENTION : {message}")


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

# --- Âge du client, calculé par rapport à DATE_REFERENCE (pas current_date()) ---
perimetre = perimetre.withColumn(
    "age_client",
    F.floor(F.datediff(DATE_REFERENCE, F.to_date("DATE_OF_BIRTH", "dd/MM/yyyy")) / 365.25)
)


# ============================================================
# ÉTAPE 1 — Vérification + correction : unicité de RADICAL sur PERIMETRE
# ============================================================
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
print(">>> Lisez la sortie ci-dessus AVANT de continuer. Confirmez avec l'encadrant :")
print("    1) que ces valeurs sont bien les 3 produits d'épargne attendus,")
print("    2) qu'aucune n'est un produit d'assurance/prévoyance hors scope.")

# GATE réel (et non plus un simple print) : tant que CONFIRME_PRODUITS_EPARGNE
# n'est pas passé à True en tête de script, on s'arrête ici -- un lancement
# en spark-submit de bout en bout ne peut plus filer directement sur une
# hypothèse de codes produit non relue.
assert CONFIRME_PRODUITS_EPARGNE, (
    "Point d'arrêt ÉTAPE 2 : relisez l'inventaire ASSI ci-dessus, confirmez "
    "avec l'encadrant que les codes 53/09/18 sont bien les 3 seuls produits "
    "d'épargne, puis passez CONFIRME_PRODUITS_EPARGNE = True en tête de "
    "script avant de relancer."
)

# Sur les 6 codes présents dans ASSI, seuls 3 sont réellement des produits
# d'épargne (09 = MaRetraite, 53 = Avenir Mes Enfants, 18 = Epargne
# Evolution) ; 86, 98, 99 sont des produits d'assurance/assistance hors
# scope. On filtre AVANT de vérifier l'exclusivité, sinon un client qui a
# une assurance ET un produit d'épargne apparaît à tort comme un
# "doublon" alors que l'exclusivité annoncée par l'encadrant ne concerne
# que les produits d'épargne entre eux.
# Confirmé sur le volume complet (voir GUIDE_MAITRE section 0) : codes
# 53/09/18 correspondent exactement aux effectifs notés par l'encadrant.
#
# NOTE (recadrage) : un client absent de ASSI (même filtré) n'est PAS un
# cas "inconnu" -- c'est un client qui ne détient aucun des 3 produits.
# C'est un négatif légitime pour le modèle d'éligibilité (étape 5bis),
# pas une population à exclure ou à mettre de côté comme "à prédire".
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
    print("Exclusivité NON respectée (ou ASSI = instantané périodique répété) "
          "-> dédoublonnage par DATE_CHARG le plus récent.")
    # FIX : tri sur la vraie date (to_date_charg), pas sur la chaîne brute
    # -- ASSI contient ~28 lignes/client en moyenne (instantané périodique,
    # comme SOLDE/FLUX), donc ce tri détermine littéralement quel produit
    # est assigné à chaque client. Un tri alphabétique aurait pu assigner
    # le produit d'un instantané de 2022 plutôt que 2025.
    w = Window.partitionBy("RADICAL").orderBy(to_date_charg().desc())
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

# Combien de clients ASSI (après filtrage produit) ne se retrouvent PAS
# dans PERIMETRE ? Un LEFT JOIN depuis PERIMETRE les exclurait
# silencieusement du dataset final.
n_cible_hors_perimetre = cible.join(perimetre.select("RADICAL"), "RADICAL", "left_anti").count()
print(f"Clients avec produit d'épargne mais absents de PERIMETRE : {n_cible_hors_perimetre} "
      f"(sur {cible.count()} au total dans la cible)")

# ============================================================
# ÉTAPE 3 — Agrégation de chaque table 1:N à 1 ligne / client
# ============================================================

# --- OPK / PACK : union des 3 années, on garde le pack le plus récent ---
opk_union = (opk_2023
             .unionByName(opk_2024, allowMissingColumns=True)
             .unionByName(opk_2025, allowMissingColumns=True))
# FIX : même bug de tri corrigé ici (3 fichiers réels, 2023/2024/2025 --
# c'est justement le cas qui casse un tri alphabétique sur chaîne).
w_opk = Window.partitionBy("RADICAL").orderBy(to_date_charg().desc())
opk_agg = (
    opk_union
    .withColumn("rang", F.row_number().over(w_opk))
    .filter("rang = 1")
    .select("RADICAL",
            F.col("CODE_PACK").alias("pack_actuel"),
            F.col("ETATC").alias("pack_etat"))
)

# --- PRODUIT_DIGITAUX ---
# FIX : cette table N'EST PAS 1:1 malgré l'hypothèse initiale du guide --
# vérifié empiriquement (1 790 451 lignes pour 1 765 730 RADICAL distincts,
# soit 24 721 clients avec plusieurs lignes, probablement des clients
# résiliés puis réabonnés). La tentative de dédoublonnage précédente
# plantait (référençait DATE_CHARG, une colonne qui n'existe pas dans ce
# fichier -- ses seules colonnes de date sont DATE_VAL_ABON/DATE_RES_ABON).
# Règle métier : priorité à l'abonnement encore ACTIF (DATE_RES_ABON
# NULL) ; à égalité, le plus récent DATE_VAL_ABON.
w_digi = Window.partitionBy("RADICAL").orderBy(
    F.col("DATE_RES_ABON").isNull().desc(),
    F.to_date(F.col("DATE_VAL_ABON"), "dd/MM/yyyy").desc(),
)
digitaux_dedup = (
    digitaux
    .withColumn("rang", F.row_number().over(w_digi))
    .filter("rang = 1")
    .drop("rang")
)
digitaux_clean = digitaux_dedup.select(
    "RADICAL",
    F.col("DATE_VAL_ABON").alias("digital_date_activation"),
    F.when(F.col("DATE_RES_ABON").isNull(), F.lit(1)).otherwise(F.lit(0))
     .alias("digital_toujours_abonne"),
)
# FIX (gap détecté à la relecture) : le pipeline de nettoyage en aval
# (EDA/feature engineering) attend une ANCIENNETÉ EN JOURS
# (anciennete_digitale_jours), pas une date brute -- cette conversion
# n'existait nulle part, ni ici ni côté EDA. Calculée par rapport à
# DATE_REFERENCE, comme age_client, PAS current_date(). Un
# digital_date_activation manquant (client jamais abonné) donne une
# ancienneté NULL, qui sera gérée comme telle en aval (flag
# jamais_active_digital + imputation), pas plafonnée arbitrairement ici.
digitaux_clean = digitaux_clean.withColumn(
    "anciennete_digitale_jours",
    F.datediff(DATE_REFERENCE, F.to_date(F.col("digital_date_activation"), "dd/MM/yyyy")),
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
# NOTE (pas corrigé ici, volontairement) : pour un client absent de SOLDE,
# solde_moyen/min/max resteront NULL après le LEFT JOIN de l'ÉTAPE 4 -- ce
# n'est PAS mis à 0 comme les compteurs/montants transactionnels, parce
# qu'une moyenne de "rien" n'est pas 0 (contrairement à un nombre
# d'opérations ou un montant total). La décision de traitement de ce NULL
# (imputation, flag séparé comme solde_volatilite_indefinie, etc.)
# appartient à l'étape de feature engineering en aval, pas à la
# construction du dataset brut -- mais elle n'est PAS encore prise dans
# le notebook EDA actuel (celui-ci ne fillna que depot_moyen et
# montant_moyen_gab). À traiter avant l'assemblage des features pour le
# modèle, sinon le VectorAssembler plantera sur ces NULL.

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
# NOTE À VÉRIFIER : les lignes échantillon ont toutes CODE_FONCT=031 et
# MONTANT=0. Si 031 = "consultation de solde" (pas un retrait),
# montant_total_gab/montant_moyen_gab pourraient être constamment nuls --
# vérifier avec gab_1.groupBy("CODE_FONCT").agg(F.count("*"), F.avg("MONTANT")).show()
gab_union = gab_1.unionByName(gab_2, allowMissingColumns=True)
gab_agg = (
    gab_union
    .withColumn("MONTANT_num", to_num("MONTANT"))
    .withColumn("DATE_OP_ts", F.to_timestamp("DATE_OP", "dd/MM/yyyy HH:mm:ss"))
    # cast avant le max, sinon comparaison alphabétique de chaînes ->
    # "dernière opération" potentiellement fausse
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_operations_gab"),
        F.sum("MONTANT_num").alias("montant_total_gab"),
        F.avg("MONTANT_num").alias("montant_moyen_gab"),
        F.max("DATE_OP_ts").alias("derniere_operation_gab"),
    )
)
# FIX (même gap que pour le digital) : le pipeline en aval attend une
# RÉCENCE EN JOURS (recence_gab_jours), pas un timestamp brut -- ajoutée
# ici, par rapport à DATE_REFERENCE. NULL pour un client sans opération
# GAB (géré en aval comme jamais_active_gab / imputation), pas 0 -- un
# client sans opération n'a pas "utilisé le GAB il y a 0 jour", il ne l'a
# simplement jamais utilisé.
gab_agg = gab_agg.withColumn(
    "recence_gab_jours",
    F.datediff(DATE_REFERENCE, F.to_date(F.col("derniere_operation_gab"))),
)

# --- RETRAITS ---
# (investigation post-hoc) : ce fichier N'EST PAS un sous-ensemble garanti
# des opérations GAB. Vérifié empiriquement : 365 122 clients présents
# dans RETRAIT sont absents de GAB. Deux sources indépendantes, à traiter
# comme deux features comportementales séparées.
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
# ÉTAPE 3bis — Vérification systématique AVANT jointure (pas après)
# ============================================================
# Ce contrôle est ce qui a permis de repérer produit_digitaux la première
# fois. Il est maintenant un ASSERT bloquant : le script s'arrête net s'il
# reste un problème, au lieu de continuer sur un dataset_final corrompu.
tables_a_verifier = {
    "cible (ASSI filtré)": cible,
    "pack (OPK)": opk_agg,
    "produit_digitaux": digitaux_clean,
    "solde": solde_agg,
    "depot_bilanciel": depot_agg,
    "flux": flux_agg,
    "gab": gab_agg,
    "retrait": retrait_agg,
    "payfac": payfac_agg,
    "vignette": vignette_agg,
}
verifier_unicite_radical(tables_a_verifier, arreter_si_probleme=True)

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
# valeur manquante à imputer par une moyenne. NOTE : anciennete_digitale_jours,
# recence_gab_jours, solde_moyen/min/max et digital_date_activation /
# derniere_operation_gab sont volontairement ABSENTS de cette liste -- un
# NULL y signifie "jamais observé", pas "zéro", et doit être traité comme
# tel (flag + imputation) à l'étape de feature engineering, pas ici.
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
# FIX : assert au lieu d'un simple avertissement -- grâce à la vérification
# de l'ÉTAPE 3bis, ce cas ne devrait normalement plus jamais se produire ;
# s'il se reproduit quand même, mieux vaut un arrêt net et bruyant qu'un
# fichier corrompu écrit silencieusement dans processed-data.
assert n_final == n_perimetre, (
    f"dataset_final ({n_final} lignes) != PERIMETRE ({n_perimetre} lignes) -- "
    f"une jointure a dupliqué des lignes malgré la vérification ÉTAPE 3bis. "
    f"Ne continuez pas tant que ce n'est pas résolu."
)
print("OK : aucune duplication introduite par les jointures.")

# ============================================================
# ÉTAPE 5bis — Label principal : ÉLIGIBILITÉ (recadrage à 4 classes)
# ============================================================
# label_eligibilite = 1 si le client détient l'un des 3 produits d'épargne
# (label_nom non nul après le LEFT JOIN avec la cible), 0 sinon. C'est un
# vrai ground truth pour TOUT PERIMETRE, pas une approximation : l'absence
# dans ASSI (filtré aux 3 produits valides) signifie "aucun produit
# d'épargne", ce qui est la 4e classe, pas une valeur inconnue.
dataset_final = dataset_final.withColumn(
    "label_eligibilite",
    F.when(F.col("label_nom").isNotNull(), F.lit(1)).otherwise(F.lit(0))
)

print("\nRépartition du label principal -- éligibilité à un produit d'épargne "
      "(1 = détient l'un des 3, 0 = aucun) :")
dataset_final.groupBy("label_eligibilite").count().orderBy("label_eligibilite").show()

print("\nRépartition des classes produit (label bonus, sous-ensemble éligible "
      "uniquement -- label_eligibilite = 1) :")
dataset_final.filter(F.col("label_eligibilite") == 1) \
    .groupBy("label_nom").count().orderBy(F.desc("count")).show(10, truncate=False)

# ============================================================
# ÉTAPE 6 — Constitution des DEUX populations d'entraînement
# ============================================================
# a) MODÈLE PRINCIPAL (éligibilité, binaire) : toute la population a un
#    label désormais -- il n'y a plus de population "à scorer" pour cette
#    tâche au sens de "cible inconnue". S'il existe malgré tout de
#    nouveaux clients hors PERIMETRE à noter plus tard, ce sera un batch
#    de scoring séparé, pas un sous-ensemble de ce dataset.
dataset_eligibilite = dataset_final

# b) MODÈLE BONUS (quel produit, multiclasse) : uniquement les clients
#    éligibles -- "lequel des 3" n'a de sens que si la réponse à la
#    question d'éligibilité est "oui". Entraîner ce modèle sur toute la
#    population (y compris les 0) introduirait une classe "aucun produit"
#    qui n'appartient pas à cette tâche et fausserait le multiclasse.
dataset_produit = dataset_final.filter(F.col("label_eligibilite") == 1)

n_eligibilite = dataset_eligibilite.count()
n_positifs = dataset_eligibilite.filter("label_eligibilite = 1").count()
n_negatifs = dataset_eligibilite.filter("label_eligibilite = 0").count()
n_produit = dataset_produit.count()

print(f"\nModèle principal (éligibilité)  : {n_eligibilite} clients "
      f"({n_positifs} positifs / {n_negatifs} négatifs -- "
      f"{n_positifs / n_eligibilite:.1%} de taux d'éligibilité)")
print(f"Modèle bonus (quel produit)     : {n_produit} clients "
      f"(sous-ensemble éligible uniquement, {n_produit / n_eligibilite:.1%} de PERIMETRE)")

# ============================================================
# ÉTAPE 7 — Écriture dans processed-data (entrée de la section 7 du guide)
# ============================================================
# Deux jeux distincts : chaque modèle a sa propre population et sa propre
# cible -- ne pas les fusionner en aval, la section 7 (Pipeline
# d'entraînement) doit les charger et les traiter séparément. Le
# notebook EDA et le GUIDE_MAITRE référencent encore les anciens chemins
# (dataset_train_produits / dataset_a_scorer) -- à mettre à jour pour
# pointer ici avant de relancer la Partie 1 du notebook EDA.
dataset_eligibilite.write.mode("overwrite").parquet(
    "s3a://processed-data/dataset_eligibilite/"
)
dataset_produit.write.mode("overwrite").parquet(
    "s3a://processed-data/dataset_produit/"
)

print("\nTerminé. Fichiers écrits :")
print("  - processed-data/dataset_eligibilite/  (modèle principal, binaire, "
      "toute la population, cible = label_eligibilite)")
print("  - processed-data/dataset_produit/      (modèle bonus, multiclasse, "
      "sous-ensemble éligible uniquement, cible = label_code / label_nom)")
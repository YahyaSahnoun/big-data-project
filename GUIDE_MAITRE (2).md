# GUIDE MAÎTRE — Projet Scoring / Éligibilité Produits d'Épargne (Binôme)

> **But de ce document** : reprendre ce projet à zéro, sans perte de contexte, que ce soit vous, votre collègue, ou une IA dans une nouvelle conversation. Tout ce qui a été décidé, cassé, corrigé et appris jusqu'ici est consigné dedans.

---

## 0. Contexte métier (mis à jour par l'encadrant)

**Objectif initial** : scorer les clients susceptibles de souscrire à "un produit d'épargne" (binaire : oui/non).

**Objectif corrigé (à jour)** : l'encadrant a précisé qu'il existe en réalité **3 produits d'épargne distincts**, et qu'**un client ne détient qu'un seul produit à la fois** (exclusif, confirmé). Une table fournie dans les fichiers indique quel client a acheté quel produit. Le but est de déterminer, pour un client donné, à quel produit il est éligible/adapté.

**Produits identifiés** (d'après les notes manuscrites de l'encadrant — noms et codes à reconfirmer une fois les fichiers étudiés, ne pas les coder en dur avant vérification) :
- Produit 1 : *"Avenir Mes Enfants"* (code ou effectif noté : 53)
- Produit 2 : nom peu lisible sur les notes (peut-être "Marhaba"), code/effectif noté : 03
- Produit 3 : *"Épargne Évolutif"* (code ou effectif noté : 18)

**Conséquence directe sur la modélisation** : ce n'est **pas** 3 modèles binaires indépendants (risque de contradictions : un client "éligible" à 2 produits alors que c'est censé être exclusif), mais **un seul modèle de classification multi-classes** avec une cible à 3 valeurs (le produit) — potentiellement 4 si certains clients n'ont aucun produit. Détails techniques complets en section 7.

**Ce qui bloque tout le reste, actuellement** : personne n'a encore étudié le contenu réel des fichiers. Tant que ce n'est pas fait, impossible de savoir quel fichier contient la table cible (client → produit acheté), quel est le format exact du `client_id`, ni quelles variables sont exploitables comme features. **C'est la priorité absolue avant toute ligne de code de modélisation.** Méthode en section 6.

---

## 1. Architecture retenue

```
Sources de données (fichiers .txt, ~21-27 fichiers ATT_PROD_EPARGNE_*/ATT_HISSAB_*)
        ↓
NiFi (ingestion) ──────────► MinIO (stockage S3, remplace HDFS)
        ↓                        buckets : raw-data / processed-data / ml-scoring
Spark (traitement + MLlib) ◄──────┘
        ↓
Hive Metastore (catalogue de tables) → Power BI (restitution, hors Docker)
```
Airflow orchestre NiFi et Spark en tâche de fond (transversal, pas dans le flux de données direct).

**Analogies pour se souvenir de chaque brique :**
- Docker = appartement meublé livré clé en main (environnement isolé, reproductible)
- NiFi = aiguilleur de gare (route les fichiers sans coder)
- MinIO = casier à colis (buckets = rangées, objets = colis)
- Spark = chef de chantier (driver) qui répartit le travail entre ouvriers (exécuteurs)
- Hive Metastore = fichier d'index d'une bibliothèque (pointe vers les données, n'en contient aucune)
- Airflow = chef d'orchestre (déclenche chaque outil au bon moment)

---

## 2. Setup technique — deux environnements en parallèle

Les deux membres du binôme font tourner **leur propre stack locale indépendante** (pas un cluster partagé) :
- **Collègue (yahya)** : Linux natif (pop-os), dossier `~/bigdata-cluster`
- **Vous (Tarouzi)** : Windows + WSL2 + Docker Desktop, dossier `~/bigdata-cluster` (dans WSL, pas `/mnt/c/...`)

### 2.1 Pour le collègue (Linux natif)
Docker + Docker Compose installés nativement, pas de couche supplémentaire. Toutes les commandes `docker compose ...` s'exécutent directement dans un terminal classique.

### 2.2 Pour vous (Windows)
1. `wsl --install` en PowerShell admin → redémarrage → créer un compte Ubuntu.
2. Docker Desktop → Settings → Resources → WSL Integration → activer sur `Ubuntu`.
3. VS Code → installer l'extension **WSL** (Microsoft) → `Ctrl+Shift+P` → **`WSL: Connect to WSL using Distro...`** → choisir `Ubuntu` (pas `docker-desktop`, qui est une distro technique interne à Docker, pas destinée au dev).
4. **Toujours travailler depuis le terminal Ubuntu**, jamais PowerShell, pour `docker compose`, l'édition de fichiers, etc.
5. Projet dans `~/bigdata-cluster` (le `$HOME` natif de WSL), jamais dans `/mnt/c/Users/...` (lent, problèmes de permissions).
6. Données dans `~/data_clients`, séparé du dossier projet (pour ne jamais versionner de données réelles avec Git par accident).
7. Pour copier depuis un disque Windows (ex. `D:`) : `cp -v "/mnt/d/chemin avec espaces/"*.txt ~/data_clients/` (guillemets obligatoires si le chemin a des espaces). Si `/mnt/d` n'existe pas : `wsl --shutdown` en PowerShell puis rouvrir Ubuntu (remonte les disques), ou montage manuel `sudo mount -t drvfs D: /mnt/d`.
8. Pour ouvrir le dossier WSL depuis l'Explorateur Windows : `\\wsl$\Ubuntu\home\<user>\data_clients` dans la barre d'adresse, ou `explorer.exe .` depuis le terminal Ubuntu.

### 2.3 docker-compose.yml (version corrigée finale, à utiliser par les deux)

```yaml
networks:
  pipeline-net:
    driver: bridge

volumes:
  minio-data:
  postgres-data:
  nifi-data:
  spark-data:

services:

  minio:
    image: minio/minio:latest
    container_name: minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    networks:
      - pipeline-net
    mem_limit: 1.5g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio-init:
    image: minio/mc:latest
    container_name: minio-init
    entrypoint: /bin/sh
    command: ["-c", "mc alias set local http://minio:9000 minioadmin minioadmin123 && mc mb -p local/raw-data && mc mb -p local/processed-data && mc mb -p local/ml-scoring && exit 0"]
    depends_on:
      minio:
        condition: service_healthy
    networks:
      - pipeline-net

  postgres:
    image: postgres:15
    container_name: postgres
    environment:
      POSTGRES_MULTIPLE_DATABASES: "hive_metastore,airflow"
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin123
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./pg-init-scripts:/docker-entrypoint-initdb.d
    networks:
      - pipeline-net
    mem_limit: 512m
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "admin"]
      interval: 5s
      timeout: 5s
      retries: 10

  hive-metastore:
    image: bitsondatadev/hive-metastore:latest
    container_name: hive-metastore
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      METASTORE_DB_HOSTNAME: postgres
      HIVE_METASTORE_DB_TYPE: postgres
      HIVE_METASTORE_URI: postgresql://admin:admin123@postgres:5432/hive_metastore
      S3_ENDPOINT: http://minio:9000
      S3_ACCESS_KEY: minioadmin
      S3_SECRET_KEY: minioadmin123
      S3_PATH_STYLE_ACCESS: "true"
    ports:
      - "9083:9083"
    networks:
      - pipeline-net
    mem_limit: 1g
    restart: unless-stopped

  spark-master:
    image: apache/spark:3.5.1
    container_name: spark-master
    command: ["/opt/spark/bin/spark-class", "org.apache.spark.deploy.master.Master"]
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin123
      SPARK_HADOOP_FS_S3A_ENDPOINT: http://minio:9000
      SPARK_HADOOP_FS_S3A_PATH_STYLE_ACCESS: "true"
      SPARK_HADOOP_FS_S3A_IMPL: org.apache.hadoop.fs.s3a.S3AFileSystem
    ports:
      - "8080:8080"
      - "7077:7077"
      - "10000:10000"
    volumes:
      - spark-data:/opt/spark/work-dir
    networks:
      - pipeline-net
    mem_limit: 1g
    restart: unless-stopped

  spark-worker:
    image: apache/spark:3.5.1
    container_name: spark-worker-1
    depends_on:
      - spark-master
    command: ["/opt/spark/bin/spark-class", "org.apache.spark.deploy.worker.Worker", "spark://spark-master:7077", "-c", "2", "-m", "1500M"]
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin123
      SPARK_HADOOP_FS_S3A_ENDPOINT: http://minio:9000
      SPARK_HADOOP_FS_S3A_PATH_STYLE_ACCESS: "true"
    networks:
      - pipeline-net
    mem_limit: 2g
    restart: unless-stopped

  nifi:
    image: apache/nifi:1.27.0
    container_name: nifi
    environment:
      NIFI_WEB_HTTP_PORT: 8443
      SINGLE_USER_CREDENTIALS_USERNAME: admin
      SINGLE_USER_CREDENTIALS_PASSWORD: admin12345678
    ports:
      - "8443:8443"
    volumes:
      - nifi-data:/opt/nifi/nifi-current/conf
      - ~/data_clients:/data/clients:ro
    networks:
      - pipeline-net
    mem_limit: 2g
    restart: unless-stopped

  airflow-webserver:
    image: apache/airflow:2.9.1
    container_name: airflow-webserver
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://admin:admin123@postgres:5432/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    command: >
      bash -c "airflow db migrate &&
      airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@example.com || true &&
      airflow webserver"
    ports:
      - "8081:8080"
    volumes:
      - ./dags:/opt/airflow/dags
    networks:
      - pipeline-net
    mem_limit: 1g
    restart: unless-stopped

  airflow-scheduler:
    image: apache/airflow:2.9.1
    container_name: airflow-scheduler
    depends_on:
      postgres:
        condition: service_healthy
      airflow-webserver:
        condition: service_started
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://admin:admin123@postgres:5432/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    command: scheduler
    volumes:
      - ./dags:/opt/airflow/dags
    networks:
      - pipeline-net
    mem_limit: 1g
    restart: unless-stopped
```

Note : `~/data_clients` dans le volume `nifi` pointe vers le `$HOME` de chacun — fonctionne tel quel pour les deux membres, tant que chacun a bien rempli son propre dossier `~/data_clients`.

### 2.4 pg-init-scripts/init-multiple-dbs.sh (obligatoire, sinon Airflow/Hive ne démarrent jamais)

```bash
#!/bin/bash
set -e

for DB in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE $DB;
EOSQL
done
```
⚠️ Ne jamais créer/éditer ce fichier depuis un éditeur Windows natif (Notepad) — uniquement via terminal Ubuntu ou VS Code connecté en WSL, sinon les fins de ligne CRLF cassent le script dans le conteneur.

### 2.5 Lancer et vérifier

```bash
mkdir -p pg-init-scripts dags
# (créer les fichiers ci-dessus dedans)
python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('YAML OK')"
docker compose up -d --build
```
Attendre 30-40s, puis `docker compose ps` (9 services `Up`).

| Service | URL | Identifiants |
|---|---|---|
| MinIO console | http://localhost:9001 | minioadmin / minioadmin123 |
| Spark master | http://localhost:8080 | — |
| NiFi | http://localhost:8443/nifi | admin / admin12345678 |
| Airflow | http://localhost:8081 | admin / admin |

---

## 3. Bugs déjà rencontrés et corrigés (ne pas les reproduire)

| Bug | Cause | Correctif |
|---|---|---|
| Airflow ne démarre pas, `database "airflow" does not exist` | `POSTGRES_MULTIPLE_DATABASES` ne fait rien sans script d'init | Script `init-multiple-dbs.sh` monté dans `/docker-entrypoint-initdb.d`, + `docker compose down -v` pour repartir d'un volume vide |
| Spark master/worker en boucle : `bin/spark-class: No such file or directory` | Chemin relatif invalide, le workdir de l'image officielle Apache Spark est `/opt/spark/work-dir` | Chemin absolu `/opt/spark/bin/spark-class` dans `command:` |
| `docker compose down -v` échoue avec `services.image must be a mapping` | Indentation YAML cassée (copier-coller) | Toujours valider avec `python3 -c "import yaml; yaml.safe_load(...)"` avant de lancer |
| `dial tcp: lookup minio on 127.0.0.11:53: server misbehaving` (Linux uniquement) | Résolveur DNS interne Docker instable sous charge (bug connu Docker Engine natif Linux) | `--add-host=minio:<IP réelle>` pour contourner le DNS, obtenue via `docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' minio` |
| `connect: connection refused` malgré la bonne IP | MinIO recevait trop de connexions concurrentes (`mc cp --recursive` en parallèle) | Boucle `for f in /data/*.txt` avec un seul fichier à la fois + retry automatique (5 tentatives, 5s d'attente) + `restart: unless-stopped` sur tous les services |
| `code` introuvable dans le terminal WSL | VS Code pas encore connecté à WSL une première fois | Ouvrir VS Code (Windows) → `Ctrl+Shift+P` → **`WSL: Connect to WSL using Distro...`** → choisir `Ubuntu` explicitement (pas `docker-desktop`) |
| NiFi : `ListFile` ne redétecte pas des fichiers déjà traités | `ListFile` retient en mémoire les fichiers déjà listés (comportement voulu en usage réel) | Clic droit sur `ListFile` → **View State** → **Clear State** → redémarrer, pour forcer un nouveau passage complet en test |
| Processeurs NiFi avec triangle ⚠, ne démarrent pas | Relations (`failure`, etc.) ni reliées ni auto-terminées | Onglet **Relationships** de chaque processeur → cocher **terminate** sur tout ce qui n'est pas `success` |

---

## 4. Ingestion des données — statut actuel

- **Méthode utilisée pour le chargement initial** : `mc cp` en boucle, un fichier à la fois avec retry (script `~/ingest.sh`), **21 fichiers texte envoyés avec succès dans le bucket `raw-data`**.
- **Flux NiFi** : construit et fonctionnel (`ListFile → FetchFile → PutS3Object`), testé avec succès sur les 21 fichiers après correction du piège "Clear State".
- **Statut** : l'ingestion initiale (backfill) est terminée. Le flux NiFi reste en place pour l'ingestion de futurs fichiers, et pour la cohérence avec l'architecture présentée en soutenance.
- **Positionnement pour le rapport** : chargement initial en masse via `mc` (backfill historique), ingestion continue automatisée via NiFi — c'est un choix d'architecture assumé, pas un contournement.

---

## 5. Checklist des sprints (avancement réel)

### Sprint 1 — Fondations et ingestion
- [x] Architecture définie et validée (MinIO + Hive Metastore remplace HDFS)
- [x] Environnement Docker fonctionnel (les deux machines, bugs section 3 corrigés)
- [x] Accès aux données anonymisées obtenu
- [x] Structure des buckets MinIO conçue (`raw-data` / `processed-data` / `ml-scoring`)
- [x] Pipeline d'ingestion construit et testé (NiFi + méthode `mc` de secours)
- [ ] Profiling / quality check des données — **pas encore fait, prochaine étape (section 6)**

### Sprint 2 — Feature engineering et modélisation
- [ ] **Étude des fichiers et de leurs champs** (bloquant, voir section 6)
- [ ] Identification de la table cible (client → produit acheté)
- [ ] Nettoyage des données
- [ ] Feature engineering
- [ ] Table Hive curated
- [ ] Encodage de la cible multi-classes (3 produits)
- [ ] Entraînement et comparaison de modèles multi-classes
- [ ] Sélection et sauvegarde du modèle final

### Sprint 3 — Industrialisation et livraison
- [ ] Script de scoring batch (adapté multi-classes)
- [ ] DAG Airflow bout en bout
- [ ] Spark Thrift Server + connexion Power BI
- [ ] Tests de bout en bout
- [ ] Rapport, schéma d'architecture, support de soutenance

---

## 6. Prochaine étape obligatoire : étudier les fichiers

**Rien en section 7 ne peut être codé avec les vrais noms de colonnes tant que cette étape n'est pas faite.**

### 6.1 Script de découverte automatique (version utilisée en pratique)

```python
"""
À lancer : docker cp discover_files.py spark-master:/opt/spark/work-dir/
puis      : docker exec -it spark-master /opt/spark/bin/spark-submit \
              --master spark://spark-master:7077 /opt/spark/work-dir/discover_files.py
"""
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

for f in sorted(files):
    nom = f.split("/")[-1]
    print(f"\n{'='*20} ANALYSE : {nom} {'='*20}")

    # Lecture en tant que CSV avec délimiteur ';'
    # header=True suppose que la première ligne contient les noms de colonnes
    # Pas de inferSchema volontairement à ce stade : tout reste en string,
    # largement suffisant pour l'exploration visuelle (voir section 6.3)
    df = spark.read.option("header", "true").option("sep", ";").csv(f)

    print(f"Nombre total de lignes : {df.count()}")
    print("Aperçu des données :")
    df.show(3, truncate=False)
```

⚠️ Ce script suppose `sep=";"` pour **tous** les fichiers. Si un fichier ressort avec une seule colonne géante contenant tout collé (voir section 6.3, cas n°1), c'est le signe qu'il utilise un autre séparateur — relancez la lecture de ce fichier précis avec `.option("sep", "|")` ou `.option("sep", "\t")` pour confirmer.

### 6.2 Grille à remplir en parcourant les résultats

| Fichier | `client_id` présent ? | Code/nom produit présent ? | Date présente ? | Rôle probable |
|---|---|---|---|---|
| ATT_PROD_EPARGNE_PERIMETRE | | | | Population de référence ? |
| ATT_PROD_EPARGNE_SOLDE_2025 | | | | État des soldes ? |
| ATT_PROD_EPARGNE_FLUX_2023/2025 | | | | Transactions ? |
| ATT_HISSAB_OPK2023/2024/2025 | | | | Ouvertures compte/produit ? |
| *(compléter pour chaque fichier)* | | | | |

**Priorité n°1** : repérer le(s) fichier(s) avec à la fois `client_id` **et** un code/nom de produit — c'est la table cible. Vérifier ensuite :
1. Format exact de `client_id` (longueur, zéros de tête) — doit être identique partout pour les jointures.
2. Si l'absence d'un client dans cette table signifie "aucun produit" (4ᵉ classe) ou si la population totale est définie ailleurs (PERIMETRE ?).
3. Les noms/codes exacts des 3 produits, pour vérifier contre les notes de l'encadrant (section 0).

### 6.3 Comment interpréter la sortie du script, fichier par fichier

Pour chaque bloc `ANALYSE : nom_du_fichier`, regardez dans cet ordre :

**a) Le `df.show(3)` s'affiche-t-il en plusieurs colonnes lisibles, ou en une seule colonne géante ?**
- Plusieurs colonnes bien séparées → le séparateur `;` est le bon pour ce fichier, continuez.
- Une seule colonne avec tout collé (souvent un nom de colonne à rallonge du style `NUM_CLI|DAT_OPE|MNT|...`) → le vrai séparateur est autre chose (`|` très probable vu ce pattern, ou `\t`). Retestez ce fichier seul avec `.option("sep", "|")`.

**b) Les noms de colonnes (première ligne du `show`) : cherchez ces familles de mots-clés typiques du bancaire marocain/français**
| Vous voyez... | Ça signifie probablement | Rôle |
|---|---|---|
| `NUM_CLI`, `CLIENT_ID`, `CIN`, `ID_CLI` | Identifiant client | Clé de jointure — notez le format exact (ex. `CL0001234` vs `1234`) |
| `COD_PROD`, `LIB_PROD`, `PRODUIT`, `TYPE_PROD` | Code ou libellé du produit d'épargne | **Candidat sérieux pour la table cible** |
| `DAT_OPE`, `DATE_TRANSACTION`, `DAT_OUV` | Une date (ouverture, opération) | Utile pour la récence (RFM) |
| `MNT`, `MONTANT`, `SOLDE` | Un montant | Utile pour la feature "montant moyen" |
| `AGENCE`, `GAB` (Guichet Automatique Bancaire) | Canal/lieu de l'opération | Feature comportementale possible |

**c) Comparez `df.count()` au nombre de `client_id` distincts** (ajoutez temporairement `df.select("NOM_COLONNE_CLIENT").distinct().count()` si une colonne ressemble à un identifiant client) :
- **`count()` ≈ nombre de clients distincts** (à peu près 1 ligne par client) → table de type "état" ou "référentiel" (ex. solde actuel, périmètre, produit détenu) — probablement pas une table de transactions.
- **`count()` très supérieur au nombre de clients distincts** (plusieurs lignes par client) → table de type "mouvements/transactions" (ex. FLUX, OPE_RETRAIT, OPERATION_GAB) — utile pour calculer fréquence/récence, pas pour la cible.

**d) Repérez le nombre de valeurs distinctes d'une colonne qui ressemble à un produit :**
```python
df.groupBy("NOM_COLONNE_PRODUIT").count().show()
```
Si vous obtenez **exactement 3 valeurs** (ou 3 + une valeur "vide"/"aucun"), avec des effectifs qui font écho aux chiffres notés par l'encadrant (53/03/18, section 0) — **c'est votre table cible**, quasi certain. Notez le nom exact de chaque valeur pour remplacer les hypothèses de la section 0.

**e) Fichiers `_COMPLEMENT`** : une fois que vous avez identifié un fichier de base (ex. `ATT_PROD_EPARGNE_SOLDE_2025`) et sa variante `..._COMPLEMENT`, comparez leurs colonnes avec `df.printSchema()` (ou juste en regardant les en-têtes du `show`). Si les colonnes sont identiques → c'est un empilement (`union`) à faire, pas une jointure. Si les colonnes diffèrent → c'est un complément d'information à joindre sur `client_id`.

**f) Ce qu'il faut noter dans la grille (section 6.2) pour chaque fichier**, au minimum : nombre de lignes, séparateur réel, présence ou non d'un `client_id`, présence ou non d'un code produit, granularité (1 ligne/client ou plusieurs), et une hypothèse de rôle. Une fois les 21 fichiers passés en revue avec cette méthode, la table cible et les tables de features candidates devraient être évidentes — revenez alors vers moi avec ce que vous avez trouvé pour écrire le vrai code de jointure et de feature engineering (section 7) avec les noms de colonnes réels.

---

## 7. Sprint 2 mis à jour — concepts et syntaxe Spark pour la classification multi-classes

Cette section explique **les méthodes disponibles et leur syntaxe**, à adapter dès que les vrais noms de colonnes sont connus (section 6).

### 7.1 Joindre plusieurs fichiers sources (`join`)

**Concept** : vos features (transactions, soldes) et votre cible (produit acheté) sont probablement dans des fichiers différents. Il faut les rassembler sur la clé commune `client_id`, comme une jointure SQL classique.

```python
df_final = df_features.join(df_target, on="client_id", how="inner")
```
- `how="inner"` : ne garde que les clients présents dans les deux tables. `"left"` garderait tous les clients de `df_features` même sans produit connu (utile si vous voulez inclure les non-acheteurs comme 4ᵉ classe).
- Une jointure est une opération de *shuffle* (comme `groupBy`) — coûteuse sur un gros volume, mais indispensable ici.

### 7.2 Encoder la cible catégorielle (`StringIndexer`)

**Concept** : MLlib ne travaille qu'avec des nombres, jamais des chaînes de caractères. Le nom du produit ("Avenir Mes Enfants", etc.) doit être converti en indices (0, 1, 2) avant l'entraînement.

```python
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="nom_produit", outputCol="label")
df_indexed = indexer.fit(df_final).transform(df_final)
```
`StringIndexer` assigne les indices par ordre de fréquence décroissante (le produit le plus courant devient 0). C'est important à savoir pour interpréter les résultats plus tard.

### 7.3 Revenir du chiffre au nom du produit (`IndexToString`)

**Concept** : une fois le modèle entraîné, ses prédictions sont des chiffres (0, 1, 2) — il faut pouvoir les retraduire en noms de produits pour le rapport ou le dashboard.

```python
from pyspark.ml.feature import IndexToString

converter = IndexToString(inputCol="prediction", outputCol="produit_predit", labels=indexer.fit(df_final).labels)
df_avec_noms = converter.transform(predictions)
```

### 7.4 Rassembler les features (`VectorAssembler`, rappel)

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["recence", "frequence", "montant_moyen"], outputCol="features")
```
Ne fait aucun calcul — regroupe simplement les colonnes numériques en un seul vecteur, format attendu par tous les algorithmes MLlib.

### 7.5 Choisir un algorithme multi-classes

Contrairement au cas binaire, plusieurs algorithmes MLlib gèrent nativement plus de 2 classes sans configuration spéciale :

| Algorithme | Syntaxe | Remarque |
|---|---|---|
| `RandomForestClassifier` | identique au binaire, MLlib détecte automatiquement le nombre de classes depuis la colonne `label` | Bon choix par défaut, robuste, gère bien les variables mixtes |
| `DecisionTreeClassifier` | `DecisionTreeClassifier(labelCol="label", featuresCol="features")` | Plus simple/interprétable, souvent moins performant seul |
| `LogisticRegression` | nécessite `family="multinomial"` explicitement | `LogisticRegression(labelCol="label", featuresCol="features", family="multinomial")` |
| `NaiveBayes` | `NaiveBayes(labelCol="label", featuresCol="features")` | Rapide, suppose les features indépendantes (rarement vrai en pratique, à tester quand même pour comparer) |

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    maxBins=32,
    weightCol="poids_classe",
    seed=42,
)
model = rf.fit(train)
```

### 7.6 Déséquilibre des classes en multi-classe (formule différente du binaire)

**Concept** : si un produit est acheté par beaucoup plus de clients que les deux autres, le modèle risque de toujours prédire le produit majoritaire. La pondération inverse la fréquence de **chaque** classe, pas juste deux comme en binaire :

```python
from pyspark.sql import functions as F

effectifs = df_indexed.groupBy("label").count()
total = df_indexed.count()
nb_classes = effectifs.count()

poids_par_classe = effectifs.withColumn(
    "poids_classe", total / (nb_classes * F.col("count"))
).select("label", "poids_classe")

df_pondere = df_indexed.join(poids_par_classe, on="label")
```
Cette formule (`total / (nb_classes × effectif_de_la_classe)`) est la pondération inverse-fréquence standard multi-classe — chaque classe rare reçoit un poids proportionnellement plus élevé.

### 7.7 Évaluer un modèle multi-classes (`MulticlassClassificationEvaluator`)

**Concept** : `BinaryClassificationEvaluator` (utilisé précédemment) ne fonctionne qu'à 2 classes. Il faut son équivalent multi-classe, avec des métriques différentes de l'AUC :

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

predictions = model.transform(test)
print("F1 pondéré :", evaluator_f1.evaluate(predictions))
print("Accuracy :", evaluator_acc.evaluate(predictions))
```
- `"f1"` : moyenne harmonique précision/rappel, pondérée par classe — **la métrique à privilégier** si les 3 produits ne sont pas vendus en quantités égales (cas très probable ici).
- `"accuracy"` : trompeuse seule si une classe domine largement les autres (mêmes limites qu'en binaire).
- Pour le détail par classe (quel produit est bien/mal prédit précisément), utilisez la matrice de confusion :
```python
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()
```

### 7.8 Tout enchaîner proprement avec un `Pipeline`

**Concept** : plutôt que d'appliquer `StringIndexer`, `VectorAssembler` et le classifieur séparément (risque d'erreur d'ordre, ou d'appliquer l'indexeur différemment entre train et test), un `Pipeline` MLlib enchaîne toutes les étapes comme une seule unité, réutilisable et sauvegardable telle quelle.

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer, assembler, rf])
pipeline_model = pipeline.fit(train)

predictions = pipeline_model.transform(test)
pipeline_model.write().overwrite().save("s3a://ml-scoring/models/pipeline_multiclasse_v1")
```
Au chargement pour le scoring batch, une seule ligne recharge tout l'enchaînement (indexation + assemblage + modèle) :
```python
from pyspark.ml import PipelineModel
pipeline_model = PipelineModel.load("s3a://ml-scoring/models/pipeline_multiclasse_v1")
```

### 7.9 Comparer plusieurs algorithmes proprement

```python
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes

candidats = {
    "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="poids_classe"),
    "LogReg": LogisticRegression(labelCol="label", featuresCol="features", family="multinomial", weightCol="poids_classe"),
    "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features", weightCol="poids_classe"),
    "NaiveBayes": NaiveBayes(labelCol="label", featuresCol="features"),  # ne supporte pas weightCol
}

for nom, algo in candidats.items():
    pipeline = Pipeline(stages=[indexer, assembler, algo])
    m = pipeline.fit(train)
    f1 = evaluator_f1.evaluate(m.transform(test))
    print(f"{nom} : F1 pondéré = {f1:.3f}")
```

---

## 8. Sprint 3 — scoring batch multi-classes (mise à jour)

```python
from pyspark.ml import PipelineModel

pipeline_model = PipelineModel.load("s3a://ml-scoring/models/pipeline_multiclasse_v1")
all_clients = spark.read.parquet("s3a://processed-data/features_clients/")

scores = pipeline_model.transform(all_clients).select("client_id", "produit_predit", "probability")
scores.coalesce(8).write.mode("overwrite").parquet("s3a://ml-scoring/scores_clients/")
```
`probability` contient désormais un **vecteur de 3 probabilités** (une par produit), pas un seul chiffre comme en binaire — utile pour montrer en dashboard non seulement le produit le plus probable, mais aussi la confiance du modèle.

Le reste du Sprint 3 (DAG Airflow, Spark Thrift Server, connexion Power BI) ne change pas dans sa structure — seul le contenu de `scoring_batch.py` est mis à jour ci-dessus.

---

## 9. Glossaire (ajouts multi-classes)

| Terme | Définition |
|---|---|
| Classification multi-classes | Prédire une cible parmi plus de 2 catégories possibles (ici : 3 produits) |
| `StringIndexer` | Convertit une colonne texte en indices numériques pour l'entraînement |
| `IndexToString` | Opération inverse : retrouve le texte à partir de l'indice prédit |
| `Pipeline` (MLlib) | Enchaîne plusieurs étapes de transformation + modèle comme une seule unité réutilisable |
| F1 pondéré | Métrique combinant précision et rappel, moyenne sur toutes les classes, pondérée par leur effectif |
| Matrice de confusion | Tableau croisant classe réelle × classe prédite, pour voir précisément quelles confusions le modèle fait |

---

## 10. Checklist finale (mise à jour)

- [x] Environnement Docker fonctionnel sur les deux machines
- [x] Données ingérées dans `raw-data` (NiFi + méthode de secours `mc`)
- [ ] Fichiers étudiés, grille de la section 6 remplie
- [ ] Table cible (client → produit) identifiée et confirmée avec l'encadrant
- [ ] Noms/codes exacts des 3 produits confirmés (vs. notes section 0)
- [ ] Jointure features + cible réalisée
- [ ] Cible encodée (`StringIndexer`), déséquilibre des classes traité
- [ ] Au moins 3 algorithmes multi-classes comparés (F1 pondéré)
- [ ] Pipeline final sauvegardé
- [ ] Scoring batch multi-classes opérationnel
- [ ] DAG Airflow, Power BI, tests bout en bout, livrables (inchangé section 8 du guide précédent)

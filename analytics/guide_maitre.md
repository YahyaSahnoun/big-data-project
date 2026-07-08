# GUIDE MAÎTRE — Projet Scoring / Éligibilité Produits d'Épargne (Binôme)

> **But de ce document** : reprendre ce projet à zéro, sans perte de contexte, que ce soit vous, votre collègue, ou une IA dans une nouvelle conversation. Tout ce qui a été décidé, cassé, corrigé et appris jusqu'ici est consigné dedans.

---

## 0. Contexte métier (mis à jour par l'encadrant, puis par l'étude réelle des fichiers)

**Objectif initial** : scorer les clients susceptibles de souscrire à "un produit d'épargne" (binaire : oui/non).

**Objectif corrigé par l'encadrant** : il existe **3 produits d'épargne distincts**, et **un client ne détient qu'un seul produit à la fois** (exclusif, confirmé). Le but est de déterminer, pour un client donné, à quel produit il est éligible/adapté.

**✅ Résolu (haute confiance, à faire valider par l'encadrant) :** le script `build_dataset_final.py` a listé les 6 valeurs distinctes de `CODE_PRODUIT`/`LIBELLE_PRODUIT` dans `ASSI` :

| Code | Libellé | Nature | Dans le périmètre ? |
|---|---|---|---|
| 86 | ATTAMINE CHAABI HISSAB | Assurance ("Attamine" = assurance en arabe) | ❌ Non |
| 98 | AL INJAD CHAABI | Assistance/secours | ❌ Non |
| 99 | AL INJAD AL MOUMTAZ | Assistance (variante premium) | ❌ Non |
| **09** | **MaRetraite** | Épargne retraite | ✅ **Oui** |
| **53** | **AVENIR MESENFANTS** | Correspond exactement au code noté par l'encadrant | ✅ **Oui** |
| **18** | **EPARGNE EVOLUTION** | Correspond exactement au code noté par l'encadrant | ✅ **Oui** |

Les chiffres notés par l'encadrant (53, 18, et probablement 09 mal lu "03") sont donc les **codes produit** eux-mêmes, pas des effectifs — ce qui explique la correspondance exacte. **Les 3 produits d'épargne cibles sont confirmés : codes `53`, `09`, `18`.** Les 3 autres (assurance/assistance) doivent être exclus du périmètre.

**Point résolu sur l'exclusivité** : 812 120 clients semblaient avoir plusieurs produits — mais ce comptage a été fait sur les 6 produits mélangés (un client peut avoir une assurance ET un produit d'épargne sans contredire l'exclusivité annoncée par l'encadrant). Il faut filtrer sur les 3 codes d'épargne **avant** de vérifier l'exclusivité, pas après.

**Point encore ouvert** : `RADICAL` n'est pas unique sur `PERIMETRE` (3 179 148 valeurs distinctes pour 3 231 609 lignes) — un dédoublonnage est nécessaire avant de l'utiliser comme clé de jointure unique (voir section 6.4 et le script `build_dataset_final.py`).

**Produits identifiés par l'encadrant** (notes manuscrites, noms/codes à confirmer) :
- Produit 1 : *"Avenir Mes Enfants"* (effectif noté : 53)
- Produit 2 : nom peu lisible (peut-être "Marhaba"), effectif noté : 03
- Produit 3 : *"Épargne Évolutif"* (effectif noté : 18)

**Conséquence sur la modélisation** : ce n'est **pas** N modèles binaires indépendants, mais **un seul modèle de classification multi-classes** (le nombre exact de classes dépend du point bloquant ci-dessus : 3 si `ASSI` est filtré aux seuls produits d'épargne, potentiellement plus si l'encadrant confirme que davantage de produits sont dans le périmètre). Détails techniques en section 7.

**Ce qui a été fait entre-temps** : les 21 fichiers ont été étudiés en détail par le collègue (voir `guide_lecture_donnees_CORRIGE.pdf`, résumé complet en section 6.4 ci-dessous) — la structure des données, la clé de jointure, et les 5 familles de fichiers sont maintenant connues. Il ne reste que le point bloquant ci-dessus à trancher avant de coder les jointures définitives.

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
- [x] Étude des fichiers et de leurs champs (`guide_lecture_donnees_CORRIGE.pdf`)
- [x] Identification de la table cible (`ASSI`, codes `53`/`09`/`18`, à faire valider par l'encadrant)
- [x] Clé de jointure identifiée et corrigée (`RADICAL`, dédoublonnage nécessaire sur `PERIMETRE`)
- [x] Script `build_dataset_final.py` : nettoyage, agrégations par famille, jointures, écriture Parquet
- [ ] Lancer le script corrigé et confirmer l'ÉTAPE 5 (`n_final == n_perimetre`)
- [ ] Encodage de la cible multi-classes (`StringIndexer`)
- [ ] Entraînement et comparaison de modèles multi-classes (section 7.5)
- [ ] Sélection et sauvegarde du modèle final (`Pipeline`)

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

### 6.4 Dictionnaire de données réel et stratégie de jointure (issu de l'étude complète des 21 fichiers)

Cette section remplace la grille générique ci-dessus, maintenant que les fichiers ont été réellement étudiés (document complet : `guide_lecture_donnees_CORRIGE.pdf`).

#### La clé de jointure : pas un `client_id` simple

Il n'y a pas de colonne `client_id` unique. Un compte est identifié par un **sextuplet** de colonnes, qui joue le rôle d'un IBAN :

| Colonne | Rôle |
|---|---|
| `BANQUE` | Code de la banque régionale |
| `AGENCE` | Code de l'agence |
| `GENERIC`/`GENERIQUE` | Code de la nature du compte (le plus souvent `21111` ; les valeurs `21117`/`21150` sont à isoler et comprendre) |
| `RADICAL` | **Le numéro de compte lui-même — candidat principal pour la jointure** |
| `PLURAL` | Indice de sous-compte/rang (souvent `0`) |
| `CLE`/`CCLE` | Clé de contrôle (comme une clé RIB) — **aucune valeur prédictive, à exclure du modèle** |

⚠️ **À vérifier avant de coder quoi que ce soit** : `RADICAL` seul est-il unique sur `PERIMETRE` ?
```python
total = df_perimetre.count()
distinct = df_perimetre.select("RADICAL").distinct().count()
print(total, distinct)
```
Si `total == distinct` → `RADICAL` seul suffit comme clé de jointure. Sinon, il faut joindre sur `(BANQUE, AGENCE, GENERIC, RADICAL, PLURAL)`.

#### Les 5 familles de fichiers et leur rôle

| Famille | Fichier(s) | Granularité | Rôle |
|---|---|---|---|
| **1. Référentiel client** | `PERIMETRE` | 1 ligne / client | Table de base (âge, ville, situation familiale, nb enfants, segment client) — définit la population à scorer |
| **2. Produits / Packs / Digital** | `ASSI` (cible candidate), `OPK*`, `PRODUIT_DIGITAUX` | 1 ligne / client (ou / client / année pour OPK) | `ASSI` = cible potentielle ; `OPK`/`PRODUIT_DIGITAUX` = features comportementales |
| **3. Soldes et flux** | `SOLDE*`, `DEPOT_BILANCEIL*`, `FLUX*` | N lignes / client (mensuel) | Séries temporelles — **à agréger** avant jointure |
| **4. Transactions** | `OPERATION_GAB*`, `OPE_RETRAIT*`, `DIGI_PAYFAC*` | N lignes / client (par transaction) | Événements individuels — **à agréger** (compteurs, sommes, récence) avant jointure |
| **5. Vignette automobile** | `VIGNETTE*` | 0 à N lignes / client | Feature comportementale indirecte (proxy patrimoine/profil) — **à agréger** |

**Principe général** : `PERIMETRE` est la table centrale. Toutes les autres se rattachent à elle par `RADICAL`. Les familles 3, 4 et 5 (plusieurs lignes par client) doivent être **agrégées à une ligne par client avant** toute jointure — sinon un client avec 50 opérations GAB apparaîtrait 50 fois dans le jeu de données final.

#### Stratégie de jointure/agrégation par table

| Table | Cardinalité | Traitement avant jointure |
|---|---|---|
| `PERIMETRE` | 1:1 | Aucun — table de base |
| `ASSI` (cible) | 1:N possible | Vérifier si un client peut avoir plusieurs produits ; si oui, définir une règle de priorité (le plus récent ? le plus significatif ?) avant de n'en garder qu'un par client |
| `OPK`/`PACK` | 1:N (par année) | Garder la dernière année, ou dériver "pack actuel" + "ancienneté dans le pack" |
| `PRODUIT_DIGITAUX` | 1:1 a priori | Vérifier l'unicité ; `DATE_RES_ABON` vide → indicateur "toujours abonné" |
| `SOLDE`/`FLUX`/`DEPOT` | 1:N (mensuel) | Agréger par client : moyenne, min, max, tendance |
| `OPERATIONS` (GAB/RETRAIT/PAYFAC) | 1:N (par transaction) | Agréger par client : nombre d'opérations, montant total, montant moyen, date de dernière opération (récence) — logique RFM |
| `VIGNETTE` | 0:N | Agréger par client : nombre de vignettes payées, montant total, ou indicateur binaire "possède un véhicule" |

#### Points de vigilance qualité, déjà identifiés (à traiter au nettoyage)

| Problème | Où | Correctif |
|---|---|---|
| Encodage cassé (`FÃ©minin` au lieu de `Féminin`) | `PERIMETRE` (`GENDER`) et probablement d'autres | Relire avec `.option("encoding", "latin1")` (ou `cp1252`), pas UTF-8 |
| Numéros de carte en notation scientifique (`4.902654e+18`) | Fichiers transactions (`NUM_CARTE`) | Forcer le typage en `StringType` dès la lecture, jamais laisser inférer en nombre |
| Montants avec virgule française (`"5126,34"`) | Tous les fichiers avec montants | `F.regexp_replace(F.col("montant"), ",", ".").cast("double")` |
| `NaN` dans `FLUX_CRED` = "aucun mouvement", pas une valeur manquante à imputer | Fichiers `FLUX` | Remplacer par `0`, jamais par une moyenne |
| `ETATC` change dans le temps (ex. `V` en 2023 → `W` en 2024) | Fichiers `OPK` par année | Traiter comme un historique, pas des doublons — garder la valeur la plus récente comme feature |

#### Récapitulatif des opérations réalisées (script `build_dataset_final.py`)

Ce tableau explique, pour chaque table source, **quelle opération** a été appliquée, **pourquoi**, et **ce qu'elle apporte** au dataset final.

| Table source | Opération appliquée | Pourquoi | Ce que ça permet ensuite |
|---|---|---|---|
| `PERIMETRE` | **Dédoublonnage** (1 ligne / `RADICAL`, via `row_number()`) | `RADICAL` seul n'était pas unique (3 179 148 distincts pour 3 231 609 lignes) — sans ce correctif, chaque jointure suivante aurait dupliqué des clients | Devient la **table pivot** : base fiable sur laquelle tout le reste vient se greffer par `LEFT JOIN` |
| `ASSI` | **Filtrage** (3 codes sur 6 : `53`, `09`, `18`) puis **dédoublonnage conditionnel** (le plus récent par `DATE_CHARG`) | Les 3 autres codes (`86`, `98`, `99`) sont des produits d'assurance/assistance hors périmètre — les inclure aurait faussé la cible et l'exclusivité annoncée par l'encadrant | Devient `cible` : **la colonne à prédire** (`label_nom`), une valeur par client |
| `OPK` (3 fichiers annuels) | **Union** des 3 années, puis **dédoublonnage** (pack le plus récent) | Même client réapparaît chaque année avec un pack potentiellement différent — on ne veut que la situation actuelle | Feature "pack actuel" + son état — proxy du niveau d'engagement du client envers la banque |
| `PRODUIT_DIGITAUX` | Sélection directe (déjà 1:1) + recodage de la date de résiliation en indicateur binaire | Pas de transformation de cardinalité nécessaire, juste rendre la colonne exploitable par un modèle | Feature "client digital actif" |
| `SOLDE` (2 fichiers) | **Union** puis **agrégation** (moyenne/min/max/nombre de mois observés) | Plusieurs lignes par client (une par mois) — il faut résumer une série temporelle en quelques chiffres | Features de niveau de richesse/stabilité financière |
| `DEPOT_BILANCEIL` (2 fichiers) | **Union** puis **agrégation** (montant moyen) | Même logique que `SOLDE` : plusieurs lignes par client | Feature complémentaire de solde, vue "bilan" |
| `FLUX` (2 fichiers) | **Union**, `NaN → 0` puis **agrégation** (moyenne, somme, nb de mois actifs) | Un mois sans flux crédit doit compter comme 0, pas être ignoré ou imputé par une moyenne | Feature de régularité des revenus (proxy salaire/pension) |
| `OPERATION_GAB` (2 fichiers) | **Union** puis **agrégation** (compte, somme, moyenne, date la plus récente) | Plusieurs lignes par client (une par retrait/consultation) — logique RFM classique | Features de fréquence et de récence d'utilisation du compte |
| `OPE_RETRAIT` (2 fichiers) | **Union** puis **agrégation** (compte, montant total) | Sous-ensemble filtré des opérations GAB (retraits uniquement) — signal plus spécifique que l'agrégat GAB global | Feature ciblée sur le comportement de retrait pur |
| `DIGI_PAYFAC` (3 fichiers) | **Union** puis **agrégation** (compte, montant total) | Trois fichiers de nommage différent pour le même type d'événement (paiement digital confirmé), à traiter comme une seule série | Feature d'adoption des paiements digitaux |
| `VIGNETTE` (2 fichiers) | **Union** (gestion du nom de colonne variable) puis **agrégation** (compte, montant total) | 0 à N lignes par client selon le nombre de véhicules | Feature comportementale indirecte (proxy patrimoine/profil socio-économique) |

**Pourquoi des `LEFT JOIN` et pas des `INNER JOIN` pour les features ?** Un client peut légitimement n'avoir aucune opération GAB, aucune vignette, etc. — ce n'est pas une donnée manquante à traiter comme une erreur, c'est une absence d'activité réelle. Un `INNER JOIN` aurait exclu ces clients du dataset ; le `LEFT JOIN` + `fillna(0)` les garde en codant correctement leur inactivité.

**Pourquoi un `INNER`/`LEFT` sur `cible` change tout ?** La jointure entre `PERIMETRE` et `cible` (`ASSI` filtré) est volontairement en `LEFT` dans le script : elle garde aussi les clients **sans** produit d'épargne connu (`label_nom` sera `NULL` pour eux). C'est ce qui permet l'étape 6 : séparer ceux qui serviront à **entraîner et évaluer** le modèle (`clients_avec_label`) de ceux qui constituent la **vraie population à scorer** (`clients_a_scorer`) — c'est-à-dire l'objectif final du projet.

#### Ce que ce dataset permet de faire ensuite

Une fois `dataset_train_produits` écrit dans `processed-data`, il devient l'entrée directe de la section 7.5 (`StringIndexer` sur `label_nom`, assemblage des features avec `VectorAssembler`, entraînement et comparaison de `RandomForestClassifier`/`LogisticRegression`/`DecisionTreeClassifier`). `dataset_a_scorer` attend simplement que le pipeline final soit entraîné et sauvegardé (section 7.5/8) pour recevoir ses prédictions.

#### État des points bloquants

1. ~~Lister les valeurs distinctes de `CODE_PRODUIT`/`LIBELLE_PRODUIT` dans `ASSI`~~ — **fait**, 3 codes confirmés (`53`, `09`, `18`), à faire valider par l'encadrant avant la soutenance.
2. ~~Vérifier l'unicité de `RADICAL` sur `PERIMETRE`~~ — **fait**, non-unique, dédoublonnage appliqué dans le script.
3. ~~Construire les agrégations par famille~~ — **fait**, voir tableau ci-dessus.
4. **Prochaine étape réelle** : lancer le script corrigé, confirmer que l'ÉTAPE 5 passe (`n_final == n_perimetre`), puis passer à l'encodage et l'entraînement (section 7.5).

**Document de référence complet** : `guide_lecture_donnees_CORRIGE.pdf` — dictionnaire de données colonne par colonne pour les 21 fichiers, avec aperçus réels des données.

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

### 6.5 Qualité des données : ce qui est traité vs ce qui reste à faire

**Déjà géré** (dans `build_dataset_final.py`) : encodage latin1, montants à virgule française, sémantique `NaN=0` pour les flux, non-unicité de `RADICAL`, exclusivité des produits ASSI.

**Pas encore géré — à ajouter avant l'entraînement :**

**a) Doublons stricts dans les tables de transactions** (lignes identiques répétées, possible chevauchement entre fichiers `_COMPLEMENT` et fichiers de base) :
```python
gab_union = gab_union.dropDuplicates()
```
À appliquer sur chaque table 1:N avant son agrégation.

**b) Âges/dates aberrants :**
```python
perimetre = perimetre.withColumn(
    "age",
    F.floor(F.datediff(F.current_date(), F.to_date("DATE_OF_BIRTH", "dd/MM/yyyy")) / 365.25)
)
perimetre = perimetre.filter((F.col("age") >= 18) & (F.col("age") <= 100))
```

**c) Montants extrêmes (outliers statistiques), sur les colonnes de montants uniquement :**
```python
bornes = dataset_final.approxQuantile("solde_moyen", [0.01, 0.99], 0.01)
dataset_final = dataset_final.filter(
    (F.col("solde_moyen") >= bornes[0]) & (F.col("solde_moyen") <= bornes[1])
)
```

**d) Valeurs manquantes dans les features démographiques — bloquant pour `VectorAssembler`** (qui plante sur un `null`) :
```python
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["NOMBRE_ENFANT", "MARITAL_STATUS"],
    outputCols=["NOMBRE_ENFANT_imp", "MARITAL_STATUS_imp"],
    strategy="median",
)
```
Cet `Imputer` s'intègre comme une étape du `Pipeline` MLlib (section 7.5), juste avant le `VectorAssembler`.

**e) Filtrage des comptes actifs (décision métier, à valider avec l'encadrant, pas juste technique) :**
```python
dataset_final = dataset_final.filter(F.col("ETATC").isin(["11"]))  # code à confirmer
```

### 6.5bis Nettoyage explicite des nulls après jointure (`clean_dataset.py`)

Après audit des valeurs `NULL` colonne par colonne sur `dataset_final`, chaque cas a été relu comme une **décision métier**, pas comme un trou à combler mécaniquement (moyenne/médiane par défaut). Script complet : `clean_dataset.py` (annexe du projet, à lancer après `build_dataset_final.py`, avant l'entraînement **et** avant le scoring batch — les deux datasets `dataset_train_produits` et `dataset_a_scorer` reçoivent le même traitement).

| Colonne | Signification du `NULL` | Décision retenue |
|---|---|---|
| `LIBELLE_VILLE` | Redondant avec `CODE_VILLE` (0 null) | **Colonne supprimée** — le code suffit, le libellé n'apporte rien de plus |
| `BPR` | Négligeable (2 lignes sur l'échantillon) | **Lignes supprimées** (`dropna`) |
| `GENDER` | Négligeable (1 ligne) | **Lignes supprimées** (`dropna`) |
| `NOMBRE_ENFANT` | Pas d'enfant, pas une valeur manquante | `fillna(0)` — pas d'imputation par médiane |
| `TAILLE_ENTREPRI` | Pas d'entreprise = compte particulier normal | `fillna("PARTICULIER")` — devient une catégorie explicite (signal particulier vs professionnel), la colonne n'est **pas** supprimée |
| `depot_moyen` | Absence d'activité observée | `fillna(0.0)` |
| `montant_moyen_gab` | Cohérent avec `nb_operations_gab=0` | `fillna(0.0)` |
| `digital_date_activation` | Jamais activé (`digital_toujours_abonne=0`) | Date brute droppée ; remplacée par un flag `jamais_active_digital` (0/1) + une ancienneté dérivée `anciennete_digitale_jours` |
| `derniere_operation_gab` | Jamais utilisé le GAB | Date brute droppée ; remplacée par un flag `jamais_utilise_gab` (0/1) + une récence dérivée `recence_gab_jours` |

**Principe général appliqué** : ne jamais imputer une date par une valeur numérique arbitraire (un `0` serait confondu avec "aujourd'hui"/"jamais de délai") — toujours dériver un flag binaire d'absence **et** une mesure numérique (ancienneté/récence en jours), le flag portant l'information "absence" et la valeur numérique restant `null` (à traiter par l'`Imputer` de la section 6.5.d avec les autres features démographiques) ou par une valeur sentinelle explicite si l'algorithme choisi ne supporte pas les `null`.

`recence_gab_jours` et `anciennete_digitale_jours` (une fois imputés) viennent s'ajouter à la liste `inputCols` du `VectorAssembler` (section 7.4/7.5.1), aux côtés de leurs flags respectifs.

---

### 7.0 Lecture correcte des fichiers (avec les corrections qualité de la section 6.4)

```python
from pyspark.sql import functions as F


# Lecture avec encodage latin1 (obligatoire, sinon les accents sont cassés)
df_perimetre = (
    spark.read
    .option("header", "true")
    .option("sep", ";")
    .option("encoding", "latin1")
    .csv("s3a://raw-data/ATT_PROD_EPARGNE_PERIMETRE.txt")
)

# Montants : virgule française -> point, puis cast en double
def parse_montant(col):
    return F.regexp_replace(F.col(col), ",", ".").cast("double")

# Numéros de carte : toujours forcer en string à la lecture pour ne pas
# perdre de chiffres via la notation scientifique
df_gab = (
    spark.read.option("header", "true").option("sep", ";").option("encoding", "latin1")
    .csv("s3a://raw-data/ATT_PROD_EPARGNE_OPERATION_GAB2023.txt")
    .withColumn("NUM_CARTE", F.col("NUM_CARTE").cast("string"))
)
```

### 7.1 Vérifier la clé de jointure avant tout

```python
total = df_perimetre.count()
distinct = df_perimetre.select("RADICAL").distinct().count()
print(f"Lignes : {total} | RADICAL distincts : {distinct}")
# Si total != distinct, la jointure devra inclure BANQUE/AGENCE/GENERIC/PLURAL en plus
```

### 7.2 Agréger les tables à plusieurs lignes par client (familles 3, 4, 5)

**Soldes/Flux (famille 3)** — agréger une série temporelle en indicateurs :
```python
features_solde = (
    df_solde
    .withColumn("SOLDEVERIF", parse_montant("SOLDEVERIF"))
    .groupBy("RADICAL")
    .agg(
        F.avg("SOLDEVERIF").alias("solde_moyen"),
        F.min("SOLDEVERIF").alias("solde_min"),
        F.max("SOLDEVERIF").alias("solde_max"),
    )
)

features_flux = (
    df_flux
    .withColumn("FLUX_CRED", F.coalesce(parse_montant("FLUX_CRED"), F.lit(0.0)))  # NaN = 0, pas une moyenne
    .groupBy("RADICAL")
    .agg(
        F.sum("FLUX_CRED").alias("flux_total"),
        F.avg("FLUX_CRED").alias("flux_moyen"),
    )
)
```

**Transactions (famille 4)** — logique RFM classique :
```python
features_transactions = (
    df_operations
    .withColumn("MONTANT", parse_montant("MONTANT"))
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_operations"),
        F.sum("MONTANT").alias("montant_total"),
        F.avg("MONTANT").alias("montant_moyen"),
        F.max("DATE_OP").alias("derniere_operation"),
    )
)
```

**Vignette (famille 5)** — feature comportementale indirecte :
```python
features_vignette = (
    df_vignette
    .withColumn("TOTAL_TTC", parse_montant("TOTAL_TTC"))
    .groupBy("RADICAL")
    .agg(
        F.count("*").alias("nb_vignettes"),
        F.sum("TOTAL_TTC").alias("montant_vignette_total"),
    )
)
```

### 7.3 Filtrer la table cible (`ASSI`) une fois le point bloquant de la section 0 tranché

```python
# Étape indispensable AVANT de continuer : lister les valeurs réelles
df_assi.groupBy("CODE_PRODUIT", "LIBELLE_PRODUIT").count().orderBy(F.desc("count")).show(20, truncate=False)

# Une fois la liste des vrais produits d'épargne confirmée avec l'encadrant,
# filtrer ASSI pour ne garder que ceux-là (exemple avec des codes fictifs à remplacer) :
codes_epargne_confirmes = ["XX", "YY", "ZZ"]  # à remplacer par les vrais codes
df_cible = df_assi.filter(F.col("CODE_PRODUIT").isin(codes_epargne_confirmes))
```

### 7.4 Joindre le tout

```python
df_final = (
    df_perimetre
    .join(df_cible.select("RADICAL", "CODE_PRODUIT", "LIBELLE_PRODUIT"), on="RADICAL", how="inner")
    .join(features_solde, on="RADICAL", how="left")
    .join(features_flux, on="RADICAL", how="left")
    .join(features_transactions, on="RADICAL", how="left")
    .join(features_vignette, on="RADICAL", how="left")
    .fillna(0, subset=["nb_operations", "montant_total", "nb_vignettes", "montant_vignette_total"])
)
```
`how="inner"` sur la jointure avec la cible : ne garde que les clients ayant un des produits d'épargne confirmés. `how="left"` sur les features : garde tous ces clients même s'ils n'ont pas d'opérations GAB ou de vignette (auquel cas `fillna(0)` comble les trous après jointure).

### 7.5 Concepts et syntaxe pour la classification multi-classes

Les sous-sections suivantes détaillent chaque brique nécessaire une fois `df_final` construit.

#### 7.5.1 Joindre plusieurs fichiers sources (`join`) — rappel du principe

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

### 7.6bis Nettoyage de la frontière par Tomek Links (complémentaire à la pondération)

**Concept.** Un lien de Tomek est une paire de points de classes différentes qui sont chacun le plus proche voisin de l'autre — un signal que ces points sont à la frontière, voire que le point majoritaire "empiète" sur la zone de la classe minoritaire. Les supprimer nettoie la frontière de décision. **Ce n'est pas une alternative à la pondération, c'est un complément** : Tomek Links agit sur les *lignes* (avant entraînement), la pondération agit sur la *fonction de coût* (pendant l'entraînement).

**Pourquoi ça ne tourne pas dans Spark.** `imbalanced-learn` (la librairie qui implémente Tomek Links) est conçue pour scikit-learn, sur une seule machine, en mémoire — MLlib n'a pas d'équivalent distribué. La bonne nouvelle : Tomek Links ne s'applique **qu'à l'ensemble étiqueté** (`clients_avec_label`, ~140K lignes), jamais à `clients_a_scorer` (3M+ lignes) — 140K lignes tiennent largement en mémoire sur une seule machine.

```python
import pandas as pd
from imblearn.under_sampling import TomekLinks
from pyspark.sql import functions as F

# 1. Ne rapatrier que le train set étiqueté, jamais la population à scorer
train_pd = clients_avec_label.select(
    "RADICAL", "age", "solde_moyen", "flux_cred_moyen", "nb_operations_gab",
    "montant_total_gab", "nb_vignettes_payees", "label_nom"  # + vos autres features
).toPandas()

X = train_pd.drop(columns=["RADICAL", "label_nom"])
y = train_pd["label_nom"]

# 2. Nettoyer la frontière (multi-classe supporté nativement)
tomek = TomekLinks(sampling_strategy="auto")  # "auto" = nettoie toutes les classes majoritaires
X_res, y_res = tomek.fit_resample(X, y)

print(f"Avant : {len(X)} lignes | Après Tomek Links : {len(X_res)} lignes")

# 3. Revenir dans Spark pour la suite du pipeline (StringIndexer, VectorAssembler, weightCol...)
train_pd_clean = X_res.copy()
train_pd_clean["label_nom"] = y_res
df_train_clean = spark.createDataFrame(train_pd_clean)
```

**Ordre d'exécution à respecter :**
```
dataset_train_produits (Parquet)
        │
        ▼
  Tomek Links (pandas, une seule fois)   ← AVANT le split train/test
        │
        ▼
  randomSplit([0.8, 0.2])                ← Tomek Links ne touche jamais le test
        │
        ▼
  Recalcul du poids par classe (section 7.6)   ← sur la distribution POST-Tomek, pas avant
        │
        ▼
  Pipeline (StringIndexer, VectorAssembler, weightCol, classifieur)
```
⚠️ Recalculez le poids par classe (section 7.6) **après** Tomek Links, pas avant — la distribution des classes a changé, l'ancien poids ne serait plus juste.

### 7.6ter Nettoyer `dataset_train_produits` maintenant, ou regénérer ?

Deux catégories de correctifs, traitées différemment :

| Nettoyage | Sur le Parquet déjà écrit, ou relancer `build_dataset_final.py` ? |
|---|---|
| Âges aberrants, valeurs manquantes démographiques, montants extrêmes (section 6.5), filtre `ETATC` | **Sur l'existant** — ce sont des colonnes déjà présentes dans le Parquet final, filtrables/imputables directement en le relisant |
| Doublons stricts dans les tables sources avant agrégation | **Nécessite de relancer l'agrégation** (`build_dataset_final.py` avec `.dropDuplicates()` ajouté) — sinon `nb_operations_gab`, `montant_total`, etc. restent gonflés par des doublons déjà comptés dans le Parquet écrit |

**Diagnostic rapide avant de relancer** (évite un rebuild inutile de 20+ minutes) :
```python
gab_test = spark.read.option("header","true").option("sep",";").option("encoding","latin1") \
    .csv("s3a://raw-data/ATT_PROD_EPARGNE_OPERATION_GAB2023*.txt")
print("Total :", gab_test.count(), "| Sans doublons :", gab_test.dropDuplicates().count())
```
Chiffres égaux → pas de doublons, pas besoin de relancer. Différents → relancez une seule fois, avec `.dropDuplicates()` **et** les correctifs de performance (`cache()`, `shuffle.partitions=8`) appliqués ensemble.

Le nettoyage est une étape **unique et en amont** : on nettoie une fois, on écrit une version propre, puis l'entraînement et le scoring consomment tous les deux cette version — pas besoin de renettoyer à chaque scoring.

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
- [x] Fichiers étudiés, grille de la section 6 remplie
- [x] Table cible (client → produit) identifiée et confirmée avec l'encadrant
- [x] Noms/codes exacts des 3 produits confirmés (codes `53`/`09`/`18`)
- [x] Jointure features + cible réalisée (`build_dataset_final.py`)
- [x] Nettoyage des nulls post-jointure (`clean_dataset.py`, section 6.5bis) — `dataset_train_produits_clean` et `dataset_a_scorer_clean` écrits dans `processed-data`
- [ ] Cible encodée (`StringIndexer`), déséquilibre des classes traité
- [ ] Au moins 3 algorithmes multi-classes comparés (F1 pondéré)
- [ ] Pipeline final sauvegardé
- [ ] Scoring batch multi-classes opérationnel
- [ ] DAG Airflow, Power BI, tests bout en bout, livrables (inchangé section 8 du guide précédent)

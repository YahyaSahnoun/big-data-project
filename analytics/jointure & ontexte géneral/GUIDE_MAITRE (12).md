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

> ### ⚠️ Recadrage du problème (postérieur au tableau ci-dessus — à mentionner dans le rapport comme un oubli corrigé)
>
> Le cadrage ci-dessus (et celui qui a suivi pendant une bonne partie du projet) traitait l'absence d'un client dans `ASSI` comme une population **"à scorer"** — une cible inconnue, à prédire une fois le modèle multi-classes entraîné. **C'était une erreur de cadrage, pas un détail technique.**
>
> Ne détenir **aucun** des 3 produits d'épargne est en réalité une information à part entière : c'est une **4ᵉ classe**, au même titre que les 3 produits. Le ground truth existe donc pour la **totalité** de `PERIMETRE`, pas seulement pour le sous-ensemble présent dans `ASSI` — il n'y a plus de population "inconnue" à ce niveau.
>
> Le besoin métier se découpe donc en **deux modèles**, pas un :
> 1. **Modèle principal — éligibilité** (binaire) : le client détient-il un produit d'épargne (n'importe lequel des 3) ou aucun ? Entraîné sur la **totalité** de `dataset_final`.
> 2. **Modèle bonus — lequel des 3 produits** (multi-classes) : n'a de sens que si la réponse à la question précédente est "oui" — entraîné **uniquement** sur le sous-ensemble éligible.
>
> `build_dataset_final.py` a été mis à jour en conséquence (section 6.4ter ci-dessous détaille le changement). **`EDA_final.ipynb` et le notebook de pipeline d'entraînement ne le sont pas encore** — ils référencent toujours les anciens chemins `dataset_train_produits`/`dataset_a_scorer` et l'ancien cadrage à un seul modèle multi-classes. C'est le prochain chantier (voir checklist, section 10).

**Conséquence sur la modélisation** : ce n'est **pas** un seul modèle multi-classes entraîné sur une cible partiellement inconnue, mais **deux modèles séparés** : un modèle binaire d'éligibilité entraîné sur toute la population, et un modèle multi-classes (3 classes : `53`/`09`/`18`) entraîné uniquement sur les clients éligibles. Détails techniques en section 7.

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
| `ClassNotFoundException: org.apache.hadoop.fs.s3a.S3AFileSystem` sur un `spark-submit` | Le connecteur S3A n'est pas natif à l'image `apache/spark`, il faut le flag `--packages` à chaque lancement | Court terme : ajouter `--packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262` à la commande. Permanent : `spark-defaults.conf` monté dans `spark-master`/`spark-worker` avec `spark.jars.packages` + la config S3A, pour ne plus jamais avoir à le répéter (voir section 7.0bis) |
| `dataset_final` a plus de lignes que `PERIMETRE` dédoublonné (ÉTAPE 5 échoue) | Une des tables agrégées à l'étape 3 (suspect n°1 : `produit_digitaux`, traité comme 1:1 sans vérification ni dédoublonnage réel) contient encore plusieurs lignes par `RADICAL` | Diagnostiquer avec une boucle `count()` vs `select("RADICAL").distinct().count()` sur chaque table agrégée pour isoler la fautive, puis lui appliquer le même dédoublonnage `Window`/`row_number()` que `PERIMETRE`/`ASSI`. **Remplacer le `print()` d'avertissement par un `assert`** pour que le script s'arrête net au lieu de continuer sur une sortie corrompue |

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

**Pourquoi un `LEFT` sur `cible` change tout (mis à jour, cf. recadrage section 0) ?** La jointure entre `PERIMETRE` et `cible` (`ASSI` filtré) est volontairement en `LEFT` dans le script : elle garde aussi les clients **sans** produit d'épargne connu (`label_nom` sera `NULL` pour eux). Ce `NULL` n'est **plus** interprété comme "cible inconnue, à scorer" — c'est directement recodé en `label_eligibilite = 0` (voir 6.4ter). Il n'y a donc plus, à ce stade, de population "à scorer" au sens d'une cible manquante : tout `PERIMETRE` a un label d'éligibilité.

#### 6.4ter Recadrage : deux datasets de sortie, pas un (mise à jour `build_dataset_final.py`)

Suite au recadrage de la section 0, l'ÉTAPE 5bis du script calcule désormais un label binaire pour **toute** la population, et l'ÉTAPE 6/7 écrit **deux** jeux de données distincts au lieu d'un seul `dataset_train_produits`/`dataset_a_scorer` :

| Dataset (Parquet) | Population | Cible | Usage |
|---|---|---|---|
| `processed-data/dataset_eligibilite/` | **Toute** `PERIMETRE` (ex-`dataset_final` complet) | `label_eligibilite` (0/1 — 1 = détient l'un des 3 produits) | Entraîner/évaluer le **modèle principal**, binaire |
| `processed-data/dataset_produit/` | Sous-ensemble filtré `label_eligibilite = 1` uniquement | `label_code`/`label_nom` (3 classes) | Entraîner/évaluer le **modèle bonus**, multi-classes |

Deux garde-fous ajoutés en même temps dans le script (à ne pas reperdre si le script est réédité) :
- Le point d'arrêt de l'ÉTAPE 2 (confirmation des 3 codes produit `53`/`09`/`18`) est passé d'un simple `print` à un vrai `assert` (`CONFIRME_PRODUITS_EPARGNE`) — il bloque l'exécution tant qu'il n'a pas été explicitement validé après relecture de l'inventaire `ASSI`.
- L'ÉTAPE 3bis (unicité `RADICAL` par table avant jointure) et l'ÉTAPE 5 (`n_final == n_perimetre`) sont elles aussi passées d'un avertissement à un `assert` bloquant.

Deux features manquantes ont aussi été comblées à cette occasion (le pipeline aval les attendait déjà sans qu'elles existent en amont) : `anciennete_digitale_jours` et `recence_gab_jours`, calculées par rapport à une `DATE_REFERENCE` fixe (dérivée du suffixe d'année le plus récent des fichiers, ex. `2025`) et non `current_date()` — cohérent avec `age_client`, déjà calculé ainsi.

#### Ce que ce dataset permet de faire ensuite

Une fois `dataset_eligibilite` et `dataset_produit` écrits dans `processed-data`, ils deviennent l'entrée de deux entraînements séparés en section 7 : `dataset_eligibilite` → modèle binaire (`BinaryClassificationEvaluator`), `dataset_produit` → modèle multi-classes (`StringIndexer` sur `label_nom`, `MulticlassClassificationEvaluator`, section 7.7). Le scoring batch (section 8) applique les deux pipelines en cascade : d'abord l'éligibilité sur tout le monde, puis le produit uniquement sur les clients prédits éligibles.

⚠️ **Pas encore fait** : `EDA_final.ipynb` et le notebook de pipeline d'entraînement lisent encore les anciens chemins (`dataset_train_produits`/`dataset_a_scorer`) et l'ancien cadrage à un seul modèle. Ils doivent être mis à jour pour pointer vers `dataset_eligibilite`/`dataset_produit` et dupliquer/adapter les étapes d'entraînement pour les deux modèles avant de relancer la Partie 1 du notebook EDA.

#### État des points bloquants

1. ~~Lister les valeurs distinctes de `CODE_PRODUIT`/`LIBELLE_PRODUIT` dans `ASSI`~~ — **fait**, 3 codes confirmés (`53`, `09`, `18`), à faire valider par l'encadrant avant la soutenance.
2. ~~Vérifier l'unicité de `RADICAL` sur `PERIMETRE`~~ — **fait**, non-unique, dédoublonnage appliqué dans le script.
3. ~~Construire les agrégations par famille~~ — **fait**, voir tableau ci-dessus.
4. ~~Recadrer éligibilité vs produit et scinder le dataset de sortie~~ — **fait** côté `build_dataset_final.py` (section 6.4ter).
5. **Prochaine étape réelle** : mettre à jour `EDA_final.ipynb` et le notebook de pipeline d'entraînement pour consommer `dataset_eligibilite`/`dataset_produit` (au lieu de `dataset_train_produits`/`dataset_a_scorer`) et entraîner les deux modèles séparément (section 7).

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

## 7. Sprint 2 mis à jour — concepts et syntaxe Spark pour les DEUX modèles (éligibilité binaire + produit multi-classes)

Cette section explique **les méthodes disponibles et leur syntaxe**, à adapter dès que les vrais noms de colonnes sont connus (section 6). Suite au recadrage (section 0/6.4ter), tout ce qui suit s'applique à **deux entraînements distincts, sur deux datasets distincts** :
- Section 7.1-7.4, 7.9 : communes aux deux modèles (jointure déjà faite dans `build_dataset_final.py`, assemblage des features).
- Section 7.5-7.6bis : le multi-classes (`RandomForestClassifier`, pondération multi-classe, Tomek Links) s'applique à `dataset_produit` (sous-ensemble éligible). Le modèle d'éligibilité (binaire) réutilise les concepts déjà connus des sprints précédents (`BinaryClassificationEvaluator`, `RandomForestClassifier`/`LogisticRegression` sans `family="multinomial"`) sur `dataset_eligibilite` (population complète) — pas redétaillé ici, cf. guide des sprints précédents.
- Section 7.7-7.8 : évaluation et `Pipeline` — à dupliquer pour les deux modèles (deux `Pipeline` MLlib distincts, deux évaluateurs différents : `BinaryClassificationEvaluator` pour l'éligibilité, `MulticlassClassificationEvaluator` pour le produit).

### 6.5 Qualité des données : doublons, nulls, outliers — implémentation finale

**Ce plan initial est dépassé — tout est maintenant implémenté dans `EDA_final.ipynb`**, le notebook consolidé qui fait référence pour cette étape. Ne pas repartir de zéro avec les extraits de code ci-dessous, qui datent d'avant l'implémentation réelle et sont conservés uniquement pour l'historique des décisions.

⚠️ **Écart connu (recadrage section 0/6.4ter)** : le tableau ci-dessous décrit `EDA_final.ipynb` tel qu'il existe **avant** le recadrage éligibilité/produit — il lit encore un seul dataset en entrée et écrit encore `dataset_train_produits_final`/`dataset_a_scorer_final`. La logique de nettoyage (doublons, nulls, imputation, plafonnement) reste valable telle quelle et n'a pas besoin d'être réécrite ; ce qui doit changer, c'est l'entrée (lire `dataset_eligibilite` en plus de/à la place de l'ancien dataset unique) et la sortie (produire les deux jeux nettoyés séparément, un par modèle) — pas encore fait.

**Ce que fait `EDA_final.ipynb`, dans l'ordre :**

| # | Étape | Détail |
|---|---|---|
| 1 | Doublons | Vérifie l'unicité de `RADICAL` (déjà garantie par `build_dataset_final.py`, contrôle de non-régression) + `dropDuplicates()` stricts, **avant** tout calcul de médiane/quantile |
| 2 | Nulls | Règles métier par colonne (section 6.5bis ci-dessous pour le détail), encodage `GENDER` corrigé |
| 3 | Imputation | `Imputer` (médiane) sur `anciennete_digitale_jours`/`recence_gab_jours`, **fit sur le train uniquement**, rechargé pour le scoring |
| 4 | Valeurs impossibles | Compteurs/montants négatifs → 0, âges aberrants : lignes supprimées sur le train, plafonnées (jamais supprimées) sur le scoring |
| 5 | Plafonnement statistique | Winsorisation IQR avec détection **zero-inflated** (quantiles calculés sur les valeurs > 0 uniquement pour les colonnes à forte proportion de zéros), bornes apprises sur le train, sauvegardées, rechargées pour le scoring |
| 6 | Réduction de dimensions | Suppression des colonnes techniques et fortement corrélées (`digital_toujours_abonne`, `flux_cred_moyen`, `solde_max` remplacé par `solde_volatilite_relative`), dérivation `age_client` sur une date de référence **fixe** (pas `current_date()`, pour un résultat stable dans le temps) |
| 7 | Encodage catégoriel | **Préparé** (`stages_encodage_categoriel()`, `StringIndexer`+`OneHotEncoder`) mais **pas appliqué** ici — le fit se fait dans le `Pipeline` d'entraînement (section 7.5), sur le vrai train complet |

**Fonctionnement local → cluster** : un seul flag `LOCAL_MODE` en haut du notebook bascule tous les chemins (données, modèle d'imputation, bornes IQR) entre un test sur un fichier local et le bucket MinIO complet. La sauvegarde des bornes IQR bascule automatiquement entre fichier JSON local (`open()`/`json.dump()`) et écriture via un DataFrame Spark sur `s3a://` (obligatoire : `open()` ne fonctionne pas sur un chemin `s3a://`).

**Sorties produites** :
- `dataset_train_produits_final` / `dataset_a_scorer_final` (Parquet, `processed-data/`)
- Modèle d'imputation : `s3a://ml-scoring/models/imputer_anciennete_recence`
- Bornes de plafonnement : `s3a://ml-scoring/models/outlier_bounds/`

### 6.5bis Règles de nettoyage détaillées (référence)

**Doublons** : `RADICAL` (contrôle), lignes strictement identiques dans chaque table (`dropDuplicates()`), avant toute agrégation/statistique.

**Nulls, par colonne :**
- `GENDER` : encodage cassé (`FÃ©minin`) normalisé — les valeurs déjà correctes sont explicitement préservées (piège identifié : un `.otherwise(None)` mal écrit peut effacer des données valides).
- `LIBELLE_VILLE` : supprimé (redondant avec `CODE_VILLE`).
- `BPR`/`GENDER` : `dropna` (nulls négligeables observés).
- `NOMBRE_ENFANT` : `fillna(0)` — absence = pas d'enfant, pas une donnée manquante.
- `TAILLE_ENTREPRI` : `fillna("PARTICULIER")` — absence = client particulier, pas professionnel.
- `pack_actuel`/`pack_etat` : catégorie explicite `"SANS_PACK"`/`"SANS_ETAT"`.
- `depot_moyen`/`montant_moyen_gab` : `fillna(0.0)` — absence d'activité, pas une moyenne à deviner.
- `digital_date_activation`/`derniere_operation_gab` : dérivées en `jamais_active_digital`/`jamais_utilise_gab` (flag binaire) + `anciennete_digitale_jours`/`recence_gab_jours` (ancienneté en jours), colonnes de dates brutes supprimées.
- `anciennete_digitale_jours`/`recence_gab_jours` : `Imputer` médiane, fit train uniquement.

**Valeurs impossibles (catégorie 1, règle métier)** : compteurs/montants négatifs → 0 ; `nb_mois_observes_solde` plafonné à 36 ; `NOMBRE_ENFANT` plafonné à 12 ; âge < 16 ou > 100 ans → ligne supprimée sur le train, plafonnée sur le scoring. Chaque plafonnement génère une colonne `_etait_extreme` (flag), pour que le modèle distingue une vraie valeur d'une valeur corrigée.

**Plafonnement statistique (catégorie 3, IQR)** : `k=1.5` (Tukey standard), sur `solde_moyen/min/max`, `depot_moyen`, `flux_cred_moyen/total`, `montant_total/moyen_gab`, `montant_total_retraits/payfac/vignette`. Détection zero-inflated si ≥50% de zéros sur la colonne.

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

### 7.0bis Configurer Spark une fois pour toutes (éviter `--packages` à chaque script)

Le connecteur S3A n'est pas inclus par défaut dans l'image `apache/spark` — sans le flag `--packages`, tout script lisant/écrivant sur `s3a://` échoue avec `ClassNotFoundException: S3AFileSystem`. Plutôt que de rallonger chaque commande `spark-submit`, configurez-le une fois :

```bash
mkdir -p spark-conf
cat > spark-conf/spark-defaults.conf << 'EOF'
spark.jars.packages    org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262
spark.hadoop.fs.s3a.endpoint          http://minio:9000
spark.hadoop.fs.s3a.access.key        minioadmin
spark.hadoop.fs.s3a.secret.key        minioadmin123
spark.hadoop.fs.s3a.path.style.access true
spark.hadoop.fs.s3a.impl              org.apache.hadoop.fs.s3a.S3AFileSystem
EOF
```

Puis montez-le dans `spark-master` et `spark-worker` du `docker-compose.yml` :
```yaml
spark-master:
  volumes:
    - spark-data:/opt/spark/work-dir
    - ./spark-conf/spark-defaults.conf:/opt/spark/conf/spark-defaults.conf

spark-worker:
  volumes:
    - ./spark-conf/spark-defaults.conf:/opt/spark/conf/spark-defaults.conf
```
```bash
docker compose up -d spark-master spark-worker
```

Ensuite, `spark-submit --master spark://spark-master:7077 mon_script.py` suffit — plus besoin de `--packages`, et les scripts eux-mêmes peuvent laisser tomber les 5 lignes `.config("spark.hadoop.fs.s3a...")` répétées partout, `SparkSession.builder.appName(...).getOrCreate()` suffit.

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

**Pourquoi ça ne tourne pas dans Spark.** `imbalanced-learn` (la librairie qui implémente Tomek Links) est conçue pour scikit-learn, sur une seule machine, en mémoire — MLlib n'a pas d'équivalent distribué. La bonne nouvelle : cette section concerne uniquement le **modèle bonus** (produit), donc uniquement `dataset_produit` — le sous-ensemble déjà filtré aux clients éligibles (~140K lignes selon les premiers comptages), jamais `dataset_eligibilite` (population complète, 3M+ lignes) qui n'a pas ce problème de frontière multi-classe. 140K lignes tiennent largement en mémoire sur une seule machine.

```python
import pandas as pd
from imblearn.under_sampling import TomekLinks
from pyspark.sql import functions as F

# 1. Ne rapatrier que dataset_produit (sous-ensemble éligible), jamais dataset_eligibilite en entier
train_pd = dataset_produit.select(
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
dataset_produit (Parquet, sous-ensemble éligible)
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

### 7.6ter Nettoyer `dataset_eligibilite`/`dataset_produit` maintenant, ou regénérer ?

S'applique aux deux jeux de données (le principe est le même, que ce soit sur la population complète ou le sous-ensemble éligible). Deux catégories de correctifs, traitées différemment :

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

## 8. Sprint 3 — scoring batch en cascade : éligibilité puis produit (mise à jour, recadrage section 0)

Suite au recadrage, le scoring batch n'applique plus un seul pipeline multi-classes à tout le monde — il enchaîne les **deux** modèles, le second ne s'appliquant qu'aux clients que le premier juge éligibles :

```python
from pyspark.ml import PipelineModel

pipeline_eligibilite = PipelineModel.load("s3a://ml-scoring/models/pipeline_eligibilite_v1")
pipeline_produit     = PipelineModel.load("s3a://ml-scoring/models/pipeline_multiclasse_v1")
all_clients = spark.read.parquet("s3a://processed-data/features_clients/")

# 1) Modèle principal : éligible ou non, sur TOUTE la population
scores_eligibilite = pipeline_eligibilite.transform(all_clients) \
    .select("client_id", "prediction", "probability") \
    .withColumnRenamed("prediction", "eligible_predit") \
    .withColumnRenamed("probability", "probabilite_eligibilite")

# 2) Modèle bonus : uniquement sur les clients prédits éligibles (pas les 0)
clients_eligibles = all_clients.join(
    scores_eligibilite.filter("eligible_predit = 1").select("client_id"), "client_id"
)
scores_produit = pipeline_produit.transform(clients_eligibles) \
    .select("client_id", "produit_predit", "probability") \
    .withColumnRenamed("probability", "probabilite_produit")

# 3) Assemblage final : un client non éligible n'a pas de produit_predit (NULL, normal)
scores = scores_eligibilite.join(scores_produit, "client_id", "left")
scores.coalesce(8).write.mode("overwrite").parquet("s3a://ml-scoring/scores_clients/")
```
`probabilite_eligibilite` est un vecteur à 2 valeurs (binaire) ; `probabilite_produit` un vecteur à 3 valeurs (une par produit), présent uniquement pour les clients prédits éligibles — les autres ont `produit_predit`/`probabilite_produit` à `NULL`, ce qui est le comportement attendu (la question "lequel des 3" ne se pose pas pour eux).

Le reste du Sprint 3 (DAG Airflow, Spark Thrift Server, connexion Power BI) ne change pas dans sa structure — seul le contenu de `scoring_batch.py` est mis à jour ci-dessus, et le dashboard Power BI doit maintenant distinguer les deux taux (taux d'éligibilité global, puis répartition produit parmi les éligibles).

---

## 9. Glossaire (ajouts multi-classes + recadrage éligibilité)

| Terme | Définition |
|---|---|
| **Modèle principal (éligibilité)** | Modèle **binaire** entraîné sur `dataset_eligibilite` (toute la population) : le client détient-il un produit d'épargne, oui/non |
| **Modèle bonus (produit)** | Modèle **multi-classes** entraîné sur `dataset_produit` (sous-ensemble éligible uniquement) : lequel des 3 produits |
| `label_eligibilite` | Colonne binaire (0/1) ajoutée à `dataset_final` à l'ÉTAPE 5bis — 1 si `label_nom` non nul après jointure avec `cible` |
| `dataset_eligibilite` | Sortie Parquet, population complète, cible = `label_eligibilite` |
| `dataset_produit` | Sortie Parquet, sous-ensemble `label_eligibilite = 1`, cible = `label_code`/`label_nom` |
| Classification multi-classes | Prédire une cible parmi plus de 2 catégories possibles (ici : 3 produits, uniquement chez les éligibles) |
| `StringIndexer` | Convertit une colonne texte en indices numériques pour l'entraînement |
| `IndexToString` | Opération inverse : retrouve le texte à partir de l'indice prédit |
| `Pipeline` (MLlib) | Enchaîne plusieurs étapes de transformation + modèle comme une seule unité réutilisable — un par modèle ici (`pipeline_eligibilite_v1`, `pipeline_multiclasse_v1`) |
| F1 pondéré | Métrique combinant précision et rappel, moyenne sur toutes les classes, pondérée par leur effectif |
| Matrice de confusion | Tableau croisant classe réelle × classe prédite, pour voir précisément quelles confusions le modèle fait |

---

## 10. Checklist finale (mise à jour — recadrage éligibilité/produit)

- [x] Environnement Docker fonctionnel sur les deux machines
- [x] Données ingérées dans `raw-data` (NiFi + méthode de secours `mc`)
- [x] Fichiers étudiés, grille de la section 6 remplie
- [x] Table cible (client → produit) identifiée et confirmée avec l'encadrant
- [x] Noms/codes exacts des 3 produits confirmés (vs. notes section 0)
- [x] **Recadrage identifié et corrigé dans `build_dataset_final.py`** : éligibilité (4ᵉ classe "aucun produit") séparée du choix du produit — deux datasets écrits (`dataset_eligibilite`, `dataset_produit`)
- [ ] **`EDA_final.ipynb` mis à jour** pour lire/nettoyer `dataset_eligibilite`/`dataset_produit` séparément (au lieu de l'ancien dataset unique) — **à faire**
- [ ] **Notebook de pipeline d'entraînement mis à jour** pour entraîner les deux modèles séparément (au lieu d'un seul multi-classes) — **à faire**
- [ ] Modèle principal (éligibilité, binaire) entraîné et évalué (`BinaryClassificationEvaluator`)
- [ ] Modèle bonus (produit, multi-classes) entraîné et évalué (`MulticlassClassificationEvaluator`, ≥3 algorithmes comparés, F1 pondéré)
- [ ] Déséquilibre des classes traité pour les deux modèles (pondération + Tomek Links pour le multi-classes, uniquement sur `dataset_produit`)
- [ ] Les deux `Pipeline` (MLlib) sauvegardés séparément (`pipeline_eligibilite_v1`, `pipeline_multiclasse_v1`)
- [ ] Scoring batch en cascade opérationnel (éligibilité sur tous, puis produit sur les éligibles uniquement — section 8)
- [ ] DAG Airflow, Power BI (dashboard distinguant taux d'éligibilité et répartition produit), tests bout en bout, livrables — **mentionner le recadrage dans le rapport comme un oubli identifié et corrigé**

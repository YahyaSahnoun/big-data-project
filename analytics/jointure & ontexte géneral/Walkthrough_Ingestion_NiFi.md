# Walkthrough — Ingestion NiFi (raw-data)

Extrait et détaillé à partir du `GUIDE_MAITRE.md` (sections 2.3, 3, 4) pour un pas-à-pas autonome. Contexte : on route les 21 fichiers texte sources (`~/data_clients` sur l'hôte) vers le bucket MinIO `raw-data`, via un flux NiFi à 3 processeurs.

---

## 0. Prérequis — accès et buckets

**NiFi** : http://localhost:8443/nifi — identifiants `admin` / `admin12345678`

Avant de construire le flux, les 3 buckets MinIO doivent exister (normalement déjà créés au démarrage de la stack, via le service `mc` dans `docker-compose.yml` — à vérifier dans la console MinIO, http://localhost:9001, `minioadmin` / `minioadmin123`) :
- `raw-data` (cible de cette ingestion)
- `processed-data`
- `ml-scoring`

Si `raw-data` n'existe pas encore :
```bash
docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin123
docker exec minio mc mb -p local/raw-data
```

**Dossier source** : le service `nifi` dans `docker-compose.yml` monte `~/data_clients` (le `$HOME` de chacun) vers `/data/clients` **en lecture seule** à l'intérieur du conteneur :
```yaml
volumes:
  - ~/data_clients:/data/clients:ro
```
Chaque personne doit avoir rempli **son propre** `~/data_clients` avec les 21 fichiers `.txt` sources avant de démarrer NiFi — le montage fonctionne tel quel pour les deux machines, pas besoin de changer le chemin dans le compose.

---

## 1. Construire le flux : 3 processeurs

Le flux retenu est volontairement simple, en 3 étapes :

```
ListFile  →  FetchFile  →  PutS3Object
(repère les fichiers)  (les lit)  (les envoie sur MinIO)
```

### 1.1 `ListFile`
- Glisser le processeur `ListFile` sur le canvas.
- Configurer la propriété **Input Directory** : `/data/clients` (le point de montage vu depuis le conteneur, pas le chemin hôte).
- Laisser les autres propriétés par défaut pour un premier test.

### 1.2 `FetchFile`
- Relier `ListFile` → `FetchFile` sur la relation `success`.
- `FetchFile` lit le contenu de chaque fichier listé par `ListFile` et le transforme en flowfile avec contenu.

### 1.3 `PutS3Object`
- Relier `FetchFile` → `PutS3Object` sur la relation `success`.
- Configurer `PutS3Object` :
  - **Bucket** : `raw-data`
  - **Endpoint Override URL** : `http://minio:9000` (nom du service Docker, pas `localhost` — NiFi et MinIO sont sur le même réseau `pipeline-net`)
  - **Access Key / Secret Key** : `minioadmin` / `minioadmin123`
  - Cocher/activer le **path-style access** si l'option est disponible dans la version de processeur utilisée (MinIO en a besoin, contrairement au vrai S3 AWS)

---

## 2. Les deux pièges déjà rencontrés — à éviter dès le premier lancement

### Piège 1 — processeurs avec un triangle ⚠, refusent de démarrer
**Cause** : chaque processeur a des relations (`success`, `failure`, etc.) qui doivent être explicitement routées ou terminées — NiFi refuse de démarrer un processeur qui a une relation non gérée.

**Correctif** : pour chaque processeur (`ListFile`, `FetchFile`, `PutS3Object`), onglet **Relationships** dans sa configuration → cocher **terminate** sur toutes les relations qui ne sont pas `success` (typiquement `failure`, `not.found`, etc., selon le processeur).

### Piège 2 — relancer un test ne redétecte aucun fichier
**Cause** : `ListFile` retient en mémoire l'état des fichiers déjà listés (c'est le comportement voulu en usage réel — ne pas re-ingérer indéfiniment les mêmes fichiers). En phase de test, ça donne l'impression que le flux ne fait plus rien.

**Correctif** : clic droit sur `ListFile` → **View State** → **Clear State**, puis redémarrer le processeur pour forcer un nouveau passage complet sur tous les fichiers du dossier.

---

## 3. Tester

1. Démarrer les 3 processeurs (sélection multiple → clic droit → **Start**, ou bouton play sur chacun individuellement).
2. Vérifier la progression : chaque processeur affiche un compteur de flowfiles traités dans son coin.
3. Confirmer côté MinIO : console http://localhost:9001 → bucket `raw-data` → les 21 fichiers doivent apparaître.
4. En cas de blocage, clic droit sur un processeur → **List Queue** (si une file d'attente existe entre deux processeurs) pour inspecter les flowfiles en attente et voir s'il y a une erreur.

---

## 4. Statut réel au moment de la rédaction de ce guide

- Le flux `ListFile → FetchFile → PutS3Object` est **construit et fonctionnel**, testé avec succès sur les 21 fichiers, après correction du piège "Clear State" ci-dessus.
- **Le chargement initial (backfill) des 21 fichiers a en réalité été fait via une méthode de secours `mc cp`** (boucle bash, un fichier à la fois avec retry), pas directement via ce flux NiFi — voir §5 ci-dessous pour le contexte.
- Le flux NiFi reste en place et fonctionnel pour l'ingestion de **futurs** fichiers, et pour la cohérence avec l'architecture présentée en soutenance (NiFi = brique d'ingestion continue dans le schéma global, `mc` = uniquement le backfill historique).

---

## 5. Contexte — pourquoi une méthode de secours `mc` existe aussi

Lors du chargement initial, `mc cp --recursive` en parallèle envoyait trop de connexions simultanées à MinIO, provoquant des `connect: connection refused`. Le correctif retenu : une boucle bash envoyant les fichiers **un par un**, avec retry automatique :

```bash
# ~/ingest.sh -- pattern retenu (script original non conservé tel quel dans ce guide,
# reconstruit ici à partir de sa description : boucle un fichier à la fois + retry)
for f in ~/data_clients/*.txt; do
  n=0
  until docker exec minio mc cp "$f" local/raw-data/ || [ $n -ge 5 ]; do
    n=$((n+1))
    echo "Échec, tentative $n/5 pour $f -- attente 5s"
    sleep 5
  done
done
```

**Positionnement pour le rapport** (déjà validé dans le guide maître) : chargement initial en masse via `mc` (backfill historique), ingestion continue automatisée via NiFi — c'est un choix d'architecture assumé, pas un contournement à cacher. Les deux méthodes coexistent pour de bonnes raisons distinctes.

---

## 6. Pour aller plus loin

Le `GUIDE_MAITRE.md` (section 3, tableau des bugs) documente aussi d'autres soucis rencontrés côté infra Docker (DNS interne, résolution `minio`, etc.) qui peuvent resurgir si l'environnement est reconstruit de zéro sur la machine du collègue — à consulter si le flux ne démarre pas pour une raison qui n'est pas l'un des deux pièges ci-dessus.

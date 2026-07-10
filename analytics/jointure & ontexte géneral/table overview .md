Comprendre les données brutes
Guide de lecture des fichiers – Projet de scoring d'éligibilité
aux produits d'épargne
Document de travail interne
9 juillet 2026
Résumé
Ce document explique, sans prérequis bancaire, ce que contiennent les fichiers bruts fournis
pour le projet de scoring d'éligibilité aux produits d'épargne. Il détaille la signification de
chaque colonne, regroupe les fichiers par famille métier, propose un schéma des jointures
possibles entre les tables, et documente le traitement des valeurs manquantes et aberrantes
appliqué avant modélisation.

Table des matières
1 Contexte et objectif de ce document 1
2 La clé de tout : comment un compte est identifié 2
3 Catalogue des fichiers par famille métier 2
3.1 Famille 1 – Référentiel client (qui est le client ?) . . . . . . . . . . . . . . . . . . . 2
3.2 Famille 2 – Produits, packs et abonnements digitaux . . . . . . . . . . . . . . . . 2
3.3 Famille 3 – Soldes et mouvements financiers . . . . . . . . . . . . . . . . . . . . . 3
3.4 Famille 4 – Opérations et transactions . . . . . . . . . . . . . . . . . . . . . . . . 3
3.5 Famille 5 – Paiement de la vignette automobile . . . . . . . . . . . . . . . . . . . 5
4 Jointures et agrégations réalisées 6
4.1 Principe général . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4.2 Schéma des relations (mis à jour) . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4.3 Opérations réalisées, table par table . . . . . . . . . . . . . . . . . . . . . . . . . 6
5 Traitement des valeurs manquantes et aberrantes 7
5.1 Principe général : deux catégories distinctes . . . . . . . . . . . . . . . . . . . . . 7
5.2 Catégorie 1 – Valeurs impossibles (règles métier) . . . . . . . . . . . . . . . . . . 7
5.3 Catégorie 2 – Plafonnement statistique (winsorisation IQR) . . . . . . . . . . . . 8
5.4 Le piège des colonnes "zero-inflated" . . . . . . . . . . . . . . . . . . . . . . . . . 8
5.5 Non-fuite train/scoring . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
5.6 Résultat du diagnostic (train) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
6 Champs exploitables retenus et raison de leur choix 10
7 Techniques de modélisation retenues 11
7.1 Chaîne de transformation (Pipeline MLlib) . . . . . . . . . . . . . . . . . . . . . . 11
7.2 Algorithmes comparés . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.3 Gestion du déséquilibre des classes . . . . . . . . . . . . . . . . . . . . . . . . . . 11
7.4 Évaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
8 Synthèse et état d'avancement 12

1 Contexte et objectif de ce document
Les fichiers fournis proviennent du système d'information d'une banque (vraisemblablement
un groupe organisé en banques régionales, la marque commerciale "Chaabi" apparaissant dans
certains libellés de produits). Chaque fichier est un export brut (format texte, séparateur ;),
correspondant à une table ou à un extrait périodique d'une table du système bancaire.
Sans culture bancaire, deux difficultés se posent en regardant ces fichiers :
1. Beaucoup de colonnes portent des noms techniques abrégés (RADICAL, CCLE, ETATC, GENERIC...)
qui ne veulent rien dire hors contexte bancaire marocain.
2. Les fichiers ne sont pas indépendants : ils décrivent tous le même client, vu sous des
angles différents (son identité, ses produits, ses soldes, ses opérations). Il faut comprendre
comment les relier pour construire un tableau unique "un client = une ligne" exploitable
par un modèle de machine learning.
Ce document répond aux deux points : la section 2 explique la structure d'identification
d'un compte, la section 3 détaille chaque fichier colonne par colonne, la section 4 explique
concrètement comment (et pourquoi) relier ces tables entre elles, et la section 5 documente le
nettoyage des valeurs manquantes et aberrantes réalisé avant l'assemblage du jeu de données
final.

2 La clé de tout : comment un compte est identifié
Avant de détailler chaque fichier, il faut comprendre un jeu de colonnes qui revient presque
partout, car c'est lui qui permet de relier les tables entre elles. Dans le système bancaire marocain, un compte n'est pas identifié par un simple numéro unique : il est identifié par une
combinaison de plusieurs segments, un peu comme un IBAN est composé de plusieurs blocs
(code banque, code guichet, numéro de compte, clé de contrôle).
À retenir : le sextuplet (BANQUE, AGENCE, GENERIC, RADICAL, PLURAL, CLE) joue
le même rôle qu'un IBAN. RADICAL seul suffit probablement à distinguer la plupart des clients, mais rien ne garantit qu'il soit unique à lui seul sur l'ensemble des
banques/agences – ce point doit être vérifié avant de l'utiliser comme clé de jointure
unique (voir section 4).
D'autres colonnes techniques reviennent aussi très souvent :

3 Catalogue des fichiers par famille métier
Les 21 fichiers peuvent être regroupés en 5 familles selon ce qu'ils décrivent. Pour chaque famille, les colonnes déjà expliquées en section 2 ne sont pas répétées : seules les colonnes spécifiques
sont détaillées.

3.1 Famille 1 – Référentiel client (qui est le client ?)
Fichier concerné : ATT_PROD_EPARGNE_PERIMETRE
C'est la table de profil client : une ligne par client, avec ses caractéristiques socio-démographiques.
C'est le point de départ naturel de tout travail de scoring, car elle définit la population de référence
(le périmètre des clients à scorer).
Granularité : une ligne par client (par RADICAL) – c'est une table de dimension, pas de
transactions.

3.2 Famille 2 – Produits, packs et abonnements digitaux
C'est la famille la plus importante pour ce projet : c'est ici que se trouve très probablement
la table cible (quel produit d'épargne le client détient).
Granularité : en général une ligne par client et par produit/pack (éventuellement plusieurs
lignes si le client a plusieurs produits, ou une ligne par année pour les fichiers OPK).

Colonne Signification
BANQUE Code de la banque régionale gestionnaire du compte
(le groupe est organisé en plusieurs banques
régionales, chacune avec son propre code).
AGENCE Code de l'agence bancaire (le guichet physique) au
sein de cette banque régionale.
GENERIC / GENERIQUE Code numérique (le plus souvent 21111, parfois
21117 ou 21150) qui désigne la nature du compte (le
type de produit auquel le compte se rattache). Ce
n'est pas un champ "générique" au sens d'inutile :
c'est un segment technique du numéro de compte. Le
fait qu'il varie peu suggère qu'il code la famille de
produit ; les valeurs différentes de 21111 méritent
d'être isolées et comparées.
RADICAL Le numéro de compte lui-même (la partie
"significative"). C'est en pratique l'identifiant qui se
rapproche le plus d'un identifiant client/compte, et
donc le candidat naturel pour faire les jointures entre
fichiers.
PLURAL Un indice de sous-compte / rang. Un même client (un
même RADICAL) peut avoir plusieurs comptes ou
co-titulaires ; PLURAL distingue ces occurrences
(souvent 0 pour le cas "normal").
CLE / CCLE La clé de contrôle (comme la clé RIB en France).
C'est un chiffre calculé automatiquement à partir des
autres segments, qui sert uniquement à détecter les
erreurs de saisie du numéro de compte. Elle n'a
aucune valeur prédictive et doit être exclue de
toute modélisation.
Table 1 – Colonnes techniques d'identification du compte, communes à la quasi-totalité des
fichiers.

3.3 Famille 3 – Soldes et mouvements financiers
Ces fichiers décrivent l'état financier du compte dans le temps : combien d'argent il y a, et
combien en entre ou en sort.
Granularité : plusieurs lignes par client (une par période) – il faudra agréger (moyenne,
min, max, tendance) avant de rejoindre ces données à la table cible pour obtenir une ligne par
client.

3.4 Famille 4 – Opérations et transactions
Ces fichiers enregistrent des événements individuels (chaque retrait, chaque paiement),
contrairement aux fichiers de soldes qui sont des photos périodiques.
Granularité : plusieurs (souvent beaucoup) de lignes par client – ces fichiers servent à calculer des compteurs et sommes (nombre d'opérations, montant total, dernière date d'opération)
avant jointure, jamais à joindre ligne à ligne tels quels.

Colonne Signification
DATE_CHARG Date de chargement de l'extrait (la date à laquelle
cette photo des données a été prise), pas
nécessairement une date d'événement métier. Souvent
la même pour toutes les lignes d'un même fichier.
ETATC État du compte (code numérique, ex. 11) – actif,
clôturé, dormant, etc. Le référentiel exact des codes
est à demander, mais cette colonne est un bon filtre
pour ne garder que les comptes actifs.
DATE_OP Date (et heure) d'une opération précise – présente
uniquement dans les fichiers de transactions (retraits,
GAB, paiements digitaux), contrairement à
DATE_CHARG qui est une date d'extraction.
Table 2 – Autres colonnes techniques récurrentes.

Colonne Signification
DATE_OF_BIRTH Date de naissance du client – permet de calculer l'âge,
une feature souvent très prédictive.
CODE_VILLE / LIBELLE_VILLE Code et nom de la ville de résidence – feature
géographique (urbain/rural, région).
BPR Vraisemblablement le code de la "Banque Populaire
Régionale" de rattachement du client – à recouper avec
BANQUE pour vérifier si c'est redondant.
GENDER Sexe du client (attention à l'encodage : le fichier contient
des caractères mal encodés comme FÃ©minin au lieu de
Féminin – problème d'encodage latin-1/UTF-8 à
corriger au nettoyage).
MARITAL_STATUS Situation familiale, encodée numériquement (1, 9, ...).
Le référentiel exact des codes (marié, célibataire, etc.)
est à demander à l'encadrant ou à déduire par fréquence.
NOMBRE_ENFANT Nombre d'enfants déclarés – feature potentiellement très
liée à un produit "famille" du type Avenir Mes Enfants
évoqué en préambule du projet.
CUSTOMER_RATING Segment ou catégorie du client (codes vus : SIL, SVC,
FNC). À l'évidence une nomenclature interne de
segmentation clientèle (par exemple type d'emploi ou
catégorie socio-professionnelle) – le référentiel des codes
doit être demandé, mais la colonne est un candidat
feature fort.
TAILLE_ENTREPRI Taille d'entreprise – rempli uniquement pour les clients
professionnels/entreprises, vide (NaN) pour les
particuliers. Peut servir de proxy pour distinguer
clientèle particulière vs. professionnelle.
Table 3 – Colonnes spécifiques du référentiel client (PERIMETRE).

Fichier Contenu
ATT_PROD_EPARGNE_ASSI Produits souscrits par le client (CODE_PRODUIT, LIBELLE_PRODUIT). Table
cible confirmée. L'inventaire complet des valeurs distinctes a révélé 6
produits, dont seuls 3 sont de l'épargne : 53 (Avenir Mes Enfants),
09 (MaRetraite), 18 (Epargne Evolution). Les trois autres (86 Attamine
Chaabi Hissab, 98 Al Injad Chaabi, 99 Al Injad Al Moumtaz ) sont des
produits d'assurance/ d'assistance hors périmètre et doivent être filtrés
avant tout calcul d'exclusivité (un client peut légitimement cumuler une
assurance et un produit d'épargne sans contredire l'exclusivité annoncée
par l'encadrant, qui ne porte que sur les 3 produits d'épargne entre eux).
ATT_PROD_EPARGNE_OPK2023/2024
ATT_HISSAB_OPK2023/2025
Pack bancaire souscrit par le client (CODE_PACK, ex. PFA01, PJUN0) – un
pack est une offre groupée de services (carte, assurance, épargne...), pas
un produit d'épargne isolé. Fichiers dupliqués par année (photo
annuelle) : un même client (RADICAL) réapparaît chaque année, ce qui
permet de suivre l'évolution de son pack dans le temps. ETATC vaut ici W
ou V (à décoder, probablement actif/résilié).
ATT_HISSAB_PRODUIT_DIGITAUX Abonnement aux services bancaires digitaux (banque en ligne / mobile).
DATE_VAL_ABON = date de début d'abonnement, DATE_RES_ABON = date de
résiliation (vide si toujours actif), S5 = indicateur Oui/Non (statut actif).
Feature comportementale utile : "client digital" ou non.
Table 4 – Fichiers de la famille Produits / Packs / Digital.

Fichier Contenu
ATT_PROD_EPARGNE_SOLDE_2025
.._SOLDE_COMPLEMENT
Solde du compte (SOLDEVERIF = solde vérifié) à une date donnée
(DATE_CHARG). Une ligne par compte et par période (mensuelle selon les
dates observées) – c'est donc une série temporelle de soldes, pas une
valeur unique.
ATT_PROD_EPARGNE_DEPOT_BILANCEIL_2023
.._DEPONT_BILANCIEL_COMPLEMENT
Vue "bilan" des dépôts : TYPE_DEPOT (type de dépôt, ex. Dépôts à vue =
compte courant classique) et MONTANT_DEPOT (montant correspondant).
Complète la vision du solde par une décomposition par type de dépôt.
ATT_PROD_EPARGNE_FLUX_2023/2025 Flux créditeurs mensuels (FLUX_CRED) : argent qui entre sur le compte
(virements de salaire, dépôts...). Beaucoup de valeurs manquantes (NaN)
signifient simplement "aucun flux ce mois-là", pas une erreur. Très utile
pour des features de type RFM (récence / fréquence / montant).
Table 5 – Fichiers de la famille Soldes / Flux.

Fichier Contenu
ATT_PROD_EPARGNE_OPERATION_GAB2023
.._OPER_GAB_COMPLEMENT
Opérations réalisées au GAB (Guichet Automatique Bancaire,
c'est-à-dire un distributeur automatique). NUM_CARTE = numéro de carte
bancaire (affiché en notation scientifique – artefact d'ouverture dans
Excel/pandas, à retraiter en texte pour ne pas perdre de chiffres). MONTANT
= montant de l'opération. CODE_FONCT = code de la fonction utilisée au
GAB (ex. 31) – retrait, consultation de solde, etc., référentiel à demander.
ATT_HISSAB_OPE_RETRAIT2023
.._OPER_RETRAIT_COMPLEMENT
Sous-ensemble des opérations GAB filtré sur les retraits uniquement
(CODE_FONCT constant à 1). Mêmes colonnes que ci-dessus.
ATT_PROD_EPARGNE_DIGI_PAYFAC_COMPLEMENT
.._DIGI_PAYFAC_2024
ATT_HISSAB_DIGI_PAYFAC2023
Paiements réalisés via un service de paiement digital (PayFac = "Payment
Facilitator"). MONTANT_TOTAL = montant payé, STATUT (C = confirmé),
DT_CONFIRMATION = date/heure de la confirmation du paiement.
Table 6 – Fichiers de la famille Transactions.

3.5 Famille 5 – Paiement de la vignette automobile
Fichiers concernés : ATT_PROD_EPARGNE_VIGNETTE et .._VIGNETTE_COMPLEMENT.
La vignette est la taxe annuelle sur les véhicules au Maroc, que la banque permet de payer
en ligne. CONTRAT identifie le paiement, TOTAL_TTC (ou C.TOTAL_TTC/100 dans la version complément, à diviser par 100 – probablement stocké en centimes) est le montant payé, STATUS_PAIEMENT
son statut, et ANNEE_PAIEMENT l'année concernée.
Ce fichier n'a l'air de rien pour un scoring de produits d'épargne, mais c'est en réalité une
feature comportementale indirecte : un client qui paie une vignette possède probablement
un véhicule, ce qui est corrélé au profil socio-économique (âge, revenu, situation familiale) –
potentiellement utile pour distinguer par exemple un profil "jeune actif" d'un profil "famille".
Granularité : zéro, une, ou plusieurs lignes par client selon le nombre de véhicules – à
agréger en "nombre de vignettes payées" et "montant total vignette" par client.

4 Jointures et agrégations réalisées
Cette section documente ce qui a été effectivement implémenté dans le script build_dataset_final.py
(plus la simple stratégie envisagée) : quelle opération a été appliquée à chaque table, pourquoi,
et ce que cela permet de faire ensuite.

4.1 Principe général
Tous les fichiers partagent la même clé de compte (section 2). La jointure se fait sur RADICAL.
Vérification faite en pratique : RADICAL seul n'est pas unique sur PERIMETRE (3 179 148
valeurs distinctes pour 3 231 609 lignes) – un dédoublonnage a donc été nécessaire avant de
l'utiliser comme clé de jointure (détail en section 6).
La difficulté centrale reste la même que pressentie : les tables n'ont pas toutes la même
granularité. Il faut agréger avant de joindre, sous peine de dupliquer des clients dans le jeu
final.

4.2 Schéma des relations (mis à jour)
PERIMETRE
(dédupliqué)
1 ligne / client
(référentiel démographique)
ASSI filtré
CIBLE confirmée : codes
OPK / PACK53 / 09 / 18
union 3 ans,
garde le + récent
PRODUIT_DIGITAUX
1 ligne / client
SOLDE / FLUX /
DEPOT_BILANCEIL
union + agrégation
OPERATIONS
GAB / RETRAIT /
PAYFAC
union + agrégation RFM
VIGNETTE
union + agrégation
RADICAL, LEFT JOIN RADICAL, LEFT JOIN
RADICAL, LEFT JOIN
RADICAL, LEFT JOIN
RADICAL, LEFT JOIN RADICAL, LEFT JOIN
Pourquoi des LEFT JOIN et pas des INNER JOIN ? Un client peut légitimement n'avoir
aucune opération GAB, aucune vignette, etc. – ce n'est pas une erreur, c'est une absence réelle
d'activité. Un INNER JOIN aurait exclu ces clients du jeu de données ; le LEFT JOIN suivi d'un
fillna(0) les garde en codant correctement leur inactivité (0 opération, 0 montant), au lieu de
les faire disparaître silencieusement.
Seule exception : la jointure entre PERIMETRE et la cible (ASSI filtré) est aussi en LEFT,
volontairement – elle conserve les clients sans produit d'épargne connu (label = NULL pour
eux). C'est ce qui permet de séparer ensuite la population d'entraînement de la vraie population
à scorer (section 6).

4.3 Opérations réalisées, table par table

Table Opération réalisée Pourquoi Ce que ça apporte
PERIMETRE Dédoublonnage
(row_number() par RADICAL)
RADICAL seul n'était
pas unique
Devient la table pivot
fiable
ASSI Filtrage (3 codes/6) puis
dédoublonnage conditionnel
(le plus récent)
Exclure les 3 produits
hors épargne avant de
juger de l'exclusivité
Devient cible : la
colonne à prédire
OPK (3 fichiers) Union puis dédoublonnage
(le plus récent)
Même client réapparaît
chaque année
Feature "pack actuel" +
engagement
PRODUIT_DIGITAUXSélection directe + recodage
binaire
Déjà 1 :1, juste rendre
exploitable
Feature "client digital
actif"
SOLDE (2 fichiers) Union puis agrégation
(moyenne/min/max/nb
mois)
Série temporelle,
plusieurs lignes/client
Niveau de
richesse/stabilité
DEPOT_BILANCEIL
(2 fichiers)
Union puis agrégation
(montant moyen)
Même logique que
SOLDE
Vue bilan
complémentaire
FLUX (2 fichiers) Union, NaN→0, puis
agrégation
Un mois sans flux = 0,
pas une valeur à
imputer
Régularité des revenus
OPERATION_GAB
(2 fichiers)
Union puis agrégation RFM
(compte, somme, moyenne,
date récente)
Plusieurs lignes/client,
logique RFM classique
Fréquence et récence
d'usage
OPE_RETRAIT (2
fichiers)
Union puis agrégation
(compte, montant)
Sous-ensemble des
GAB (retraits)
Signal ciblé sur les
retraits
DIGI_PAYFAC (3
fichiers)
Union puis agrégation
(compte, montant)
Trois nommages
différents, même type
d'événement
Adoption du paiement
digital
VIGNETTE (2
fichiers)
Union (gestion nom de
colonne variable) puis
agrégation
0 à N lignes/client
selon nb véhicules
Proxy patrimoine/profil
Table 7 – Opérations réalisées par le script build_dataset_final.py.

Champs explicitement exclus : CLE/CCLE (clé de contrôle, aucune valeur prédictive),
NUM_CARTE brut (identifiant technique, pas une feature), DATE_CHARG brute (date d'extraction, pas
un événement métier – seule sa comparaison entre fichiers sert, via les fenêtres de dédoublonnage).

Sur la population finale : le LEFT JOIN avec la cible sépare le jeu en deux :
— Clients avec produit connu (label non nul) : population d'entraînement et d'évaluation
du modèle.
— Clients sans produit connu (label nul) : la vraie population à scorer – l'objectif
final du projet.

5 Traitement des valeurs manquantes et aberrantes
Une fois le jeu de données assemblé (une ligne par client, cf. section 4), deux problèmes
distincts restaient à traiter avant modélisation : les valeurs manquantes (déjà gérées via un
Imputer MLlib fit sur le train et réappliqué au scoring, non détaillé ici) et les valeurs aberrantes,
objet de cette section. Le script correspondant est clean_dataset.py.

5.1 Principe général : deux catégories distinctes
Toutes les valeurs aberrantes ne se traitent pas de la même façon. On distingue deux
catégories, appliquées dans cet ordre :
— Catégorie 1 – Valeurs impossibles : des erreurs de données pures (un compteur
négatif, une date de naissance dans le futur), qui n'ont aucune interprétation métier
valide. Corrigées par des règles métier explicites (bornes dures), pas par une méthode
statistique.
— Catégorie 2 – Valeurs statistiquement extrêmes : des valeurs plausibles individuellement (un très gros solde n'est pas "impossible") mais qui s'écartent fortement de la
distribution du reste de la population, et risquent de dominer l'apprentissage du modèle
si elles ne sont pas contenues. Traitées par winsorisation (plafonnement) selon la méthode
IQR de Tukey.
Cette distinction importe : appliquer une méthode statistique (IQR) à une valeur qui est en
réalité une erreur de saisie masquerait le problème plutôt que de le corriger ; à l'inverse, supprimer ou plafonner arbitrairement une valeur extrême mais plausible ferait perdre un signal utile
au modèle.

5.2 Catégorie 1 – Valeurs impossibles (règles métier)
Fonction : corriger_valeurs_impossibles(df, is_train). Trois familles de corrections :
— Compteurs négatifs (COLS_COMPTAGE_NON_NEGATIVES : nb_mois_observes_solde,
nb_mois_avec_flux, nb_operations_gab, nb_retraits, nb_paiements_digitaux,
nb_vignettes_payees, NOMBRE_ENFANT) : ramenés à 0. Un compteur ne peut structurellement pas être négatif.
— Montants négatifs illégitimes (COLS_MONTANT_NON_NEGATIFS : depot_moyen,
flux_cred_moyen, flux_cred_total, montant_total_gab, montant_moyen_gab,
montant_total_retraits, montant_total_payfac, montant_total_vignette) : ramenés à 0. Attention : solde_moyen, solde_min et solde_max sont volontairement
exclus de cette liste, car un solde négatif (découvert bancaire) est un état de compte
parfaitement légitime, pas une erreur.
— nb_mois_observes_solde anormalement élevé : plafonné à un seuil métier de 36 mois
(12×3), au-delà duquel la valeur est jugée non plausible plutôt que statistiquement extrême. Investigation détaillée : la médiane et le 3e quartile de cette colonne valent tous
deux 24 (cohérent avec une fenêtre d'extraction attendue d'environ 2 ans), mais le maximum observé atteignait 144 mois. Comme Q1 = Q3 = 24, l'IQR de cette colonne est nul :
la méthode statistique (catégorie 2) est donc inopérante ici et ne peut pas être utilisée
pour la plafonner automatiquement, d'où le recours à une règle métier dure. L'origine la
plus probable est un doublon de jointure lors de l'union des fichiers SOLDE_2025 et
SOLDE_COMPLEMENT (plusieurs sous-comptes ou lignes non filtrées comptés comme des
mois d'observation distincts pour un même RADICAL) plutôt qu'un historique réellement
long.
— Dates de naissance impossibles (âge < 16 ans ou > 100 ans) : sur le train, les lignes
concernées sont supprimées (erreurs de saisie présumées, effectif marginal). Sur le scoring, aucune ligne n'est jamais supprimée : un client à scorer reste dans la population
quelle que soit la qualité de ce champ isolé.
Chaque correction plafonnée conserve un flag binaire (<colonne>_etait_extreme) qui indique au modèle qu'une valeur a été modifiée, sans lui faire perdre l'information que quelque
chose d'inhabituel a été observé.

5.3 Catégorie 2 – Plafonnement statistique (winsorisation IQR)
Fonctions : apprendre_bornes_plafonnement (fit, train uniquement) / charger_bornes_plafonnement
+ appliquer_plafonnement (réapplication, train et scoring).
Pour chaque colonne retenue (COLS_A_PLAFONNER : solde_moyen, solde_min, solde_max,
depot_moyen, flux_cred_moyen, flux_cred_total, montant_total_gab,
montant_moyen_gab, montant_total_retraits, montant_total_payfac,
montant_total_vignette), les bornes de Tukey sont calculées sur le train :
borne_basse = Q1 − k × IQR   borne_haute = Q3 + k × IQR   (IQR = Q3 − Q1)
Toute valeur en dehors de [borne_basse, borne_haute] est plafonnée (winsorisée) à la borne
la plus proche – jamais supprimée, ni en train ni en scoring. Un flag <colonne>_etait_extreme
est ajouté pour chaque colonne plafonnée, dans la même logique que les flags catégorie 1.

5.4 Le piège des colonnes "zero-inflated"
Un problème spécifique a été identifié et corrigé sur les colonnes où une large majorité des
clients affichent une valeur de 0 (absence d'usage légitime, pas une erreur) : montant_total_gab,
montant_moyen_gab, montant_total_payfac, montant_total_vignette. Pour ces colonnes,
Q1 et Q3 valent souvent 0 eux-mêmes, ce qui effondre les bornes de Tukey à [0, 0] : toute valeur
non nulle – c'est-à-dire tout client réellement actif sur ce canal – se retrouvait alors plafonnée
à 0, détruisant le seul signal utile de la colonne (fréquentation GAB, adoption du paiement
digital, possession de véhicule via la vignette).

Correction appliquée : détection automatique des colonnes zero-inflated sur le train (fonction
detecter_colonnes_zero_inflated, seuil : ≥ 50% de valeurs à 0), puis calcul des quantiles
Q1/Q3 uniquement sur le sous-ensemble des valeurs strictement positives pour ces colonnes.
La borne basse est en outre systématiquement plafonnée à 0 (jamais positive), puisque les zéros
représentent une absence d'usage légitime et ne doivent jamais être traités comme hors bornes.

Sur le jeu de données du projet, 4 colonnes ont été détectées comme zero-inflated sur le
train : montant_total_gab (91% de zéros), montant_moyen_gab (91%), montant_total_payfac
(67%), montant_total_vignette (88%).

5.5 Non-fuite train/scoring
Même logique de non-fuite que l'Imputer déjà en place pour les valeurs manquantes :
— traiter_valeurs_aberrantes_train(df_train) : diagnostic, correction des valeurs
impossibles (avec suppression de lignes autorisée), apprentissage des bornes IQR sur le
train, sauvegarde en JSON (./models/outlier_bounds.json), puis application.
— traiter_valeurs_aberrantes_scoring(df_scorer) : correction des valeurs impossibles
(sans jamais supprimer de ligne), chargement des bornes apprises sur le train (jamais
recalculées), puis application à l'identique.
Les bornes utilisées sur la population à scorer sont donc strictement celles apprises sur le
train, garantissant qu'aucune information de la population de scoring ne contamine le calcul
des seuils.

5.6 Résultat du diagnostic (train)
Colonne Bornes apprises Val. plafonnées
solde_moyen [-47 414.3, 79 296.3] 576
solde_min [-4 270.2, 4 078.6] 1 142
solde_max [-170 523.9, 291 488.4] 518
depot_moyen [-100 695.9, 169 849.8] 602
flux_cred_moyen [-26 545.4, 45 751.0] 469
flux_cred_total [-419 138.8, 719 364.7] 507
montant_total_gab* [-2 013.2, 3 456.8] 27
montant_moyen_gab* [-94.9, 159.7] 61
montant_total_retraits [-94 200.0, 157 000.0] 198
montant_total_payfac* [-22 821.6, 43 440.4] 117
montant_total_vignette* [-2 450.0, 5 950.0] 31
nb_mois_observes_solde (règle métier, seuil 36) 668

* Colonnes zero-inflated : quantiles calculés sur les valeurs strictement positives uniquement.

Total : 4 248 valeurs plafonnées (catégorie 2) sur les 11 colonnes statistiques, plus 668
valeurs corrigées par la règle métier sur nb_mois_observes_solde (catégorie 1). Aucune ligne
supplémentaire supprimée à cette étape (4 141 → 4 141, hors les 4 141 lignes déjà filtrées lors du
contrôle des dates de naissance impossibles en amont).

À noter : avant correction du piège zero-inflated (cf. 5.4), le même diagnostic plafonnait
à tort 386 valeurs de montant_total_gab, 386 de montant_moyen_gab, 883 de
montant_total_payfac et 517 de montant_total_vignette vers des bornes dégénérées
[0, 0] ou quasi nulles – écrasant systématiquement le signal des clients réellement actifs sur
ces canaux. La correction a réduit ce nombre à respectivement 27, 61, 117 et 31, tout en ne
plafonnant plus que les valeurs réellement extrêmes.

6 Champs exploitables retenus et raison de leur choix
Une fois toutes les tables assemblées et nettoyées, voici les colonnes qui composent le jeu de
données final, et pourquoi chacune a été retenue.

Champ retenu Raison
Âge (dérivé de
DATE_OF_BIRTH), ville,
situation familiale, nombre
d'enfants, segment client
(CUSTOMER_RATING)
Profil socio-démographique – base de tout scoring client, et
NOMBRE_ENFANT est directement lié au produit Avenir Mes Enfants.
pack_actuel, pack_etat
(depuis OPK)
Le pack bancaire détenu reflète le niveau d'engagement du client
envers la banque, souvent corrélé à la propension à souscrire un
produit supplémentaire.
digital_toujours_abonne
(depuis
PRODUIT_DIGITAUX)
Un client digital actif est plus facilement atteignable par des
campagnes de ciblage in-app, et son comportement digital est souvent
corrélé à un profil plus jeune/actif.
solde_moyen/min/max,
depot_moyen (depuis
SOLDE/DEPOT)
Niveau de richesse et de stabilité financière – un solde élevé et stable
est un signal direct de capacité d'épargne.
flux_cred_moyen/total,
nb_mois_avec_flux (depuis
FLUX)
Régularité des revenus (proxy salaire/pension) – un revenu régulier
est un signal de capacité à épargner dans la durée.
nb_operations_gab,
montant_total/moyen_gab,
derniere_operation_gab
(depuis GAB)
Fréquence et récence d'utilisation du compte – logique RFM classique,
un client actif est plus susceptible d'être réceptif à une offre.
nb_retraits,
montant_total_retraits
(depuis RETRAIT)
Signal plus spécifique que l'agrégat GAB global : un usage important
en retraits cash peut indiquer un profil différent d'un usage orienté
épargne.
nb_paiements_digitaux,
montant_total_payfac
(depuis DIGI_PAYFAC)
Adoption des paiements digitaux – autre proxy du profil "client
actif/digital".
nb_vignettes_payees,
montant_total_vignette
(depuis VIGNETTE)
Proxy indirect de possession d'un véhicule, corrélé au profil
socio-économique (âge actif, famille).
*_etait_extreme (flags,
catégorie 1 et 2)
Indique au modèle qu'une valeur a été corrigée/plafonnée, préservant
le signal "profil inhabituel" sans laisser la valeur brute dominer
l'apprentissage.
label_code, label_nom
(depuis ASSI filtré)
La cible : le produit d'épargne réellement détenu, parmi les 3
confirmés.
Table 8 – Champs retenus dans le dataset final et justification.

7 Techniques de modélisation retenues
La cible a 3 valeurs possibles et est exclusive (un seul produit par client) : il s'agit d'une
classification multi-classes, pas de 3 modèles binaires indépendants.

7.1 Chaîne de transformation (Pipeline MLlib)
1. StringIndexer : convertit label_nom (texte) en indices numériques (0, 1, 2) – MLlib
n'entraîne que sur des labels numériques.
2. VectorAssembler : rassemble toutes les colonnes numériques retenues (section 6) en un
seul vecteur features, sans aucun calcul.
3. Un classifieur multi-classes (voir ci-dessous), entraîné sur ce vecteur.
4. IndexToString : reconvertit la prédiction numérique en nom de produit lisible pour la
restitution finale.
Ces étapes sont chaînées dans un Pipeline MLlib unique, entraîné, évalué et sauvegardé
comme un seul objet réutilisable.

7.2 Algorithmes comparés

Algorithme Remarque
RandomForestClassifier Choix par défaut : robuste, gère bien les variables mixtes
(montants, compteurs, catégorielles encodées), détecte le
nombre de classes automatiquement.
LogisticRegression (family="multinomial") Modèle linéaire de référence, plus interprétable, utile
pour comparer.
DecisionTreeClassifier Encore plus interprétable qu'une forêt, généralement
moins performant seul.
NaiveBayes Rapide, teste l'hypothèse (souvent fausse en pratique)
d'indépendance des features – gardé comme
comparaison basse.
Table 9 – Algorithmes de classification multi-classes comparés.

7.3 Gestion du déséquilibre des classes
La répartition observée dans ASSI filtré n'est pas équilibrée entre les 3 produits (MaRetraite
et Avenir Mes Enfants nettement plus fréquents qu'Epargne Evolution, d'après les effectifs bruts
vus en section 3). Une pondération inverse-fréquence par classe est appliquée via weightCol,
pour que le modèle ne se contente pas de toujours prédire le produit majoritaire.

7.4 Évaluation
MulticlassClassificationEvaluator (pas BinaryClassificationEvaluator, réservé à 2
classes), avec le F1 pondéré comme métrique principale (plus robuste au déséquilibre que l'accuracy brute), complété par une matrice de confusion (groupBy("label","prediction").count())
pour voir précisément quelles confusions le modèle fait entre les 3 produits.

8 Synthèse et état d'avancement
✓ Valeurs distinctes de CODE_PRODUIT/LIBELLE_PRODUIT listées : 3 produits d'épargne confirmés (53, 09, 18), 3 produits hors périmètre identifiés et exclus. À faire valider par l'encadrant avant la soutenance.
✓ Unicité de RADICAL vérifiée sur PERIMETRE (non unique) et corrigée par dédoublonnage.
✓ Agrégation par client réalisée pour toutes les tables 1 :N (familles Soldes/Flux, Transactions, Vignette).
✓ Jeu de données final assemblé et scindé en clients_avec_label / clients_a_scorer,
écrit dans processed-data/.
✓ Valeurs manquantes traitées (Imputer MLlib, fit train / réapplication scoring, sans fuite).
✓ Valeurs aberrantes traitées en deux catégories (règles métier + winsorisation IQR), avec
détection et correction spécifique des colonnes zero-inflated (GAB, payfac, vignette) et
d'un cas de doublon probable de jointure (nb_mois_observes_solde). Détail en section 5.
□ Encodage de la cible, entraînement et comparaison des 4 algorithmes de la section 7.
□ Sélection du modèle final, sauvegarde du Pipeline, scoring de clients_a_scorer.

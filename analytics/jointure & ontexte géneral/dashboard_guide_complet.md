# Guide complet de reconstruction — Dashboard éligibilité + produits

Ce document reprend tout depuis le début, avec les corrections déjà identifiées (colonnes calculées pour les tranches au lieu du regroupement automatique — indisponible en DirectQuery — et filtres Top N pour les listes longues).

**Rappel avant de commencer** : appliquez le thème (`Affichage → Thèmes → Parcourir pour les thèmes` → `theme_eligibilite.json`) avant de construire quoi que ce soit — tout héritera automatiquement des couleurs.

---

## Étape 0 — Toutes les mesures à créer d'abord

Clic droit sur `dataset_eligibilite` → **Nouvelle mesure** → coller → Entrée. Répétez pour chacune :

```dax
Nb Clients Total = COUNTROWS ( dataset_eligibilite )
```
```dax
Âge Moyen = AVERAGE ( dataset_eligibilite[age_client] )
```
```dax
Solde Moyen Global = AVERAGE ( dataset_eligibilite[solde_moyen] )
```
```dax
Ancienneté Digitale Moyenne = AVERAGE ( dataset_eligibilite[anciennete_digitale_jours_imp] )
```
```dax
Nb Villes Distinctes = DISTINCTCOUNT ( dataset_eligibilite[CODE_VILLE] )
```
```dax
Nb Clients Éligibles =
CALCULATE ( [Nb Clients Total], dataset_eligibilite[label_eligibilite] = 1 )
```
```dax
Taux d'Éligibilité Global =
DIVIDE ( [Nb Clients Éligibles], [Nb Clients Total] )
```
```dax
Nb Opérations GAB Moyen = AVERAGE ( dataset_eligibilite[nb_operations_gab] )
```
```dax
Nb Produits Distincts = DISTINCTCOUNT ( dataset_eligibilite[label_nom] )
```

*(si `label_eligibilite = 1` renvoie une erreur ou zéro : remplacez `1` par `"1"` entre guillemets — dépend si Power BI a détecté la colonne comme numérique ou texte)*

## Étape 0bis — Colonnes calculées pour les distributions

Ces trois-là sont des **colonnes**, pas des mesures : clic droit sur la table → **Nouvelle colonne** cette fois.

```dax
Tranche Age =
SWITCH (
    TRUE (),
    dataset_eligibilite[age_client] < 26, "18-25",
    dataset_eligibilite[age_client] < 36, "26-35",
    dataset_eligibilite[age_client] < 46, "36-45",
    dataset_eligibilite[age_client] < 56, "46-55",
    dataset_eligibilite[age_client] < 66, "56-65",
    "66+"
)
```
```dax
Tranche Ancienneté Digitale =
SWITCH (
    TRUE (),
    dataset_eligibilite[anciennete_digitale_jours_imp] < 365, "< 1 an",
    dataset_eligibilite[anciennete_digitale_jours_imp] < 730, "1-2 ans",
    dataset_eligibilite[anciennete_digitale_jours_imp] < 1825, "2-5 ans",
    dataset_eligibilite[anciennete_digitale_jours_imp] < 3650, "5-10 ans",
    "10+ ans"
)
```

Pour `Tranche Solde`, trouvez d'abord vos vraies bornes : ajoutez temporairement deux cartes avec les mesures rapides `MIN(dataset_eligibilite[solde_moyen])` et `MAX(dataset_eligibilite[solde_moyen])`, notez les valeurs, ajustez les seuils ci-dessous en conséquence, puis supprimez ces deux cartes de reconnaissance.

```dax
Tranche Solde =
SWITCH (
    TRUE (),
    dataset_eligibilite[solde_moyen] < 1000, "< 1K",
    dataset_eligibilite[solde_moyen] < 5000, "1K-5K",
    dataset_eligibilite[solde_moyen] < 10000, "5K-10K",
    dataset_eligibilite[solde_moyen] < 25000, "10K-25K",
    dataset_eligibilite[solde_moyen] < 50000, "25K-50K",
    "50K+"
)
```

---

## Technique Top N (à réutiliser partout où indiqué)

1. Sélectionnez le visuel.
2. Volet **Filtres** → le champ concerné (ex. `CODE_VILLE`) → **Type de filtre** → **Top N**.
3. **Afficher les éléments** : `Haut` / `10`.
4. Glissez la mesure pertinente (ex. `Nb Clients Total`) dans **Par valeur**.
5. **Appliquer le filtre**.

---

## Page : Accueil

- Navigateur de pages (Insertion → Boutons → Navigateur → Navigateur de pages) — rail vertical à gauche.
- 6 cartes : `Nb Clients Total`, `Nb Clients Éligibles`, `Taux d'Éligibilité Global`, `Solde Moyen Global`, `Âge Moyen`, `Nb Villes Distinctes`.
- **Rien d'autre** — pas de graphiques ici, c'est le résumé exécutif.

---

## Page : Démographie

| Visuel | Axe | Légende | Valeurs |
|---|---|---|---|
| Donut | — | `GENDER` | `Nb Clients Total` |
| Colonnes groupées | `Tranche Age` | — | `Nb Clients Total` |
| Barres horizontales | `MARITAL_STATUS` | — | `Nb Clients Total` |
| Barres horizontales | `CODE_VILLE` **+ Top N (10)** | — | `Nb Clients Total` |

---

## Page : Comportement financier

| Visuel | Axe | Légende | Valeurs |
|---|---|---|---|
| Colonnes (remplace "histogramme") | `Tranche Solde` | — | `Nb Clients Total` |
| Colonnes | `CODE_VILLE` **+ Top N (10)** | — | `Solde Moyen Global` |
| Barres | `TAILLE_ENTREPRI` *(substitut à "profession", champ inexistant)* | — | `Solde Moyen Global` |
| Box Plot *(optionnel — importer via Insertion → Obtenir plus de visuels → "Box and Whisker Chart")* | — | — | `solde_moyen` **(colonne brute, pas la mesure)** |

---

## Page : Engagement digital

| Visuel | Axe | Légende | Valeurs |
|---|---|---|---|
| Donut | — | `jamais_active_digital` | `Nb Clients Total` |
| Barres | `TAILLE_ENTREPRI` | — | `Nb Opérations GAB Moyen` |
| Colonnes (remplace "histogramme") | `Tranche Ancienneté Digitale` | — | `Nb Clients Total` |
| Colonnes empilées | `GENDER` | `jamais_active_digital` | `Nb Clients Total` |

---

## Page : Éligibilité

| Visuel | Axe | Légende | Valeurs |
|---|---|---|---|
| Jauge | — | — | `Taux d'Éligibilité Global` |
| Barres empilées 100% | `label_eligibilite` | `GENDER` | `Nb Clients Total` |
| Colonnes | `CODE_VILLE` **+ Top N (10)** | `label_eligibilite` | `Nb Clients Total` |
| Donut | — | `label_eligibilite` | `Nb Clients Total` |

---

## Page : Produit *(nouvelle)*

**Important à comprendre avant de construire cette page** : `pack_actuel`, `pack_etat`, `label_nom`, `label_code` sont des colonnes déjà présentes dans `dataset_eligibilite` — elles décrivent les produits que les clients **détiennent actuellement**. Ce n'est **pas** une prédiction — le modèle produit (multi-classes, qui recommanderait le *prochain* produit à proposer) n'est pas encore construit, séparément de tout ça. Cette page est donc une vue **descriptive** de l'existant, ce qui est déjà utile et honnête, à ne pas confondre avec un futur dashboard de recommandations.

| Visuel | Axe | Légende | Valeurs |
|---|---|---|---|
| Carte | — | — | `Nb Produits Distincts` |
| Donut — Répartition des produits actuels | — | `pack_actuel` | `Nb Clients Total` |
| Barres — Produits par nom **+ Top N (10)** | `label_nom` | — | `Nb Clients Total` |
| Barres — État des packs | `pack_etat` | — | `Nb Clients Total` |
| Colonnes empilées — Produits par ville | `CODE_VILLE` **+ Top N (10)** | `pack_actuel` | `Nb Clients Total` |

N'oubliez pas d'ajouter cette page au navigateur de pages sur Accueil s'il ne l'a pas détectée automatiquement (parfois nécessaire de rafraîchir le visuel navigateur après l'ajout d'une nouvelle page).

---

## Checklist finale

- [ ] Thème appliqué
- [ ] Les 9 mesures créées (Étape 0)
- [ ] Les 3 colonnes calculées créées (Étape 0bis), avec les vraies bornes pour `Tranche Solde`
- [ ] Accueil : 6 cartes + navigateur, rien d'autre
- [ ] Démographie : 4 visuels
- [ ] Comportement financier : 3-4 visuels (box plot optionnel)
- [ ] Engagement digital : 4 visuels
- [ ] Éligibilité : 4 visuels
- [ ] Produit : 5 visuels
- [ ] Enregistré (`Ctrl+S`) — **faites-le après chaque page terminée cette fois**, pas seulement à la fin

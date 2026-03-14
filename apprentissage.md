# Questions d'entretien — Projet IA Energie UEMOA (BCEAO)

---

## 1. Pourquoi toute l'UEMOA ensemble et pas un modèle par pays ?

**Réponse courte** : Avec seulement **34 observations par pays** (1990-2023), un modèle individuel n'a pas assez de données pour apprendre correctement. En regroupant les 8 pays, on passe à **272 observations** — 8x plus de données pour capturer les relations entre PIB, population, accès à l'électricité et consommation.

**Arguments techniques** :
- **Transfert d'apprentissage** : la relation "plus de PIB → plus de consommation" est universelle dans la zone UEMOA. Ce que le modèle apprend sur la Côte d'Ivoire (gros consommateur) aide à prédire le Togo (petit consommateur) et inversement
- **Robustesse statistique** : 34 points avec 80 features = sous-déterminé (plus de variables que d'observations). Le modèle ne peut pas converger correctement
- **Zone monétaire commune** : les pays UEMOA partagent le FCFA, la même banque centrale (BCEAO), des politiques économiques similaires. Les dynamiques énergétiques sont structurellement comparables
- **Validation croisée temporelle** : avec 272 obs, on peut faire 5 folds de 32 obs chacun. Avec 34 obs par pays, ce serait impossible

---

## 2. L'écrasement des petits pays — problème et solution

**Le problème** :
La Côte d'Ivoire consomme **10 088 GWh**, la Guinée-Bissau **213 GWh** — un rapport de **47x**. Quand le modèle minimise l'erreur quadratique (RMSE), se tromper de 500 GWh sur la Côte d'Ivoire (5% d'erreur) est "acceptable", mais 500 GWh sur la Guinée-Bissau c'est **2.3x sa consommation réelle**. Résultat : le modèle optimise pour les gros pays et ignore les petits. C'est ce qu'on voyait avec la Guinée-Bissau à R²=-17.4 et des prédictions **négatives**.

**La solution — 2 techniques combinées** :

**a) Transformation logarithmique de la cible** (`log1p`)
- Au lieu de prédire 213 GWh vs 10 088 GWh (écart 47x), le modèle prédit log(214)=5.37 vs log(10 089)=9.22 (écart 1.7x)
- L'erreur est maintenant **proportionnelle** : se tromper de 0.1 en log-espace, c'est ~10% d'erreur que ce soit pour un petit ou un grand pays
- C'est comme prédire un **taux de croissance** plutôt qu'une valeur absolue

**b) Encodage one-hot des pays** (8 variables binaires)
- Le modèle reçoit `country_BEN=1, country_CIV=0, ...` comme features
- Il peut ainsi apprendre un **niveau de base différent** pour chaque pays
- La Guinée-Bissau a son propre "intercept", la Côte d'Ivoire le sien
- C'est conceptuellement proche d'un **effet fixe pays** en économétrie panel

**Résultat** : Guinée-Bissau R²=-17.4 → **R²=0.72**, MAPE 105% → **8.7%**

---

## 3. Comment j'ai géré le surapprentissage

**Le diagnostic** : Le premier modèle (XGBoost, 300 arbres, depth=6) avait une erreur <2 GWh sur le train (1990-2016) mais prédisait **2-3x la réalité** sur le test (2017-2023) pour le Bénin. Signe classique d'overfitting.

**Les solutions appliquées** :

| Technique | Avant | Après | Pourquoi |
|---|---|---|---|
| **Split temporel** | Train 1990-2016, Test 2017-2023 | idem | Pas de fuite du futur vers le passé |
| **CV par années** | 5 folds temporels | idem | Chaque fold respecte la chronologie |
| **Log-transform** | cible brute (GWh) | `log1p(GWh)` | Réduit la variance, stabilise les résidus |
| **Modèle linéaire** | XGBoost (non-linéaire, mémorise) | **Ridge** (linéaire, régularisé) | Biais inductif fort → généralise mieux |
| **Régularisation L2** | aucune | `alpha=10.0` | Pénalise les coefficients trop grands |
| **Réduction profondeur** | max_depth=6-8 | max_depth=4 | Moins de complexité des arbres |
| **Augmentation min_samples** | min_samples_leaf=3 | min_samples_leaf=5-8 | Empêche les feuilles trop spécifiques |

**Pourquoi Ridge a gagné** : Avec 272 observations et 80 features (dont 8 dummies pays), un modèle linéaire régularisé est le bon choix. Les relations log(PIB) → log(GWh) sont quasi-linéaires en Afrique de l'Ouest (phase d'industrialisation). Les arbres de décision, eux, créent des frontières artificielles qui ne généralisent pas au-delà de la période d'entraînement.

**Preuve** : CV R² = 0.86 ± 0.14 sur 5 folds, avec un fold 5 (le plus récent) à **R²=0.99** — le modèle s'améliore avec plus de données, signe d'un bon biais-variance tradeoff.

---

## Phrase de synthèse pour l'entretien

> *"J'ai regroupé les 8 pays pour avoir assez de données (272 vs 34 obs), puis géré l'écrasement des petits pays par une transformation log et un encodage pays. Face au surapprentissage des modèles complexes, j'ai adopté une régression Ridge régularisée qui généralise bien grâce à son biais inductif linéaire — adapté à la phase de développement énergétique de la zone UEMOA."*

---

## Résultats finaux — R² par pays

| Pays | R² avant | R² après | MAPE après |
|---|---|---|---|
| Bénin | -0.81 | **0.98** | 5.4% |
| Burkina Faso | 0.90 | **0.98** | 9.6% |
| Côte d'Ivoire | 0.87 | **0.94** | 7.0% |
| Guinée-Bissau | -17.40 | **0.72** | 8.7% |
| Mali | 0.54 | **0.83** | 9.9% |
| Niger | 0.88 | **0.99** | 5.6% |
| Sénégal | 0.96 | **0.99** | 3.9% |
| Togo | 0.87 | **0.99** | 4.0% |
| **Global** | **0.94** | **0.98** | **7.7%** |

# ⚡ Prévision de la Demande Électrique — Zone UEMOA (Horizon 2045)

> **La population croît — combien d'électricité faudra-t-il demain ?**

Projet personnel | Intelligence Artificielle & Analyse Prédictive
Déc. 2023 – Févr. 2024

---

## Résumé

Conception d'un **pipeline end-to-end d'intelligence artificielle** pour anticiper la demande électrique au Togo à l'horizon 2045 :

- **Extraction automatisée** de 21 indicateurs macroéconomiques via l'API Banque Mondiale (8 pays UEMOA, 1990–2023)
- **Ingénierie de 74 features** (lags temporels, moyennes mobiles, ratios démographiques, interactions)
- **Entraînement d'un Stacking Regressor** (Random Forest + Gradient Boosting + LightGBM / méta-modèle Ridge) atteignant un **R² de 0.89**
- **Dashboard interactif** (Streamlit, Plotly) avec visualisations analytiques et interprétations IA des trajectoires énergétiques — projections avec intervalles de confiance à 95%

**Perspective BCEAO** : Maîtrise de la modélisation prédictive sur séries temporelles macroéconomiques et du transfer learning régional (entraînement multi-pays), directement transposable à la projection d'agrégats monétaires, financiers et économiques de la zone UEMOA.

---

## Résultats Clés

| Métrique | Valeur |
|---|---|
| **Meilleur modèle** | Stacking Regressor |
| **R²** | 0.887 |
| **Variable cible** | Consommation totale (GWh) |
| **Données d'entraînement** | 272 observations (8 pays × 34 ans) |
| **Features** | 74 variables construites à partir de 21 indicateurs bruts |
| **Projection Togo 2045** | ~6 600 GWh (×5 vs 2023) |
| **Population projetée 2045** | ~13.9 millions d'habitants |

### Comparaison des 4 modèles

| Modèle | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| **Stacking** | **0.887** | — | — | — |
| Gradient Boosting | 0.787 | — | — | — |
| Random Forest | 0.696 | — | — | — |
| LightGBM | 0.657 | — | — | — |

---

## Sources de Données (Open Data)

| Source | Volume | Couverture |
|---|---|---|
| **Banque Mondiale (WDI)** | 5 175 observations brutes | 21 indicateurs × 8 pays × 34 ans |

**Pays UEMOA couverts** : 🇹🇬 Togo (focus), 🇸🇳 Sénégal, 🇧🇯 Bénin, 🇨🇮 Côte d'Ivoire, 🇧🇫 Burkina Faso, 🇲🇱 Mali, 🇳🇪 Niger, 🇬🇼 Guinée-Bissau

**Période** : 1990 – 2023

### 21 Indicateurs par domaine

| Domaine | Indicateurs |
|---|---|
| **Démographie** (7) | Population totale, croissance, urbanisation, fécondité, espérance de vie, pop 0-14 ans, pop 15-64 ans |
| **Énergie** (6) | kWh/hab, accès total/urbain/rural, renouvelable (%), énergie kg pétrole/hab |
| **Économie** (5) | PIB/hab, PIB total, croissance PIB, industrie (% PIB), inflation |
| **Social** (3) | Abonnements mobile, alphabétisation, chômage |

---

## Architecture du Projet

```
energy-prediction-togo/
├── data/
│   ├── raw/                          # 5 175 observations brutes (API BM)
│   ├── processed/                    # 272 × 74 features transformées
│   └── predictions/                  # Prédictions + projections 2024-2045
├── src/
│   ├── etl/
│   │   ├── extract.py                # Extraction API WDI (21 indicateurs)
│   │   ├── transform.py              # Feature engineering (74 variables)
│   │   └── load.py                   # Split temporel + préparation
│   ├── models/
│   │   ├── train.py                  # 4 modèles (RF, GB, LightGBM, Stacking)
│   │   └── predict.py                # Projections 2024-2045 avec IC 95%
│   └── utils/
│       └── config.py                 # Configuration centralisée
├── dashboard/
│   └── app.py                        # Dashboard Streamlit (4 onglets)
├── models/
│   ├── model_energy.joblib           # Modèle Stacking sauvegardé
│   └── results.csv                   # Métriques comparatives
├── requirements.txt
└── README.md
```

---

## Technologies

| Catégorie | Technologies |
|---|---|
| **ETL & Data** | Python, Pandas, NumPy, API REST Banque Mondiale (WDI) |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting, Stacking), LightGBM |
| **Feature Engineering** | Lags (t-1, t-2), moyennes mobiles (MA3, MA5), ratios démographiques, interactions croisées |
| **Visualisation** | Plotly (graphiques interactifs, thème plotly_dark) |
| **Dashboard** | Streamlit (4 onglets, KPIs, interprétations IA) |
| **Versioning** | Git, GitHub |

---

## Pipeline

### 1. Extract — Collecte automatisée

Extraction de **21 indicateurs** via l'API Banque Mondiale pour **8 pays UEMOA** (1990–2023).

→ **5 175 observations** brutes extraites

### 2. Transform — Feature Engineering

Pipeline de transformation en plusieurs étapes :
1. **Pivot** des indicateurs en colonnes (format tabulaire)
2. **Imputation** des valeurs manquantes (interpolation + forward/backward fill + médiane)
3. **Variable cible** : `conso_totale_gwh = Population × kWh/hab / 10⁶`
4. **Features temporelles** : variation annuelle (%), lags (t-1, t-2), moyennes mobiles (MA3, MA5)
5. **Features d'ingénierie** : population active, ratio de dépendance, PIB/hab calculé, interaction renouvelable × accès, total mobile
6. **Validation** et nettoyage final

→ **272 lignes × 74 colonnes** après transformation

### 3. Train — Modélisation

**Cible unique** : `conso_totale_gwh` (demande totale en GWh)

**Stratégie** : Entraînement sur les **8 pays UEMOA** simultanément (transfer learning régional) — 272 observations au lieu de 34 pour le seul Togo.

**4 algorithmes comparés** :
- Random Forest
- Gradient Boosting
- LightGBM
- **Stacking Ensemble** (RF + GB + LightGBM → méta-modèle Ridge) ✅ retenu

**Validation** : Split temporel (80/20) — pas de fuite de données temporelles.

### 4. Predict — Projections 2045

- Prédictions historiques sur l'ensemble du jeu de données
- **Projections 2024–2045** (22 ans) avec intervalles de confiance à 95% (±2σ croissant)
- Projection démographique intégrée pour estimer la consommation par habitant future

### 5. Dashboard — Visualisations & Interprétations IA

| Onglet | Contenu |
|---|---|
| **1. Données collectées** | Répartition par domaine, couverture UEMOA, explorateur interactif d'indicateurs |
| **2. Exploration** | Co-évolution population/demande, fracture urbain/rural, corrélations structurelles, contexte socio-économique |
| **3. Modèle IA** | Comparaison des algorithmes (R², MAPE), radar multi-critères, validation observé vs prédit, dispersion |
| **4. Prédictions 2045** | Trajectoire historique + projections avec IC 95%, démographie future, jauge par année sélectionnable |

Chaque graphique est accompagné d'une **interprétation analytique** : élasticité, corrélation de Pearson, analyse de la fracture énergétique, implications pour la planification.

---

## Installation

```bash
# Cloner le projet
git clone https://github.com/Theobaw01/energy-prediction-togo.git
cd energy-prediction-togo

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Extraction des données (API Banque Mondiale)
python src/etl/extract.py

# 2. Transformation & feature engineering
python src/etl/transform.py

# 3. Entraînement des modèles (4 algorithmes)
python src/models/train.py

# 4. Génération des prédictions & projections 2024-2045
python src/models/predict.py

# 5. Lancement du dashboard interactif
python -m streamlit run dashboard/app.py
```

Le dashboard sera accessible sur **http://localhost:8501**.

---

## Pertinence BCEAO / UEMOA

Ce projet démontre des compétences directement transposables aux missions de la **Banque Centrale des États de l'Afrique de l'Ouest** :

| Compétence | Application dans le projet | Transposition BCEAO |
|---|---|---|
| **Pipeline de données** | ETL automatisé (API → transformation → modèle → dashboard) | Intégration de flux de données macroéconomiques |
| **ML sur séries temporelles** | Stacking Regressor sur 33 ans de données | Projection d'agrégats monétaires et financiers |
| **Transfer learning régional** | Entraînement sur 8 pays pour prédire sur 1 | Modélisation multi-pays de la zone franc |
| **Feature engineering** | 74 variables construites à partir de 21 brutes | Création d'indicateurs dérivés pour l'analyse économique |
| **Communication des résultats** | Dashboard avec interprétations IA | Reporting analytique pour les organes décisionnels |

---

## Auteur

**Théodore Bawana** — Développeur en Intelligence Artificielle
[GitHub](https://github.com/Theobaw01)

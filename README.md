# ⚡ Prévision de la Demande Électrique — Zone UEMOA (Horizon 2045)

> **La population croît — combien d'électricité faudra-t-il demain ?**

> *Ce projet a été réalisé dans l'objectif de maîtriser les concepts liés à l'ingénierie de données et au Machine Learning, transposables dans des situations réelles de modélisation macroéconomique.*

Projet personnel | Intelligence Artificielle & Analyse Prédictive

---

## Résumé

Conception d'un **pipeline end-to-end d'intelligence artificielle** pour anticiper la demande électrique des **8 pays de l'UEMOA** à l'horizon 2045 :

- **Extraction automatisée** de 21 indicateurs macroéconomiques via l'API Banque Mondiale (8 pays UEMOA, 1990–2023)
- **Ingénierie de 80+ features** (lags temporels, moyennes mobiles, transformations log, ratios démographiques, interactions croisées)
- **Entraînement d'un Stacking Regressor** (Random Forest + Gradient Boosting + XGBoost + LightGBM / méta-modèle Ridge) avec **cross-validation temporelle (5 folds)**
- **Dashboard interactif multi-pays** (Streamlit, Plotly) : sélection de n'importe quel pays UEMOA, benchmark comparatif, projections avec intervalles de confiance à 95%

**Perspective BCEAO** : Maîtrise de la modélisation prédictive sur séries temporelles macroéconomiques et du transfer learning régional (entraînement multi-pays), directement transposable à la projection d'agrégats monétaires, financiers et économiques de la zone UEMOA.

---

## Résultats Clés

| Métrique | Valeur |
|---|---|
| **Meilleur modèle** | Stacking Regressor |
| **Variable cible** | Consommation totale (GWh) |
| **Données d'entraînement** | 272 observations (8 pays × 34 ans) |
| **Features** | 80+ variables construites à partir de 21 indicateurs bruts |
| **Validation** | Split temporel 80/20 + cross-validation temporelle 5 folds |
| **Projections** | 2024–2045 pour les 8 pays UEMOA |

### Comparaison des 5 modèles

| Modèle | Description |
|---|---|
| **Stacking Ensemble** ✅ | RF + GB + XGBoost + LightGBM → méta-modèle Ridge |
| Gradient Boosting | 200 arbres, profondeur 5 |
| Random Forest | 300 arbres, profondeur 8 |
| XGBoost | 300 arbres, profondeur 6 |
| LightGBM | 300 arbres, profondeur 6 |

> Les métriques exactes (R², RMSE, MAE, MAPE) sont recalculées à chaque exécution du pipeline et affichées dans le dashboard.

---

## Sources de Données (Open Data)

| Source | Volume | Couverture |
|---|---|---|
| **Banque Mondiale (WDI)** | ~5 175 observations brutes | 21 indicateurs × 8 pays × 34 ans |

**Pays UEMOA couverts** : 🇹🇬 Togo, 🇸🇳 Sénégal, 🇧🇯 Bénin, 🇨🇮 Côte d'Ivoire, 🇧🇫 Burkina Faso, 🇲🇱 Mali, 🇳🇪 Niger, 🇬🇼 Guinée-Bissau

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
│   ├── raw/                          # ~5 175 observations brutes (API BM)
│   ├── processed/                    # 272 × 80+ features transformées
│   └── predictions/                  # Prédictions + projections 2024-2045
├── src/
│   ├── etl/
│   │   ├── extract.py                # Extraction API WDI (21 indicateurs, 8 pays)
│   │   ├── transform.py              # Feature engineering (80+ variables)
│   │   └── load.py                   # Split temporel + préparation
│   ├── models/
│   │   ├── train.py                  # 5 modèles + CV temporelle + feature importance
│   │   └── predict.py                # Projections 2024-2045 × 8 pays avec IC 95%
│   └── utils/
│       └── config.py                 # Configuration centralisée (21 indicateurs, 8 pays)
├── dashboard/
│   └── app.py                        # Dashboard Streamlit multi-pays (4 onglets)
├── models/
│   ├── model_energy.joblib           # Modèle Stacking sauvegardé
│   ├── results.csv                   # Métriques comparatives des 5 modèles
│   ├── cv_scores.csv                 # Scores de cross-validation par fold
│   └── feature_importance.csv        # Top 15 features les plus influentes
├── requirements.txt
└── README.md
```

---

## Technologies

| Catégorie | Technologies |
|---|---|
| **ETL & Data** | Python, Pandas, NumPy, API REST Banque Mondiale (WDI) |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting, Stacking), XGBoost, LightGBM |
| **Validation** | TimeSeriesSplit (5 folds), split temporel 80/20, feature importance |
| **Feature Engineering** | Lags (t-1, t-2), MA (3, 5), log-transforms, ratios, interactions croisées |
| **Visualisation** | Plotly (graphiques interactifs, thème plotly_dark) |
| **Dashboard** | Streamlit (4 onglets, sélecteur multi-pays, benchmark UEMOA) |
| **Versioning** | Git, GitHub |

---

## Pipeline

### 1. Extract — Collecte automatisée

Extraction de **21 indicateurs** via l'API Banque Mondiale pour **8 pays UEMOA** (1990–2023).

→ **~5 175 observations** brutes extraites automatiquement

### 2. Transform — Feature Engineering

Pipeline de transformation en plusieurs étapes :
1. **Pivot** des indicateurs en colonnes (format tabulaire)
2. **Imputation** des valeurs manquantes (interpolation + forward/backward fill + médiane)
3. **Variable cible** : `conso_totale_gwh = Population × kWh/hab / 10⁶`
4. **Features temporelles** : variation annuelle (%), lags (t-1, t-2), moyennes mobiles (MA3, MA5)
5. **Features classiques** : population active, ratio de dépendance, PIB/hab calculé, interactions
6. **Features avancées** : log(population), log(PIB), PIB/actif, GWh/PIB, population électrifiée, industrie absolue, interaction croissance PIB × population
7. **Validation** et nettoyage final

→ **272 lignes × 80+ colonnes** après transformation

### 3. Train — Modélisation

**Cible unique** : `conso_totale_gwh` (demande totale en GWh)

**Stratégie** : Entraînement sur les **8 pays UEMOA** simultanément (transfer learning régional) — 272 observations au lieu de 34 pour un seul pays.

**5 algorithmes comparés** :
- Random Forest (300 arbres, profondeur 8)
- Gradient Boosting (200 arbres, profondeur 5, lr 0.05)
- XGBoost (300 arbres, profondeur 6)
- LightGBM (300 arbres, profondeur 6)
- **Stacking Ensemble** (RF + GB + XGB + LGBM → méta-modèle Ridge) ✅ retenu

**Validation** :
- Split temporel (80/20) — pas de fuite de données temporelles
- Cross-validation temporelle (TimeSeriesSplit, 5 folds)
- Export automatique des scores CV et de l'importance des features

### 4. Predict — Projections 2045

- Prédictions historiques sur l'ensemble du jeu de données (8 pays)
- **Projections 2024–2045** (22 ans) pour **chaque pays UEMOA** avec intervalles de confiance à 95% (±2σ croissant)
- Projection démographique intégrée pour estimer la consommation par habitant future
- Résumé console avec MAE/MAPE par pays et tableau de projection finale

### 5. Dashboard — Visualisations Multi-pays & Interprétations IA

**Sélecteur de pays** dans la barre latérale : tout pays UEMOA est consultable individuellement.

| Onglet | Contenu |
|---|---|
| **1. Données & Extraction** | Répartition par domaine, couverture par pays UEMOA, explorateur d'indicateurs avec comparaison multi-pays |
| **2. Exploration** | Co-évolution population/demande, fracture urbain/rural, benchmark GWh UEMOA, matrice de corrélation |
| **3. Modèle IA** | Comparaison des 5 algorithmes (R², MAPE), cross-validation temporelle, radar multi-critères, feature importance, validation observé vs prédit |
| **4. Prédictions 2045** | Trajectoire historique + projections avec IC 95%, comparaison des 8 pays, jauge interactive par année, classement projeté UEMOA |

Chaque graphique est accompagné d'une **interprétation analytique** : élasticité, corrélation de Pearson, analyse de la fracture énergétique, classement régional, implications pour la planification.

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

# 3. Entraînement des modèles (5 algorithmes + cross-validation)
python src/models/train.py

# 4. Génération des prédictions & projections 2024-2045 (8 pays)
python src/models/predict.py

# 5. Lancement du dashboard interactif multi-pays
python -m streamlit run dashboard/app.py
```

Le dashboard sera accessible sur **http://localhost:8501**.

---

## Pertinence BCEAO / UEMOA

> Ce projet démontre des compétences directement transposables aux missions de la **Banque Centrale des États de l'Afrique de l'Ouest**.

| Compétence | Application dans le projet | Transposition BCEAO |
|---|---|---|
| **Pipeline de données** | ETL automatisé (API → transformation → modèle → dashboard) | Intégration de flux de données macroéconomiques |
| **ML sur séries temporelles** | Stacking Regressor sur 34 ans de données avec CV temporelle | Projection d'agrégats monétaires et financiers |
| **Transfer learning régional** | Entraînement sur 8 pays pour prédire sur chacun | Modélisation multi-pays de la zone franc |
| **Feature engineering avancé** | 80+ variables (log, interactions, lags, MA) à partir de 21 brutes | Création d'indicateurs dérivés pour l'analyse économique |
| **Dashboard multi-pays** | Sélecteur de pays, benchmark comparatif, interprétations IA | Reporting analytique régional pour les organes décisionnels |
| **Validation rigoureuse** | TimeSeriesSplit 5 folds, feature importance, IC 95% | Fiabilité des modèles de prévision institutionnels |

---

## Auteur

**Théodore Bawana** — Développeur en Intelligence Artificielle
[GitHub](https://github.com/Theobaw01)

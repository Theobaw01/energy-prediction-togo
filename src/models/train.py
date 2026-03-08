"""
Entraînement des modèles prédictifs : Random Forest, XGBoost, LightGBM.
Validation croisée temporelle et sauvegarde du meilleur modèle.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import MODELS_DIR, RANDOM_STATE
from etl.load import load_processed_data, temporal_train_test_split

# Import optionnel pour XGBoost et LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Calcule les métriques d'évaluation."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n   {model_name}:")
    print(f"     RMSE : {rmse:.2f}")
    print(f"     MAE  : {mae:.2f}")
    print(f"     R²   : {r2:.4f}")

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2}


def get_feature_importance(model, feature_names: list, top_n: int = 10):
    """Affiche les features les plus importantes."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        print(f"\n   Top {top_n} features :")
        for i, idx in enumerate(indices):
            print(f"     {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def train():
    """
    Pipeline d'entraînement complet.
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES PRÉDICTIFS")
    print("=" * 60)

    # 1. Charger les données
    print("\n1. Chargement des données...")
    df = load_processed_data()
    X_train, X_test, y_train, y_test, feature_names, train_df, test_df = \
        temporal_train_test_split(df)

    # Vérifier qu'on a des données valides
    valid_train = ~np.isnan(y_train) & ~np.isinf(y_train)
    valid_test = ~np.isnan(y_test) & ~np.isinf(y_test)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    # Remplacer NaN dans X
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ Pas assez de données valides pour l'entraînement.")
        return

    # 2. Définir les modèles
    print("\n2. Entraînement des modèles...")
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    # 3. Entraîner et évaluer
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n   Entraînement de {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        trained_models[name] = model

        get_feature_importance(model, feature_names, top_n=8)

    # 4. Sélectionner le meilleur modèle
    print("\n" + "=" * 60)
    print("RÉSULTATS COMPARATIFS")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]['model']
    best_model = trained_models[best_name]
    print(f"\n🏆 Meilleur modèle : {best_name} (R² = {results_df.iloc[0]['r2']:.4f})")

    # 5. Sauvegarder le meilleur modèle
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump({
        'model': best_model,
        'model_name': best_name,
        'feature_names': feature_names,
        'metrics': results_df.iloc[0].to_dict(),
    }, model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}")

    # Sauvegarder aussi les résultats
    results_path = os.path.join(MODELS_DIR, 'results.csv')
    results_df.to_csv(results_path, index=False)

    return best_model, results_df


if __name__ == '__main__':
    train()

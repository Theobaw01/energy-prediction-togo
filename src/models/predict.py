"""
Prédictions avec le modèle entraîné.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import MODELS_DIR, PREDICTIONS_DIR, COUNTRIES
from etl.load import load_processed_data, prepare_features


def load_model():
    """Charge le meilleur modèle sauvegardé."""
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modèle non trouvé. Exécutez d'abord train.py")
    return joblib.load(model_path)


def predict():
    """
    Génère des prédictions pour les dernières années disponibles
    et projections futures.
    """
    print("=" * 60)
    print("PRÉDICTIONS")
    print("=" * 60)

    # Charger modèle et données
    saved = load_model()
    model = saved['model']
    model_name = saved['model_name']
    feature_names = saved['feature_names']
    print(f"Modèle : {model_name}")

    df = load_processed_data()
    X, y, _ = prepare_features(df)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Prédictions sur tout le dataset
    predictions = model.predict(X)

    # Ajouter au DataFrame
    results = df[['country_code', 'country_name', 'year']].copy()
    results['actual'] = y
    results['predicted'] = predictions
    results['error'] = results['actual'] - results['predicted']
    results['error_pct'] = (results['error'] / results['actual'].replace(0, np.nan) * 100).round(2)

    # Sauvegarder
    output_path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
    results.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✅ Prédictions sauvegardées : {output_path}")

    # Résumé par pays
    print("\n--- Erreur moyenne par pays ---")
    for code, name in COUNTRIES.items():
        country = results[results['country_code'] == code]
        if not country.empty:
            mae = country['error'].abs().mean()
            print(f"   {name}: MAE = {mae:.2f} kWh/hab")

    return results


if __name__ == '__main__':
    predict()

"""
Entrainement — Prediction de la demande electrique (GWh) a partir de la
croissance demographique.

Modeles : Random Forest, Gradient Boosting, XGBoost, LightGBM, Stacking.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import MODELS_DIR, RANDOM_STATE
from etl.load import load_processed, temporal_split

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def get_models():
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_split=5,
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        )
    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
    return models


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    print(f"  {name:22s}  RMSE {rmse:10.1f}  MAE {mae:10.1f}  R2 {r2:.4f}  MAPE {mape:.1f}%")
    return {'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def train():
    print("=" * 60)
    print("  ENTRAINEMENT — Demande electrique (GWh)")
    print("=" * 60)

    df = load_processed()
    X_train, X_test, y_train, y_test, feat_names, train_df, test_df = temporal_split(df)

    # Nettoyage
    valid_tr = ~np.isnan(y_train) & ~np.isinf(y_train)
    valid_te = ~np.isnan(y_test) & ~np.isinf(y_test)
    X_train, y_train = X_train[valid_tr], y_train[valid_tr]
    X_test, y_test = X_test[valid_te], y_test[valid_te]
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    models = get_models()
    results = []
    trained = {}

    print()
    for name, model in models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        res = evaluate(y_test, y_pred, name)
        results.append(res)
        trained[name] = model

    # Stacking
    if len(trained) >= 2:
        fresh = get_models()
        stack = StackingRegressor(
            estimators=list(fresh.items()),
            final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
        )
        try:
            stack.fit(X_tr, y_train)
            y_pred = stack.predict(X_te)
            res = evaluate(y_test, y_pred, 'Stacking')
            results.append(res)
            trained['Stacking'] = stack
        except Exception as e:
            print(f"  Stacking echoue : {e}")

    # Meilleur modele
    res_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    best_name = res_df.iloc[0]['model']
    best_model = trained[best_name]
    best_r2 = res_df.iloc[0]['r2']
    print(f"\n  Meilleur : {best_name} (R2 = {best_r2:.4f})")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
        top = np.argsort(fi)[::-1][:8]
        print("\n  Features cles :")
        for i in top:
            print(f"    {feat_names[i]:35s}  {fi[i]:.3f}")

    # Sauvegarder modele
    model_path = os.path.join(MODELS_DIR, 'model_energy.joblib')
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'model_name': best_name,
        'feature_names': feat_names,
        'metrics': res_df.iloc[0].to_dict(),
    }, model_path)
    print(f"\n  Modele : {model_path}")

    # Sauvegarder resultats
    res_df['target'] = 'energy'
    res_df['target_name'] = 'Demande electrique (GWh)'
    res_path = os.path.join(MODELS_DIR, 'results.csv')
    res_df.to_csv(res_path, index=False)
    print(f"  Resultats : {res_path}")
    print(f"  Termine.")
    return res_df


if __name__ == '__main__':
    train()

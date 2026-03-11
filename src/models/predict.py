"""
Predictions historiques + Projections 2024-2030
Cible unique : demande electrique totale (GWh) = f(population, contexte)
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    MODELS_DIR, PREDICTIONS_DIR, COUNTRIES, FOCUS_COUNTRY, FORECAST_HORIZON
)
from etl.load import load_processed, prepare_features


def load_model():
    path = os.path.join(MODELS_DIR, 'model_energy.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manquant : {path}\n  -> python src/models/train.py")
    return joblib.load(path)


def predict_historical(df):
    """Predictions sur les donnees historiques (validation visuelle)."""
    saved = load_model()
    model = saved['model']
    scaler = saved['scaler']

    X, y, _ = prepare_features(df)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)

    out = df[['country_code', 'country_name', 'year']].copy()
    out['actual'] = y
    out['predicted'] = preds
    out['error'] = out['actual'] - out['predicted']
    out['error_pct'] = np.where(out['actual'] != 0,
                                 (out['error'] / out['actual'] * 100).round(2), 0)
    return out


def project_future(df, horizon=FORECAST_HORIZON):
    """
    Projections futures par pays.
    On extrapole les features par tendance lineaire sur les 5 dernieres annees,
    puis on applique le modele.
    """
    saved = load_model()
    model = saved['model']
    scaler = saved['scaler']
    feat_names = saved['feature_names']

    max_year = int(df['year'].max())
    rows = []

    for code, name in COUNTRIES.items():
        cdf = df[df['country_code'] == code].sort_values('year')
        if cdf.empty or len(cdf) < 5:
            continue
        recent = cdf.tail(5)

        for h in range(1, horizon + 1):
            future = {}
            for f in feat_names:
                if f in recent.columns:
                    vals = recent[f].values
                    if len(vals) >= 2 and not np.all(np.isnan(vals)):
                        t = np.polyfit(range(len(vals)), vals, 1)
                        future[f] = t[0] * (len(vals) - 1 + h) + t[1]
                    else:
                        future[f] = vals[-1] if len(vals) > 0 else 0
                else:
                    future[f] = 0

            X = np.array([[future.get(f, 0) for f in feat_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            pred = model.predict(scaler.transform(X))[0]

            # IC simplifie
            if 'conso_totale_gwh' in recent.columns:
                std = np.std(recent['conso_totale_gwh'].values)
                ci = 1.96 * std * np.sqrt(h)
            else:
                ci = 0

            # Population projetee (pour info)
            pop_proj = None
            if 'SP.POP.TOTL' in recent.columns:
                pvals = recent['SP.POP.TOTL'].values
                if len(pvals) >= 2:
                    pt = np.polyfit(range(len(pvals)), pvals, 1)
                    pop_proj = pt[0] * (len(pvals) - 1 + h) + pt[1]

            rows.append({
                'country_code': code,
                'country_name': name,
                'year': max_year + h,
                'predicted_gwh': round(pred, 2),
                'ci_lower': round(pred - ci, 2),
                'ci_upper': round(pred + ci, 2),
                'pop_projected': round(pop_proj) if pop_proj else None,
                'horizon': h,
            })

    return pd.DataFrame(rows)


def predict():
    print("=" * 60)
    print("  PREDICTIONS — Demande electrique (GWh)")
    print("=" * 60)

    df = load_processed()

    # 1. Historique
    hist = predict_historical(df)
    hist_path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
    hist.to_csv(hist_path, index=False)
    print(f"  Predictions historiques : {len(hist)} lignes -> {hist_path}")

    tg_h = hist[hist['country_code'] == FOCUS_COUNTRY]
    if not tg_h.empty:
        mae = tg_h['error'].abs().mean()
        print(f"  MAE Togo : {mae:.1f} GWh")

    # 2. Projections
    proj = project_future(df)
    proj_path = os.path.join(PREDICTIONS_DIR, 'projections.csv')
    proj.to_csv(proj_path, index=False)
    print(f"  Projections : {len(proj)} lignes -> {proj_path}")

    tg_p = proj[proj['country_code'] == FOCUS_COUNTRY]
    if not tg_p.empty:
        print(f"\n  Togo — Projections :")
        for _, r in tg_p.iterrows():
            pop_str = f"  pop {r['pop_projected']/1e6:.1f}M" if r['pop_projected'] else ""
            print(f"    {int(r['year'])} : {r['predicted_gwh']:.1f} GWh "
                  f"[{r['ci_lower']:.1f} - {r['ci_upper']:.1f}]{pop_str}")

    print(f"  Termine.")


if __name__ == '__main__':
    predict()

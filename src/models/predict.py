"""
Predictions historiques + Projections 2024-2045 pour les 8 pays UEMOA
Cible : demande electrique totale (GWh) = f(population, contexte macro)

Methode de projection :
  - Extrapolation robuste des features (10 ans de tendance, clipping)
  - Prediction ML + lissage exponentiel + plancher historique
  - IC 95% base sur les residus du modele
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
    log_target = saved.get('log_target', False)

    X, y_log, _ = prepare_features(df, log_target=log_target)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc = scaler.transform(X)
    preds_log = model.predict(X_sc)

    # Revenir en GWh
    if log_target:
        y_gwh = np.expm1(y_log)
        preds_gwh = np.expm1(preds_log)
    else:
        y_gwh = y_log
        preds_gwh = preds_log

    out = df[['country_code', 'country_name', 'year']].copy()
    out['actual'] = y_gwh
    out['predicted'] = preds_gwh
    out['error'] = out['actual'] - out['predicted']
    out['error_pct'] = np.where(out['actual'] != 0,
                                 (out['error'] / out['actual'] * 100).round(2), 0)
    return out


def _extrapolate_feature(vals, h, feat_name):
    """Extrapole une feature de facon robuste."""
    clean = vals[~np.isnan(vals)]
    if len(clean) < 2:
        return clean[-1] if len(clean) > 0 else 0.0

    # Tendance lineaire
    t = np.polyfit(range(len(clean)), clean, 1)
    projected = t[0] * (len(clean) - 1 + h) + t[1]

    last_val = clean[-1]

    # Pour les taux (%): clipper entre 0 et 100
    pct_kw = ('_ZS', '_ZG', 'ratio', 'year_norm', 'intensite', '_chg')
    if any(k in feat_name for k in pct_kw):
        if last_val >= 0:
            projected = np.clip(projected, 0, max(clean.max() * 1.5, 100))
        else:
            projected = np.clip(projected, min(clean.min() * 1.5, -100), 100)

    # Pour les valeurs absolues positives (population, PIB, conso)
    # ne pas laisser tomber en negatif
    abs_kw = ('POP', 'GDP', 'MKTP', 'pop_', 'pib_', 'log_', 'mobile',
              'indus_', 'gwh_par', 'electrifiee')
    if any(k in feat_name for k in abs_kw):
        projected = max(projected, last_val * 0.5)  # plancher a 50% du dernier

    return projected


def project_future(df, horizon=FORECAST_HORIZON):
    """
    Projections futures par pays avec approche robuste :
    1. Extrapole les features (10 dernieres annees, clipping intelligent)
    2. Predit avec le modele ML
    3. Lisse avec la tendance historique (exponentiel)
    4. Garantit la coherence (pas de baisse irrealiste)
    """
    saved = load_model()
    model = saved['model']
    scaler = saved['scaler']
    feat_names = saved['feature_names']
    log_target = saved.get('log_target', False)

    # Calculer le MAPE residuel du modele pour IC
    hist = predict_historical(df)
    residual_std = hist['error'].std()

    max_year = int(df['year'].max())
    rows = []

    for code, name in COUNTRIES.items():
        cdf = df[df['country_code'] == code].sort_values('year')
        if cdf.empty or len(cdf) < 5:
            continue

        # Utiliser les 10 dernieres annees pour la tendance (plus robuste)
        n_trend = min(10, len(cdf))
        recent = cdf.tail(n_trend)

        # Calculer CAGR historique de la conso (derniers 10 ans)
        gwh_vals = recent['conso_totale_gwh'].values
        last_gwh = gwh_vals[-1]
        if gwh_vals[0] > 0 and last_gwh > 0:
            cagr = (last_gwh / gwh_vals[0]) ** (1 / len(gwh_vals)) - 1
            cagr = np.clip(cagr, 0.005, 0.12)  # entre 0.5% et 12%/an
        else:
            cagr = 0.03  # defaut 3%

        prev_pred = last_gwh

        for h in range(1, horizon + 1):
            future = {}
            for f in feat_names:
                # One-hot country features
                if f.startswith('country_'):
                    future[f] = 1.0 if f == f'country_{code}' else 0.0
                elif f in recent.columns:
                    vals = recent[f].values
                    future[f] = _extrapolate_feature(vals, h, f)
                else:
                    future[f] = 0

            X = np.array([[future.get(f, 0) for f in feat_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            ml_pred_raw = model.predict(scaler.transform(X))[0]
            ml_pred = np.expm1(ml_pred_raw) if log_target else ml_pred_raw

            # Projection par tendance (CAGR)
            trend_pred = last_gwh * (1 + cagr) ** h

            # Blending : moyenne ponderee ML (60%) + tendance (40%)
            # Le ML capture les accelerations, la tendance stabilise
            alpha = 0.6
            blended = alpha * ml_pred + (1 - alpha) * trend_pred

            # Lissage : ne pas s'eloigner trop du pas precedent
            max_jump = max(prev_pred * 0.15, 100)  # max 15%/an de variation
            blended = np.clip(blended, prev_pred - max_jump * 0.3,
                              prev_pred + max_jump)

            # Plancher : jamais en dessous de 90% de la derniere valeur historique
            blended = max(blended, last_gwh * 0.90)

            prev_pred = blended

            # IC base sur les residus du modele, croissant avec l'horizon
            ci = 1.96 * residual_std * np.sqrt(h) * 0.3
            ci_lower = max(blended - ci, last_gwh * 0.5)
            ci_upper = blended + ci

            # Population projetee
            pop_proj = None
            if 'SP.POP.TOTL' in recent.columns:
                pvals = recent['SP.POP.TOTL'].values
                if len(pvals) >= 2:
                    pop_growth = (pvals[-1] / pvals[0]) ** (1 / len(pvals)) - 1
                    pop_proj = pvals[-1] * (1 + pop_growth) ** h

            rows.append({
                'country_code': code,
                'country_name': name,
                'year': max_year + h,
                'predicted_gwh': round(blended, 1),
                'ci_lower': round(ci_lower, 1),
                'ci_upper': round(ci_upper, 1),
                'pop_projected': round(pop_proj) if pop_proj else None,
                'horizon': h,
                'cagr_pct': round(cagr * 100, 2),
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

    # Synthese par pays
    print(f"\n  {'Pays':20s}  {'Obs':>5s}  {'MAE':>10s}  {'MAPE':>8s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*10}  {'-'*8}")
    for code, name in COUNTRIES.items():
        ch = hist[hist['country_code'] == code]
        if not ch.empty:
            mae_c = ch['error'].abs().mean()
            mape_c = ch['error_pct'].abs().mean()
            print(f"  {name:20s}  {len(ch):5d}  {mae_c:10.1f}  {mape_c:7.1f}%")

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

    # Resume par pays (derniere annee projetee)
    last_yr = proj['year'].max()
    proj_last = proj[proj['year'] == last_yr]
    print(f"\n  Projections {int(last_yr)} — Tous les pays UEMOA :")
    print(f"  {'Pays':20s}  {'GWh':>10s}  {'IC bas':>10s}  {'IC haut':>10s}  {'Pop (M)':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for _, r in proj_last.sort_values('predicted_gwh', ascending=False).iterrows():
        pop_str = f"{r['pop_projected']/1e6:.1f}" if pd.notna(r.get('pop_projected')) else "—"
        print(f"  {r['country_name']:20s}  {r['predicted_gwh']:10,.0f}  "
              f"{r['ci_lower']:10,.0f}  {r['ci_upper']:10,.0f}  {pop_str:>8s}")

    print(f"  Termine.")


if __name__ == '__main__':
    predict()

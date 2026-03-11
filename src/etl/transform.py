"""
ETL — Transform
Pivot, nettoyage, feature engineering cible :
  Population -> Consommation electrique totale
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import RAW_DIR, PROCESSED_DIR, FOCUS_COUNTRY


def load_raw():
    path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant : {path}\n  -> python src/etl/extract.py")
    return pd.read_csv(path)


def pivot(df):
    """Une ligne = un pays x une annee, une colonne par indicateur."""
    p = df.pivot_table(
        index=['country_code', 'country_name', 'year'],
        columns='indicator_code', values='value', aggfunc='first',
    ).reset_index()
    p.columns.name = None
    return p


def fill_missing(df):
    """Interpolation lineaire par pays, puis fill, puis mediane regionale."""
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'year']
    for cc in df['country_code'].unique():
        m = df['country_code'] == cc
        df.loc[m, num] = df.loc[m, num].interpolate(method='linear', limit_direction='both')
        df.loc[m, num] = df.loc[m, num].ffill().bfill()
    for c in num:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def engineer(df):
    """Features derives pour la prediction de demande energetique."""
    out = df.copy()

    # Consommation totale = population x kWh/hab  (en GWh)
    if 'SP.POP.TOTL' in out.columns and 'EG.USE.ELEC.KH.PC' in out.columns:
        out['conso_totale_gwh'] = out['SP.POP.TOTL'] * out['EG.USE.ELEC.KH.PC'] / 1e6

    # Gap acces urbain / rural
    if 'EG.ELC.ACCS.UR.ZS' in out.columns and 'EG.ELC.ACCS.RU.ZS' in out.columns:
        out['gap_acces_urb_rur'] = out['EG.ELC.ACCS.UR.ZS'] - out['EG.ELC.ACCS.RU.ZS']

    # Tendance temporelle normalisee
    y_min, y_max = out['year'].min(), out['year'].max()
    out['year_norm'] = (out['year'] - y_min) / (y_max - y_min) if y_max > y_min else 0

    # Features par pays : lags, variations, moyennes mobiles
    key_cols = ['SP.POP.TOTL', 'EG.USE.ELEC.KH.PC', 'EG.ELC.ACCS.ZS', 'NY.GDP.PCAP.CD']
    key_cols = [c for c in key_cols if c in out.columns]

    for col in key_cols:
        grp = out.groupby('country_code')[col]
        out[f'{col}_lag1'] = grp.shift(1)
        out[f'{col}_chg'] = grp.pct_change() * 100
        out[f'{col}_ma3'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Population urbaine estimee
    if 'SP.POP.TOTL' in out.columns and 'SP.URB.TOTL.IN.ZS' in out.columns:
        out['pop_urbaine'] = out['SP.POP.TOTL'] * out['SP.URB.TOTL.IN.ZS'] / 100

    # Ratio conso / PIB  (intensite energetique)
    if 'EG.USE.ELEC.KH.PC' in out.columns and 'NY.GDP.PCAP.CD' in out.columns:
        out['intensite_kwh_pib'] = out['EG.USE.ELEC.KH.PC'] / out['NY.GDP.PCAP.CD'].replace(0, np.nan)

    return out


def transform():
    print("=" * 60)
    print("  TRANSFORMATION — Population & Energie")
    print("=" * 60)

    raw = load_raw()
    print(f"  Brut : {len(raw):,} lignes")

    df = pivot(raw)
    print(f"  Pivot : {df.shape[0]} x {df.shape[1]}")

    df = fill_missing(df)
    print(f"  Missing combles")

    df = engineer(df)
    print(f"  Features : {df.shape[1]} colonnes")

    # Nettoyage final
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"  Dimensions : {df.shape[0]} x {df.shape[1]}")
    print(f"  Fichier : {path}")

    # Apercu Togo
    tg = df[df['country_code'] == FOCUS_COUNTRY]
    if not tg.empty:
        last = tg[tg['year'] == tg['year'].max()].iloc[0]
        print(f"\n  Togo (derniere annee) :")
        for col, label in [
            ('SP.POP.TOTL', 'Population'),
            ('EG.USE.ELEC.KH.PC', 'kWh/hab'),
            ('conso_totale_gwh', 'Conso totale (GWh)'),
            ('EG.ELC.ACCS.ZS', 'Acces electr. (%)'),
        ]:
            if col in last.index:
                print(f"    {label:25s} : {last[col]:,.1f}")

    print(f"  Termine.")
    return df


if __name__ == '__main__':
    transform()

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

    # Population urbaine estimee
    if 'SP.POP.TOTL' in out.columns and 'SP.URB.TOTL.IN.ZS' in out.columns:
        out['pop_urbaine'] = out['SP.POP.TOTL'] * out['SP.URB.TOTL.IN.ZS'] / 100

    # Ratio conso / PIB  (intensite energetique)
    if 'EG.USE.ELEC.KH.PC' in out.columns and 'NY.GDP.PCAP.CD' in out.columns:
        out['intensite_kwh_pib'] = out['EG.USE.ELEC.KH.PC'] / out['NY.GDP.PCAP.CD'].replace(0, np.nan)

    # Part population active (15-64) absolute
    if 'SP.POP.TOTL' in out.columns and 'SP.POP.1564.TO.ZS' in out.columns:
        out['pop_active'] = out['SP.POP.TOTL'] * out['SP.POP.1564.TO.ZS'] / 100

    # Ratio dependance (0-14 / 15-64)
    if 'SP.POP.0014.TO.ZS' in out.columns and 'SP.POP.1564.TO.ZS' in out.columns:
        out['ratio_dependance'] = out['SP.POP.0014.TO.ZS'] / out['SP.POP.1564.TO.ZS'].replace(0, np.nan)

    # PIB total par habitant verifie
    if 'NY.GDP.MKTP.CD' in out.columns and 'SP.POP.TOTL' in out.columns:
        out['pib_par_hab_calc'] = out['NY.GDP.MKTP.CD'] / out['SP.POP.TOTL'].replace(0, np.nan)

    # Energie renouvelable x acces
    if 'EG.FEC.RNEW.ZS' in out.columns and 'EG.ELC.ACCS.ZS' in out.columns:
        out['renew_x_acces'] = out['EG.FEC.RNEW.ZS'] * out['EG.ELC.ACCS.ZS'] / 100

    # Mobile par hab (proxy modernisation)
    if 'IT.CEL.SETS.P2' in out.columns and 'SP.POP.TOTL' in out.columns:
        out['mobile_total'] = out['IT.CEL.SETS.P2'] * out['SP.POP.TOTL'] / 100

    # Features par pays : lags, variations, moyennes mobiles
    key_cols = ['SP.POP.TOTL', 'EG.USE.ELEC.KH.PC', 'EG.ELC.ACCS.ZS',
                'NY.GDP.PCAP.CD', 'SP.URB.TOTL.IN.ZS', 'IT.CEL.SETS.P2',
                'EG.USE.PCAP.KG.OE', 'NY.GDP.MKTP.KD.ZG']
    key_cols = [c for c in key_cols if c in out.columns]

    for col in key_cols:
        grp = out.groupby('country_code')[col]
        out[f'{col}_lag1'] = grp.shift(1)
        out[f'{col}_lag2'] = grp.shift(2)
        out[f'{col}_chg'] = grp.pct_change() * 100
        out[f'{col}_ma3'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
        out[f'{col}_ma5'] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())

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

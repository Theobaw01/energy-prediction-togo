"""
ETL — Load
Prepare X (features demographiques) et y (consommation electrique)
pour predire la demande energetique a partir de la croissance de population.
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, FOCUS_COUNTRY

TARGET = 'conso_totale_gwh'

# Liste fixe des pays UEMOA pour encodage one-hot coherent
_COUNTRY_LIST = ['BEN', 'BFA', 'CIV', 'GNB', 'MLI', 'NER', 'SEN', 'TGO']


def load_processed():
    path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manquant : {path}\n  -> python src/etl/transform.py")
    return pd.read_csv(path)


def prepare_features(df, log_target=True):
    """
    X = features demographiques + contexte + one-hot pays
    y = log(conso_totale_gwh) si log_target=True, sinon GWh brut

    Le one-hot pays permet au modele de distinguer les niveaux par pays.
    Le log-target reduit l'ecrasement des petits pays par les grands.
    """
    exclude = {'country_code', 'country_name', 'year', TARGET,
                'EG.USE.ELEC.KH.PC'}  # exclure kWh/hab pour eviter data leak

    # Exclure aussi les derives directs de la cible
    exclude.update(c for c in df.columns if c.startswith('EG.USE.ELEC.KH.PC'))

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = df[feature_cols].values

    # One-hot encoding pays (colonnes fixes)
    codes = df['country_code'].values if 'country_code' in df.columns else None
    if codes is not None:
        dummies = np.zeros((len(df), len(_COUNTRY_LIST)), dtype=np.float64)
        for i, cc in enumerate(_COUNTRY_LIST):
            dummies[:, i] = (codes == cc).astype(float)
        X = np.hstack([X, dummies])
        country_cols = [f'country_{cc}' for cc in _COUNTRY_LIST]
        feature_cols = feature_cols + country_cols

    y_raw = df[TARGET].values
    if log_target:
        y = np.log1p(np.clip(y_raw, 0, None))  # log(1+x), safe pour 0
    else:
        y = y_raw
    return X, y, feature_cols


def temporal_split(df):
    """Split temporel : annees recentes en test."""
    df_valid = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    df_valid = df_valid.sort_values(['country_code', 'year'])

    years = sorted(df_valid['year'].unique())
    split = int(len(years) * (1 - TEST_SIZE))
    train_years = years[:split]
    test_years = years[split:]

    train_df = df_valid[df_valid['year'].isin(train_years)]
    test_df = df_valid[df_valid['year'].isin(test_years)]

    X_train, y_train, feat = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    print(f"  Train : {len(train_df)} obs ({min(train_years)}-{max(train_years)})")
    print(f"  Test  : {len(test_df)} obs ({min(test_years)}-{max(test_years)})")
    print(f"  Features : {len(feat)}")
    return X_train, X_test, y_train, y_test, feat, train_df, test_df


if __name__ == '__main__':
    df = load_processed()
    X_tr, X_te, y_tr, y_te, feats, _, _ = temporal_split(df)
    print(f"\n  X_train {X_tr.shape}  X_test {X_te.shape}")
    print(f"  y range : {y_tr.min():.0f} — {y_tr.max():.0f} GWh")

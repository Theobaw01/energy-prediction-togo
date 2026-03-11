"""
ETL — Extract
Source : API Banque Mondiale (WDI)
Indicateurs : Population + Consommation electrique + contexte (8 pays UEMOA)
"""
import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    COUNTRIES, INDICATORS, COUNTRY_NAME_FR,
    START_YEAR, END_YEAR, RAW_DIR, FOCUS_COUNTRY_NAME
)

API_BASE = "https://api.worldbank.org/v2"
MAX_RETRIES = 3
RETRY_DELAY = 2
PER_PAGE = 5000


def fetch_indicator(indicator_code, country_codes, start_year, end_year):
    """Recupere un indicateur via l'API Banque Mondiale."""
    countries = ';'.join(country_codes)
    url = (
        f"{API_BASE}/country/{countries}/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page={PER_PAGE}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'EnergyBI/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    ECHEC {indicator_code}: {e}")
                return pd.DataFrame()

    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return pd.DataFrame()

    records = []
    for entry in data[1]:
        if entry.get('value') is not None:
            en_name = entry['country']['value']
            records.append({
                'country_code': entry['country']['id'],
                'country_name': COUNTRY_NAME_FR.get(en_name, en_name),
                'year': int(entry['date']),
                'indicator_code': indicator_code,
                'value': float(entry['value']),
            })

    return pd.DataFrame(records)


def extract_all():
    """Extraction de tous les indicateurs pour les 8 pays UEMOA."""
    print("=" * 60)
    print(f"  EXTRACTION — Population & Energie ({FOCUS_COUNTRY_NAME} + UEMOA)")
    print("=" * 60)
    print(f"  Pays : {len(COUNTRIES)}  |  Periode : {START_YEAR}-{END_YEAR}")
    print(f"  Indicateurs : {len(INDICATORS)}")
    print("-" * 60)

    country_codes = list(COUNTRIES.keys())
    all_data = []

    for code, name in INDICATORS.items():
        df = fetch_indicator(code, country_codes, START_YEAR, END_YEAR)
        if df.empty:
            print(f"  [ ] {name}")
        else:
            all_data.append(df)
            print(f"  [x] {name} — {len(df)} obs.")

    if not all_data:
        print("\n  Aucune donnee extraite.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['country_code', 'year', 'indicator_code'])
    combined = combined.reset_index(drop=True)

    output_path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    combined.to_csv(output_path, index=False, encoding='utf-8')

    print("-" * 60)
    print(f"  Total : {len(combined):,} enregistrements")
    print(f"  Fichier : {output_path}")
    print(f"  Termine.")
    return combined


if __name__ == '__main__':
    extract_all()

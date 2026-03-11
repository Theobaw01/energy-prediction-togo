"""
Configuration — Demande energetique et croissance demographique
Togo & UEMOA | Cadre BCEAO

Question : la population croit, combien d'electricite faudra-t-il demain ?
ETL cible : Population (habitants) -> Consommation electrique (kWh)
"""
import os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

for d in [RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Pays UEMOA ──────────────────────────────────────────────────────────────
FOCUS_COUNTRY = 'TG'
FOCUS_COUNTRY_NAME = 'Togo'

COUNTRIES = {
    'TG': 'Togo',
    'SN': 'Senegal',
    'CI': "Cote d'Ivoire",
    'BJ': 'Benin',
    'BF': 'Burkina Faso',
    'ML': 'Mali',
    'NE': 'Niger',
    'GW': 'Guinee-Bissau',
}

# ── Indicateurs Banque Mondiale ──────────────────────────────────────────────
# On ne garde que ce qui sert : population et consommation
INDICATORS = {
    # Demographie
    'SP.POP.TOTL':        'Population totale',
    'SP.POP.GROW':        'Croissance demographique (%)',
    'SP.URB.TOTL.IN.ZS':  'Taux urbanisation (%)',
    # Energie
    'EG.USE.ELEC.KH.PC':  'Consommation electrique (kWh/hab)',
    'EG.ELC.ACCS.ZS':     'Acces electricite (%)',
    'EG.ELC.ACCS.UR.ZS':  'Acces electricite urbain (%)',
    'EG.ELC.ACCS.RU.ZS':  'Acces electricite rural (%)',
    # Economie (contexte)
    'NY.GDP.PCAP.CD':     'PIB par habitant (USD)',
}

COUNTRY_NAME_FR = {
    'Togo': 'Togo',
    'Senegal': 'Senegal',
    "Cote d'Ivoire": "Cote d'Ivoire",
    'Benin': 'Benin',
    'Burkina Faso': 'Burkina Faso',
    'Mali': 'Mali',
    'Niger': 'Niger',
    'Guinea-Bissau': 'Guinee-Bissau',
}

# ── Parametres ───────────────────────────────────────────────────────────────
START_YEAR = 2000
END_YEAR = 2023
RANDOM_STATE = 42
TEST_SIZE = 0.2
FORECAST_HORIZON = 7  # Projections 2024-2030

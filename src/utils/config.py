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
# 21 indicateurs — demographie, energie, economie, social
INDICATORS = {
    # --- Demographie (7) ---
    'SP.POP.TOTL':        'Population totale',
    'SP.POP.GROW':        'Croissance demographique (%)',
    'SP.URB.TOTL.IN.ZS':  'Taux urbanisation (%)',
    'SP.DYN.TFRT.IN':     'Taux de fecondite (naissances/femme)',
    'SP.DYN.LE00.IN':     'Esperance de vie (annees)',
    'SP.POP.0014.TO.ZS':  'Population 0-14 ans (%)',
    'SP.POP.1564.TO.ZS':  'Population 15-64 ans (%)',
    # --- Energie (6) ---
    'EG.USE.ELEC.KH.PC':  'Consommation electrique (kWh/hab)',
    'EG.ELC.ACCS.ZS':     'Acces electricite (%)',
    'EG.ELC.ACCS.UR.ZS':  'Acces electricite urbain (%)',
    'EG.ELC.ACCS.RU.ZS':  'Acces electricite rural (%)',
    'EG.FEC.RNEW.ZS':     'Energie renouvelable (% conso finale)',
    'EG.USE.PCAP.KG.OE':  'Utilisation energie (kg petrole eq./hab)',
    # --- Economie (5) ---
    'NY.GDP.PCAP.CD':     'PIB par habitant (USD)',
    'NY.GDP.MKTP.CD':     'PIB total (USD courants)',
    'NY.GDP.MKTP.KD.ZG':  'Croissance PIB (%)',
    'NV.IND.TOTL.ZS':     'Part industrie (% PIB)',
    'FP.CPI.TOTL.ZG':     'Inflation IPC (%)',
    # --- Social / Infrastructure (3) ---
    'IT.CEL.SETS.P2':     'Abonnements mobile (/100 hab)',
    'SE.ADT.LITR.ZS':     'Taux alphabetisation adultes (%)',
    'SL.UEM.TOTL.ZS':     'Chomage (%)',
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
START_YEAR = 1990
END_YEAR = 2023
RANDOM_STATE = 42
TEST_SIZE = 0.2
FORECAST_HORIZON = 22  # Projections 2024-2045

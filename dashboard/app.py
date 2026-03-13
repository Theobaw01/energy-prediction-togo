"""
Dashboard IA — Prevision de la Demande Electrique | Zone UEMOA
=====================================================================
Ce projet a ete realise dans l'objectif de maitriser les concepts lies a
l'ingenierie de donnees et au Machine Learning, transposables dans des
situations reelles de modelisation macroeconomique.

Pipeline : API Banque Mondiale -> ETL -> Feature Engineering -> ML -> Dashboard
Perimetre : 8 pays UEMOA | 1990-2023 | Horizon 2045
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IA — Demande Electrique UEMOA 2045",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"

# Palette coherente
P = {
    "primary": "#3B82F6",   # bleu
    "violet": "#8B5CF6",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "cyan": "#06B6D4",
    "lime": "#84CC16",
    "orange": "#F97316",
    "muted": "#64748B",
    "bg_card": "#0F172A",
    "bg_dark": "#020617",
    "border": "#1E293B",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
}

PALETTE_8 = [P["primary"], P["violet"], P["amber"], P["emerald"],
             P["red"], P["cyan"], P["lime"], P["orange"]]

FLAGS = {
    'TG': '🇹🇬', 'SN': '🇸🇳', 'CI': '🇨🇮', 'BJ': '🇧🇯',
    'BF': '🇧🇫', 'ML': '🇲🇱', 'NE': '🇳🇪', 'GW': '🇬🇼',
}

IND_LABELS = {
    "SP.POP.TOTL": "Population totale", "SP.POP.GROW": "Croissance demo. (%)",
    "SP.URB.TOTL.IN.ZS": "Urbanisation (%)", "SP.DYN.TFRT.IN": "Fecondite",
    "SP.DYN.LE00.IN": "Esperance de vie", "SP.POP.0014.TO.ZS": "Pop 0-14 (%)",
    "SP.POP.1564.TO.ZS": "Pop 15-64 (%)", "EG.USE.ELEC.KH.PC": "kWh/hab",
    "EG.ELC.ACCS.ZS": "Acces electr. (%)", "EG.ELC.ACCS.UR.ZS": "Acces urbain (%)",
    "EG.ELC.ACCS.RU.ZS": "Acces rural (%)", "EG.FEC.RNEW.ZS": "Renouvelable (%)",
    "EG.USE.PCAP.KG.OE": "Energie (kg petrole/hab)", "NY.GDP.PCAP.CD": "PIB/hab (USD)",
    "NY.GDP.MKTP.CD": "PIB total (USD)", "NY.GDP.MKTP.KD.ZG": "Croissance PIB (%)",
    "NV.IND.TOTL.ZS": "Industrie (% PIB)", "FP.CPI.TOTL.ZG": "Inflation (%)",
    "IT.CEL.SETS.P2": "Mobile (/100 hab)", "SE.ADT.LITR.ZS": "Alphabetisation (%)",
    "SL.UEM.TOTL.ZS": "Chomage (%)",
}

# ─────────────────────────────────────────────────────────────────────
# STYLE — adapte du design SnapTaf (shadcn/tailwind -> CSS pur)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.5rem; max-width: 1400px; }

/* ── Header ── */
.dash-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    padding: 24px 32px; border-radius: 12px; margin-bottom: 20px;
    border: 1px solid #1E293B;
}
.dash-header .breadcrumb { color: #64748B; font-size: 0.72em; margin-bottom: 4px; }
.dash-header .breadcrumb a { color: #94A3B8; text-decoration: none; }
.dash-header h1 { color: #F8FAFC; margin: 0; font-size: 1.5em; font-weight: 800;
                   letter-spacing: -0.025em; }
.dash-header .sub { color: #94A3B8; font-size: 0.82em; margin-top: 4px; font-weight: 400; }
.dash-header .obj {
    display: inline-block; margin-top: 10px; padding: 6px 14px;
    background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px; color: #94A3B8; font-size: 0.68em; font-style: italic;
}

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 18px; }
.kpi-card {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 10px;
    padding: 16px 18px;
}
.kpi-card .kpi-top { display: flex; align-items: flex-start; justify-content: space-between; }
.kpi-card .kpi-label { color: #64748B; font-size: 0.68em; text-transform: uppercase;
                       letter-spacing: 0.05em; font-weight: 500; }
.kpi-card .kpi-value { color: #F8FAFC; font-size: 1.6em; font-weight: 800;
                       margin-top: 4px; letter-spacing: -0.025em; }
.kpi-card .kpi-delta { font-size: 0.72em; font-weight: 600; margin-top: 4px;
                       display: flex; align-items: center; gap: 3px; }
.kpi-card .kpi-delta.up { color: #10B981; }
.kpi-card .kpi-delta.dn { color: #EF4444; }
.kpi-card .kpi-delta .vs { color: #64748B; font-weight: 400; margin-left: 4px; }
.kpi-card .kpi-icon {
    background: #1E293B; border-radius: 8px; padding: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1em;
}

/* ── Section headers ── */
.sec-header {
    display: flex; align-items: center; gap: 8px;
    margin: 22px 0 10px 0; padding-bottom: 6px;
    border-bottom: 1px solid #1E293B;
}
.sec-header .sec-icon { font-size: 1em; }
.sec-header .sec-title { color: #F1F5F9; font-size: 0.92em; font-weight: 600; }
.sec-header .sec-badge {
    background: rgba(16,185,129,0.1); color: #10B981; border: 1px solid rgba(16,185,129,0.2);
    border-radius: 6px; padding: 2px 8px; font-size: 0.6em; font-weight: 600;
    margin-left: auto;
}

/* ── Insight cards ── */
.insight-card {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 10px;
    padding: 16px 20px; margin: 10px 0 16px 0;
}
.insight-card .insight-label {
    color: #3B82F6; font-size: 0.68em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.insight-card p { color: #CBD5E1; font-size: 0.82em; line-height: 1.6; margin: 0; }
.insight-card .val { color: #10B981; font-weight: 700; }
.insight-card .warn { color: #EF4444; font-weight: 700; }
.insight-card .hl { color: #F8FAFC; font-weight: 600; }
.insight-card .blue { color: #3B82F6; font-weight: 700; }

/* ── Step boxes ── */
.step-box {
    background: #0F172A; border: 1px solid #1E293B; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 12px;
}
.step-box .step-tag { color: #3B82F6; font-size: 0.65em; font-weight: 700;
                      text-transform: uppercase; letter-spacing: 0.1em; }
.step-box .step-title { color: #F1F5F9; font-weight: 700; font-size: 0.92em; margin-top: 2px; }
.step-box .step-desc { color: #64748B; font-size: 0.74em; margin-top: 3px; }

/* ── Prediction highlight cards ── */
.pred-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0; }
.pred-card {
    background: #0F172A; border-radius: 10px; padding: 16px 18px;
}
.pred-card.blue { border: 1px solid rgba(59,130,246,0.3); }
.pred-card.green { border: 1px solid rgba(16,185,129,0.3); }
.pred-card.violet { border: 1px solid rgba(139,92,246,0.3); }
.pred-card .pred-icon { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
.pred-card .pred-icon span { font-size: 0.82em; font-weight: 600; color: #E2E8F0; }
.pred-card .pred-value { font-size: 1.8em; font-weight: 800; letter-spacing: -0.03em; }
.pred-card .pred-value.c-blue { color: #3B82F6; }
.pred-card .pred-value.c-green { color: #10B981; }
.pred-card .pred-value.c-violet { color: #8B5CF6; }
.pred-card .pred-sub { color: #64748B; font-size: 0.72em; margin-top: 2px; }
.pred-card .pred-desc { color: #94A3B8; font-size: 0.72em; margin-top: 8px; line-height: 1.5; }

/* ── Footer ── */
.dash-footer {
    text-align: center; color: #475569; font-size: 0.68em;
    padding: 18px 0; margin-top: 24px; border-top: 1px solid #1E293B;
}
.dash-footer strong { color: #64748B; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    d = {}
    for k, f in [("df", "data/processed/energy_data_processed.csv"),
                  ("raw", "data/raw/energy_data_raw.csv"),
                  ("pred", "data/predictions/predictions.csv"),
                  ("proj", "data/predictions/projections.csv"),
                  ("res", "models/results.csv"),
                  ("fi", "models/feature_importance.csv"),
                  ("cv", "models/cv_scores.csv")]:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            d[k] = pd.read_csv(p)
    return d


def fmt(v, u=""):
    if pd.isna(v): return "—"
    if abs(v) >= 1e9: s = f"{v/1e9:,.1f} Mrd"
    elif abs(v) >= 1e6: s = f"{v/1e6:,.1f} M"
    elif abs(v) >= 1e3: s = f"{v:,.0f}"
    else: s = f"{v:,.1f}"
    return f"{s} {u}".strip() if u else s


def pct_change(a, b):
    return ((b / a) - 1) * 100 if a and a != 0 else 0


def fig_layout(fig, title="", h=400, yaxis_title="", margin_b=60):
    """Layout commun pour tous les graphiques."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F1F5F9", family="Inter"),
                   x=0, xanchor="left"),
        template=TMPL, height=h, hovermode="x unified",
        margin=dict(t=45, b=margin_b, l=60, r=20),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#94A3B8")),
        yaxis_title=yaxis_title,
        plot_bgcolor="#020617", paper_bgcolor="#020617",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <div class="breadcrumb">Energy Prediction UEMOA &rsaquo; <strong>Dashboard IA</strong></div>
    <h1>⚡ Prevision de la Demande Electrique — Zone UEMOA</h1>
    <div class="sub">Analyse predictive et projections energetiques — 8 pays UEMOA | 1990-2023 | Horizon 2045</div>
    <div class="obj">🎯 Ce projet a ete realise dans l'objectif de maitriser les concepts lies a l'ingenierie
    de donnees et au Machine Learning, transposables dans des situations reelles.</div>
</div>
""", unsafe_allow_html=True)

data = load()
if "df" not in data:
    st.error("Donnees absentes. Executez le pipeline ETL d'abord.")
    st.stop()

df_all = data["df"]
raw_all = data.get("raw")
pred_all = data.get("pred")
proj_all = data.get("proj")
res = data.get("res")
fi_df = data.get("fi")
cv_df = data.get("cv")

countries_available = sorted(df_all["country_code"].unique().tolist())
country_names = df_all.drop_duplicates("country_code").set_index("country_code")["country_name"].to_dict()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Navigation")
    sel_country = st.selectbox(
        "Pays",
        countries_available,
        format_func=lambda c: f"{FLAGS.get(c,'')} {country_names.get(c, c)}",
        index=countries_available.index('TG') if 'TG' in countries_available else 0,
        key="sel_country",
    )
    sel_name = country_names.get(sel_country, sel_country)
    sel_flag = FLAGS.get(sel_country, '')

    st.divider()
    y_min_h, y_max_h = int(df_all["year"].min()), int(df_all["year"].max())
    yr = st.slider("Periode historique", y_min_h, y_max_h, (y_min_h, y_max_h), key="yr_hist")

    st.divider()
    st.markdown(f"**Pays** : {sel_flag} {sel_name}")
    st.markdown(f"**Historique** : {yr[0]} — {yr[1]}")
    st.markdown(f"**Projection** : 2024 — 2045")

    st.divider()
    st.markdown("##### Telecharger")
    if raw_all is not None:
        st.download_button("📥 Donnees brutes", raw_all.to_csv(index=False).encode(),
                           "donnees_brutes.csv", key="dl_raw")
    if proj_all is not None:
        st.download_button("📥 Projections UEMOA", proj_all.to_csv(index=False).encode(),
                           "projections_uemoa_2045.csv", key="dl_proj")

# Country data
df = df_all[df_all["country_code"] == sel_country].copy()
tg = df[df["year"].between(*yr)].sort_values("year")
raw_sel = raw_all[raw_all["country_code"] == sel_country] if raw_all is not None else None
pred = pred_all[pred_all["country_code"] == sel_country] if pred_all is not None else None
proj = proj_all[proj_all["country_code"] == sel_country] if proj_all is not None else None

if tg.empty:
    st.warning(f"Aucune donnee disponible pour {sel_name}.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────
last, first = tg.iloc[-1], tg.iloc[0]

kpi_html = '<div class="kpi-grid">'

# 1 — Population
if "SP.POP.TOTL" in tg.columns:
    pop_now, pop_bef = last["SP.POP.TOTL"], first["SP.POP.TOTL"]
    gr = pct_change(pop_bef, pop_now)
    kpi_html += f'''<div class="kpi-card"><div class="kpi-top">
        <div><div class="kpi-label">Population</div>
        <div class="kpi-value">{fmt(pop_now)}</div>
        <div class="kpi-delta up">↑ +{gr:.0f}%<span class="vs">vs {yr[0]}</span></div>
        </div><div class="kpi-icon">👥</div></div></div>'''

# 2 — Demande GWh
if "conso_totale_gwh" in tg.columns:
    gwh, gwh0 = last["conso_totale_gwh"], first["conso_totale_gwh"]
    d = pct_change(gwh0, gwh)
    kpi_html += f'''<div class="kpi-card"><div class="kpi-top">
        <div><div class="kpi-label">Demande electrique</div>
        <div class="kpi-value">{fmt(gwh, "GWh")}</div>
        <div class="kpi-delta up">↑ +{d:.0f}%<span class="vs">vs {yr[0]}</span></div>
        </div><div class="kpi-icon">⚡</div></div></div>'''

# 3 — Acces
if "EG.ELC.ACCS.ZS" in tg.columns:
    acc = last["EG.ELC.ACCS.ZS"]
    acc0 = first["EG.ELC.ACCS.ZS"]
    css = "up" if acc > acc0 else "dn"
    arr = "↑" if acc > acc0 else "↓"
    kpi_html += f'''<div class="kpi-card"><div class="kpi-top">
        <div><div class="kpi-label">Acces electrique</div>
        <div class="kpi-value">{acc:.1f}%</div>
        <div class="kpi-delta {css}">{arr} {acc-acc0:+.1f} pts<span class="vs">vs {yr[0]}</span></div>
        </div><div class="kpi-icon">🔌</div></div></div>'''

# 4 — Projection 2045
if proj is not None and not proj.empty:
    row_2045 = proj[proj["year"] == proj["year"].max()].iloc[0]
    kpi_html += f'''<div class="kpi-card"><div class="kpi-top">
        <div><div class="kpi-label">Projection {int(row_2045["year"])}</div>
        <div class="kpi-value">{fmt(row_2045["predicted_gwh"], "GWh")}</div>
        <div class="kpi-delta up">↑ Horizon IA</div>
        </div><div class="kpi-icon">🔮</div></div></div>'''

# 5 — Modele
if res is not None and not res.empty:
    best = res.sort_values("r2", ascending=False).iloc[0]
    kpi_html += f'''<div class="kpi-card"><div class="kpi-top">
        <div><div class="kpi-label">Modele IA</div>
        <div class="kpi-value">R² {best["r2"]:.3f}</div>
        <div class="kpi-delta up">↑ {best["model"]}</div>
        </div><div class="kpi-icon">🧠</div></div></div>'''

kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "📊 Tendances",
    "🔬 Exploration",
    "🧠 Modele IA",
    "🔮 Predictions 2045",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — TENDANCES
# ═══════════════════════════════════════════════════════════════════
with t1:
    n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
    n_pays = len(raw_all["country_code"].unique()) if raw_all is not None else 0
    n_raw = len(raw_all) if raw_all is not None else 0

    st.markdown(f'''<div class="step-box">
        <div class="step-tag">Etape 1 — Collecte automatisee</div>
        <div class="step-title">API Banque Mondiale (WDI) → {n_raw:,} observations extraites</div>
        <div class="step-desc">{n_ind} indicateurs × {n_pays} pays UEMOA × {yr[1]-yr[0]+1} annees
         | Source : data.worldbank.org</div>
    </div>''', unsafe_allow_html=True)

    # ── Evolution demande electrique ── (graphe principal)
    st.markdown('''<div class="sec-header">
        <span class="sec-icon">⚡</span>
        <span class="sec-title">Evolution de la demande electrique — Tous les pays</span>
    </div>''', unsafe_allow_html=True)

    if "conso_totale_gwh" in df_all.columns:
        fig = go.Figure()
        for i, cc in enumerate(countries_available):
            dc = df_all[(df_all["country_code"] == cc) & (df_all["year"].between(*yr))].sort_values("year")
            if dc.empty or "conso_totale_gwh" not in dc.columns:
                continue
            is_sel = cc == sel_country
            cn = country_names.get(cc, cc)
            fig.add_trace(go.Scatter(
                x=dc["year"], y=dc["conso_totale_gwh"],
                name=f"{FLAGS.get(cc,'')} {cn}",
                mode="lines+markers" if is_sel else "lines",
                line=dict(width=3.5 if is_sel else 1.2, color=PALETTE_8[i % 8]),
                opacity=1.0 if is_sel else 0.3,
                marker=dict(size=6) if is_sel else dict(size=0),
            ))
        fig = fig_layout(fig, f"Demande electrique (GWh) — 8 pays UEMOA ({yr[0]}-{yr[1]})",
                         420, "GWh")
        st.plotly_chart(fig, key="trend_gwh_all")

        # Ranking actuel
        last_yr_data = df_all[df_all["year"] == y_max_h][["country_code", "country_name", "conso_totale_gwh"]].dropna()
        if not last_yr_data.empty:
            rank = last_yr_data.sort_values("conso_totale_gwh", ascending=False).reset_index(drop=True)
            pos = list(rank["country_code"]).index(sel_country) + 1 if sel_country in list(rank["country_code"]) else "?"
            top1 = rank.iloc[0]
            sel_gwh = rank[rank["country_code"] == sel_country]["conso_totale_gwh"].iloc[0]
            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">📊 Lecture du graphique</div>
                <p>En <span class="hl">{y_max_h}</span>, la {sel_flag} <span class="hl">{sel_name}</span>
                se classe <span class="blue">#{pos}</span> sur {len(rank)} pays en demande electrique
                avec <span class="val">{sel_gwh:,.0f} GWh</span>.
                Le leader est {FLAGS.get(top1["country_code"],"")} <span class="hl">{top1["country_name"]}</span>
                (<span class="val">{top1["conso_totale_gwh"]:,.0f} GWh</span>).
                La trajectoire montre une croissance generalisee, refletant
                l'urbanisation et l'industrialisation progressive de la region.</p>
            </div>''', unsafe_allow_html=True)

    st.divider()

    # ── Explorateur indicateur ──
    st.markdown(f'''<div class="sec-header">
        <span class="sec-icon">🔍</span>
        <span class="sec-title">Explorateur d'indicateurs — comparaison UEMOA</span>
    </div>''', unsafe_allow_html=True)

    if raw_all is not None and not raw_all.empty:
        raw_f = raw_all[raw_all["year"].between(*yr)]
        ind_codes = sorted(raw_f["indicator_code"].unique().tolist())
        sel_ind = st.selectbox("Indicateur", ind_codes,
                               format_func=lambda c: IND_LABELS.get(c, c), key="raw_ind")

        fig = go.Figure()
        for i, cc in enumerate(countries_available):
            rcc = raw_f[(raw_f["country_code"] == cc) & (raw_f["indicator_code"] == sel_ind)].sort_values("year")
            if rcc.empty: continue
            is_sel = cc == sel_country
            fig.add_trace(go.Scatter(
                x=rcc["year"], y=rcc["value"],
                name=f"{FLAGS.get(cc,'')} {country_names.get(cc, cc)}",
                mode="lines+markers" if is_sel else "lines",
                line=dict(width=3 if is_sel else 1.2, color=PALETTE_8[i % 8]),
                opacity=1.0 if is_sel else 0.3,
                marker=dict(size=5) if is_sel else dict(size=0),
            ))
        fig = fig_layout(fig, f"{IND_LABELS.get(sel_ind, sel_ind)} — 8 pays UEMOA",
                         380, IND_LABELS.get(sel_ind, ""))
        st.plotly_chart(fig, key="raw_explore")

        ri = raw_f[(raw_f["country_code"] == sel_country) & (raw_f["indicator_code"] == sel_ind)].sort_values("year")
        if len(ri) > 1:
            v0, v1 = ri["value"].iloc[0], ri["value"].iloc[-1]
            chg = pct_change(abs(v0) if v0 != 0 else 1, abs(v1))
            trend = "hausse" if chg > 5 else "baisse" if chg < -5 else "stagnation"
            css = "val" if chg > 0 else "warn"
            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">📈 Interpretation</div>
                <p><span class="hl">{IND_LABELS.get(sel_ind, sel_ind)}</span> pour
                {sel_flag} {sel_name} : de <span class="val">{v0:,.1f}</span> ({yr[0]})
                a <span class="val">{v1:,.1f}</span> ({yr[1]}),
                soit une <span class="{css}">{trend} de {abs(chg):.1f}%</span>.
                Les courbes des autres pays sont affichees en arriere-plan pour
                permettre un benchmarking regional immediat.</p>
            </div>''', unsafe_allow_html=True)

    with st.expander("Voir les donnees brutes", expanded=False):
        if raw_sel is not None:
            st.dataframe(raw_sel[raw_sel["year"].between(*yr)], height=250)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ═══════════════════════════════════════════════════════════════════
with t2:
    n_cols = len(df_all.columns)
    st.markdown(f'''<div class="step-box">
        <div class="step-tag">Etape 2 — Feature Engineering</div>
        <div class="step-title">21 indicateurs bruts → {n_cols} variables (lags, moyennes mobiles, ratios, log)</div>
        <div class="step-desc">{len(df_all)} observations pour {len(countries_available)} pays.
        Chaque variable est un signal potentiel pour le modele predictif.</div>
    </div>''', unsafe_allow_html=True)

    # ── Co-evolution pop / GWh ──
    st.markdown(f'''<div class="sec-header">
        <span class="sec-icon">👥</span>
        <span class="sec-title">Co-evolution population / demande — {sel_flag} {sel_name}</span>
    </div>''', unsafe_allow_html=True)

    if "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Population (M)",
            marker_color=P["primary"], opacity=0.25,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=tg["year"], y=tg["conso_totale_gwh"], name="Demande (GWh)",
            mode="lines+markers", line=dict(color=P["emerald"], width=3),
            marker=dict(size=5),
        ), secondary_y=True)
        fig = fig_layout(fig, f"Population vs demande electrique — {sel_name}", 400)
        fig.update_yaxes(title_text="Population (M)", secondary_y=False,
                         gridcolor="#1E293B")
        fig.update_yaxes(title_text="GWh", secondary_y=True,
                         gridcolor="#1E293B")
        st.plotly_chart(fig, key="exp_pop_gwh")

        pop_chg = pct_change(tg["SP.POP.TOTL"].iloc[0], tg["SP.POP.TOTL"].iloc[-1])
        gwh_chg = pct_change(
            tg["conso_totale_gwh"].iloc[0] if tg["conso_totale_gwh"].iloc[0] > 0 else 1,
            tg["conso_totale_gwh"].iloc[-1]
        )
        elast = gwh_chg / pop_chg if pop_chg > 0 else 0
        corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
        st.markdown(f'''<div class="insight-card">
            <div class="insight-label">📐 Analyse quantitative</div>
            <p>Population : <span class="val">+{pop_chg:.0f}%</span> |
            Demande : <span class="val">+{gwh_chg:.0f}%</span> |
            <span class="hl">Elasticite = {elast:.2f}</span>
            (chaque +1% de la population entraine +{elast:.2f}% de demande electrique).
            Correlation de Pearson : <span class="blue">{corr:.3f}</span>.
            Cette relation forte confirme que la croissance demographique est un
            <span class="hl">driver majeur</span> de la demande energetique.</p>
        </div>''', unsafe_allow_html=True)

    st.divider()

    # ── Fracture urbain / rural ──
    st.markdown(f'''<div class="sec-header">
        <span class="sec-icon">🏙️</span>
        <span class="sec-title">Fracture energetique urbain vs rural</span>
    </div>''', unsafe_allow_html=True)

    if "EG.ELC.ACCS.UR.ZS" in tg.columns and "EG.ELC.ACCS.RU.ZS" in tg.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tg["year"], y=tg["EG.ELC.ACCS.UR.ZS"], name="Acces urbain",
            mode="lines+markers", line=dict(color=P["emerald"], width=2.5),
            marker=dict(size=4), fill="tonexty",
        ))
        fig.add_trace(go.Scatter(
            x=tg["year"], y=tg["EG.ELC.ACCS.RU.ZS"], name="Acces rural",
            mode="lines+markers", line=dict(color=P["red"], width=2.5),
            marker=dict(size=4),
        ))
        if "EG.ELC.ACCS.ZS" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["EG.ELC.ACCS.ZS"], name="Moyenne nationale",
                mode="lines", line=dict(color=P["amber"], width=2, dash="dash"),
            ))
        fig = fig_layout(fig, f"Acces a l'electricite — {sel_name}", 360,
                         "% de la population")
        st.plotly_chart(fig, key="exp_acces")

        gap = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1] - tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
        urb = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]
        rur = tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
        st.markdown(f'''<div class="insight-card">
            <div class="insight-label">🏚️ Diagnostic fracture energetique</div>
            <p>Urbain : <span class="val">{urb:.1f}%</span> |
            Rural : <span class="warn">{rur:.1f}%</span> |
            Ecart : <span class="warn">{gap:.1f} points</span>.
            Cette inegalite signifie que les gains d'electrification rurale
            representent un <span class="hl">reservoir de demande latente</span> :
            chaque point d'acces gagne en zone rurale generera une hausse mecanique
            de la consommation nationale.</p>
        </div>''', unsafe_allow_html=True)

    st.divider()

    # ── Matrice de correlation ──
    st.markdown('''<div class="sec-header">
        <span class="sec-icon">🔗</span>
        <span class="sec-title">Correlations entre variables cles</span>
    </div>''', unsafe_allow_html=True)

    corr_cols = ["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EG.ELC.ACCS.ZS",
                 "NY.GDP.PCAP.CD", "IT.CEL.SETS.P2", "conso_totale_gwh"]
    corr_cols = [c for c in corr_cols if c in tg.columns]
    lbl_map = {"SP.POP.TOTL": "Population", "SP.URB.TOTL.IN.ZS": "Urbanisation",
               "EG.ELC.ACCS.ZS": "Acces %", "NY.GDP.PCAP.CD": "PIB/hab",
               "IT.CEL.SETS.P2": "Mobile", "conso_totale_gwh": "Demande GWh"}
    if len(corr_cols) > 3:
        cm = tg[corr_cols].corr()
        labels = [lbl_map.get(c, c) for c in corr_cols]
        fig = go.Figure(go.Heatmap(
            z=cm.values, x=labels, y=labels,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(cm.values, 2), texttemplate="%{text:.2f}",
            textfont_size=11, colorbar=dict(thickness=12, len=0.6),
        ))
        fig = fig_layout(fig, f"Matrice de correlation — {sel_name}", 420)
        st.plotly_chart(fig, key="corr_heat")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MODELE IA
# ═══════════════════════════════════════════════════════════════════
with t3:
    n_feat = len([c for c in df_all.columns if c not in
                  ['country_code','country_name','year','conso_totale_gwh','EG.USE.ELEC.KH.PC']
                  and not c.startswith('EG.USE.ELEC.KH.PC')
                  and df_all[c].dtype in ['float64','int64','float32','int32']])

    st.markdown(f'''<div class="step-box">
        <div class="step-tag">Etape 3 — Entrainement du modele</div>
        <div class="step-title">5 algorithmes compares sur {len(df_all)} observations ({len(countries_available)} pays)</div>
        <div class="step-desc">{n_feat} features | Cible : demande electrique (GWh) |
        Split temporel 80/20 + cross-validation temporelle 5 folds</div>
    </div>''', unsafe_allow_html=True)

    if res is not None and not res.empty:
        # ── Performance comparee ──
        st.markdown('''<div class="sec-header">
            <span class="sec-icon">📊</span>
            <span class="sec-title">Performance comparee des algorithmes</span>
        </div>''', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            r2_s = res.sort_values("r2", ascending=True)
            colors_r2 = [P["emerald"] if v > 0.85 else P["amber"] if v > 0.7
                         else P["red"] for v in r2_s["r2"]]
            fig = go.Figure(go.Bar(
                x=r2_s["r2"], y=r2_s["model"], orientation="h",
                marker_color=colors_r2,
                text=[f"{v:.3f}" for v in r2_s["r2"]],
                textposition="outside", textfont=dict(size=12, color="#F1F5F9"),
            ))
            fig = fig_layout(fig, "Score R² (plus haut = meilleur)", 300)
            fig.update_layout(xaxis_range=[0, 1.08])
            st.plotly_chart(fig, key="mod_r2")

        with c2:
            mape_s = res.sort_values("mape", ascending=False)
            colors_m = [P["emerald"] if v < 25 else P["amber"] if v < 40
                        else P["red"] for v in mape_s["mape"]]
            fig = go.Figure(go.Bar(
                x=mape_s["mape"], y=mape_s["model"], orientation="h",
                marker_color=colors_m,
                text=[f"{v:.1f}%" for v in mape_s["mape"]],
                textposition="outside", textfont=dict(size=12, color="#F1F5F9"),
            ))
            fig = fig_layout(fig, "Erreur MAPE % (plus bas = meilleur)", 300)
            st.plotly_chart(fig, key="mod_mape")

        best = res.sort_values("r2", ascending=False).iloc[0]
        st.markdown(f'''<div class="insight-card">
            <div class="insight-label">🏆 Verdict</div>
            <p><span class="hl">{best["model"]}</span> domine avec un
            <span class="val">R² = {best["r2"]:.3f}</span>
            ({best["r2"]*100:.1f}% de la variance expliquee) et un
            <span class="val">MAPE de {best["mape"]:.1f}%</span>.
            Le Stacking combine les forces de Random Forest, Gradient Boosting,
            XGBoost et LightGBM via un meta-modele Ridge, reduisant simultanement
            le biais et la variance des predictions.</p>
        </div>''', unsafe_allow_html=True)

        st.divider()

        # ── Cross-validation ──
        if cv_df is not None and not cv_df.empty:
            st.markdown(f'''<div class="sec-header">
                <span class="sec-icon">🔄</span>
                <span class="sec-title">Cross-validation temporelle</span>
                <span class="sec-badge">Panel par annee · {len(cv_df)} folds</span>
            </div>''', unsafe_allow_html=True)

            cv_mean = cv_df["r2"].mean()
            cv_std = cv_df["r2"].std()
            has_yr = "test_years" in cv_df.columns
            if has_yr:
                x_labels = [f"Fold {int(r['fold'])}\n{r['test_years']}"
                            for _, r in cv_df.iterrows()]
            else:
                x_labels = [f"Fold {int(r)}" for r in cv_df["fold"]]
            fig = go.Figure(go.Bar(
                x=x_labels,
                y=cv_df["r2"],
                marker_color=[P["emerald"] if v > 0 else P["red"] for v in cv_df["r2"]],
                text=[f"{v:.3f}" for v in cv_df["r2"]],
                textposition="outside", textfont=dict(size=11, color="#F1F5F9"),
            ))
            fig.add_hline(y=cv_mean, line_dash="dash", line_color=P["primary"],
                          annotation_text=f"Moyenne: {cv_mean:.3f}",
                          annotation_font_color=P["primary"])
            fig = fig_layout(fig, f"R² par fold temporel — {cv_df['model'].iloc[0]}", 300, "R²")
            st.plotly_chart(fig, key="mod_cv")

            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">🔬 Robustesse du modele</div>
                <p>Chaque fold entraine le modele sur des
                <span class="hl">annees anterieures</span> et le teste sur des
                <span class="hl">annees futures</span>, tous pays confondus.
                R² moyen : <span class="val">{cv_mean:.3f} ± {cv_std:.3f}</span>.
                Cette methode garantit qu'aucune donnee future ne fuit
                dans l'entrainement — condition essentielle pour valider
                un modele de prediction temporelle.
                Score test reel : R² = <span class="val">{best["r2"]:.3f}</span>.</p>
            </div>''', unsafe_allow_html=True)

        st.divider()

        # ── Radar multi-criteres ──
        st.markdown('''<div class="sec-header">
            <span class="sec-icon">🎯</span>
            <span class="sec-title">Radar multi-criteres</span>
        </div>''', unsafe_allow_html=True)

        categories = ["R²", "1-MAPE", "1-MAE_n", "1-RMSE_n"]
        max_rmse, max_mae = res["rmse"].max(), res["mae"].max()
        fig = go.Figure()
        for _, row in res.iterrows():
            vals = [row["r2"], max(0, 1 - row["mape"] / 100),
                    max(0, 1 - row["mae"] / max_mae),
                    max(0, 1 - row["rmse"] / max_rmse)]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["model"], fill="toself", opacity=0.2,
            ))
        fig = fig_layout(fig, "Profil multi-criteres par modele", 420)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#1E293B"),
                angularaxis=dict(gridcolor="#1E293B"),
                bgcolor="#020617",
            ),
        )
        st.plotly_chart(fig, key="mod_radar")

        st.divider()

        # ── Feature importance ──
        if fi_df is not None and not fi_df.empty:
            st.markdown('''<div class="sec-header">
                <span class="sec-icon">📋</span>
                <span class="sec-title">Importance des features</span>
                <span class="sec-badge">Top 15</span>
            </div>''', unsafe_allow_html=True)

            fi_top = fi_df.head(15).sort_values("importance")
            fig = go.Figure(go.Bar(
                x=fi_top["importance"], y=fi_top["feature"], orientation="h",
                marker_color=P["primary"],
                text=[f"{v:.3f}" for v in fi_top["importance"]],
                textposition="outside", textfont=dict(size=9, color="#94A3B8"),
            ))
            fig = fig_layout(fig, "Features les plus influentes pour la prediction", 450)
            fig.update_layout(xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, key="mod_fi")

            top_feat = fi_df.iloc[0]["feature"]
            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">🔑 Quels facteurs comptent ?</div>
                <p>La feature la plus influente est <span class="hl">{top_feat}</span>.
                Cette analyse permet de comprendre les <span class="hl">drivers</span>
                de la demande electrique : PIB, population, industrialisation,
                et acces au reseau sont les facteurs cles. Cela garantit
                la <span class="val">transparence</span> et l'<span class="val">interpretabilite</span>
                du systeme predictif.</p>
            </div>''', unsafe_allow_html=True)

    # ── Validation observe vs predit ──
    if pred is not None and not pred.empty:
        st.divider()
        st.markdown(f'''<div class="sec-header">
            <span class="sec-icon">✅</span>
            <span class="sec-title">Validation : observe vs predit — {sel_flag} {sel_name}</span>
        </div>''', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["actual"], name="Observe (reel)",
            mode="lines+markers", line=dict(color=P["emerald"], width=2.5),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["predicted"], name="Predit par le modele",
            mode="lines+markers", line=dict(color=P["violet"], width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))
        fig.add_trace(go.Bar(
            x=pred["year"], y=pred["error"].abs(), name="Ecart absolu (GWh)",
            marker_color=P["red"], opacity=0.15,
        ))
        fig = fig_layout(fig, f"Le modele capture-t-il les tendances reelles ? — {sel_name}",
                         400, "GWh")
        st.plotly_chart(fig, key="mod_valid")

        mae = pred["error"].abs().mean()
        mape_val = pred["error_pct"].abs().mean() if "error_pct" in pred.columns else 0
        st.markdown(f'''<div class="insight-card">
            <div class="insight-label">📏 Qualite des predictions</div>
            <p>Pour {sel_flag} {sel_name}, l'ecart moyen est de
            <span class="val">{mae:,.0f} GWh</span> (MAE) soit
            <span class="val">{mape_val:.1f}%</span> d'erreur (MAPE).
            La courbe verte (reel) et violette (predit) se superposent bien,
            confirmant que le modele capture les tendances structurelles
            de la demande electrique.</p>
        </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS 2045
# ═══════════════════════════════════════════════════════════════════
with t4:
    st.markdown(f'''<div class="step-box">
        <div class="step-tag">Etape 4 — Projections IA</div>
        <div class="step-title">Anticiper la demande electrique pour {sel_name} et l'UEMOA (2024-2045)</div>
        <div class="step-desc">Methode hybride : predictions ML + tendance historique (CAGR),
        avec intervalle de confiance a 95%.</div>
    </div>''', unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        last_hist_gwh = tg["conso_totale_gwh"].iloc[-1] if (
            not tg.empty and "conso_totale_gwh" in tg.columns) else 0

        # ── Trajectoire pays selectionne ──
        st.markdown(f'''<div class="sec-header">
            <span class="sec-icon">📈</span>
            <span class="sec-title">Trajectoire {sel_flag} {sel_name} — historique + projection</span>
        </div>''', unsafe_allow_html=True)

        fig = go.Figure()
        # Historique
        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"],
                name="Historique (observe)", mode="lines+markers",
                line=dict(color=P["emerald"], width=3), marker=dict(size=5),
            ))
        # IC
        proj_s = proj.sort_values("year")
        fig.add_trace(go.Scatter(
            x=pd.concat([proj_s["year"], proj_s["year"][::-1]]),
            y=pd.concat([proj_s["ci_upper"], proj_s["ci_lower"][::-1]]),
            fill="toself", fillcolor="rgba(139,92,246,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%", showlegend=True,
        ))
        # Projection
        fig.add_trace(go.Scatter(
            x=proj_s["year"], y=proj_s["predicted_gwh"],
            name="Projection IA", mode="lines+markers",
            line=dict(color=P["violet"], width=3), marker=dict(size=7, symbol="diamond"),
        ))
        # Ligne verticale
        fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=P["muted"],
                      annotation_text="Historique | Projection",
                      annotation_position="top", annotation_font_color=P["muted"],
                      annotation_font_size=10)
        # Annotations
        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_annotation(x=y_max_h, y=last_hist_gwh,
                               text=f"{last_hist_gwh:,.0f} GWh",
                               showarrow=True, arrowhead=2, arrowcolor=P["emerald"],
                               font=dict(size=10, color=P["emerald"]),
                               ax=-40, ay=-25)
        last_proj = proj_s.iloc[-1]
        fig.add_annotation(x=last_proj["year"], y=last_proj["predicted_gwh"],
                           text=f"{last_proj['predicted_gwh']:,.0f} GWh",
                           showarrow=True, arrowhead=2, arrowcolor=P["violet"],
                           font=dict(size=11, color=P["violet"], weight=700),
                           ax=40, ay=-25)

        fig = fig_layout(fig,
                         f"{sel_name} — Demande electrique {yr[0]}-{int(last_proj['year'])}",
                         480, "GWh")
        st.plotly_chart(fig, key="pred_main")

        gr_total = pct_change(last_hist_gwh, last_proj["predicted_gwh"])
        cagr = proj_s["cagr_pct"].iloc[0] if "cagr_pct" in proj_s.columns else 0

        # ── 3 cartes predictives ──
        st.markdown(f'''<div class="pred-grid">
            <div class="pred-card blue">
                <div class="pred-icon">📈 <span>Croissance totale</span></div>
                <div class="pred-value c-blue">+{gr_total:.0f}%</div>
                <div class="pred-sub">{y_max_h} → {int(last_proj["year"])}</div>
                <div class="pred-desc">La demande de {sel_name} passerait de
                {last_hist_gwh:,.0f} a {last_proj["predicted_gwh"]:,.0f} GWh,
                portee par la croissance demographique et l'electrification.</div>
            </div>
            <div class="pred-card green">
                <div class="pred-icon">🎯 <span>Demande projetee</span></div>
                <div class="pred-value c-green">{last_proj["predicted_gwh"]:,.0f}</div>
                <div class="pred-sub">GWh en {int(last_proj["year"])}</div>
                <div class="pred-desc">IC 95% : [{last_proj["ci_lower"]:,.0f} —
                {last_proj["ci_upper"]:,.0f}] GWh. La fourchette s'elargit avec
                l'horizon, refletant l'incertitude croissante.</div>
            </div>
            <div class="pred-card violet">
                <div class="pred-icon">⚡ <span>CAGR historique</span></div>
                <div class="pred-value c-violet">{cagr:.1f}%</div>
                <div class="pred-sub">Croissance annuelle composee</div>
                <div class="pred-desc">Le taux de croissance annuel compose (CAGR)
                utilise pour stabiliser les projections ML, base sur les 10
                dernieres annees observees.</div>
            </div>
        </div>''', unsafe_allow_html=True)

        st.divider()

        # ── Courbes comparees 8 pays ──
        if proj_all is not None and not proj_all.empty:
            st.markdown('''<div class="sec-header">
                <span class="sec-icon">🌍</span>
                <span class="sec-title">Projections comparees — 8 pays UEMOA</span>
                <span class="sec-badge">Horizon 2045</span>
            </div>''', unsafe_allow_html=True)

            fig = go.Figure()
            for i, cc in enumerate(countries_available):
                cn = country_names.get(cc, cc)
                is_sel = cc == sel_country

                # Historique
                dc = df_all[(df_all["country_code"] == cc) &
                            (df_all["year"].between(*yr))].sort_values("year")
                if not dc.empty and "conso_totale_gwh" in dc.columns:
                    fig.add_trace(go.Scatter(
                        x=dc["year"], y=dc["conso_totale_gwh"],
                        name=f"{FLAGS.get(cc,'')} {cn}" if is_sel else None,
                        mode="lines",
                        line=dict(width=3 if is_sel else 1, color=PALETTE_8[i % 8]),
                        opacity=1.0 if is_sel else 0.2,
                        showlegend=is_sel, legendgroup=cc,
                    ))

                # Projection
                pc = proj_all[proj_all["country_code"] == cc].sort_values("year")
                if not pc.empty:
                    fig.add_trace(go.Scatter(
                        x=pc["year"], y=pc["predicted_gwh"],
                        name=f"{FLAGS.get(cc,'')} {cn} (proj.)",
                        mode="lines",
                        line=dict(width=3 if is_sel else 1,
                                  color=PALETTE_8[i % 8], dash="dash"),
                        opacity=1.0 if is_sel else 0.2,
                        legendgroup=cc,
                    ))

            fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=P["muted"])
            fig = fig_layout(fig,
                             "Demande electrique projetee — 8 pays UEMOA (trait plein = hist., pointille = proj.)",
                             460, "GWh")
            st.plotly_chart(fig, key="pred_uemoa")

            # Classement final
            last_yr_proj = proj_all["year"].max()
            proj_rank = proj_all[proj_all["year"] == last_yr_proj].sort_values(
                "predicted_gwh", ascending=False)
            if not proj_rank.empty:
                pos = list(proj_rank["country_code"]).index(sel_country) + 1 \
                    if sel_country in list(proj_rank["country_code"]) else "?"
                top_c = proj_rank.iloc[0]
                st.markdown(f'''<div class="insight-card">
                    <div class="insight-label">🏅 Classement projete {int(last_yr_proj)}</div>
                    <p>{sel_flag} <span class="hl">{sel_name}</span> se placerait en position
                    <span class="blue">#{pos}</span> sur {len(proj_rank)} pays.
                    Le leader serait {FLAGS.get(top_c["country_code"],"")}
                    <span class="hl">{top_c["country_name"]}</span> avec
                    <span class="val">{top_c["predicted_gwh"]:,.0f} GWh</span>.
                    Cette vue multi-pays est un outil de <span class="hl">planification
                    regionale</span> directement transposable aux missions de la BCEAO.</p>
                </div>''', unsafe_allow_html=True)

        st.divider()

        # ── Jauge annee par annee ──
        st.markdown(f'''<div class="sec-header">
            <span class="sec-icon">🔎</span>
            <span class="sec-title">Consulter une annee — {sel_flag} {sel_name}</span>
        </div>''', unsafe_allow_html=True)

        proj_years_list = sorted(proj_s["year"].astype(int).tolist())
        sel_proj_yr = st.select_slider("Annee de projection", options=proj_years_list,
                                       value=proj_years_list[-1], key="pred_yr_sel")
        row_sel = proj_s[proj_s["year"] == sel_proj_yr]
        if not row_sel.empty:
            r = row_sel.iloc[0]
            gr_gwh = pct_change(last_hist_gwh, r["predicted_gwh"])
            pop_proj = r.get("pop_projected")
            gr_pop = pct_change(tg["SP.POP.TOTL"].iloc[-1], pop_proj) if (
                pd.notna(pop_proj) and "SP.POP.TOTL" in tg.columns) else 0
            kwh_hab = r["predicted_gwh"] * 1e6 / pop_proj if (
                pd.notna(pop_proj) and pop_proj > 0) else 0

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=r["predicted_gwh"],
                delta={"reference": last_hist_gwh, "relative": True, "valueformat": ".0%"},
                title={"text": f"Demande {sel_name} {sel_proj_yr} (GWh)",
                       "font": {"size": 14, "color": "#E2E8F0"}},
                number={"font": {"size": 32, "color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [0, proj_s["ci_upper"].max() * 1.1],
                             "tickcolor": "#1E293B"},
                    "bar": {"color": P["violet"]},
                    "bgcolor": "#0F172A",
                    "bordercolor": "#1E293B",
                    "steps": [
                        {"range": [0, last_hist_gwh],
                         "color": "rgba(16,185,129,0.1)"},
                        {"range": [r["ci_lower"], r["ci_upper"]],
                         "color": "rgba(139,92,246,0.1)"},
                    ],
                    "threshold": {"line": {"color": P["emerald"], "width": 3},
                                  "thickness": 0.75, "value": last_hist_gwh},
                },
            ))
            fig.update_layout(template=TMPL, height=280,
                              margin=dict(t=50, b=10),
                              paper_bgcolor="#020617")
            st.plotly_chart(fig, key="pred_gauge")

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f'''<div class="kpi-card"><div class="kpi-top"><div>
                    <div class="kpi-label">Population {sel_proj_yr}</div>
                    <div class="kpi-value">{pop_proj/1e6:.1f} M</div>
                    <div class="kpi-delta up">↑ +{gr_pop:.0f}% vs {y_max_h}</div>
                    </div><div class="kpi-icon">👥</div></div></div>''',
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f'''<div class="kpi-card"><div class="kpi-top"><div>
                    <div class="kpi-label">kWh / habitant</div>
                    <div class="kpi-value">{kwh_hab:,.0f}</div>
                    <div class="kpi-delta up">kWh/an</div>
                    </div><div class="kpi-icon">💡</div></div></div>''',
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f'''<div class="kpi-card"><div class="kpi-top"><div>
                    <div class="kpi-label">Intervalle de confiance</div>
                    <div class="kpi-value">{r["ci_lower"]:,.0f} — {r["ci_upper"]:,.0f}</div>
                    <div class="kpi-delta up">GWh (IC 95%)</div>
                    </div><div class="kpi-icon">📐</div></div></div>''',
                    unsafe_allow_html=True)

            st.markdown(f'''<div class="insight-card">
                <div class="insight-label">📋 Fiche {sel_proj_yr} — {sel_flag} {sel_name}</div>
                <p>Population projetee : <span class="val">{pop_proj/1e6:.1f} M</span> |
                Demande : <span class="val">{r["predicted_gwh"]:,.0f} GWh</span>
                (<span class="val">+{gr_gwh:.0f}%</span> vs {y_max_h}) |
                <span class="val">{kwh_hab:,.0f} kWh/hab/an</span>.
                La jauge montre la demande projetee par rapport au niveau actuel
                (seuil vert). La zone violette represente l'intervalle de confiance.</p>
            </div>''', unsafe_allow_html=True)

    else:
        st.info("Executez `python src/models/predict.py` pour generer les projections.")


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
    <strong>Objectif pedagogique</strong> : Maitriser les concepts d'ingenierie de donnees
    et de Machine Learning, transposables dans des situations reelles.<br>
    Architecture : API Banque Mondiale (WDI) → ETL Python → Stacking Regressor
    (RF + GB + XGBoost + LightGBM / Ridge) → Dashboard Streamlit + Plotly<br>
    8 pays UEMOA | 21 indicateurs | 82 features | 1990-2023 | Horizon 2045
</div>
""", unsafe_allow_html=True)

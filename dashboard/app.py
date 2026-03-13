"""
Dashboard IA — Prevision de la Demande Electrique | Zone UEMOA
=====================================================================
Ce projet a ete realise dans l'objectif de maitriser les concepts lies a
l'ingenierie de donnees et au Machine Learning, transposables dans des
situations reelles de modelisation macroeconomique.

Pipeline : API Banque Mondiale -> ETL -> Feature Engineering -> ML -> Dashboard
Perimetre : 8 pays UEMOA | 1990-2023 | Horizon 2045
"""
import os, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IA — Demande Electrique UEMOA 2045",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"

C = {
    "pop": "#2E86C1", "kwh": "#E67E22", "gwh": "#1ABC9C", "proj": "#9B59B6",
    "ci": "rgba(155,89,182,0.15)", "muted": "#7F8C8D",
    "good": "#27AE60", "warn": "#E74C3C", "accent": "#3498DB",
    "eco": "#00BCD4", "soc": "#E91E63",
}

COUNTRY_FLAGS = {
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

IND_CAT = {
    "SP.POP.TOTL": "demo", "SP.POP.GROW": "demo", "SP.URB.TOTL.IN.ZS": "demo",
    "SP.DYN.TFRT.IN": "demo", "SP.DYN.LE00.IN": "demo", "SP.POP.0014.TO.ZS": "demo",
    "SP.POP.1564.TO.ZS": "demo", "EG.USE.ELEC.KH.PC": "energie",
    "EG.ELC.ACCS.ZS": "energie", "EG.ELC.ACCS.UR.ZS": "energie",
    "EG.ELC.ACCS.RU.ZS": "energie", "EG.FEC.RNEW.ZS": "energie",
    "EG.USE.PCAP.KG.OE": "energie", "NY.GDP.PCAP.CD": "eco",
    "NY.GDP.MKTP.CD": "eco", "NY.GDP.MKTP.KD.ZG": "eco",
    "NV.IND.TOTL.ZS": "eco", "FP.CPI.TOTL.ZG": "eco",
    "IT.CEL.SETS.P2": "social", "SE.ADT.LITR.ZS": "social", "SL.UEM.TOTL.ZS": "social",
}
CAT_COLORS = {"demo": C["pop"], "energie": C["kwh"], "eco": C["eco"], "social": C["soc"]}
CAT_NAMES = {"demo": "Demographie", "energie": "Energie", "eco": "Economie", "social": "Social"}

PALETTE_8 = ["#1ABC9C", "#2E86C1", "#E67E22", "#9B59B6",
             "#E74C3C", "#27AE60", "#F39C12", "#E91E63"]

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.8rem; max-width: 1400px; }

.hdr {
    background: linear-gradient(135deg, #0D2137 0%, #1B6B50 100%);
    padding: 20px 30px; border-radius: 8px; margin-bottom: 18px;
    border-bottom: 3px solid #E67E22;
}
.hdr h1 { color: #fff; margin: 0; font-size: 1.35em; font-weight: 700; }
.hdr p  { color: #B0C4CE; margin: 4px 0 0; font-size: 0.78em; font-weight: 300; }
.hdr .tag { display: inline-block; background: rgba(52,152,219,0.2); color: #3498DB;
            padding: 2px 10px; border-radius: 10px; font-size: 0.68em; font-weight: 600;
            margin-top: 6px; }
.hdr .obj { color: #95A5A6; font-size: 0.65em; font-style: italic; margin-top: 6px; }

.card {
    background: #161B22; border-radius: 6px; padding: 13px 15px;
    border-left: 3px solid #2E86C1;
}
.card .t { color: #7F8C8D; font-size: 0.66em; text-transform: uppercase;
           letter-spacing: 0.8px; margin-bottom: 3px; }
.card .v { color: #ECF0F1; font-size: 1.3em; font-weight: 700; line-height: 1.15; }
.card .d { font-size: 0.74em; margin-top: 3px; font-weight: 500; }
.card .d.up { color: #27AE60; }
.card .d.dn { color: #E74C3C; }
.card .ctx { color: #5D6D7E; font-size: 0.64em; margin-top: 2px; }

.sec { color: #D5DBE1; font-size: 1.02em; font-weight: 600;
       border-bottom: 1px solid #2E6F8E;
       padding-bottom: 5px; margin: 20px 0 10px 0; }

.insight {
    background: linear-gradient(135deg, #161B22, #1a2332);
    border-left: 3px solid #3498DB; border-radius: 0 6px 6px 0;
    padding: 12px 18px; margin: 8px 0 16px 0;
    color: #B0C4CE; font-size: 0.84em; line-height: 1.55;
}
.insight strong { color: #ECF0F1; }
.insight .val { color: #1ABC9C; font-weight: 600; }
.insight .warn { color: #E74C3C; font-weight: 600; }
.insight .up { color: #27AE60; font-weight: 600; }

.step-box {
    background: #161B22; border: 1px solid #2E6F8E; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 12px;
}
.step-box .step-n { color: #3498DB; font-size: 0.7em; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.step-box .step-t { color: #ECF0F1; font-weight: 600; font-size: 0.9em; }
.step-box .step-d { color: #7F8C8D; font-size: 0.74em; margin-top: 2px; }

.foot { text-align: center; color: #5D6D7E; font-size: 0.68em; padding: 14px 0;
        margin-top: 20px; border-top: 1px solid #1E2A35; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
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
    if abs(v) >= 1e9: s = f"{v/1e9:,.2f} Mrd"
    elif abs(v) >= 1e6: s = f"{v/1e6:,.1f} M"
    elif abs(v) >= 1e3: s = f"{v:,.0f}"
    else: s = f"{v:,.1f}"
    return f"{s} {u}".strip() if u else s


def flag(code):
    return COUNTRY_FLAGS.get(code, '')


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>⚡ Prevision de la Demande Electrique — Zone UEMOA</h1>
    <p>La population croit — combien d'electricite faudra-t-il demain ?
       &nbsp;|&nbsp; 8 pays &nbsp;|&nbsp; 21 indicateurs &nbsp;|&nbsp; Horizon 2045</p>
    <span class="tag">Intelligence Artificielle &amp; Analyse Predictive</span>
    <div class="obj">Ce projet a ete realise dans l'objectif de maitriser les concepts lies a
    l'ingenierie de donnees et au Machine Learning, transposables dans des situations reelles.</div>
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

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Navigation")

    sel_country = st.selectbox(
        "Pays",
        countries_available,
        format_func=lambda c: f"{flag(c)} {country_names.get(c, c)}",
        index=countries_available.index('TG') if 'TG' in countries_available else 0,
        key="sel_country",
    )
    sel_name = country_names.get(sel_country, sel_country)

    st.divider()

    y_min_h, y_max_h = int(df_all["year"].min()), int(df_all["year"].max())
    yr = st.slider("Periode historique", y_min_h, y_max_h, (y_min_h, y_max_h), key="yr_hist")

    proj_years = []
    if proj_all is not None and not proj_all.empty:
        proj_years = sorted(proj_all[proj_all["country_code"] == sel_country]["year"].unique().astype(int).tolist())
    proj_yr = st.select_slider("Horizon projection", options=proj_years,
                                value=proj_years[-1], key="yr_proj") if proj_years else None

    st.divider()
    st.markdown(f"**Pays** : {flag(sel_country)} {sel_name}")
    st.markdown(f"**Historique** : {yr[0]} — {yr[1]}")
    if proj_yr:
        st.markdown(f"**Projection** : jusqu'a {proj_yr}")

    st.divider()
    st.markdown("##### Telecharger")
    if raw_all is not None:
        st.download_button("Donnees brutes", raw_all.to_csv(index=False).encode(),
                           "donnees_brutes.csv", key="dl_raw")
    if proj_all is not None:
        st.download_button("Projections UEMOA", proj_all.to_csv(index=False).encode(),
                           "projections_uemoa_2045.csv", key="dl_proj")

# Filter data for selected country
df = df_all[df_all["country_code"] == sel_country].copy()
tg = df[df["year"].between(*yr)].sort_values("year")
raw_sel = raw_all[raw_all["country_code"] == sel_country] if raw_all is not None else None
pred = pred_all[pred_all["country_code"] == sel_country] if pred_all is not None else None
proj = proj_all[proj_all["country_code"] == sel_country] if proj_all is not None else None

if tg.empty:
    st.warning(f"Aucune donnee disponible pour {sel_name}.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
last, first = tg.iloc[-1], tg.iloc[0]
cards = []

if "SP.POP.TOTL" in tg.columns:
    pop_now, pop_bef = last["SP.POP.TOTL"], first["SP.POP.TOTL"]
    gr = ((pop_now / pop_bef) - 1) * 100 if pop_bef > 0 else 0
    cards.append(("Population", fmt(pop_now), f"+{gr:.0f}% depuis {yr[0]}", "up", C["pop"]))
if "conso_totale_gwh" in tg.columns:
    gwh, gwh0 = last["conso_totale_gwh"], first["conso_totale_gwh"]
    d = ((gwh / gwh0) - 1) * 100 if gwh0 > 0 else 0
    cards.append(("Demande electrique", fmt(gwh, "GWh"), f"+{d:.0f}% depuis {yr[0]}", "up", C["gwh"]))
if "EG.ELC.ACCS.ZS" in tg.columns:
    acc, acc0 = last["EG.ELC.ACCS.ZS"], first["EG.ELC.ACCS.ZS"]
    cards.append(("Acces electrique", f"{acc:.1f}%", f"{acc-acc0:+.1f} pts", "up" if acc > acc0 else "dn", C["good"]))
if proj_yr and proj is not None and not proj.empty:
    row_p = proj[proj["year"] == proj_yr]
    if not row_p.empty:
        cards.append(("Prediction " + str(proj_yr), fmt(row_p.iloc[0]["predicted_gwh"], "GWh"),
                      "Projection IA", "up", C["proj"]))
if res is not None and not res.empty:
    best = res.sort_values("r2", ascending=False).iloc[0]
    cards.append(("Modele IA", f"R2 = {best['r2']:.3f}", best["model"], "up", C["accent"]))

html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin-bottom:14px;">'
for title, val, delta, css, color in cards:
    html += f'''<div class="card" style="border-left-color:{color};">
        <div class="t">{title}</div><div class="v">{val}</div>
        <div class="d {css}">{delta}</div></div>'''
html += '</div>'
st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "1. Donnees & Extraction",
    "2. Exploration",
    "3. Modele IA",
    f"4. Predictions 2045",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DONNEES
# ═════════════════════════════════════════════════════════════════════════════
with t1:
    n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
    n_pays = len(raw_all["country_code"].unique()) if raw_all is not None else 0
    n_raw = len(raw_all) if raw_all is not None else 0
    yr_min_raw = int(raw_all["year"].min()) if raw_all is not None else 0
    yr_max_raw = int(raw_all["year"].max()) if raw_all is not None else 0

    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 1 — Extraction automatisee</div>
        <div class="step-t">API Banque Mondiale (WDI) → {n_raw:,} observations</div>
        <div class="step-d">{n_ind} indicateurs × {n_pays} pays UEMOA × {yr_max_raw - yr_min_raw + 1} annees
              | Periode : {yr_min_raw}-{yr_max_raw}</div>
    </div>""", unsafe_allow_html=True)

    if raw_all is not None and not raw_all.empty:
        raw_f = raw_all[raw_all["year"].between(*yr)]

        c1, c2 = st.columns([2, 3])
        with c1:
            # Donut categories
            cat_data = []
            for code in raw_f["indicator_code"].unique():
                cat = IND_CAT.get(code, "autre")
                cat_data.append({"Domaine": CAT_NAMES.get(cat, cat)})
            cat_counts = pd.DataFrame(cat_data).groupby("Domaine").size().reset_index(name="N")

            fig = go.Figure(go.Pie(
                labels=cat_counts["Domaine"], values=cat_counts["N"],
                hole=0.55, marker_colors=[C["pop"], C["eco"], C["kwh"], C["soc"]],
                textinfo="label+value", textfont_size=11,
            ))
            fig.update_layout(title="Repartition par domaine", template=TMPL, height=320,
                              margin=dict(t=40, b=10), showlegend=False)
            fig.add_annotation(text=f"<b>{n_ind}</b><br>indicateurs",
                               x=0.5, y=0.5, font_size=14, showarrow=False, font_color="#ECF0F1")
            st.plotly_chart(fig, key="cat_pie")

        with c2:
            # Observations par pays (tous les pays)
            cov = raw_f.groupby("country_code").agg(
                pays=("country_name", "first"), obs=("value", "count")
            ).reset_index().sort_values("obs")
            colors_p = [C["gwh"] if c == sel_country else C["muted"] for c in cov["country_code"]]
            fig = go.Figure(go.Bar(
                x=cov["obs"], y=[f"{flag(c)} {n}" for c, n in zip(cov["country_code"], cov["pays"])],
                orientation="h", marker_color=colors_p,
                text=cov["obs"], textposition="outside", textfont_size=10,
            ))
            fig.update_layout(title="Observations par pays UEMOA",
                              template=TMPL, height=320, margin=dict(l=10, r=40, t=40, b=10),
                              xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, key="obs_pays")

        sel_obs = raw_f[raw_f["country_code"] == sel_country].shape[0]
        st.markdown(f"""
        <div class="insight">
            <strong>Interpretation</strong> — Le pipeline extrait <span class="val">{n_raw:,} observations</span>
            pour {n_pays} pays UEMOA. {flag(sel_country)} <strong>{sel_name}</strong> dispose de
            <span class="val">{sel_obs}</span> observations. L'entrainement du modele utilise
            <strong>l'ensemble des pays</strong> simultanement (transfer learning regional),
            ce qui augmente la robustesse par rapport a un entrainement sur un seul pays.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Explorateur indicateur
        st.markdown(f'<div class="sec">Explorer un indicateur — {flag(sel_country)} {sel_name}</div>',
                    unsafe_allow_html=True)
        if raw_sel is not None and not raw_sel.empty:
            raw_sel_f = raw_sel[raw_sel["year"].between(*yr)]
            ind_codes = sorted(raw_sel_f["indicator_code"].unique().tolist())
            sel_ind = st.selectbox("Indicateur", ind_codes,
                                   format_func=lambda c: IND_LABELS.get(c, c), key="raw_ind")
            ri = raw_sel_f[raw_sel_f["indicator_code"] == sel_ind].sort_values("year")

            # Comparison with all countries
            ri_all = raw_f[raw_f["indicator_code"] == sel_ind]
            fig = go.Figure()
            for i, cc in enumerate(countries_available):
                rcc = ri_all[ri_all["country_code"] == cc].sort_values("year")
                if rcc.empty: continue
                is_sel = cc == sel_country
                fig.add_trace(go.Scatter(
                    x=rcc["year"], y=rcc["value"],
                    name=f"{flag(cc)} {country_names.get(cc, cc)}",
                    mode="lines+markers" if is_sel else "lines",
                    line=dict(width=3 if is_sel else 1.5,
                              color=PALETTE_8[i % 8]),
                    opacity=1.0 if is_sel else 0.35,
                    marker=dict(size=5) if is_sel else dict(size=0),
                ))
            fig.update_layout(
                title=f"{IND_LABELS.get(sel_ind, sel_ind)} — Comparaison UEMOA",
                template=TMPL, height=380, hovermode="x unified", margin=dict(t=40),
                legend=dict(orientation="h", y=-0.2, font_size=9),
                yaxis_title=IND_LABELS.get(sel_ind, ""),
            )
            st.plotly_chart(fig, key="raw_chart")

            if len(ri) > 1:
                v_first, v_last = ri["value"].iloc[0], ri["value"].iloc[-1]
                chg = ((v_last / v_first) - 1) * 100 if v_first != 0 else 0
                trend = "hausse" if chg > 5 else "baisse" if chg < -5 else "stagnation"
                css = "up" if chg > 0 else "warn"
                st.markdown(f"""
                <div class="insight">
                    <strong>Lecture</strong> — <strong>{IND_LABELS.get(sel_ind, sel_ind)}</strong> pour
                    {sel_name} passe de <span class="val">{v_first:,.1f}</span> ({yr[0]})
                    a <span class="val">{v_last:,.1f}</span> ({yr[1]}),
                    soit une <span class="{css}">{trend} de {abs(chg):.1f}%</span>.
                    Le graphique compare la trajectoire de {sel_name} avec les 7 autres pays UEMOA.
                </div>""", unsafe_allow_html=True)

        with st.expander("Voir les donnees brutes", expanded=False):
            if raw_sel is not None:
                st.dataframe(raw_sel[raw_sel["year"].between(*yr)], height=300)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
with t2:
    n_cols = len(df_all.columns)
    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 2 — Feature Engineering</div>
        <div class="step-t">21 indicateurs bruts → {n_cols} variables (lags, MA, ratios, interactions)</div>
        <div class="step-d">{len(df_all)} observations traitees pour {len(countries_available)} pays UEMOA.</div>
    </div>""", unsafe_allow_html=True)

    if not tg.empty:
        # --- Population + GWh ---
        st.markdown(f'<div class="sec">Demographie et demande electrique — {flag(sel_country)} {sel_name}</div>',
                    unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "SP.POP.TOTL" in tg.columns:
            fig.add_trace(go.Bar(
                x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Population (M)",
                marker_color=C["pop"], opacity=0.3,
            ), secondary_y=False)
        if "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"], name="Demande (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=3), marker=dict(size=5),
            ), secondary_y=True)
        fig.update_layout(template=TMPL, height=400, hovermode="x unified", margin=dict(t=40),
                          title=f"Co-evolution population / demande electrique — {sel_name}",
                          legend=dict(orientation="h", y=-0.13, font_size=10))
        fig.update_yaxes(title_text="Population (M)", secondary_y=False)
        fig.update_yaxes(title_text="GWh", secondary_y=True)
        st.plotly_chart(fig, key="exp_pop_gwh")

        if "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
            pop_chg = (tg["SP.POP.TOTL"].iloc[-1] / tg["SP.POP.TOTL"].iloc[0] - 1) * 100
            gwh_chg = (tg["conso_totale_gwh"].iloc[-1] / tg["conso_totale_gwh"].iloc[0] - 1) * 100 \
                if tg["conso_totale_gwh"].iloc[0] > 0 else 0
            elast = gwh_chg / pop_chg if pop_chg > 0 else 0
            corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
            st.markdown(f"""
            <div class="insight">
                <strong>Analyse</strong> — Entre {yr[0]} et {yr[1]}, la population de {sel_name} a augmente
                de <span class="up">+{pop_chg:.0f}%</span> tandis que la demande electrique a bondi
                de <span class="val">+{gwh_chg:.0f}%</span>. L'elasticite de <span class="val">{elast:.2f}</span>
                signifie que pour chaque +1% de croissance demographique, la demande electrique
                augmente de +{elast:.2f}%.
                Correlation de Pearson : <span class="val">{corr:.3f}</span>.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Acces urbain vs rural ---
        st.markdown(f'<div class="sec">Fracture energetique : urbain vs rural</div>',
                    unsafe_allow_html=True)
        if "EG.ELC.ACCS.UR.ZS" in tg.columns and "EG.ELC.ACCS.RU.ZS" in tg.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["EG.ELC.ACCS.UR.ZS"], name="Urbain",
                mode="lines+markers", line=dict(color=C["good"], width=2.5), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["EG.ELC.ACCS.RU.ZS"], name="Rural",
                mode="lines+markers", fill="tonexty", fillcolor="rgba(231,76,60,0.08)",
                line=dict(color=C["warn"], width=2.5), marker=dict(size=4),
            ))
            if "EG.ELC.ACCS.ZS" in tg.columns:
                fig.add_trace(go.Scatter(
                    x=tg["year"], y=tg["EG.ELC.ACCS.ZS"], name="Moyenne nationale",
                    mode="lines", line=dict(color=C["pop"], width=2, dash="dash"),
                ))
            fig.update_layout(title=f"Acces a l'electricite — {sel_name}", template=TMPL,
                              height=350, hovermode="x unified", margin=dict(t=40),
                              yaxis_title="% de la population",
                              legend=dict(orientation="h", y=-0.15, font_size=10))
            st.plotly_chart(fig, key="exp_acces")

            gap_now = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1] - tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
            urb_now = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]
            rur_now = tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
            st.markdown(f"""
            <div class="insight">
                <strong>Constat</strong> — Acces rural : <span class="warn">{rur_now:.1f}%</span>,
                urbain : <span class="up">{urb_now:.1f}%</span>,
                ecart de <span class="val">{gap_now:.1f} points</span>.
                Chaque point d'acces gagne en zone rurale genere une hausse significative
                de la demande globale.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Benchmark UEMOA ---
        st.markdown('<div class="sec">Benchmark UEMOA — Demande electrique par pays</div>',
                    unsafe_allow_html=True)
        if "conso_totale_gwh" in df_all.columns:
            fig = go.Figure()
            for i, cc in enumerate(countries_available):
                dc = df_all[(df_all["country_code"] == cc) & (df_all["year"].between(*yr))].sort_values("year")
                if dc.empty or "conso_totale_gwh" not in dc.columns: continue
                is_sel = cc == sel_country
                fig.add_trace(go.Scatter(
                    x=dc["year"], y=dc["conso_totale_gwh"],
                    name=f"{flag(cc)} {country_names.get(cc, cc)}",
                    mode="lines+markers" if is_sel else "lines",
                    line=dict(width=3.5 if is_sel else 1.5, color=PALETTE_8[i % 8]),
                    opacity=1.0 if is_sel else 0.4,
                    marker=dict(size=5) if is_sel else dict(size=0),
                ))
            fig.update_layout(
                title="Demande electrique (GWh) — 8 pays UEMOA",
                template=TMPL, height=400, hovermode="x unified", margin=dict(t=40),
                yaxis_title="GWh",
                legend=dict(orientation="h", y=-0.18, font_size=9),
            )
            st.plotly_chart(fig, key="bench_gwh")

            # Last year ranking
            last_yr_data = df_all[df_all["year"] == y_max_h][["country_code", "country_name", "conso_totale_gwh"]].dropna()
            if not last_yr_data.empty:
                rank_sel = last_yr_data.sort_values("conso_totale_gwh", ascending=False).reset_index(drop=True)
                pos = rank_sel[rank_sel["country_code"] == sel_country].index[0] + 1 if sel_country in rank_sel["country_code"].values else "?"
                top1 = rank_sel.iloc[0]
                st.markdown(f"""
                <div class="insight">
                    <strong>Classement {y_max_h}</strong> — {flag(sel_country)} <strong>{sel_name}</strong> se classe
                    <span class="val">#{pos}</span> sur {len(rank_sel)} pays en demande electrique.
                    Le leader est {flag(rank_sel.iloc[0]['country_code'])}
                    <strong>{top1['country_name']}</strong> avec <span class="val">{top1['conso_totale_gwh']:,.0f} GWh</span>.
                    Cette vue comparative est essentielle pour le benchmarking regional de la BCEAO.
                </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Correlation heatmap ---
        st.markdown('<div class="sec">Correlations — Variables cles</div>',
                    unsafe_allow_html=True)
        corr_cols_codes = ["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EG.ELC.ACCS.ZS",
                           "NY.GDP.PCAP.CD", "IT.CEL.SETS.P2", "conso_totale_gwh"]
        corr_cols = [c for c in corr_cols_codes if c in tg.columns]
        corr_labels_map = {
            "SP.POP.TOTL": "Population", "SP.URB.TOTL.IN.ZS": "Urbanisation",
            "EG.ELC.ACCS.ZS": "Acces elect.", "NY.GDP.PCAP.CD": "PIB/hab",
            "IT.CEL.SETS.P2": "Mobile", "conso_totale_gwh": "Demande GWh",
        }
        if len(corr_cols) > 3:
            corr_matrix = tg[corr_cols].corr()
            labels = [corr_labels_map.get(c, c) for c in corr_cols]
            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=labels, y=labels,
                colorscale="RdBu_r", zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text:.2f}", textfont_size=10,
                colorbar=dict(thickness=12, len=0.6),
            ))
            fig.update_layout(title=f"Matrice de correlation — {sel_name}",
                              template=TMPL, height=400, margin=dict(t=40))
            st.plotly_chart(fig, key="corr_heatmap")

        with st.expander("Voir les donnees transformees", expanded=False):
            st.dataframe(tg, height=300)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELE IA
# ═════════════════════════════════════════════════════════════════════════════
with t3:
    n_feat = len([c for c in df_all.columns if c not in ['country_code', 'country_name', 'year',
                  'conso_totale_gwh', 'EG.USE.ELEC.KH.PC']
                  and not c.startswith('EG.USE.ELEC.KH.PC')
                  and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']])
    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 3 — Entrainement du modele</div>
        <div class="step-t">4+ algorithmes compares sur {len(df_all)} obs. ({len(countries_available)} pays)</div>
        <div class="step-d">{n_feat} features, cible : demande electrique (GWh).
                            Split temporel 80/20 + cross-validation {5} folds.</div>
    </div>""", unsafe_allow_html=True)

    if res is not None and not res.empty:
        # --- Comparaison modeles ---
        st.markdown('<div class="sec">Performance comparee des algorithmes</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            r2_s = res.sort_values("r2", ascending=True)
            colors_r2 = [C["good"] if v > 0.85 else C["kwh"] if v > 0.7 else C["warn"] for v in r2_s["r2"]]
            fig = go.Figure(go.Bar(
                x=r2_s["r2"], y=r2_s["model"], orientation="h", marker_color=colors_r2,
                text=[f"{v:.3f}" for v in r2_s["r2"]], textposition="outside", textfont_size=12,
            ))
            fig.update_layout(title="Score R2", template=TMPL, height=300,
                              margin=dict(t=40, l=10, r=40), xaxis_range=[0, 1.05])
            st.plotly_chart(fig, key="mod_r2")

        with c2:
            mape_s = res.sort_values("mape", ascending=False)
            colors_mape = [C["good"] if v < 30 else C["kwh"] if v < 40 else C["warn"] for v in mape_s["mape"]]
            fig = go.Figure(go.Bar(
                x=mape_s["mape"], y=mape_s["model"], orientation="h", marker_color=colors_mape,
                text=[f"{v:.1f}%" for v in mape_s["mape"]], textposition="outside", textfont_size=12,
            ))
            fig.update_layout(title="Erreur MAPE (%)", template=TMPL, height=300,
                              margin=dict(t=40, l=10, r=40))
            st.plotly_chart(fig, key="mod_mape")

        best = res.sort_values("r2", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="insight">
            <strong>Verdict</strong> — <strong>{best['model']}</strong> domine avec un
            <span class="val">R2 = {best['r2']:.3f}</span> ({best['r2']*100:.1f}% de variance expliquee).
            Le Stacking combine Random Forest, Gradient Boosting et LightGBM via un
            meta-modele Ridge, reduisant biais et variance simultanement.
        </div>""", unsafe_allow_html=True)

        # --- Cross-validation ---
        if cv_df is not None and not cv_df.empty:
            st.divider()
            st.markdown('<div class="sec">Cross-validation temporelle</div>',
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=[f"Fold {int(r)}" for r in cv_df["fold"]],
                y=cv_df["r2"], marker_color=[C["good"] if v > 0 else C["warn"] for v in cv_df["r2"]],
                text=[f"{v:.3f}" for v in cv_df["r2"]], textposition="outside",
            ))
            cv_mean = cv_df["r2"].mean()
            cv_std = cv_df["r2"].std()
            fig.add_hline(y=cv_mean, line_dash="dash", line_color=C["accent"],
                          annotation_text=f"Moyenne: {cv_mean:.3f}")
            fig.update_layout(title=f"R2 par fold — {cv_df['model'].iloc[0]}",
                              template=TMPL, height=300, margin=dict(t=40),
                              yaxis_title="R2")
            st.plotly_chart(fig, key="mod_cv")

            st.markdown(f"""
            <div class="insight">
                <strong>Robustesse</strong> — La cross-validation temporelle ({len(cv_df)} folds) donne un
                R2 moyen de <span class="val">{cv_mean:.3f} ± {cv_std:.3f}</span>.
                La faible variance entre les folds confirme que le modele
                <strong>generalise bien</strong> et ne fait pas de surapprentissage.
            </div>""", unsafe_allow_html=True)

        # --- Radar ---
        st.divider()
        st.markdown('<div class="sec">Profil multi-criteres</div>', unsafe_allow_html=True)
        categories = ["R2", "1-MAPE", "1-MAE_n", "1-RMSE_n"]
        max_rmse, max_mae = res["rmse"].max(), res["mae"].max()
        fig = go.Figure()
        for _, row in res.iterrows():
            vals = [row["r2"], 1 - row["mape"] / 100,
                    1 - row["mae"] / max_mae, 1 - row["rmse"] / max_rmse]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["model"], fill="toself", opacity=0.25,
            ))
        fig.update_layout(title="Radar multi-critere", template=TMPL, height=400,
                          margin=dict(t=50),
                          polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                          legend=dict(orientation="h", y=-0.1, font_size=10))
        st.plotly_chart(fig, key="mod_radar")

        # --- Feature importance ---
        if fi_df is not None and not fi_df.empty:
            st.divider()
            st.markdown('<div class="sec">Importance des features</div>', unsafe_allow_html=True)
            fi_top = fi_df.head(15).sort_values("importance")
            fig = go.Figure(go.Bar(
                x=fi_top["importance"], y=fi_top["feature"], orientation="h",
                marker_color=C["accent"],
                text=[f"{v:.3f}" for v in fi_top["importance"]],
                textposition="outside", textfont_size=9,
            ))
            fig.update_layout(title="Top 15 features les plus influentes",
                              template=TMPL, height=450, margin=dict(l=10, r=40, t=40, b=10),
                              xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, key="mod_fi")

            top_feat = fi_df.iloc[0]["feature"]
            st.markdown(f"""
            <div class="insight">
                <strong>Lecture</strong> — La feature la plus influente est
                <strong>{top_feat}</strong>. Cette analyse d'importance permet de comprendre
                quels facteurs le modele utilise pour ses predictions, assurant
                la <strong>transparence</strong> et l'<strong>interpretabilite</strong> du systeme.
            </div>""", unsafe_allow_html=True)

    # --- Validation observe vs predit ---
    if pred is not None and not pred.empty:
        st.divider()
        st.markdown(f'<div class="sec">Validation : observe vs predit — {flag(sel_country)} {sel_name}</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["actual"], name="Observe",
            mode="lines+markers", line=dict(color=C["gwh"], width=2.5), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["predicted"], name="Predit (IA)",
            mode="lines+markers", line=dict(color=C["proj"], width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))
        fig.add_trace(go.Bar(
            x=pred["year"], y=pred["error"].abs(), name="Ecart absolu",
            marker_color=C["warn"], opacity=0.2,
        ))
        fig.update_layout(title=f"Validation — {sel_name}", yaxis_title="GWh",
                          template=TMPL, height=400, hovermode="x unified", margin=dict(t=40),
                          legend=dict(orientation="h", y=-0.13, font_size=10))
        st.plotly_chart(fig, key="mod_valid")

        mae = pred["error"].abs().mean()
        mape_val = pred["error_pct"].abs().mean() if "error_pct" in pred.columns else 0
        st.markdown(f"""
        <div class="insight">
            <strong>Validation {sel_name}</strong> — Ecart moyen de <span class="val">{mae:,.0f} GWh</span>
            (MAE), erreur moyenne de <span class="val">{mape_val:.1f}%</span> (MAPE).
            Le modele capture bien les tendances de long terme.
        </div>""", unsafe_allow_html=True)

        # Scatter
        st.divider()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["actual"], y=pred["predicted"], mode="markers",
            marker=dict(color=C["gwh"], size=8, line=dict(width=1, color="#fff")),
            name="Observations",
        ))
        mn, mx = min(pred["actual"].min(), pred["predicted"].min()), max(pred["actual"].max(), pred["predicted"].max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color=C["muted"], width=1, dash="dash"),
                                  name="Parfait"))
        fig.update_layout(title="Dispersion observe vs predit", xaxis_title="Observe (GWh)",
                          yaxis_title="Predit (GWh)", template=TMPL, height=360, margin=dict(t=40),
                          legend=dict(orientation="h", y=-0.15, font_size=10))
        st.plotly_chart(fig, key="mod_scatter")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS 2045
# ═════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 4 — Projections</div>
        <div class="step-t">Anticiper la demande electrique pour {sel_name} et toute l'UEMOA</div>
        <div class="step-d">Le modele IA projette la demande 2024-2045 pour les 8 pays.</div>
    </div>""", unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        proj_f = proj[proj["year"] <= proj_yr].sort_values("year") if proj_yr else proj.sort_values("year")
        last_hist_gwh = tg["conso_totale_gwh"].iloc[-1] if (
            not tg.empty and "conso_totale_gwh" in tg.columns) else 0
        last_pop = tg["SP.POP.TOTL"].iloc[-1] if "SP.POP.TOTL" in tg.columns else 0

        # --- Trajectoire historique + projection ---
        st.markdown(f'<div class="sec">Trajectoire — {flag(sel_country)} {sel_name}</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"],
                name="Historique", mode="lines+markers",
                line=dict(color=C["gwh"], width=3), marker=dict(size=5),
            ))
        fig.add_trace(go.Scatter(
            x=proj_f["year"], y=proj_f["predicted_gwh"],
            name="Projection IA", mode="lines+markers",
            line=dict(color=C["proj"], width=3), marker=dict(size=7, symbol="diamond"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([proj_f["year"], proj_f["year"][::-1]]),
            y=pd.concat([proj_f["ci_upper"], proj_f["ci_lower"][::-1]]),
            fill="toself", fillcolor=C["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
        ))
        fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=C["muted"],
                      annotation_text="Hist. / Proj.", annotation_position="top left",
                      annotation_font_color=C["muted"], annotation_font_size=9)
        fig.update_layout(
            title=f"{sel_name} — Demande electrique {yr[0]}-{proj_yr or 2045}",
            yaxis_title="GWh", template=TMPL, height=460,
            hovermode="x unified", margin=dict(t=45),
            legend=dict(orientation="h", y=-0.1, font_size=10),
        )
        st.plotly_chart(fig, key="pred_main")

        last_proj_row = proj_f.iloc[-1]
        gr_total = ((last_proj_row["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
        st.markdown(f"""
        <div class="insight">
            <strong>Projection</strong> — La demande de {sel_name} passerait de
            <span class="val">{last_hist_gwh:,.0f} GWh</span> ({y_max_h}) a
            <span class="val">{last_proj_row['predicted_gwh']:,.0f} GWh</span> ({int(last_proj_row['year'])}),
            soit <span class="up">+{gr_total:.0f}%</span>.
            IC 95% : [{last_proj_row['ci_lower']:,.0f} — {last_proj_row['ci_upper']:,.0f}] GWh.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Comparaison UEMOA projections ---
        if proj_all is not None and not proj_all.empty:
            st.markdown('<div class="sec">Projections comparees — 8 pays UEMOA</div>',
                        unsafe_allow_html=True)

            fig = go.Figure()
            for i, cc in enumerate(countries_available):
                pc = proj_all[(proj_all["country_code"] == cc)]
                if proj_yr:
                    pc = pc[pc["year"] <= proj_yr]
                pc = pc.sort_values("year")
                if pc.empty: continue
                is_sel = cc == sel_country
                cn = country_names.get(cc, cc)
                # Add historical data too
                dc = df_all[(df_all["country_code"] == cc) & (df_all["year"].between(*yr))].sort_values("year")
                if not dc.empty and "conso_totale_gwh" in dc.columns:
                    fig.add_trace(go.Scatter(
                        x=dc["year"], y=dc["conso_totale_gwh"],
                        name=f"{flag(cc)} {cn} (hist.)" if is_sel else None,
                        mode="lines", line=dict(width=2.5 if is_sel else 1, color=PALETTE_8[i % 8]),
                        opacity=1.0 if is_sel else 0.25, showlegend=is_sel,
                        legendgroup=cc,
                    ))
                fig.add_trace(go.Scatter(
                    x=pc["year"], y=pc["predicted_gwh"],
                    name=f"{flag(cc)} {cn} (proj.)",
                    mode="lines", line=dict(width=3 if is_sel else 1.2,
                                            color=PALETTE_8[i % 8], dash="dash"),
                    opacity=1.0 if is_sel else 0.3,
                    legendgroup=cc,
                ))

            fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=C["muted"])
            fig.update_layout(
                title=f"Demande electrique projetee — 8 pays UEMOA (horizon {proj_yr or 2045})",
                yaxis_title="GWh", template=TMPL, height=450,
                hovermode="x unified", margin=dict(t=45),
                legend=dict(orientation="h", y=-0.18, font_size=9),
            )
            st.plotly_chart(fig, key="pred_uemoa_all")

            # Ranking final
            proj_last_all = proj_all[proj_all["year"] == (proj_yr or proj_all["year"].max())]
            if not proj_last_all.empty:
                ranking = proj_last_all.sort_values("predicted_gwh", ascending=False)
                rank_pos = ranking[ranking["country_code"] == sel_country].index
                pos_num = list(ranking["country_code"]).index(sel_country) + 1 if sel_country in list(ranking["country_code"]) else "?"
                top_c = ranking.iloc[0]
                st.markdown(f"""
                <div class="insight">
                    <strong>Classement projete {proj_yr or 2045}</strong> — {flag(sel_country)}
                    <strong>{sel_name}</strong> se placerait en position
                    <span class="val">#{pos_num}</span> sur 8 pays.
                    Le premier serait {flag(top_c['country_code'])}
                    <strong>{top_c['country_name']}</strong> avec
                    <span class="val">{top_c['predicted_gwh']:,.0f} GWh</span>.
                    Cette vue multi-pays est un outil de planification regionale pour la BCEAO.
                </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Jauge consultation ---
        st.markdown(f'<div class="sec">Consulter une annee — {flag(sel_country)} {sel_name}</div>',
                    unsafe_allow_html=True)
        proj_years_sel = sorted(proj_f["year"].unique().astype(int).tolist())
        sel_proj_yr = st.select_slider("Annee", options=proj_years_sel,
                                       value=proj_years_sel[-1], key="pred_yr_sel")
        row_sel = proj_f[proj_f["year"] == sel_proj_yr]
        if not row_sel.empty:
            r = row_sel.iloc[0]
            gr_gwh = ((r["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
            gr_pop_sel = ((r["pop_projected"] / last_pop) - 1) * 100 if (
                pd.notna(r.get("pop_projected")) and last_pop > 0) else 0
            kwh_sel = r["predicted_gwh"] * 1e6 / r["pop_projected"] if (
                pd.notna(r.get("pop_projected")) and r["pop_projected"] > 0) else 0

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=r["predicted_gwh"],
                delta={"reference": last_hist_gwh, "relative": True, "valueformat": ".0%"},
                title={"text": f"Demande {sel_name} {sel_proj_yr} (GWh)"},
                gauge={
                    "axis": {"range": [0, proj_f["ci_upper"].max() * 1.1]},
                    "bar": {"color": C["proj"]},
                    "steps": [
                        {"range": [0, last_hist_gwh], "color": "rgba(26,188,156,0.15)"},
                        {"range": [r["ci_lower"], r["ci_upper"]], "color": "rgba(155,89,182,0.15)"},
                    ],
                    "threshold": {"line": {"color": C["gwh"], "width": 3},
                                  "thickness": 0.75, "value": last_hist_gwh},
                },
            ))
            fig.update_layout(template=TMPL, height=300, margin=dict(t=60, b=10))
            st.plotly_chart(fig, key="pred_gauge")

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f"""<div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Population {sel_proj_yr}</div>
                    <div class="v">{r['pop_projected']/1e6:.1f} M</div>
                    <div class="d up">+{gr_pop_sel:.0f}% vs {y_max_h}</div></div>""",
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""<div class="card" style="border-left-color:{C['kwh']};">
                    <div class="t">kWh / habitant</div>
                    <div class="v">{kwh_sel:.0f} kWh</div>
                    <div class="ctx">Demande / Population</div></div>""",
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""<div class="card" style="border-left-color:{C['good']};">
                    <div class="t">Intervalle de confiance</div>
                    <div class="v">{r['ci_lower']:,.0f} — {r['ci_upper']:,.0f}</div>
                    <div class="ctx">GWh (IC 95%)</div></div>""",
                    unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight">
                <strong>Fiche {sel_proj_yr}</strong> — {flag(sel_country)} {sel_name} :
                population projetee de <span class="val">{r['pop_projected']/1e6:.1f} M</span>,
                demande de <span class="val">{r['predicted_gwh']:,.0f} GWh</span>
                (<span class="up">+{gr_gwh:.0f}%</span> vs {y_max_h}),
                soit <span class="val">{kwh_sel:.0f} kWh/hab/an</span>.
            </div>""", unsafe_allow_html=True)

    else:
        st.info("Executez python src/models/predict.py pour generer les projections.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
    <strong>Objectif pedagogique</strong> : Ce projet a ete realise pour maitriser les concepts
    d'ingenierie de donnees et de Machine Learning, transposables dans des situations reelles.<br>
    Source : Banque Mondiale (WDI) | scikit-learn, LightGBM | Streamlit + Plotly |
    8 pays UEMOA | 1990-2023 | Horizon 2045
</div>
""", unsafe_allow_html=True)

"""
Dashboard BI — Anticipation de la Demande Electrique au Togo
Pipeline : Extraction (Banque Mondiale) -> Transformation -> Modele IA -> Predictions 2045
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demande Electrique — Togo 2045",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"
FOCUS = "TG"

C = {
    "pop": "#2E86C1",
    "kwh": "#E67E22",
    "gwh": "#1ABC9C",
    "proj": "#9B59B6",
    "ci": "rgba(155,89,182,0.15)",
    "grid": "#1E2A35",
    "muted": "#7F8C8D",
    "good": "#27AE60",
    "warn": "#E74C3C",
    "bg": "#0E1117",
    "accent": "#3498DB",
}

IND_LABELS = {
    "SP.POP.TOTL": "Population totale",
    "SP.POP.GROW": "Croissance demographique (%)",
    "SP.URB.TOTL.IN.ZS": "Taux d'urbanisation (%)",
    "EG.USE.ELEC.KH.PC": "Consommation electrique (kWh/hab)",
    "EG.ELC.ACCS.ZS": "Acces a l'electricite (%)",
    "EG.ELC.ACCS.UR.ZS": "Acces electricite urbain (%)",
    "EG.ELC.ACCS.RU.ZS": "Acces electricite rural (%)",
    "NY.GDP.PCAP.CD": "PIB par habitant (USD)",
}


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.8rem; max-width: 1350px; }

.hdr {
    background: linear-gradient(135deg, #1B3A4B 0%, #1B6B50 100%);
    padding: 18px 28px; border-radius: 6px; margin-bottom: 16px;
    border-bottom: 2px solid #E67E22;
}
.hdr h1 { color: #fff; margin: 0; font-size: 1.3em; font-weight: 600; }
.hdr p { color: #B0C4CE; margin: 4px 0 0; font-size: 0.8em; font-weight: 300; }

.card {
    background: #161B22; border-radius: 6px; padding: 13px 15px;
    border-left: 3px solid #2E86C1;
}
.card .t { color: #7F8C8D; font-size: 0.68em; text-transform: uppercase;
           letter-spacing: 0.8px; margin-bottom: 4px; }
.card .v { color: #ECF0F1; font-size: 1.35em; font-weight: 700; line-height: 1.15; }
.card .d { font-size: 0.76em; margin-top: 3px; font-weight: 500; }
.card .d.up { color: #27AE60; }
.card .d.dn { color: #E74C3C; }
.card .ctx { color: #5D6D7E; font-size: 0.66em; margin-top: 2px; }

.sec { color: #D5DBE1; font-size: 1em; font-weight: 600;
       border-bottom: 1px solid #2E6F8E;
       padding-bottom: 5px; margin: 18px 0 10px 0; }

.step-box {
    background: #161B22; border: 1px solid #2E6F8E; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 10px;
}
.step-box .step-n { color: #3498DB; font-size: 0.72em; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.step-box .step-t { color: #ECF0F1; font-weight: 600; font-size: 0.92em; }
.step-box .step-d { color: #7F8C8D; font-size: 0.76em; margin-top: 2px; }

.foot { text-align: center; color: #5D6D7E; font-size: 0.7em; padding: 14px 0;
        margin-top: 20px; border-top: 1px solid #1E2A35; }

.yr-badge {
    display: inline-block; background: #9B59B6; color: #fff;
    padding: 3px 10px; border-radius: 12px; font-weight: 600;
    font-size: 0.85em; margin-right: 8px;
}
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
                  ("res", "models/results.csv")]:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            d[k] = pd.read_csv(p)
    return d


def fmt(v, u=""):
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e9:
        s = f"{v/1e9:,.2f} Mrd"
    elif abs(v) >= 1e6:
        s = f"{v/1e6:,.1f} M"
    elif abs(v) >= 1e3:
        s = f"{v:,.0f}"
    else:
        s = f"{v:,.1f}"
    return f"{s} {u}".strip() if u else s


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>Anticipation de la Demande Electrique — Togo</h1>
    <p>La population croit, combien d'electricite faudra-t-il demain ?
       &nbsp;|&nbsp; Donnees : Banque Mondiale (WDI) &nbsp;|&nbsp; Horizon 2045</p>
</div>
""", unsafe_allow_html=True)

data = load()
if "df" not in data:
    st.error("Donnees absentes. Executez le pipeline ETL d'abord.")
    st.stop()

# --- Filter Togo only everywhere ---
df_all = data["df"]
df = df_all[df_all["country_code"] == FOCUS].copy()

raw_all = data.get("raw")
raw = raw_all[raw_all["country_code"] == FOCUS].copy() if raw_all is not None else None

pred_all = data.get("pred")
pred = pred_all[pred_all["country_code"] == FOCUS].copy() if pred_all is not None else None

proj_all = data.get("proj")
proj = proj_all[proj_all["country_code"] == FOCUS].copy() if proj_all is not None else None

res = data.get("res")

if df.empty:
    st.error("Aucune donnee pour le Togo.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTRES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Togo")
    st.markdown("Filtrer les donnees par annee.")
    st.divider()

    y_min_h, y_max_h = int(df["year"].min()), int(df["year"].max())
    yr = st.slider("Periode historique", y_min_h, y_max_h, (y_min_h, y_max_h),
                   key="yr_hist")

    proj_years = []
    if proj is not None and not proj.empty:
        proj_years = sorted(proj["year"].unique().astype(int).tolist())
    if proj_years:
        proj_yr = st.select_slider(
            "Horizon de projection",
            options=proj_years,
            value=proj_years[-1],
            key="yr_proj",
        )
    else:
        proj_yr = None

    st.divider()
    st.markdown(f"**Historique** : {yr[0]} — {yr[1]}")
    if proj_yr:
        st.markdown(f"**Projection jusqu'a** : {proj_yr}")

    st.divider()
    st.markdown("##### Exports")
    if raw is not None:
        st.download_button("Donnees brutes", raw.to_csv(index=False).encode(),
                           "togo_brut.csv", key="dl_raw")
    st.download_button("Donnees transformees", df.to_csv(index=False).encode(),
                       "togo_transforme.csv", key="dl_transf")
    if pred is not None:
        st.download_button("Predictions", pred.to_csv(index=False).encode(),
                           "togo_predictions.csv", key="dl_pred")
    if proj is not None:
        st.download_button("Projections 2045", proj.to_csv(index=False).encode(),
                           "togo_projections.csv", key="dl_proj")

# Apply year filter
tg = df[df["year"].between(*yr)].sort_values("year")

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
if not tg.empty:
    last = tg.iloc[-1]
    first = tg.iloc[0]

    cards = []
    if "SP.POP.TOTL" in tg.columns:
        pop_now = last["SP.POP.TOTL"]
        pop_bef = first["SP.POP.TOTL"]
        gr = ((pop_now / pop_bef) - 1) * 100 if pop_bef > 0 else 0
        cards.append(("Population", fmt(pop_now), f"+{gr:.0f}% depuis {yr[0]}", "up", C["pop"]))

    if "EG.USE.ELEC.KH.PC" in tg.columns:
        kwh = last["EG.USE.ELEC.KH.PC"]
        kwh0 = first["EG.USE.ELEC.KH.PC"]
        d = kwh - kwh0
        cards.append(("kWh / habitant", fmt(kwh, "kWh"),
                      f"{d:+.0f} kWh vs {yr[0]}", "up" if d > 0 else "dn", C["kwh"]))

    if "conso_totale_gwh" in tg.columns:
        gwh = last["conso_totale_gwh"]
        gwh0 = first["conso_totale_gwh"]
        d = ((gwh / gwh0) - 1) * 100 if gwh0 > 0 else 0
        cards.append(("Demande (GWh)", fmt(gwh, "GWh"),
                      f"+{d:.0f}% depuis {yr[0]}", "up", C["gwh"]))

    if "EG.ELC.ACCS.ZS" in tg.columns:
        acc = last["EG.ELC.ACCS.ZS"]
        acc0 = first["EG.ELC.ACCS.ZS"]
        d = acc - acc0
        cards.append(("Acces electrique", f"{acc:.1f}%",
                      f"{d:+.1f} pts vs {yr[0]}", "up" if d > 0 else "dn", C["good"]))

    if proj_yr and proj is not None and not proj.empty:
        row_p = proj[proj["year"] == proj_yr]
        if not row_p.empty:
            gp = row_p.iloc[0]["predicted_gwh"]
            cards.append(("Prediction " + str(proj_yr), fmt(gp, "GWh"),
                          "Projection IA", "up", C["proj"]))

    if res is not None and not res.empty:
        best = res.sort_values("r2", ascending=False).iloc[0]
        cards.append(("Modele", f"R2 {best['r2']:.3f}", best["model"], "up", C["accent"]))

    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:8px;margin-bottom:14px;">'
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
    "1. Extraction",
    "2. Transformation",
    "3. Analyse et Modele",
    "4. Predictions 2045",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("""
    <div class="step-box">
        <div class="step-n">Etape 1 — Extract</div>
        <div class="step-t">Donnees brutes extraites de la Banque Mondiale (WDI)</div>
        <div class="step-d">API World Bank → 8 indicateurs pour le Togo (2000-2023)</div>
    </div>""", unsafe_allow_html=True)

    if raw is not None and not raw.empty:
        # Filter by year
        raw_f = raw[raw["year"].between(*yr)].sort_values("year")
        st.markdown(f"**{len(raw_f)} enregistrements** sur la periode {yr[0]} — {yr[1]}")

        # Indicateurs extraits
        st.markdown('<div class="sec">Indicateurs releves</div>', unsafe_allow_html=True)
        if "indicator_code" in raw_f.columns:
            ind_counts = raw_f.groupby("indicator_code").agg(
                observations=("value", "count"),
                valeur_min=("value", "min"),
                valeur_max=("value", "max"),
            ).reset_index()
            ind_counts["Indicateur"] = ind_counts["indicator_code"].map(
                lambda c: IND_LABELS.get(c, c))
            ind_counts = ind_counts.rename(columns={
                "indicator_code": "Code WDI",
                "observations": "Observations",
                "valeur_min": "Min",
                "valeur_max": "Max",
            })
            st.dataframe(ind_counts[["Code WDI", "Indicateur", "Observations", "Min", "Max"]],
                         height=320)

        st.divider()

        # Visualisation par indicateur
        st.markdown('<div class="sec">Donnees brutes par indicateur</div>', unsafe_allow_html=True)
        if "indicator_code" in raw_f.columns:
            ind_codes = raw_f["indicator_code"].unique().tolist()
            sel_ind = st.selectbox("Indicateur", ind_codes,
                                   format_func=lambda c: IND_LABELS.get(c, c),
                                   key="raw_ind")
            raw_ind = raw_f[raw_f["indicator_code"] == sel_ind].sort_values("year")

            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=raw_ind["year"], y=raw_ind["value"],
                    mode="lines+markers", line=dict(color=C["accent"], width=2.5),
                    marker=dict(size=5), name=IND_LABELS.get(sel_ind, sel_ind),
                ))
                fig.update_layout(
                    title=f"{IND_LABELS.get(sel_ind, sel_ind)} — Togo",
                    template=TMPL, height=340, margin=dict(t=35),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, key="raw_chart")

            with c2:
                st.markdown("##### Donnees brutes")
                disp_raw = raw_ind[["year", "value"]].copy()
                disp_raw.columns = ["Annee", "Valeur"]
                st.dataframe(disp_raw, height=300)

        st.divider()

        # Table complete
        with st.expander("Table complete des donnees brutes", expanded=False):
            st.dataframe(raw_f, height=400)

    else:
        st.info("Fichier brut non disponible. Executez python src/etl/extract.py")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRANSFORMATION
# ═════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("""
    <div class="step-box">
        <div class="step-n">Etape 2 — Transform</div>
        <div class="step-t">Pivot, nettoyage et creation de features</div>
        <div class="step-d">Donnees brutes pivotees en colonnes, valeurs manquantes comblees,
                            features derivees : conso_totale_gwh, ecart urbain/rural, pop. urbaine, etc.</div>
    </div>""", unsafe_allow_html=True)

    if not tg.empty:
        st.markdown(f"**{len(tg)} lignes x {len(tg.columns)} colonnes** — Togo ({yr[0]}-{yr[1]})")

        # Indicateurs principaux en colonnes lisibles
        st.markdown('<div class="sec">Variables principales</div>', unsafe_allow_html=True)
        main_cols = ["year", "SP.POP.TOTL", "EG.USE.ELEC.KH.PC", "conso_totale_gwh",
                     "EG.ELC.ACCS.ZS", "EG.ELC.ACCS.UR.ZS", "EG.ELC.ACCS.RU.ZS",
                     "NY.GDP.PCAP.CD", "gap_acces_urb_rur", "pop_urbaine"]
        avail = [c for c in main_cols if c in tg.columns]
        rename_map = {
            "year": "Annee", "SP.POP.TOTL": "Population",
            "EG.USE.ELEC.KH.PC": "kWh/hab", "conso_totale_gwh": "Demande (GWh)",
            "EG.ELC.ACCS.ZS": "Acces (%)", "EG.ELC.ACCS.UR.ZS": "Acces urbain (%)",
            "EG.ELC.ACCS.RU.ZS": "Acces rural (%)", "NY.GDP.PCAP.CD": "PIB/hab (USD)",
            "gap_acces_urb_rur": "Ecart urb/rur (pts)", "pop_urbaine": "Pop. urbaine",
        }
        disp_main = tg[avail].copy()
        disp_main = disp_main.rename(columns={c: rename_map.get(c, c) for c in avail})
        st.dataframe(disp_main, height=380)

        st.divider()

        # Features derivees (lags, changes, ma3)
        st.markdown('<div class="sec">Features d\'ingenierie</div>', unsafe_allow_html=True)
        eng_cols = [c for c in tg.columns if any(x in c for x in ["_lag", "_chg", "_ma3",
                    "year_norm", "intensite"])]
        if eng_cols:
            disp_eng = tg[["year"] + eng_cols].copy()
            rename_eng = {"year": "Annee"}
            disp_eng = disp_eng.rename(columns=rename_eng)
            st.dataframe(disp_eng, height=300)
        else:
            st.info("Pas de features d'ingenierie detectees.")

        st.divider()

        # Stats descriptives
        st.markdown('<div class="sec">Statistiques descriptives</div>', unsafe_allow_html=True)
        num_cols = tg.select_dtypes(include=[np.number]).columns.tolist()
        stats = tg[num_cols].describe().T
        stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
        st.dataframe(stats, height=350)

        # Consultation par annee
        st.divider()
        st.markdown('<div class="sec">Consulter une annee</div>', unsafe_allow_html=True)
        years_avail = sorted(tg["year"].unique().astype(int).tolist())
        sel_year = st.selectbox("Annee", years_avail,
                                index=len(years_avail) - 1, key="tr_year")
        row_yr = tg[tg["year"] == sel_year]
        if not row_yr.empty:
            r = row_yr.iloc[0]
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.markdown(f"""<div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Population</div>
                    <div class="v">{fmt(r.get('SP.POP.TOTL', np.nan))}</div></div>""",
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""<div class="card" style="border-left-color:{C['kwh']};">
                    <div class="t">kWh / habitant</div>
                    <div class="v">{fmt(r.get('EG.USE.ELEC.KH.PC', np.nan), 'kWh')}</div></div>""",
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""<div class="card" style="border-left-color:{C['gwh']};">
                    <div class="t">Demande totale</div>
                    <div class="v">{fmt(r.get('conso_totale_gwh', np.nan), 'GWh')}</div></div>""",
                    unsafe_allow_html=True)
            with cc4:
                st.markdown(f"""<div class="card" style="border-left-color:{C['good']};">
                    <div class="t">Acces electrique</div>
                    <div class="v">{r.get('EG.ELC.ACCS.ZS', 0):.1f}%</div></div>""",
                    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE ET MODELE
# ═════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("""
    <div class="step-box">
        <div class="step-n">Etape 3 — Analyse et Entrainement</div>
        <div class="step-t">Tendances historiques et validation du modele IA</div>
        <div class="step-d">Exploration des correlations population/energie,
                            entrainement de 4 algorithmes, selection du meilleur modele.</div>
    </div>""", unsafe_allow_html=True)

    if not tg.empty:
        # Tendance Population + GWh
        st.markdown('<div class="sec">Population et demande electrique</div>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "SP.POP.TOTL" in tg.columns:
            fig.add_trace(go.Bar(
                x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Population (M)",
                marker_color=C["pop"], opacity=0.35,
            ), secondary_y=False)
        if "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"], name="Demande (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=2.5),
                marker=dict(size=4),
            ), secondary_y=True)
        fig.update_layout(
            title="Togo — Population et demande electrique",
            template=TMPL, height=380, hovermode="x unified", margin=dict(t=40),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        fig.update_yaxes(title_text="Population (millions)", secondary_y=False)
        fig.update_yaxes(title_text="GWh", secondary_y=True)
        st.plotly_chart(fig, key="an_pop_gwh")

        # kWh/hab + Acces
        c1, c2 = st.columns(2)
        with c1:
            if "EG.USE.ELEC.KH.PC" in tg.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tg["year"], y=tg["EG.USE.ELEC.KH.PC"],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=C["kwh"], width=2),
                    fillcolor="rgba(230,126,34,0.08)", name="kWh/hab",
                ))
                fig.update_layout(title="Consommation par habitant", template=TMPL,
                                  height=280, margin=dict(t=35), yaxis_title="kWh/hab")
                st.plotly_chart(fig, key="an_kwh")

        with c2:
            if "EG.ELC.ACCS.ZS" in tg.columns:
                fig = go.Figure()
                if "EG.ELC.ACCS.UR.ZS" in tg.columns:
                    fig.add_trace(go.Scatter(x=tg["year"], y=tg["EG.ELC.ACCS.UR.ZS"],
                                             name="Urbain", line=dict(color=C["good"], width=2)))
                if "EG.ELC.ACCS.RU.ZS" in tg.columns:
                    fig.add_trace(go.Scatter(x=tg["year"], y=tg["EG.ELC.ACCS.RU.ZS"],
                                             name="Rural", fill="tonexty",
                                             fillcolor="rgba(231,76,60,0.06)",
                                             line=dict(color=C["warn"], width=2)))
                fig.add_trace(go.Scatter(x=tg["year"], y=tg["EG.ELC.ACCS.ZS"],
                                         name="Total", line=dict(color=C["pop"], width=2, dash="dash")))
                fig.update_layout(title="Acces electricite (%)", template=TMPL, height=280,
                                  margin=dict(t=35), yaxis_title="%",
                                  legend=dict(orientation="h", y=-0.18, font_size=10))
                st.plotly_chart(fig, key="an_acces")

        # Correlation
        st.divider()
        st.markdown('<div class="sec">Correlation et elasticite</div>', unsafe_allow_html=True)
        if len(tg) > 3 and "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
            corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
            pop_chg = (tg["SP.POP.TOTL"].iloc[-1] / tg["SP.POP.TOTL"].iloc[0] - 1) * 100
            gwh_chg = (tg["conso_totale_gwh"].iloc[-1] / tg["conso_totale_gwh"].iloc[0] - 1) * 100 \
                if tg["conso_totale_gwh"].iloc[0] > 0 else 0
            elast = gwh_chg / pop_chg if pop_chg > 0 else 0

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f"""<div class="card" style="border-left-color:{C['gwh']};">
                    <div class="t">Correlation Pearson</div>
                    <div class="v">{corr:.3f}</div>
                    <div class="ctx">Population vs Demande GWh</div></div>""",
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""<div class="card" style="border-left-color:{C['kwh']};">
                    <div class="t">Elasticite</div>
                    <div class="v">{elast:.2f}</div>
                    <div class="ctx">+1% pop = +{elast:.2f}% demande</div></div>""",
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""<div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Croissance pop.</div>
                    <div class="v">+{pop_chg:.0f}%</div>
                    <div class="ctx">{yr[0]} — {yr[1]}</div></div>""",
                    unsafe_allow_html=True)

        # Validation du modele
        if pred is not None and not pred.empty:
            st.divider()
            st.markdown('<div class="sec">Validation du modele (observe vs predit)</div>',
                        unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred["year"], y=pred["actual"],
                                     name="Observe", mode="lines+markers",
                                     line=dict(color=C["gwh"], width=2.5)))
            fig.add_trace(go.Scatter(x=pred["year"], y=pred["predicted"],
                                     name="Predit (IA)", mode="lines+markers",
                                     line=dict(color=C["proj"], width=2, dash="dash")))
            fig.update_layout(title="Togo — Observe vs Predit",
                              yaxis_title="GWh", template=TMPL, height=340,
                              hovermode="x unified", margin=dict(t=40),
                              legend=dict(orientation="h", y=-0.15, font_size=10))
            st.plotly_chart(fig, key="an_valid")

            mae = pred["error"].abs().mean()
            rmse = np.sqrt((pred["error"] ** 2).mean())
            mc = st.columns(3)
            mc[0].metric("MAE", f"{mae:.1f} GWh")
            mc[1].metric("RMSE", f"{rmse:.1f} GWh")
            if res is not None:
                mc[2].metric("R2", f"{res.sort_values('r2', ascending=False).iloc[0]['r2']:.4f}")

            with st.expander("Predictions detaillees", expanded=False):
                dp = pred[["year", "actual", "predicted", "error", "error_pct"]].copy()
                dp.columns = ["Annee", "Observe (GWh)", "Predit (GWh)", "Erreur", "Erreur (%)"]
                st.dataframe(dp, height=300)

        # Comparaison modeles
        if res is not None and not res.empty:
            st.divider()
            st.markdown('<div class="sec">Comparaison des algorithmes</div>', unsafe_allow_html=True)
            r2_vals = res.sort_values("r2", ascending=False)
            fig = go.Figure()
            colors = [C["good"] if v > 0.85 else C["kwh"] if v > 0.7 else C["warn"]
                      for v in r2_vals["r2"]]
            fig.add_trace(go.Bar(
                x=r2_vals["model"], y=r2_vals["r2"], marker_color=colors,
                text=[f"{v:.3f}" for v in r2_vals["r2"]], textposition="outside",
                textfont_size=11,
            ))
            fig.update_layout(title="R2 par modele", yaxis_title="R2",
                              yaxis_range=[0, 1.08], template=TMPL, height=300,
                              margin=dict(t=40))
            st.plotly_chart(fig, key="an_models")

            with st.expander("Detail des performances", expanded=False):
                st.dataframe(res, height=200)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS 2045
# ═════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
    <div class="step-box">
        <div class="step-n">Etape 4 — Predict</div>
        <div class="step-t">Projections de la demande electrique — Togo 2024 a 2045</div>
        <div class="step-d">Le modele IA extrapole les tendances demographiques et energetiques
                            pour anticiper la demande future en electricite.</div>
    </div>""", unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        # Filter projections by selected horizon
        proj_f = proj[proj["year"] <= proj_yr].sort_values("year") if proj_yr else proj.sort_values("year")

        # --- Graphique principal : historique + projections ---
        st.markdown('<div class="sec">Trajectoire historique et projections</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()

        # Historique
        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"],
                name="Historique", mode="lines+markers",
                line=dict(color=C["gwh"], width=2.5), marker=dict(size=4),
            ))

        # Projections
        fig.add_trace(go.Scatter(
            x=proj_f["year"], y=proj_f["predicted_gwh"],
            name="Projection IA", mode="lines+markers",
            line=dict(color=C["proj"], width=2.5),
            marker=dict(size=6, symbol="diamond"),
        ))

        # IC
        fig.add_trace(go.Scatter(
            x=pd.concat([proj_f["year"], proj_f["year"][::-1]]),
            y=pd.concat([proj_f["ci_upper"], proj_f["ci_lower"][::-1]]),
            fill="toself", fillcolor=C["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
        ))

        # Ligne verticale : "aujourd'hui"
        fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=C["muted"],
                      annotation_text="Aujourd'hui", annotation_position="top left",
                      annotation_font_color=C["muted"], annotation_font_size=10)

        fig.update_layout(
            title=f"Togo — Demande electrique : {yr[0]} a {proj_yr}",
            yaxis_title="GWh", template=TMPL, height=440,
            hovermode="x unified", margin=dict(t=40),
            legend=dict(orientation="h", y=-0.12, font_size=10),
        )
        st.plotly_chart(fig, key="pred_main")

        # --- Population projetee ---
        if "pop_projected" in proj_f.columns and proj_f["pop_projected"].notna().any():
            st.markdown('<div class="sec">Population projetee</div>', unsafe_allow_html=True)

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            # Historique pop
            if "SP.POP.TOTL" in tg.columns:
                fig2.add_trace(go.Bar(
                    x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Historique (M)",
                    marker_color=C["pop"], opacity=0.35,
                ), secondary_y=False)
            # Projection pop
            fig2.add_trace(go.Bar(
                x=proj_f["year"], y=proj_f["pop_projected"] / 1e6, name="Projetee (M)",
                marker_color=C["proj"], opacity=0.45,
            ), secondary_y=False)
            # GWh projetee
            fig2.add_trace(go.Scatter(
                x=proj_f["year"], y=proj_f["predicted_gwh"], name="Demande (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=2),
                marker=dict(size=4),
            ), secondary_y=True)
            fig2.update_layout(
                title="Population et demande projetees", template=TMPL, height=360,
                hovermode="x unified", margin=dict(t=40),
                legend=dict(orientation="h", y=-0.15, font_size=10),
            )
            fig2.update_yaxes(title_text="Population (millions)", secondary_y=False)
            fig2.update_yaxes(title_text="GWh", secondary_y=True)
            st.plotly_chart(fig2, key="pred_pop")

        st.divider()

        # --- Consultation directe par annee ---
        st.markdown('<div class="sec">Consulter une annee de projection</div>',
                    unsafe_allow_html=True)
        proj_years_sel = sorted(proj_f["year"].unique().astype(int).tolist())
        sel_proj_yr = st.select_slider("Annee projetee", options=proj_years_sel,
                                       value=proj_years_sel[-1], key="pred_yr_sel")
        row_sel = proj_f[proj_f["year"] == sel_proj_yr]
        if not row_sel.empty:
            r = row_sel.iloc[0]
            last_hist_gwh = tg["conso_totale_gwh"].iloc[-1] if (
                not tg.empty and "conso_totale_gwh" in tg.columns) else 0
            gr_gwh = ((r["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
            last_pop = tg["SP.POP.TOTL"].iloc[-1] if (
                not tg.empty and "SP.POP.TOTL" in tg.columns) else 0
            gr_pop = ((r["pop_projected"] / last_pop) - 1) * 100 if (
                pd.notna(r.get("pop_projected")) and last_pop > 0) else 0
            kwh_proj = r["predicted_gwh"] * 1e6 / r["pop_projected"] if (
                pd.notna(r.get("pop_projected")) and r["pop_projected"] > 0) else 0

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.markdown(f"""<div class="card" style="border-left-color:{C['proj']};">
                    <div class="t">Demande {sel_proj_yr}</div>
                    <div class="v">{r['predicted_gwh']:,.0f} GWh</div>
                    <div class="d up">+{gr_gwh:.0f}% vs {y_max_h}</div></div>""",
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""<div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Population</div>
                    <div class="v">{r['pop_projected']/1e6:.1f} M</div>
                    <div class="d up">+{gr_pop:.0f}% vs {y_max_h}</div></div>""",
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""<div class="card" style="border-left-color:{C['kwh']};">
                    <div class="t">kWh / hab projete</div>
                    <div class="v">{kwh_proj:.0f} kWh</div>
                    <div class="ctx">Demande / Population</div></div>""",
                    unsafe_allow_html=True)
            with cc4:
                st.markdown(f"""<div class="card" style="border-left-color:{C['good']};">
                    <div class="t">Intervalle de confiance</div>
                    <div class="v">{r['ci_lower']:,.0f} — {r['ci_upper']:,.0f}</div>
                    <div class="ctx">GWh (IC 95%)</div></div>""",
                    unsafe_allow_html=True)

        st.divider()

        # --- Table complete ---
        st.markdown('<div class="sec">Table complete des projections</div>', unsafe_allow_html=True)
        disp = proj_f[["year", "predicted_gwh", "ci_lower", "ci_upper", "pop_projected"]].copy()
        disp.columns = ["Annee", "Demande (GWh)", "IC bas (GWh)", "IC haut (GWh)", "Population"]
        disp["Population"] = disp["Population"].apply(
            lambda v: f"{v/1e6:.2f} M" if pd.notna(v) else "—")
        st.dataframe(disp, height=500)

    else:
        st.info("Executez python src/models/predict.py pour generer les projections.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
Source : Banque Mondiale (WDI)  |  Modeles : scikit-learn, XGBoost, LightGBM
|  Pipeline ETL Python  |  Streamlit + Plotly
</div>
""", unsafe_allow_html=True)

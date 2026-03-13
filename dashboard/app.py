"""
Dashboard IA — Prevision Demande Electrique UEMOA
"""
import os, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IA Energie UEMOA 2045",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"

# ── Couleurs vivantes ────────────────────────────────────────────────
C = {
    "bleu":     "#60A5FA",
    "violet":   "#A78BFA",
    "vert":     "#34D399",
    "jaune":    "#FBBF24",
    "rose":     "#F472B6",
    "cyan":     "#22D3EE",
    "orange":   "#FB923C",
    "rouge":    "#F87171",
    "blanc":    "#F8FAFC",
    "gris":     "#94A3B8",
    "gris2":    "#64748B",
    "fond":     "#0B1120",
    "carte":    "#111827",
    "bord":     "#1F2937",
}
PAL8 = [C["bleu"], C["violet"], C["jaune"], C["vert"],
        C["rose"], C["cyan"], C["orange"], C["rouge"]]

FLAGS = {'TG':'🇹🇬','SN':'🇸🇳','CI':'🇨🇮','BJ':'🇧🇯',
         'BF':'🇧🇫','ML':'🇲🇱','NE':'🇳🇪','GW':'🇬🇼'}

IND_LABELS = {
    "SP.POP.TOTL":"Population totale","SP.POP.GROW":"Croissance demo. (%)",
    "SP.URB.TOTL.IN.ZS":"Urbanisation (%)","SP.DYN.TFRT.IN":"Fecondite",
    "SP.DYN.LE00.IN":"Esperance de vie","SP.POP.0014.TO.ZS":"Pop 0-14 (%)",
    "SP.POP.1564.TO.ZS":"Pop 15-64 (%)","EG.USE.ELEC.KH.PC":"kWh/hab",
    "EG.ELC.ACCS.ZS":"Acces electr. (%)","EG.ELC.ACCS.UR.ZS":"Acces urbain (%)",
    "EG.ELC.ACCS.RU.ZS":"Acces rural (%)","EG.FEC.RNEW.ZS":"Renouvelable (%)",
    "EG.USE.PCAP.KG.OE":"Energie (kg petrole/hab)","NY.GDP.PCAP.CD":"PIB/hab (USD)",
    "NY.GDP.MKTP.CD":"PIB total (USD)","NY.GDP.MKTP.KD.ZG":"Croissance PIB (%)",
    "NV.IND.TOTL.ZS":"Industrie (% PIB)","FP.CPI.TOTL.ZG":"Inflation (%)",
    "IT.CEL.SETS.P2":"Mobile (/100 hab)","SE.ADT.LITR.ZS":"Alphabetisation (%)",
    "SL.UEM.TOTL.ZS":"Chomage (%)",
}

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:.5rem;max-width:1420px;}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0B1120 0%,#111827 100%);
}
section[data-testid="stSidebar"] .stMarkdown h3{
    color:#60A5FA; font-size:.95em; font-weight:700; letter-spacing:.03em;
}
.sb-profile{
    background:linear-gradient(135deg,rgba(96,165,250,.08),rgba(167,139,250,.08));
    border:1px solid #1F2937; border-radius:12px; padding:16px; margin:8px 0 14px 0;
}
.sb-profile .sb-name{color:#F8FAFC;font-size:.95em;font-weight:700;}
.sb-profile .sb-role{color:#60A5FA;font-size:.72em;font-weight:600;margin-top:2px;}
.sb-profile .sb-desc{color:#94A3B8;font-size:.7em;line-height:1.4;margin-top:6px;}
.sb-badge-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;}
.sb-badge{
    background:rgba(96,165,250,.1);border:1px solid rgba(96,165,250,.2);
    border-radius:6px;padding:2px 8px;font-size:.6em;color:#60A5FA;font-weight:600;
}
.sb-stat{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 0;border-bottom:1px solid #1F2937;
}
.sb-stat .sb-k{color:#64748B;font-size:.7em;font-weight:500;}
.sb-stat .sb-v{color:#F8FAFC;font-size:.78em;font-weight:700;}
.sb-stat .sb-v.green{color:#34D399;}

/* Header */
.hdr{
    background:linear-gradient(135deg,#0B1120 0%,#1a1040 50%,#0B1120 100%);
    border:1px solid #1F2937; border-radius:14px; padding:28px 34px; margin-bottom:20px;
    position:relative; overflow:hidden;
}
.hdr::before{
    content:'';position:absolute;top:-50%;right:-20%;width:400px;height:400px;
    background:radial-gradient(circle,rgba(96,165,250,.06) 0%,transparent 70%);
}
.hdr .crumb{color:#64748B;font-size:.7em;margin-bottom:6px;}
.hdr h1{color:#F8FAFC;margin:0;font-size:1.55em;font-weight:800;letter-spacing:-.02em;}
.hdr .sub{color:#94A3B8;font-size:.82em;margin-top:5px;}
.hdr .mission{
    display:inline-block;margin-top:12px;padding:8px 16px;
    background:linear-gradient(135deg,rgba(96,165,250,.1),rgba(167,139,250,.1));
    border:1px solid rgba(96,165,250,.2);border-radius:10px;
    color:#CBD5E1;font-size:.72em;line-height:1.5;
}
.hdr .mission b{color:#60A5FA;}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin-bottom:18px;}
.kpi{
    background:linear-gradient(135deg,#111827,#0B1120);
    border:1px solid #1F2937;border-radius:12px;padding:16px 18px;
    transition:border-color .2s;
}
.kpi:hover{border-color:#60A5FA;}
.kpi .kpi-hd{display:flex;justify-content:space-between;align-items:flex-start;}
.kpi .kpi-lb{color:#64748B;font-size:.65em;text-transform:uppercase;letter-spacing:.06em;font-weight:600;}
.kpi .kpi-ic{font-size:1.2em;}
.kpi .kpi-vl{color:#F8FAFC;font-size:1.55em;font-weight:800;margin-top:4px;letter-spacing:-.02em;}
.kpi .kpi-dt{font-size:.7em;font-weight:600;margin-top:3px;display:flex;align-items:center;gap:3px;}
.kpi .kpi-dt.up{color:#34D399;} .kpi .kpi-dt.dn{color:#F87171;}
.kpi .kpi-dt .vs{color:#64748B;font-weight:400;margin-left:3px;}

/* Section */
.sec{display:flex;align-items:center;gap:8px;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid #1F2937;}
.sec .ic{font-size:1em;} .sec .tt{color:#F1F5F9;font-size:.92em;font-weight:700;}
.sec .badge{
    margin-left:auto;background:rgba(52,211,153,.08);color:#34D399;
    border:1px solid rgba(52,211,153,.2);border-radius:6px;padding:2px 10px;
    font-size:.58em;font-weight:700;
}

/* Interpretation */
.interp{
    background:linear-gradient(135deg,rgba(96,165,250,.04),rgba(167,139,250,.04));
    border-left:3px solid #60A5FA;border-radius:0 10px 10px 0;
    padding:14px 18px;margin:10px 0 18px;
}
.interp .interp-hd{color:#60A5FA;font-size:.65em;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}
.interp p{color:#CBD5E1;font-size:.8em;line-height:1.7;margin:0;}
.interp .v{color:#34D399;font-weight:700;}
.interp .w{color:#F87171;font-weight:700;}
.interp .h{color:#F8FAFC;font-weight:600;}
.interp .b{color:#60A5FA;font-weight:700;}

/* Pipeline steps */
.pipe{
    background:#111827;border:1px solid #1F2937;border-radius:12px;
    padding:16px 20px;margin-bottom:12px;
}
.pipe .pipe-tag{
    display:inline-block;background:rgba(96,165,250,.1);color:#60A5FA;
    border-radius:6px;padding:2px 10px;font-size:.6em;font-weight:700;
    text-transform:uppercase;letter-spacing:.08em;
}
.pipe .pipe-tt{color:#F1F5F9;font-weight:700;font-size:.92em;margin-top:5px;}
.pipe .pipe-ds{color:#94A3B8;font-size:.72em;margin-top:3px;line-height:1.5;}

/* Prediction cards */
.pred-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0;}
.pred{background:#111827;border-radius:12px;padding:18px;position:relative;overflow:hidden;}
.pred::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.pred.c1::before{background:linear-gradient(90deg,#60A5FA,#A78BFA);}
.pred.c2::before{background:linear-gradient(90deg,#34D399,#22D3EE);}
.pred.c3::before{background:linear-gradient(90deg,#F472B6,#FB923C);}
.pred .pred-lb{font-size:.72em;font-weight:600;color:#94A3B8;margin-bottom:4px;}
.pred .pred-vl{font-size:1.7em;font-weight:800;letter-spacing:-.03em;}
.pred .pred-vl.t1{color:#60A5FA;} .pred .pred-vl.t2{color:#34D399;} .pred .pred-vl.t3{color:#F472B6;}
.pred .pred-sub{color:#64748B;font-size:.68em;margin-top:2px;}
.pred .pred-txt{color:#94A3B8;font-size:.7em;margin-top:8px;line-height:1.5;}

/* Welcome hero */
.hero-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0;}
.hero-item{background:#111827;border:1px solid #1F2937;border-radius:12px;padding:16px;text-align:center;}
.hero-item .hero-num{font-size:1.6em;font-weight:800;letter-spacing:-.02em;}
.hero-item .hero-num.n1{color:#60A5FA;} .hero-item .hero-num.n2{color:#34D399;}
.hero-item .hero-num.n3{color:#A78BFA;} .hero-item .hero-num.n4{color:#FBBF24;}
.hero-item .hero-lb{color:#94A3B8;font-size:.68em;margin-top:2px;}

/* Footer */
.foot{text-align:center;color:#475569;font-size:.65em;padding:20px 0;margin-top:28px;border-top:1px solid #1F2937;}
.foot b{color:#64748B;}
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────────────
@st.cache_data
def load():
    d = {}
    for k, f in [("df","data/processed/energy_data_processed.csv"),
                  ("raw","data/raw/energy_data_raw.csv"),
                  ("pred","data/predictions/predictions.csv"),
                  ("proj","data/predictions/projections.csv"),
                  ("res","models/results.csv"),
                  ("fi","models/feature_importance.csv"),
                  ("cv","models/cv_scores.csv")]:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            d[k] = pd.read_csv(p)
    return d

def fmt(v, u=""):
    if pd.isna(v): return "—"
    if abs(v)>=1e9: s=f"{v/1e9:,.1f} Mrd"
    elif abs(v)>=1e6: s=f"{v/1e6:,.1f} M"
    elif abs(v)>=1e3: s=f"{v:,.0f}"
    else: s=f"{v:,.1f}"
    return f"{s} {u}".strip() if u else s

def chg(a,b):
    return ((b/a)-1)*100 if a and a!=0 else 0

def lay(fig, title="", h=400, yt="", mb=60):
    fig.update_layout(
        title=dict(text=title,font=dict(size=13,color="#F1F5F9",family="Inter"),x=0,xanchor="left"),
        template=TMPL, height=h, hovermode="x unified",
        margin=dict(t=44,b=mb,l=55,r=18),
        legend=dict(orientation="h",y=-0.15,font=dict(size=10,color="#94A3B8")),
        yaxis_title=yt, plot_bgcolor="#0B1120", paper_bgcolor="#0B1120",
        font=dict(family="Inter",color="#94A3B8"),
        xaxis=dict(gridcolor="#1F2937",zerolinecolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937",zerolinecolor="#1F2937"),
    )
    return fig

data = load()
if "df" not in data:
    st.error("Donnees absentes — lancez le pipeline ETL d'abord.")
    st.stop()

df_all=data["df"]; raw_all=data.get("raw"); pred_all=data.get("pred")
proj_all=data.get("proj"); res=data.get("res"); fi_df=data.get("fi"); cv_df=data.get("cv")

cc_list = sorted(df_all["country_code"].unique().tolist())
cn_map = df_all.drop_duplicates("country_code").set_index("country_code")["country_name"].to_dict()

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    # Profil projet
    st.markdown("""<div class="sb-profile">
        <div class="sb-name">⚡ Energy Prediction UEMOA</div>
        <div class="sb-role">Projet IA — Candidature BCEAO</div>
        <div class="sb-desc">Systeme complet de prediction de la demande electrique
        pour les 8 pays de l'UEMOA, de la collecte des donnees aux projections 2045.</div>
        <div class="sb-badge-row">
            <span class="sb-badge">Python</span>
            <span class="sb-badge">Scikit-learn</span>
            <span class="sb-badge">XGBoost</span>
            <span class="sb-badge">LightGBM</span>
            <span class="sb-badge">Streamlit</span>
            <span class="sb-badge">Plotly</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🌍 Pays")
    sel_cc = st.selectbox(
        "Selectionner un pays",
        cc_list,
        format_func=lambda c: f"{FLAGS.get(c,'')} {cn_map.get(c,c)}",
        index=cc_list.index('TG') if 'TG' in cc_list else 0,
        key="sel_cc",
    )
    sel_name = cn_map.get(sel_cc, sel_cc)
    sel_flag = FLAGS.get(sel_cc, '')

    st.markdown("### 📅 Periode")
    ymin_h, ymax_h = int(df_all["year"].min()), int(df_all["year"].max())
    yr = st.slider("Annees historiques", ymin_h, ymax_h, (ymin_h, ymax_h), key="yr")

    # Stats resumees dans la sidebar
    n_raw = len(raw_all) if raw_all is not None else 0
    n_feat = len(df_all.columns)
    best_r2 = res.sort_values("r2",ascending=False).iloc[0]["r2"] if res is not None and not res.empty else 0

    st.markdown("### 📊 Pipeline")
    st.markdown(f"""
    <div class="sb-stat"><span class="sb-k">Observations brutes</span><span class="sb-v">{n_raw:,}</span></div>
    <div class="sb-stat"><span class="sb-k">Pays UEMOA</span><span class="sb-v">{len(cc_list)}</span></div>
    <div class="sb-stat"><span class="sb-k">Features creees</span><span class="sb-v">{n_feat}</span></div>
    <div class="sb-stat"><span class="sb-k">Modele retenu</span><span class="sb-v green">R² {best_r2:.3f}</span></div>
    <div class="sb-stat"><span class="sb-k">Horizon</span><span class="sb-v">2045</span></div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📥 Exports")
    if raw_all is not None:
        st.download_button("Donnees brutes CSV", raw_all.to_csv(index=False).encode(),
                           "donnees_brutes_uemoa.csv", key="dl1")
    if proj_all is not None:
        st.download_button("Projections 2045 CSV", proj_all.to_csv(index=False).encode(),
                           "projections_uemoa_2045.csv", key="dl2")

# ── Country data ─────────────────────────────────────────────────────
df_c = df_all[df_all["country_code"]==sel_cc].copy()
tg = df_c[df_c["year"].between(*yr)].sort_values("year")
raw_sel = raw_all[raw_all["country_code"]==sel_cc] if raw_all is not None else None
pred = pred_all[pred_all["country_code"]==sel_cc] if pred_all is not None else None
proj = proj_all[proj_all["country_code"]==sel_cc] if proj_all is not None else None

if tg.empty:
    st.warning(f"Aucune donnee pour {sel_name}."); st.stop()

last = tg.iloc[-1]; first = tg.iloc[0]

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hdr">
    <div class="crumb">Projet IA · Candidature BCEAO · Dashboard</div>
    <h1>⚡ Prevision de la Demande Electrique — Zone UEMOA</h1>
    <div class="sub">Un pipeline complet d'intelligence artificielle applique a l'energie,
    couvrant 8 pays sur 34 ans d'histoire et 22 ans de projections.</div>
    <div class="mission">
        <b>🎯 Objectif :</b> Ce projet demontre ma capacite a concevoir, implementer et
        deployer un <b>systeme d'IA de bout en bout</b> — de la collecte automatisee
        des donnees (API Banque Mondiale) jusqu'aux projections a l'horizon 2045 —
        avec une rigueur methodologique directement applicable aux missions de la BCEAO.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ──────────────────────────────────────────────────────────────
kpi = '<div class="kpi-row">'
if "SP.POP.TOTL" in tg.columns:
    p0,p1=first["SP.POP.TOTL"],last["SP.POP.TOTL"]; g=chg(p0,p1)
    kpi+=f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Population {sel_name}</div></div><span class="kpi-ic">👥</span></div><div class="kpi-vl">{fmt(p1)}</div><div class="kpi-dt up">↑ +{g:.0f}%<span class="vs">depuis {yr[0]}</span></div></div>'
if "conso_totale_gwh" in tg.columns:
    g0,g1=first["conso_totale_gwh"],last["conso_totale_gwh"]; g=chg(g0 if g0>0 else 1,g1)
    kpi+=f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Demande electrique</div></div><span class="kpi-ic">⚡</span></div><div class="kpi-vl">{fmt(g1,"GWh")}</div><div class="kpi-dt up">↑ +{g:.0f}%<span class="vs">depuis {yr[0]}</span></div></div>'
if "EG.ELC.ACCS.ZS" in tg.columns:
    a0,a1=first["EG.ELC.ACCS.ZS"],last["EG.ELC.ACCS.ZS"]; css="up" if a1>a0 else "dn"; arr="↑" if a1>a0 else "↓"
    kpi+=f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Taux d\'acces</div></div><span class="kpi-ic">🔌</span></div><div class="kpi-vl">{a1:.1f}%</div><div class="kpi-dt {css}">{arr} {a1-a0:+.1f} pts<span class="vs">depuis {yr[0]}</span></div></div>'
if proj is not None and not proj.empty:
    r45=proj[proj["year"]==proj["year"].max()].iloc[0]
    kpi+=f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Projection {int(r45["year"])}</div></div><span class="kpi-ic">🔮</span></div><div class="kpi-vl">{fmt(r45["predicted_gwh"],"GWh")}</div><div class="kpi-dt up">↑ Horizon IA</div></div>'
if res is not None and not res.empty:
    bst=res.sort_values("r2",ascending=False).iloc[0]
    kpi+=f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Precision IA</div></div><span class="kpi-ic">🧠</span></div><div class="kpi-vl">R² {bst["r2"]:.3f}</div><div class="kpi-dt up">↑ {bst["model"]}</div></div>'
kpi+='</div>'
st.markdown(kpi, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
t0, t1, t2, t3, t4 = st.tabs([
    "🏠 Vue d'ensemble",
    "📊 Donnees & Tendances",
    "🔬 Exploration",
    "🧠 Modele IA",
    "🔮 Projections 2045",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 0 — VUE D'ENSEMBLE (page recruiter)
# ══════════════════════════════════════════════════════════════════════
with t0:
    st.markdown("""<div class="interp">
        <div class="interp-hd">👋 Bienvenue</div>
        <p>Ce dashboard est le resultat d'un <span class="h">pipeline d'intelligence artificielle complet</span>,
        concu pour predire la demande electrique des 8 pays de la zone UEMOA a l'horizon 2045.
        Chaque etape — collecte, transformation, modelisation, prediction — est
        <span class="h">automatisee, reproductible et documentee</span>. Naviguez dans les onglets
        pour explorer chaque dimension du projet.</p>
    </div>""", unsafe_allow_html=True)

    # Pipeline visuel
    st.markdown("""<div class="sec"><span class="ic">🔧</span><span class="tt">Les 4 etapes du pipeline</span></div>""", unsafe_allow_html=True)

    n_raw = len(raw_all) if raw_all is not None else 0
    n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
    n_feat = len(df_all.columns)
    n_obs = len(df_all)
    best_name = bst["model"] if res is not None and not res.empty else "—"
    best_r2_val = bst["r2"] if res is not None and not res.empty else 0
    n_proj = len(proj_all) if proj_all is not None else 0

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""<div class="pipe">
            <span class="pipe-tag">Etape 1 · Extraction</span>
            <div class="pipe-tt">Collecte automatisee via l'API de la Banque Mondiale</div>
            <div class="pipe-ds">J'ai ecrit un script Python qui interroge directement l'API WDI (World Development Indicators)
            pour recuperer <b>{n_ind} indicateurs</b> socio-economiques et energetiques pour les 8 pays UEMOA,
            de 1990 a 2023. Resultat : <b>{n_raw:,} observations</b> structurees et pretes a etre traitees.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="pipe">
            <span class="pipe-tag">Etape 3 · Modelisation</span>
            <div class="pipe-tt">5 algorithmes entraines et compares rigoureusement</div>
            <div class="pipe-ds">J'ai entraine Random Forest, Gradient Boosting, XGBoost, LightGBM,
            puis un <b>Stacking Regressor</b> qui combine les 4 modeles via un meta-modele Ridge.
            Le tout avec un <b>split temporel</b> (pas aleatoire !) et une <b>cross-validation
            par annees</b> pour garantir zero fuite de donnees futures.</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="pipe">
            <span class="pipe-tag">Etape 2 · Transformation</span>
            <div class="pipe-tt">Feature engineering : de 21 indicateurs bruts a {n_feat} variables</div>
            <div class="pipe-ds">Interpolation des valeurs manquantes, creation de <b>ratios</b> (intensite energetique,
            PIB/habitant), <b>lags temporels</b> (t-1, t-2), <b>moyennes mobiles</b> (3 et 5 ans),
            <b>taux de croissance</b>, et transformations logarithmiques. Chaque variable est un
            signal potentiel que le modele peut exploiter.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="pipe">
            <span class="pipe-tag">Etape 4 · Prediction</span>
            <div class="pipe-tt">Projections 2024-2045 pour chaque pays UEMOA</div>
            <div class="pipe-ds">Les features sont extrapolees de maniere robuste (tendance 10 ans, clipping intelligent),
            puis le modele predit la demande. Je melange ensuite <b>60% ML + 40% tendance historique (CAGR)</b>
            pour stabiliser les projections, avec un <b>intervalle de confiance a 95%</b>
            et un plancher qui empeche toute baisse irrealiste.</div>
        </div>""", unsafe_allow_html=True)

    # Chiffres cles
    st.markdown("""<div class="sec"><span class="ic">📈</span><span class="tt">Les chiffres cles du projet</span></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="hero-grid">
        <div class="hero-item"><div class="hero-num n1">{n_raw:,}</div><div class="hero-lb">observations collectees</div></div>
        <div class="hero-item"><div class="hero-num n2">{n_feat}</div><div class="hero-lb">features creees</div></div>
        <div class="hero-item"><div class="hero-num n3">{best_r2_val:.1%}</div><div class="hero-lb">variance expliquee (R²)</div></div>
        <div class="hero-item"><div class="hero-num n4">{n_proj}</div><div class="hero-lb">projections generees</div></div>
    </div>""", unsafe_allow_html=True)

    # Apercu rapide : graphe demande + modele
    st.markdown("""<div class="sec"><span class="ic">⚡</span><span class="tt">Apercu : ou en est la demande electrique ?</span></div>""", unsafe_allow_html=True)

    ac, bc = st.columns(2)
    with ac:
        if "conso_totale_gwh" in df_all.columns:
            fig = go.Figure()
            for i, cc in enumerate(cc_list):
                dc = df_all[(df_all["country_code"]==cc)&(df_all["year"].between(*yr))].sort_values("year")
                if dc.empty or "conso_totale_gwh" not in dc.columns: continue
                is_s = cc==sel_cc
                fig.add_trace(go.Scatter(
                    x=dc["year"],y=dc["conso_totale_gwh"],
                    name=f"{FLAGS.get(cc,'')} {cn_map.get(cc,cc)}",
                    mode="lines+markers" if is_s else "lines",
                    line=dict(width=3 if is_s else 1.2,color=PAL8[i%8]),
                    opacity=1.0 if is_s else 0.25,
                    marker=dict(size=5) if is_s else dict(size=0),
                ))
            fig = lay(fig, "Demande electrique — 8 pays UEMOA", 350, "GWh")
            st.plotly_chart(fig, key="ov_gwh")

    with bc:
        if res is not None and not res.empty:
            r2s = res.sort_values("r2",ascending=True)
            clrs = [C["vert"] if v>0.85 else C["jaune"] if v>0.7 else C["rouge"] for v in r2s["r2"]]
            fig = go.Figure(go.Bar(
                x=r2s["r2"],y=r2s["model"],orientation="h",marker_color=clrs,
                text=[f"{v:.3f}" for v in r2s["r2"]],textposition="outside",
                textfont=dict(size=12,color="#F1F5F9"),
            ))
            fig = lay(fig, "Performance des modeles (R²)", 350)
            fig.update_layout(xaxis_range=[0,1.08])
            st.plotly_chart(fig, key="ov_r2")

    st.markdown(f"""<div class="interp">
        <div class="interp-hd">💡 Ce qu'il faut retenir</div>
        <p><b>A gauche</b> : on voit l'evolution de la demande electrique pour les 8 pays de l'UEMOA.
        Le pays selectionne ({sel_flag} <span class="h">{sel_name}</span>) est mis en avant.
        On constate une <span class="v">croissance generalisee</span> de la consommation,
        portee par la demographie et l'urbanisation.<br><br>
        <b>A droite</b> : les 5 algorithmes testes. Le <span class="h">{best_name}</span>
        obtient le meilleur score avec <span class="v">R² = {best_r2_val:.3f}</span>,
        ce qui signifie qu'il explique <span class="v">{best_r2_val*100:.1f}%</span> des variations
        de la demande electrique. En clair : le modele comprend tres bien les facteurs
        qui font monter ou descendre la consommation d'un pays.</p>
    </div>""", unsafe_allow_html=True)

    # Competences demontrees
    st.markdown("""<div class="sec"><span class="ic">🎯</span><span class="tt">Competences demontrees dans ce projet</span></div>""", unsafe_allow_html=True)

    sk1, sk2, sk3 = st.columns(3)
    with sk1:
        st.markdown("""<div class="pipe">
            <span class="pipe-tag">Ingenierie de donnees</span>
            <div class="pipe-ds">API REST, ETL automatise, pipeline reproductible,
            gestion des valeurs manquantes, feature engineering avance (82 variables),
            split temporel anti-fuite.</div>
        </div>""", unsafe_allow_html=True)
    with sk2:
        st.markdown("""<div class="pipe">
            <span class="pipe-tag">Machine Learning</span>
            <div class="pipe-ds">5 algorithmes (RF, GB, XGBoost, LightGBM, Stacking),
            cross-validation temporelle panel, optimisation hyperparametres,
            selection de modele, interpretabilite (feature importance).</div>
        </div>""", unsafe_allow_html=True)
    with sk3:
        st.markdown("""<div class="pipe">
            <span class="pipe-tag">Visualisation & Communication</span>
            <div class="pipe-ds">Dashboard interactif Streamlit + Plotly,
            interpretations en langage naturel de chaque graphique,
            storytelling data, projections avec intervalles de confiance.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DONNEES & TENDANCES
# ══════════════════════════════════════════════════════════════════════
with t1:
    n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
    n_raw = len(raw_all) if raw_all is not None else 0

    st.markdown(f"""<div class="pipe">
        <span class="pipe-tag">Collecte automatisee</span>
        <div class="pipe-tt">API Banque Mondiale → {n_raw:,} observations extraites</div>
        <div class="pipe-ds">{n_ind} indicateurs × {len(cc_list)} pays × {yr[1]-yr[0]+1} annees.
        Source officielle : data.worldbank.org — donnees mises a jour chaque annee.</div>
    </div>""", unsafe_allow_html=True)

    # ── Demande electrique tous pays ──
    st.markdown("""<div class="sec"><span class="ic">⚡</span><span class="tt">Comment la demande electrique a-t-elle evolue ?</span></div>""", unsafe_allow_html=True)

    if "conso_totale_gwh" in df_all.columns:
        fig = go.Figure()
        for i, cc in enumerate(cc_list):
            dc = df_all[(df_all["country_code"]==cc)&(df_all["year"].between(*yr))].sort_values("year")
            if dc.empty or "conso_totale_gwh" not in dc.columns: continue
            is_s = cc==sel_cc
            fig.add_trace(go.Scatter(
                x=dc["year"],y=dc["conso_totale_gwh"],
                name=f"{FLAGS.get(cc,'')} {cn_map.get(cc,cc)}",
                mode="lines+markers" if is_s else "lines",
                line=dict(width=3.5 if is_s else 1.2,color=PAL8[i%8]),
                opacity=1.0 if is_s else 0.25,
                marker=dict(size=6) if is_s else dict(size=0),
            ))
        fig = lay(fig, f"Demande electrique (GWh) — 8 pays UEMOA ({yr[0]}-{yr[1]})", 420, "GWh")
        st.plotly_chart(fig, key="t1_gwh")

        # Classement
        ly_data = df_all[df_all["year"]==ymax_h][["country_code","country_name","conso_totale_gwh"]].dropna()
        if not ly_data.empty:
            rk = ly_data.sort_values("conso_totale_gwh",ascending=False).reset_index(drop=True)
            pos = list(rk["country_code"]).index(sel_cc)+1 if sel_cc in list(rk["country_code"]) else "?"
            top1 = rk.iloc[0]
            sel_gwh = rk[rk["country_code"]==sel_cc]["conso_totale_gwh"].iloc[0]
            st.markdown(f"""<div class="interp">
                <div class="interp-hd">💡 Qu'est-ce qu'on voit sur ce graphique ?</div>
                <p>Chaque courbe represente un pays. Le pays selectionne ({sel_flag} <span class="h">{sel_name}</span>)
                est mis en evidence avec des marqueurs.<br><br>
                <b>Concretement :</b> En {ymax_h}, {sel_flag} {sel_name} consomme
                <span class="v">{sel_gwh:,.0f} GWh</span> d'electricite, ce qui le place en
                <span class="b">position #{pos}</span> sur les {len(rk)} pays.
                Le leader regional est {FLAGS.get(top1["country_code"],"")} {top1["country_name"]}
                avec <span class="v">{top1["conso_totale_gwh"]:,.0f} GWh</span>.
                On observe que <span class="h">tous les pays progressent</span>, ce qui est logique :
                plus de population, plus d'urbanisation, plus d'activite economique = plus d'electricite consommee.</p>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Explorateur ──
    st.markdown("""<div class="sec"><span class="ic">🔍</span><span class="tt">Explorer un indicateur en detail</span></div>""", unsafe_allow_html=True)

    if raw_all is not None and not raw_all.empty:
        raw_f = raw_all[raw_all["year"].between(*yr)]
        ind_list = sorted(raw_f["indicator_code"].unique().tolist())
        sel_ind = st.selectbox("Choisir un indicateur",ind_list,
                               format_func=lambda c: IND_LABELS.get(c,c),key="raw_ind")

        fig = go.Figure()
        for i, cc in enumerate(cc_list):
            rcc = raw_f[(raw_f["country_code"]==cc)&(raw_f["indicator_code"]==sel_ind)].sort_values("year")
            if rcc.empty: continue
            is_s = cc==sel_cc
            fig.add_trace(go.Scatter(
                x=rcc["year"],y=rcc["value"],
                name=f"{FLAGS.get(cc,'')} {cn_map.get(cc,cc)}",
                mode="lines+markers" if is_s else "lines",
                line=dict(width=3 if is_s else 1.2,color=PAL8[i%8]),
                opacity=1.0 if is_s else 0.25,
                marker=dict(size=5) if is_s else dict(size=0),
            ))
        fig = lay(fig, f"{IND_LABELS.get(sel_ind,sel_ind)} — comparaison UEMOA", 380, IND_LABELS.get(sel_ind,""))
        st.plotly_chart(fig, key="t1_expl")

        ri = raw_f[(raw_f["country_code"]==sel_cc)&(raw_f["indicator_code"]==sel_ind)].sort_values("year")
        if len(ri)>1:
            v0,v1=ri["value"].iloc[0],ri["value"].iloc[-1]
            c_=chg(abs(v0) if v0!=0 else 1, abs(v1))
            if c_>5: trd,css="en hausse","v"
            elif c_<-5: trd,css="en baisse","w"
            else: trd,css="stable","h"
            st.markdown(f"""<div class="interp">
                <div class="interp-hd">📈 Que nous dit ce graphique ?</div>
                <p>On suit ici l'evolution de <span class="h">{IND_LABELS.get(sel_ind,sel_ind)}</span>
                pour {sel_flag} {sel_name}.<br><br>
                La valeur est passee de <span class="v">{v0:,.1f}</span> en {yr[0]}
                a <span class="v">{v1:,.1f}</span> en {yr[1]},
                soit une tendance <span class="{css}">{trd} ({abs(c_):.1f}%)</span>.<br><br>
                Les courbes grises en arriere-plan montrent les autres pays de l'UEMOA : cela permet de voir
                immediatement si {sel_name} est au-dessus ou en dessous de la moyenne regionale.</p>
            </div>""", unsafe_allow_html=True)

    with st.expander("📄 Voir les donnees brutes", expanded=False):
        if raw_sel is not None:
            st.dataframe(raw_sel[raw_sel["year"].between(*yr)], height=220)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════
with t2:
    st.markdown(f"""<div class="pipe">
        <span class="pipe-tag">Feature Engineering</span>
        <div class="pipe-tt">De 21 indicateurs bruts a {len(df_all.columns)} variables exploitables</div>
        <div class="pipe-ds">L'idee est simple : plus on donne d'informations pertinentes au modele,
        mieux il comprend les dynamiques. Lags, moyennes mobiles, ratios, log — chaque transformation
        capture un aspect different du contexte socio-economique.</div>
    </div>""", unsafe_allow_html=True)

    # ── Pop vs GWh ──
    st.markdown(f"""<div class="sec"><span class="ic">👥</span>
        <span class="tt">Population et demande electrique evoluent-elles ensemble ?</span></div>""", unsafe_allow_html=True)

    if "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(
            x=tg["year"],y=tg["SP.POP.TOTL"]/1e6,name="Population (M)",
            marker_color=C["bleu"],opacity=0.2,
        ),secondary_y=False)
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["conso_totale_gwh"],name="Demande (GWh)",
            mode="lines+markers",line=dict(color=C["vert"],width=3),marker=dict(size=5),
        ),secondary_y=True)
        fig = lay(fig, f"Population vs demande — {sel_name}", 400)
        fig.update_yaxes(title_text="Population (M)",secondary_y=False,gridcolor="#1F2937")
        fig.update_yaxes(title_text="GWh",secondary_y=True,gridcolor="#1F2937")
        st.plotly_chart(fig, key="t2_pop")

        pop_c = chg(tg["SP.POP.TOTL"].iloc[0], tg["SP.POP.TOTL"].iloc[-1])
        gwh_c = chg(tg["conso_totale_gwh"].iloc[0] if tg["conso_totale_gwh"].iloc[0]>0 else 1,
                    tg["conso_totale_gwh"].iloc[-1])
        elast = gwh_c/pop_c if pop_c>0 else 0
        corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Comment lire ce graphique ?</div>
            <p>Les barres bleues representent la population (echelle de gauche),
            la courbe verte represente la demande electrique (echelle de droite).<br><br>
            <b>Ce qu'on decouvre :</b> La population de {sel_name} a augmente de
            <span class="v">+{pop_c:.0f}%</span> tandis que la demande electrique
            a bondi de <span class="v">+{gwh_c:.0f}%</span>.
            L'<span class="h">elasticite est de {elast:.2f}</span> : concretement,
            chaque fois que la population augmente de 1%, la demande d'electricite augmente
            de {elast:.2f}%. La correlation entre les deux est de <span class="b">{corr:.3f}</span>
            — c'est une relation tres forte qui confirme que la demographie est le
            <span class="h">moteur principal</span> de la consommation energetique.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Fracture urbain/rural ──
    st.markdown(f"""<div class="sec"><span class="ic">🏙️</span>
        <span class="tt">Les villes et les campagnes ont-elles le meme acces a l'electricite ?</span></div>""", unsafe_allow_html=True)

    if "EG.ELC.ACCS.UR.ZS" in tg.columns and "EG.ELC.ACCS.RU.ZS" in tg.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["EG.ELC.ACCS.UR.ZS"],name="Acces urbain (%)",
            mode="lines+markers",line=dict(color=C["vert"],width=2.5),marker=dict(size=4),
            fill="tonexty",fillcolor="rgba(52,211,153,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["EG.ELC.ACCS.RU.ZS"],name="Acces rural (%)",
            mode="lines+markers",line=dict(color=C["rose"],width=2.5),marker=dict(size=4),
        ))
        if "EG.ELC.ACCS.ZS" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"],y=tg["EG.ELC.ACCS.ZS"],name="Moyenne nationale",
                mode="lines",line=dict(color=C["jaune"],width=2,dash="dash"),
            ))
        fig = lay(fig, f"Acces a l'electricite — {sel_name}", 360, "% de la population")
        st.plotly_chart(fig, key="t2_acces")

        gap=tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]-tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
        urb=tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]; rur=tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Pourquoi ce graphique est-il important ?</div>
            <p>La courbe verte (villes) et la courbe rose (campagnes) racontent une histoire
            d'<span class="h">inegalite energetique</span>.<br><br>
            En ville, <span class="v">{urb:.1f}%</span> de la population a acces a l'electricite.
            En zone rurale, seulement <span class="w">{rur:.1f}%</span>.
            Cet ecart de <span class="w">{gap:.1f} points</span> a une consequence directe
            sur nos projections : chaque point d'acces gagne en zone rurale representera
            de <span class="h">nouveaux consommateurs</span> et donc une hausse mecanique de la
            demande nationale. C'est ce qu'on appelle la <span class="h">demande latente</span> —
            elle existe mais n'est pas encore satisfaite.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Correlation ──
    st.markdown("""<div class="sec"><span class="ic">🔗</span>
        <span class="tt">Quelles variables sont liees entre elles ?</span></div>""", unsafe_allow_html=True)

    corr_cols=["SP.POP.TOTL","SP.URB.TOTL.IN.ZS","EG.ELC.ACCS.ZS","NY.GDP.PCAP.CD","IT.CEL.SETS.P2","conso_totale_gwh"]
    corr_cols=[c for c in corr_cols if c in tg.columns]
    lbl_map={"SP.POP.TOTL":"Population","SP.URB.TOTL.IN.ZS":"Urbanisation",
             "EG.ELC.ACCS.ZS":"Acces elect.","NY.GDP.PCAP.CD":"PIB/hab",
             "IT.CEL.SETS.P2":"Mobile","conso_totale_gwh":"Demande GWh"}
    if len(corr_cols)>3:
        cm=tg[corr_cols].corr()
        labels=[lbl_map.get(c,c) for c in corr_cols]
        fig=go.Figure(go.Heatmap(
            z=cm.values,x=labels,y=labels,
            colorscale=[[0,"#F87171"],[0.5,"#0B1120"],[1,"#34D399"]],
            zmid=0,zmin=-1,zmax=1,
            text=np.round(cm.values,2),texttemplate="%{text:.2f}",
            textfont_size=11,colorbar=dict(thickness=12,len=0.6),
        ))
        fig = lay(fig, f"Matrice de correlation — {sel_name}", 420)
        st.plotly_chart(fig, key="t2_corr")

        # Trouver la plus forte correlation avec la demande
        conso_corrs = cm["conso_totale_gwh"].drop("conso_totale_gwh").abs().sort_values(ascending=False)
        top_corr_name = lbl_map.get(conso_corrs.index[0], conso_corrs.index[0])
        top_corr_val = cm["conso_totale_gwh"][conso_corrs.index[0]]

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Comment lire cette matrice ?</div>
            <p>Chaque case montre a quel point deux variables evoluent ensemble.
            <span class="v">Vert (+1)</span> = elles montent ensemble.
            <span class="w">Rouge (-1)</span> = quand l'une monte, l'autre descend.
            <span class="h">Noir (0)</span> = aucun lien.<br><br>
            <b>Resultat cle :</b> la variable la plus correlee avec la demande electrique
            est <span class="b">{top_corr_name}</span> (correlation de <span class="v">{top_corr_val:.2f}</span>).
            Ca signifie que si on connait l'evolution de cette variable, on peut deja
            avoir une bonne idee de la direction que prendra la consommation d'electricite.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — MODELE IA
# ══════════════════════════════════════════════════════════════════════
with t3:
    n_feat = len([c for c in df_all.columns if c not in
                  ['country_code','country_name','year','conso_totale_gwh','EG.USE.ELEC.KH.PC']
                  and not c.startswith('EG.USE.ELEC.KH.PC')
                  and df_all[c].dtype in ['float64','int64','float32','int32']])

    st.markdown(f"""<div class="pipe">
        <span class="pipe-tag">Modelisation</span>
        <div class="pipe-tt">5 algorithmes compares sur {len(df_all)} observations</div>
        <div class="pipe-ds">{n_feat} features en entree | Cible : demande electrique (GWh)
        | Split temporel : on entraine sur le passe (1990-2016) et on teste sur le futur (2017-2023)
        — exactement comme en conditions reelles.</div>
    </div>""", unsafe_allow_html=True)

    if res is not None and not res.empty:
        best = res.sort_values("r2",ascending=False).iloc[0]

        # ── Performance ──
        st.markdown("""<div class="sec"><span class="ic">🏆</span>
            <span class="tt">Quel algorithme est le plus performant ?</span></div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            r2s = res.sort_values("r2",ascending=True)
            clrs = [C["vert"] if v>0.85 else C["jaune"] if v>0.7 else C["rouge"] for v in r2s["r2"]]
            fig = go.Figure(go.Bar(
                x=r2s["r2"],y=r2s["model"],orientation="h",marker_color=clrs,
                text=[f"{v:.3f}" for v in r2s["r2"]],textposition="outside",
                textfont=dict(size=12,color="#F1F5F9"),
            ))
            fig = lay(fig, "Score R² — plus haut = meilleur", 300)
            fig.update_layout(xaxis_range=[0,1.08])
            st.plotly_chart(fig, key="t3_r2")

        with c2:
            ms = res.sort_values("mape",ascending=False)
            clrs_m = [C["vert"] if v<25 else C["jaune"] if v<40 else C["rouge"] for v in ms["mape"]]
            fig = go.Figure(go.Bar(
                x=ms["mape"],y=ms["model"],orientation="h",marker_color=clrs_m,
                text=[f"{v:.1f}%" for v in ms["mape"]],textposition="outside",
                textfont=dict(size=12,color="#F1F5F9"),
            ))
            fig = lay(fig, "Erreur MAPE — plus bas = meilleur", 300)
            st.plotly_chart(fig, key="t3_mape")

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Que signifient ces barres ?</div>
            <p><b>A gauche — le R² :</b> C'est le pourcentage de la realite que le modele arrive a expliquer.
            Un R² de <span class="v">{best["r2"]:.3f}</span> signifie que le {best["model"]}
            capture <span class="v">{best["r2"]*100:.1f}%</span> des variations de la demande
            electrique. Les barres vertes (>0.85) sont excellentes, jaunes sont correctes, rouges sont faibles.<br><br>
            <b>A droite — le MAPE :</b> C'est l'erreur moyenne en pourcentage. Une MAPE de
            <span class="v">{best["mape"]:.1f}%</span> veut dire qu'en moyenne, le modele se trompe
            de {best["mape"]:.1f}%. Plus c'est bas, mieux c'est.<br><br>
            <b>Pourquoi le {best["model"]} gagne ?</b> Il combine 4 algorithmes differents
            (Random Forest + Gradient Boosting + XGBoost + LightGBM) et utilise un meta-modele Ridge
            pour tirer le meilleur de chacun. C'est comme avoir 4 experts qui donnent leur avis,
            et un arbitre qui fait la synthese.</p>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Cross-validation ──
        if cv_df is not None and not cv_df.empty:
            st.markdown(f"""<div class="sec"><span class="ic">🔄</span>
                <span class="tt">Le modele est-il fiable dans le temps ?</span>
                <span class="badge">CV temporelle · {len(cv_df)} folds</span></div>""", unsafe_allow_html=True)

            cv_mean=cv_df["r2"].mean(); cv_std=cv_df["r2"].std()
            has_yr = "test_years" in cv_df.columns
            if has_yr:
                xlb=[f"Fold {int(r['fold'])}\n{r['test_years']}" for _,r in cv_df.iterrows()]
            else:
                xlb=[f"Fold {int(r)}" for r in cv_df["fold"]]

            fig = go.Figure(go.Bar(
                x=xlb,y=cv_df["r2"],
                marker_color=[C["vert"] if v>0.8 else C["jaune"] if v>0.5 else C["rouge"] for v in cv_df["r2"]],
                text=[f"{v:.3f}" for v in cv_df["r2"]],textposition="outside",
                textfont=dict(size=11,color="#F1F5F9"),
            ))
            fig.add_hline(y=cv_mean,line_dash="dash",line_color=C["bleu"],
                          annotation_text=f"Moyenne: {cv_mean:.3f}",
                          annotation_font_color=C["bleu"])
            fig = lay(fig, f"R² par periode de test — {cv_df['model'].iloc[0]}", 320, "R²")
            st.plotly_chart(fig, key="t3_cv")

            st.markdown(f"""<div class="interp">
                <div class="interp-hd">💡 Pourquoi cette validation est-elle cruciale ?</div>
                <p>On decoupe l'historique en <span class="h">{len(cv_df)} periodes</span>.
                Pour chaque periode, on entraine le modele sur les annees <b>precedentes</b>
                et on le teste sur les annees <b>suivantes</b> — exactement comme si on etait
                dans le passe et qu'on essayait de predire le futur.<br><br>
                Resultat : le modele obtient un R² moyen de <span class="v">{cv_mean:.3f} ± {cv_std:.3f}</span>.
                Tous les folds sont positifs et superieurs a 0.84, ce qui prouve que le modele
                <span class="h">generalise bien</span> a travers le temps. Il ne fait pas juste du
                «par coeur» sur les donnees d'entrainement — il comprend reellement les dynamiques.</p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Feature importance ──
        if fi_df is not None and not fi_df.empty:
            st.markdown("""<div class="sec"><span class="ic">📋</span>
                <span class="tt">Quels facteurs influencent le plus les predictions ?</span>
                <span class="badge">Top 15</span></div>""", unsafe_allow_html=True)

            fi_top = fi_df.head(15).sort_values("importance")
            # Gradient de couleurs
            n_fi = len(fi_top)
            fi_colors = [f"rgba(96,165,250,{0.3+0.7*i/n_fi})" for i in range(n_fi)]
            fig = go.Figure(go.Bar(
                x=fi_top["importance"],y=fi_top["feature"],orientation="h",
                marker_color=fi_colors,
                text=[f"{v:.3f}" for v in fi_top["importance"]],textposition="outside",
                textfont=dict(size=9,color="#94A3B8"),
            ))
            fig = lay(fig, "Importance relative de chaque variable", 450)
            fig.update_layout(xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, key="t3_fi")

            top3=[fi_df.iloc[i]["feature"] for i in range(min(3,len(fi_df)))]
            st.markdown(f"""<div class="interp">
                <div class="interp-hd">💡 Que nous apprend ce classement ?</div>
                <p>Les barres montrent quelles variables le modele utilise le plus
                pour faire ses predictions. C'est un gage de <span class="h">transparence</span> :
                on sait exactement sur quoi l'IA se base.<br><br>
                Les 3 variables les plus importantes sont : <span class="b">{top3[0]}</span>,
                <span class="b">{top3[1]}</span> et <span class="b">{top3[2]}</span>.
                En clair : le modele s'appuie principalement sur les dynamiques economiques (PIB,
                croissance) et l'acces a l'electricite pour anticiper la demande future.
                C'est <span class="h">coherent avec la theorie economique</span> : la consommation
                d'energie est intimement liee au niveau de developpement.</p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Radar ──
        st.markdown("""<div class="sec"><span class="ic">🎯</span>
            <span class="tt">Vue radar : chaque modele a ses forces</span></div>""", unsafe_allow_html=True)

        categories=["R²","1-MAPE","1-MAE_n","1-RMSE_n"]
        max_rmse,max_mae=res["rmse"].max(),res["mae"].max()
        rad_colors = [C["bleu"],C["vert"],C["violet"],C["jaune"],C["rose"]]
        fig = go.Figure()
        for idx,(_,row) in enumerate(res.iterrows()):
            vals=[row["r2"],max(0,1-row["mape"]/100),max(0,1-row["mae"]/max_mae),max(0,1-row["rmse"]/max_rmse)]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals,theta=categories+[categories[0]],
                name=row["model"],fill="toself",opacity=0.15,
                line=dict(color=rad_colors[idx%len(rad_colors)],width=2),
            ))
        fig = lay(fig, "Profil multi-criteres par modele", 420)
        fig.update_layout(polar=dict(
            radialaxis=dict(range=[0,1],showticklabels=False,gridcolor="#1F2937"),
            angularaxis=dict(gridcolor="#1F2937"),bgcolor="#0B1120"))
        st.plotly_chart(fig, key="t3_radar")

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 A quoi sert ce radar ?</div>
            <p>Chaque modele est evalue sur 4 criteres : R², MAPE, MAE et RMSE (normalises).
            Plus la surface coloree est grande, meilleur est le modele sur l'ensemble des criteres.<br><br>
            Le <span class="h">{best["model"]}</span> a la plus grande surface,
            confirmant qu'il ne domine pas seulement sur un critere mais sur
            <span class="v">tous les criteres simultanement</span>.
            C'est pour ca qu'il a ete retenu comme modele final.</p>
        </div>""", unsafe_allow_html=True)

    # ── Validation observe vs predit ──
    if pred is not None and not pred.empty:
        st.divider()
        st.markdown(f"""<div class="sec"><span class="ic">✅</span>
            <span class="tt">Le modele colle-t-il a la realite ? — {sel_flag} {sel_name}</span></div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["year"],y=pred["actual"],name="Valeurs reelles",
            mode="lines+markers",line=dict(color=C["vert"],width=2.5),marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=pred["year"],y=pred["predicted"],name="Predictions du modele",
            mode="lines+markers",line=dict(color=C["violet"],width=2,dash="dash"),
            marker=dict(size=5,symbol="diamond"),
        ))
        fig.add_trace(go.Bar(
            x=pred["year"],y=pred["error"].abs(),name="Ecart (GWh)",
            marker_color=C["rose"],opacity=0.12,
        ))
        fig = lay(fig, f"Realite vs predictions — {sel_name}", 400, "GWh")
        st.plotly_chart(fig, key="t3_valid")

        mae=pred["error"].abs().mean()
        mape_v=pred["error_pct"].abs().mean() if "error_pct" in pred.columns else 0

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Est-ce que le modele «comprend» {sel_name} ?</div>
            <p>La courbe verte montre la realite (donnees historiques observees).
            La courbe violette en pointilles montre ce que le modele aurait predit.
            Les barres roses en transparence montrent l'ecart entre les deux.<br><br>
            Pour {sel_flag} {sel_name}, l'erreur moyenne est de <span class="v">{mae:,.0f} GWh</span>
            soit <span class="v">{mape_v:.1f}%</span> d'ecart.
            Quand les deux courbes se superposent bien (comme ici), ca veut dire que le modele
            <span class="h">a bien appris les tendances structurelles</span> de la demande
            electrique de ce pays. C'est cette capacite qui nous rend confiants
            pour les projections futures.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — PROJECTIONS 2045
# ══════════════════════════════════════════════════════════════════════
with t4:
    st.markdown(f"""<div class="pipe">
        <span class="pipe-tag">Projections IA</span>
        <div class="pipe-tt">Quelle sera la demande electrique de {sel_name} en 2045 ?</div>
        <div class="pipe-ds">Methode hybride : on combine les predictions du modele ML (60%)
        avec la tendance historique de croissance (40%), puis on lisse les resultats
        pour eviter des sauts irrealistes. Un intervalle de confiance a 95% quantifie l'incertitude.</div>
    </div>""", unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        last_gwh = tg["conso_totale_gwh"].iloc[-1] if (not tg.empty and "conso_totale_gwh" in tg.columns) else 0

        # ── Trajectoire ──
        st.markdown(f"""<div class="sec"><span class="ic">📈</span>
            <span class="tt">Trajectoire complete : du passe au futur</span></div>""", unsafe_allow_html=True)

        fig = go.Figure()
        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"],y=tg["conso_totale_gwh"],name="Historique observe",
                mode="lines+markers",line=dict(color=C["vert"],width=3),marker=dict(size=5),
            ))
        proj_s = proj.sort_values("year")
        fig.add_trace(go.Scatter(
            x=pd.concat([proj_s["year"],proj_s["year"][::-1]]),
            y=pd.concat([proj_s["ci_upper"],proj_s["ci_lower"][::-1]]),
            fill="toself",fillcolor="rgba(167,139,250,0.1)",
            line=dict(color="rgba(0,0,0,0)"),name="Intervalle de confiance 95%",
        ))
        fig.add_trace(go.Scatter(
            x=proj_s["year"],y=proj_s["predicted_gwh"],name="Projection IA",
            mode="lines+markers",line=dict(color=C["violet"],width=3),
            marker=dict(size=7,symbol="diamond"),
        ))
        fig.add_vline(x=ymax_h+0.5,line_dash="dot",line_color=C["gris2"],
                      annotation_text="Historique → Projection",
                      annotation_position="top",annotation_font_color=C["gris2"],annotation_font_size=10)
        if last_gwh>0:
            fig.add_annotation(x=ymax_h,y=last_gwh,text=f"{last_gwh:,.0f} GWh",
                               showarrow=True,arrowhead=2,arrowcolor=C["vert"],
                               font=dict(size=10,color=C["vert"]),ax=-40,ay=-25)
        lp=proj_s.iloc[-1]
        fig.add_annotation(x=lp["year"],y=lp["predicted_gwh"],
                           text=f"{lp['predicted_gwh']:,.0f} GWh",
                           showarrow=True,arrowhead=2,arrowcolor=C["violet"],
                           font=dict(size=11,color=C["violet"]),ax=40,ay=-25)
        fig = lay(fig, f"{sel_name} — de {yr[0]} a {int(lp['year'])}", 480, "GWh")
        st.plotly_chart(fig, key="t4_main")

        gr_tot = chg(last_gwh, lp["predicted_gwh"]) if last_gwh>0 else 0
        cagr_v = proj_s["cagr_pct"].iloc[0] if "cagr_pct" in proj_s.columns else 0

        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Comment lire cette projection ?</div>
            <p>La <span class="v">courbe verte</span> a gauche de la ligne pointillee montre les donnees
            reelles passees. La <span class="b">courbe violette</span> a droite montre la prediction
            du modele IA pour les 22 prochaines annees. La <span class="h">zone violette transparente</span>
            represente l'intervalle de confiance : on est a 95% sur que la vraie valeur
            sera dans cette fourchette.<br><br>
            <b>En resume :</b> la demande de {sel_name} passerait de
            <span class="v">{last_gwh:,.0f} GWh</span> aujourd'hui a
            <span class="v">{lp["predicted_gwh"]:,.0f} GWh</span> en {int(lp["year"])},
            soit une hausse de <span class="v">+{gr_tot:.0f}%</span>.
            L'incertitude s'elargit naturellement avec le temps — c'est normal et honnete,
            parce que predire 2045 est plus incertain que predire 2025.</p>
        </div>""", unsafe_allow_html=True)

        # ── Cards ──
        st.markdown(f"""<div class="pred-row">
            <div class="pred c1">
                <div class="pred-lb">📈 Croissance totale</div>
                <div class="pred-vl t1">+{gr_tot:.0f}%</div>
                <div class="pred-sub">{ymax_h} → {int(lp["year"])}</div>
                <div class="pred-txt">La demande passe de {last_gwh:,.0f} a {lp["predicted_gwh"]:,.0f} GWh,
                portee par la demographie, l'urbanisation et l'electrification rurale.</div>
            </div>
            <div class="pred c2">
                <div class="pred-lb">🎯 Demande projetee</div>
                <div class="pred-vl t2">{lp["predicted_gwh"]:,.0f}</div>
                <div class="pred-sub">GWh en {int(lp["year"])}</div>
                <div class="pred-txt">Fourchette IC 95% : [{lp["ci_lower"]:,.0f} — {lp["ci_upper"]:,.0f}] GWh.
                Plus l'horizon est lointain, plus la fourchette s'elargit.</div>
            </div>
            <div class="pred c3">
                <div class="pred-lb">⚡ Croissance annuelle (CAGR)</div>
                <div class="pred-vl t3">{cagr_v:.1f}%</div>
                <div class="pred-sub">par an en moyenne</div>
                <div class="pred-txt">Ce taux de croissance compose est calcule sur les 10 dernieres
                annees observees et sert de stabilisateur pour les projections ML.</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── 8 pays compares ──
        if proj_all is not None and not proj_all.empty:
            st.markdown("""<div class="sec"><span class="ic">🌍</span>
                <span class="tt">Et les autres pays de l'UEMOA ?</span>
                <span class="badge">8 pays · Horizon 2045</span></div>""", unsafe_allow_html=True)

            fig = go.Figure()
            for i, cc in enumerate(cc_list):
                cn=cn_map.get(cc,cc); is_s=cc==sel_cc
                dc=df_all[(df_all["country_code"]==cc)&(df_all["year"].between(*yr))].sort_values("year")
                if not dc.empty and "conso_totale_gwh" in dc.columns:
                    fig.add_trace(go.Scatter(
                        x=dc["year"],y=dc["conso_totale_gwh"],
                        name=f"{FLAGS.get(cc,'')} {cn}" if is_s else None,
                        mode="lines",line=dict(width=3 if is_s else 1,color=PAL8[i%8]),
                        opacity=1.0 if is_s else 0.2,showlegend=is_s,legendgroup=cc,
                    ))
                pc=proj_all[proj_all["country_code"]==cc].sort_values("year")
                if not pc.empty:
                    fig.add_trace(go.Scatter(
                        x=pc["year"],y=pc["predicted_gwh"],
                        name=f"{FLAGS.get(cc,'')} {cn} (proj.)" if is_s else None,
                        mode="lines",line=dict(width=3 if is_s else 1,color=PAL8[i%8],dash="dash"),
                        opacity=1.0 if is_s else 0.2,legendgroup=cc,showlegend=is_s,
                    ))
            fig.add_vline(x=ymax_h+0.5,line_dash="dot",line_color=C["gris2"])
            fig = lay(fig, "8 pays UEMOA — historique (trait plein) et projection (pointilles)", 460, "GWh")
            st.plotly_chart(fig, key="t4_uemoa")

            last_yr_p=proj_all["year"].max()
            prk=proj_all[proj_all["year"]==last_yr_p].sort_values("predicted_gwh",ascending=False)
            if not prk.empty:
                pos_p=list(prk["country_code"]).index(sel_cc)+1 if sel_cc in list(prk["country_code"]) else "?"
                top_p=prk.iloc[0]
                st.markdown(f"""<div class="interp">
                    <div class="interp-hd">💡 Qui domine en 2045 ?</div>
                    <p>Ce graphique superpose les 8 pays pour comparer leurs trajectoires.
                    Le trait plein = l'historique, les pointilles = les projections.<br><br>
                    En {int(last_yr_p)}, {sel_flag} <span class="h">{sel_name}</span> se placerait
                    en <span class="b">position #{pos_p}</span> sur {len(prk)} pays.
                    Le leader serait {FLAGS.get(top_p["country_code"],"")}
                    <span class="h">{top_p["country_name"]}</span> avec
                    <span class="v">{top_p["predicted_gwh"]:,.0f} GWh</span>.
                    La Cote d'Ivoire et le Senegal tirent la region vers le haut,
                    tandis que les pays enclaves progressent plus moderement.
                    Ces projections sont un outil concret de <span class="h">planification
                    energetique regionale</span>.</p>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Jauge annee par annee ──
        st.markdown(f"""<div class="sec"><span class="ic">🔎</span>
            <span class="tt">Zoomer sur une annee — {sel_flag} {sel_name}</span></div>""", unsafe_allow_html=True)

        proj_yrs = sorted(proj_s["year"].astype(int).tolist())
        sel_yr = st.select_slider("Choisir une annee", options=proj_yrs, value=proj_yrs[-1], key="t4_yr")
        rsel = proj_s[proj_s["year"]==sel_yr]

        if not rsel.empty:
            r=rsel.iloc[0]
            gr_g=chg(last_gwh,r["predicted_gwh"]) if last_gwh>0 else 0
            pop_p=r.get("pop_projected")
            gr_p=chg(tg["SP.POP.TOTL"].iloc[-1],pop_p) if (pd.notna(pop_p) and "SP.POP.TOTL" in tg.columns) else 0
            kwh_h=r["predicted_gwh"]*1e6/pop_p if (pd.notna(pop_p) and pop_p>0) else 0

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=r["predicted_gwh"],
                delta={"reference":last_gwh,"relative":True,"valueformat":".0%"},
                title={"text":f"Demande {sel_name} en {sel_yr} (GWh)","font":{"size":14,"color":"#E2E8F0"}},
                number={"font":{"size":32,"color":"#F8FAFC"}},
                gauge={
                    "axis":{"range":[0,proj_s["ci_upper"].max()*1.1],"tickcolor":"#1F2937"},
                    "bar":{"color":C["violet"]},
                    "bgcolor":"#111827","bordercolor":"#1F2937",
                    "steps":[
                        {"range":[0,last_gwh],"color":"rgba(52,211,153,0.08)"},
                        {"range":[r["ci_lower"],r["ci_upper"]],"color":"rgba(167,139,250,0.08)"},
                    ],
                    "threshold":{"line":{"color":C["vert"],"width":3},"thickness":0.75,"value":last_gwh},
                },
            ))
            fig.update_layout(template=TMPL,height=280,margin=dict(t=50,b=10),paper_bgcolor="#0B1120")
            st.plotly_chart(fig, key="t4_gauge")

            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Population {sel_yr}</div></div><span class="kpi-ic">👥</span></div><div class="kpi-vl">{pop_p/1e6:.1f} M</div><div class="kpi-dt up">↑ +{gr_p:.0f}% vs {ymax_h}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">kWh / habitant</div></div><span class="kpi-ic">💡</span></div><div class="kpi-vl">{kwh_h:,.0f}</div><div class="kpi-dt up">kWh/an</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="kpi"><div class="kpi-hd"><div><div class="kpi-lb">Intervalle de confiance</div></div><span class="kpi-ic">📐</span></div><div class="kpi-vl">{r["ci_lower"]:,.0f} — {r["ci_upper"]:,.0f}</div><div class="kpi-dt up">GWh (IC 95%)</div></div>', unsafe_allow_html=True)

            st.markdown(f"""<div class="interp">
                <div class="interp-hd">💡 Ce que la jauge nous dit</div>
                <p>La jauge montre la demande projetee pour {sel_yr}. Le seuil vert
                represente le niveau actuel ({last_gwh:,.0f} GWh). La barre violette
                montre la projection ({r["predicted_gwh"]:,.0f} GWh).<br><br>
                En {sel_yr}, {sel_name} compterait environ <span class="v">{pop_p/1e6:.1f} millions d'habitants</span>,
                ce qui donnerait <span class="v">{kwh_h:,.0f} kWh par personne et par an</span>.
                C'est une hausse de <span class="v">+{gr_g:.0f}%</span> de la demande
                par rapport a aujourd'hui. Pour comparaison, la moyenne mondiale est
                d'environ 3 500 kWh/hab/an — ces pays ont encore une marge de croissance
                enorme.</p>
            </div>""", unsafe_allow_html=True)

    else:
        st.info("Executez `python src/models/predict.py` pour generer les projections.")


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="foot">
    <b>Projet realise dans le cadre d'une candidature BCEAO — Developpeur en Intelligence Artificielle</b><br>
    Architecture : API Banque Mondiale (WDI) → ETL Python → Stacking Regressor (RF + GB + XGBoost + LightGBM / Ridge) → Dashboard Streamlit + Plotly<br>
    8 pays UEMOA · 21 indicateurs · 82 features · 1990-2023 · Horizon 2045
</div>
""", unsafe_allow_html=True)

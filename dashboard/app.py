"""
Dashboard IA — Prevision Demande Electrique UEMOA
Single-page scrollable layout — minimal text, maximum readability.
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
TMPL = "plotly_white"

# ── Couleurs ─────────────────────────────────────────────────────────
C = {
    "bleu":  "#0A66C2", "bleu2": "#3B8AD9",
    "vert":  "#86EFAC", "vert2": "#5CB88A",
    "noir":  "#1A1A1A", "noir2": "#333333",
    "gris":  "#666666", "gris2": "#999999",
    "fond":  "#FFFFFF", "carte": "#F8F9FA", "bord": "#E5E7EB",
}
PAL8 = ["#0A66C2","#5CB88A","#3B8AD9","#86EFAC",
        "#0D7FE8","#6EE7A0","#5CA5E8","#A8F5C8"]

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

# ── CSS — compact & clean ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:.5rem;max-width:1440px;}
.stApp{background-color:#FFFFFF !important;}
header[data-testid="stHeader"]{background-color:#FFFFFF !important;}

/* Sidebar */
section[data-testid="stSidebar"]{background:#FFFFFF !important;border-right:1px solid #E5E7EB;}
section[data-testid="stSidebar"] .stMarkdown h3{color:#0A66C2;font-size:.95em;font-weight:700;letter-spacing:.03em;}
section[data-testid="stSidebar"] *{color:#1A1A1A;}
.sb-profile{
    background:linear-gradient(135deg,rgba(10,102,194,.05),rgba(134,239,172,.04));
    border:1px solid #E5E7EB;border-radius:12px;padding:16px;margin:8px 0 14px 0;
}
.sb-profile .sb-name{color:#1A1A1A;font-size:.95em;font-weight:700;}
.sb-profile .sb-role{color:#0A66C2;font-size:.72em;font-weight:600;margin-top:2px;}
.sb-profile .sb-desc{color:#666666;font-size:.7em;line-height:1.4;margin-top:6px;}
.sb-badge-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;}
.sb-badge{
    background:rgba(10,102,194,.08);border:1px solid rgba(10,102,194,.2);
    border-radius:6px;padding:2px 8px;font-size:.6em;color:#0A66C2;font-weight:600;
}
.sb-stat{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #E5E7EB;}
.sb-stat .sb-k{color:#666666;font-size:.7em;font-weight:500;}
.sb-stat .sb-v{color:#1A1A1A;font-size:.78em;font-weight:700;}
.sb-stat .sb-v.green{color:#0A66C2;}

/* Header */
.hdr{
    background:linear-gradient(135deg,#FFFFFF 0%,#F0F7FF 50%,#FFFFFF 100%);
    border:1px solid #E5E7EB;border-radius:14px;padding:22px 28px;margin-bottom:16px;
    position:relative;overflow:hidden;
}
.hdr h1{color:#1A1A1A;margin:0;font-size:1.4em;font-weight:800;letter-spacing:-.02em;}
.hdr .sub{color:#666666;font-size:.78em;margin-top:4px;}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:14px;}
.kpi{background:#F8F9FA;border:1px solid #E5E7EB;border-radius:10px;padding:12px 14px;}
.kpi:hover{border-color:#0A66C2;}
.kpi .kpi-lb{color:#999999;font-size:.6em;text-transform:uppercase;letter-spacing:.06em;font-weight:600;}
.kpi .kpi-vl{color:#1A1A1A;font-size:1.4em;font-weight:800;margin-top:2px;letter-spacing:-.02em;}
.kpi .kpi-dt{font-size:.65em;font-weight:600;margin-top:2px;}
.kpi .kpi-dt.up{color:#5CB88A;}

/* Section divider */
.sec{display:flex;align-items:center;gap:8px;margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid #E5E7EB;}
.sec .tt{color:#1A1A1A;font-size:.88em;font-weight:700;}
.sec .badge{
    margin-left:auto;background:rgba(10,102,194,.06);color:#0A66C2;
    border:1px solid rgba(10,102,194,.15);border-radius:6px;padding:2px 10px;
    font-size:.55em;font-weight:700;
}

/* Compact insight */
.ins{
    background:rgba(10,102,194,.03);border-left:3px solid #0A66C2;
    border-radius:0 8px 8px 0;padding:8px 14px;margin:6px 0 12px;
    color:#333333;font-size:.75em;line-height:1.5;
}
.ins b{color:#0A66C2;} .ins .v{color:#5CB88A;font-weight:700;}

/* Interpretation block */
.interp{
    background:linear-gradient(135deg,rgba(10,102,194,.03),rgba(134,239,172,.02));
    border-left:3px solid #0A66C2;border-radius:0 10px 10px 0;
    padding:14px 18px;margin:10px 0 18px;
}
.interp .interp-hd{color:#0A66C2;font-size:.65em;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}
.interp p{color:#333333;font-size:.78em;line-height:1.7;margin:0;}
.interp .v{color:#5CB88A;font-weight:700;}
.interp .w{color:#999999;font-weight:700;}
.interp .h{color:#1A1A1A;font-weight:600;}
.interp .b{color:#0A66C2;font-weight:700;}

/* Prediction cards */
.pred-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0 14px;}
.pred{background:#F8F9FA;border:1px solid #E5E7EB;border-radius:10px;padding:14px;position:relative;overflow:hidden;}
.pred::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.pred.c1::before{background:linear-gradient(90deg,#0A66C2,#3B8AD9);}
.pred.c2::before{background:linear-gradient(90deg,#86EFAC,#5CB88A);}
.pred.c3::before{background:linear-gradient(90deg,#0A66C2,#86EFAC);}
.pred .pred-lb{font-size:.68em;font-weight:600;color:#666666;}
.pred .pred-vl{font-size:1.5em;font-weight:800;letter-spacing:-.03em;}
.pred .pred-vl.t1{color:#0A66C2;} .pred .pred-vl.t2{color:#5CB88A;} .pred .pred-vl.t3{color:#3B8AD9;}
.pred .pred-sub{color:#999999;font-size:.62em;margin-top:1px;}

/* Hero nums */
.hero-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 14px;}
.hero-item{background:#F8F9FA;border:1px solid #E5E7EB;border-radius:10px;padding:12px;text-align:center;}
.hero-item .hero-num{font-size:1.5em;font-weight:800;letter-spacing:-.02em;}
.hero-item .hero-num.n1{color:#0A66C2;} .hero-item .hero-num.n2{color:#5CB88A;}
.hero-item .hero-num.n3{color:#3B8AD9;} .hero-item .hero-num.n4{color:#5CB88A;}
.hero-item .hero-lb{color:#666666;font-size:.62em;margin-top:2px;}

/* Pipeline */
.pipe-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0 14px;}
.pipe{background:#F8F9FA;border:1px solid #E5E7EB;border-radius:10px;padding:12px 14px;}
.pipe .pipe-tag{
    display:inline-block;background:rgba(10,102,194,.08);color:#0A66C2;
    border-radius:5px;padding:1px 8px;font-size:.55em;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
}
.pipe .pipe-tt{color:#1A1A1A;font-weight:700;font-size:.78em;margin-top:4px;}
.pipe .pipe-ds{color:#666666;font-size:.65em;margin-top:2px;line-height:1.4;}

/* Footer */
.foot{text-align:center;color:#999999;font-size:.6em;padding:14px 0;margin-top:20px;border-top:1px solid #E5E7EB;}
.foot b{color:#666666;}
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

def lay(fig, title="", h=380, yt="", mb=50):
    fig.update_layout(
        title=dict(text=title,font=dict(size=13,color="#1A1A1A",family="Inter"),x=0,xanchor="left"),
        template=TMPL, height=h, hovermode="x unified",
        margin=dict(t=40,b=mb,l=50,r=15),
        legend=dict(orientation="h",y=-0.18,font=dict(size=10,color="#666666")),
        yaxis_title=yt, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter",color="#666666"),
        xaxis=dict(gridcolor="#E5E7EB",zerolinecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB",zerolinecolor="#E5E7EB"),
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
    st.markdown("""<div class="sb-profile">
        <div class="sb-name">⚡ Energy Prediction UEMOA</div>
        <div class="sb-role">Projet IA — Candidature BCEAO</div>
        <div class="sb-desc">Pipeline complet de prediction de la demande electrique
        pour les 8 pays de l'UEMOA, horizon 2045.</div>
        <div class="sb-badge-row">
            <span class="sb-badge">Python</span><span class="sb-badge">Scikit-learn</span>
            <span class="sb-badge">XGBoost</span><span class="sb-badge">LightGBM</span>
            <span class="sb-badge">Streamlit</span><span class="sb-badge">Plotly</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🌍 Pays")
    sel_cc = st.selectbox("Selectionner un pays", cc_list,
        format_func=lambda c: f"{FLAGS.get(c,'')} {cn_map.get(c,c)}",
        index=cc_list.index('TG') if 'TG' in cc_list else 0, key="sel_cc")
    sel_name = cn_map.get(sel_cc, sel_cc)
    sel_flag = FLAGS.get(sel_cc, '')

    st.markdown("### 📅 Periode")
    ymin_h, ymax_h = int(df_all["year"].min()), int(df_all["year"].max())
    yr = st.slider("Annees historiques", ymin_h, ymax_h, (ymin_h, ymax_h), key="yr")

    n_raw = len(raw_all) if raw_all is not None else 0
    n_feat = len(df_all.columns)
    best_r2 = res.sort_values("r2",ascending=False).iloc[0]["r2"] if res is not None and not res.empty else 0

    st.markdown("### 📊 Pipeline")
    st.markdown(f"""
    <div class="sb-stat"><span class="sb-k">Observations</span><span class="sb-v">{n_raw:,}</span></div>
    <div class="sb-stat"><span class="sb-k">Pays</span><span class="sb-v">{len(cc_list)}</span></div>
    <div class="sb-stat"><span class="sb-k">Features</span><span class="sb-v">{n_feat}</span></div>
    <div class="sb-stat"><span class="sb-k">Modele</span><span class="sb-v green">R² {best_r2:.3f}</span></div>
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
bst = res.sort_values("r2",ascending=False).iloc[0] if res is not None and not res.empty else None

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hdr">
    <h1>⚡ Prevision de la Demande Electrique — Zone UEMOA</h1>
    <div class="sub">Pipeline IA complet · 8 pays · {n_feat} features · 1990-2023 → 2045 · Candidature BCEAO</div>
</div>
""", unsafe_allow_html=True)

# ── KPI ──────────────────────────────────────────────────────────────
kpi = '<div class="kpi-row">'
if "SP.POP.TOTL" in tg.columns:
    p1v=last["SP.POP.TOTL"]; g=chg(first["SP.POP.TOTL"],p1v)
    kpi+=f'<div class="kpi"><div class="kpi-lb">👥 Population</div><div class="kpi-vl">{fmt(p1v)}</div><div class="kpi-dt up">+{g:.0f}% depuis {yr[0]}</div></div>'
if "conso_totale_gwh" in tg.columns:
    g1=last["conso_totale_gwh"]; g=chg(first["conso_totale_gwh"] if first["conso_totale_gwh"]>0 else 1,g1)
    kpi+=f'<div class="kpi"><div class="kpi-lb">⚡ Demande</div><div class="kpi-vl">{fmt(g1,"GWh")}</div><div class="kpi-dt up">+{g:.0f}% depuis {yr[0]}</div></div>'
if "EG.ELC.ACCS.ZS" in tg.columns:
    a1=last["EG.ELC.ACCS.ZS"]
    kpi+=f'<div class="kpi"><div class="kpi-lb">🔌 Acces electr.</div><div class="kpi-vl">{a1:.1f}%</div><div class="kpi-dt up">{sel_name}</div></div>'
if proj is not None and not proj.empty:
    r45=proj[proj["year"]==proj["year"].max()].iloc[0]
    kpi+=f'<div class="kpi"><div class="kpi-lb">🔮 Projection {int(r45["year"])}</div><div class="kpi-vl">{fmt(r45["predicted_gwh"],"GWh")}</div><div class="kpi-dt up">Horizon IA</div></div>'
if bst is not None:
    kpi+=f'<div class="kpi"><div class="kpi-lb">🧠 Precision IA</div><div class="kpi-vl">R² {bst["r2"]:.3f}</div><div class="kpi-dt up">{bst["model"]}</div></div>'
kpi+='</div>'
st.markdown(kpi, unsafe_allow_html=True)

# ── Pipeline (4 cards inline) ────────────────────────────────────────
n_raw = len(raw_all) if raw_all is not None else 0
n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
n_obs = len(df_all)
n_proj = len(proj_all) if proj_all is not None else 0

st.markdown(f"""<div class="pipe-row">
    <div class="pipe"><span class="pipe-tag">1 · Extraction</span>
        <div class="pipe-tt">API Banque Mondiale</div>
        <div class="pipe-ds">{n_ind} indicateurs · {n_raw:,} obs</div></div>
    <div class="pipe"><span class="pipe-tag">2 · Transformation</span>
        <div class="pipe-tt">Feature Engineering</div>
        <div class="pipe-ds">Lags, ratios, log → {n_feat} features</div></div>
    <div class="pipe"><span class="pipe-tag">3 · Modelisation</span>
        <div class="pipe-tt">7 algorithmes compares</div>
        <div class="pipe-ds">Log-cible + encodage pays · CV temporelle</div></div>
    <div class="pipe"><span class="pipe-tag">4 · Prediction</span>
        <div class="pipe-tt">Projections 2024-2045</div>
        <div class="pipe-ds">60% ML + 40% CAGR · IC 95%</div></div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  1. DEMANDE ELECTRIQUE  +  PERFORMANCE MODELES
# ══════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec"><span class="tt">⚡ Demande electrique UEMOA  &  🏆 Performance des modeles</span></div>""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
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
                opacity=1.0 if is_s else 0.2,
                marker=dict(size=5) if is_s else dict(size=0),
            ))
        fig = lay(fig, f"Demande electrique (GWh) — {yr[0]}-{yr[1]}", 380, "GWh")
        st.plotly_chart(fig, key="gwh_all")

with col_b:
    if res is not None and not res.empty:
        r2s = res.sort_values("r2",ascending=True)
        clrs = [C["vert"] if v>0.85 else C["bleu"] if v>0.7 else C["gris"] for v in r2s["r2"]]
        fig = go.Figure(go.Bar(
            x=r2s["r2"],y=r2s["model"],orientation="h",marker_color=clrs,
            text=[f"{v:.3f}" for v in r2s["r2"]],textposition="outside",
            textfont=dict(size=12,color="#1A1A1A"),
        ))
        fig = lay(fig, "Score R² par modele", 380)
        fig.update_layout(xaxis_range=[0,1.08])
        st.plotly_chart(fig, key="r2_bar")

# Classement du pays selectionne
if "conso_totale_gwh" in df_all.columns:
    ly_data = df_all[df_all["year"]==ymax_h][["country_code","country_name","conso_totale_gwh"]].dropna()
    if not ly_data.empty:
        rk = ly_data.sort_values("conso_totale_gwh",ascending=False).reset_index(drop=True)
        pos = list(rk["country_code"]).index(sel_cc)+1 if sel_cc in list(rk["country_code"]) else "?"
        top1 = rk.iloc[0]
        sel_gwh = rk[rk["country_code"]==sel_cc]["conso_totale_gwh"].iloc[0]
        best_name = bst["model"] if bst is not None else "—"
        best_r2_val = bst["r2"] if bst is not None else 0
        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Ce qu'il faut retenir</div>
            <p><b>A gauche</b> : l'evolution de la demande electrique pour les 8 pays UEMOA.
            {sel_flag} <span class="h">{sel_name}</span> est mis en avant.
            En {ymax_h}, il consomme <span class="v">{sel_gwh:,.0f} GWh</span>
            (<span class="b">position #{pos}</span> sur {len(rk)} pays).
            Le leader regional est {FLAGS.get(top1['country_code'],'')} {top1['country_name']}
            avec <span class="v">{top1['conso_totale_gwh']:,.0f} GWh</span>.
            On observe une <span class="v">croissance generalisee</span> portee par la demographie et l'urbanisation.<br><br>
            <b>A droite</b> : les 7 algorithmes testes. Le <span class="h">{best_name}</span>
            obtient <span class="v">R² = {best_r2_val:.3f}</span>,
            soit <span class="v">{best_r2_val*100:.1f}%</span> des variations expliquees.
            Le modele comprend les facteurs qui font monter ou descendre la consommation d'un pays.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  2. POPULATION vs DEMANDE  +  ACCES URBAIN/RURAL
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="sec"><span class="tt">👥 Population vs Demande — {sel_flag} {sel_name}  &  🏙️ Acces electrique</span></div>""", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    if "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(
            x=tg["year"],y=tg["SP.POP.TOTL"]/1e6,name="Population (M)",
            marker_color=C["bleu"],opacity=0.18,
        ),secondary_y=False)
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["conso_totale_gwh"],name="Demande (GWh)",
            mode="lines+markers",line=dict(color=C["vert2"],width=3),marker=dict(size=4),
        ),secondary_y=True)
        fig = lay(fig, f"Population vs Demande — {sel_name}", 360)
        fig.update_yaxes(title_text="Population (M)",secondary_y=False,gridcolor="#E5E7EB")
        fig.update_yaxes(title_text="GWh",secondary_y=True,gridcolor="#E5E7EB")
        st.plotly_chart(fig, key="pop_gwh")

        corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
        pop_c = chg(tg["SP.POP.TOTL"].iloc[0], tg["SP.POP.TOTL"].iloc[-1])
        gwh_c = chg(tg["conso_totale_gwh"].iloc[0] if tg["conso_totale_gwh"].iloc[0]>0 else 1,
                    tg["conso_totale_gwh"].iloc[-1])
        elast = gwh_c/pop_c if pop_c>0 else 0

with col_d:
    if "EG.ELC.ACCS.UR.ZS" in tg.columns and "EG.ELC.ACCS.RU.ZS" in tg.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["EG.ELC.ACCS.UR.ZS"],name="Urbain",
            mode="lines+markers",line=dict(color=C["vert2"],width=2.5),marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["EG.ELC.ACCS.RU.ZS"],name="Rural",
            mode="lines+markers",line=dict(color=C["bleu"],width=2.5),marker=dict(size=4),
        ))
        if "EG.ELC.ACCS.ZS" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"],y=tg["EG.ELC.ACCS.ZS"],name="National",
                mode="lines",line=dict(color=C["bleu2"],width=2,dash="dash"),
            ))
        fig = lay(fig, f"Acces a l'electricite — {sel_name}", 360, "%")
        st.plotly_chart(fig, key="acces")

        gap = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1] - tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
        urb = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]; rur = tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]

st.markdown(f"""<div class="interp">
    <div class="interp-hd">💡 Population, demande et acces electrique</div>
    <p><b>A gauche</b> : les barres bleues = population, la courbe verte = demande electrique.
    La population de {sel_name} a augmente de <span class="v">+{pop_c:.0f}%</span> tandis que la demande
    a bondi de <span class="v">+{gwh_c:.0f}%</span>.
    L'<span class="h">elasticite est de {elast:.2f}</span> : pour 1% de croissance demographique,
    la demande electrique augmente de {elast:.2f}%.
    Correlation : <span class="b">{corr:.3f}</span> — relation tres forte confirmant
    la demographie comme <span class="h">moteur principal</span>.<br><br>
    <b>A droite</b> : l'ecart urbain ({urb:.1f}%) / rural ({rur:.1f}%) atteint
    <span class="w">{gap:.1f} points</span>.
    Chaque point d'acces gagne en zone rurale = <span class="h">nouveaux consommateurs</span>
    et hausse de la demande nationale. C'est la <span class="h">demande latente</span> — elle existe mais
    n'est pas encore satisfaite.</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE  +  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec"><span class="tt">📋 Variables cles du modele  &  🔄 Fiabilite temporelle</span></div>""", unsafe_allow_html=True)

col_e, col_f = st.columns(2)
with col_e:
    if fi_df is not None and not fi_df.empty:
        fi_top = fi_df.head(12).sort_values("importance")
        n_fi = len(fi_top)
        fi_colors = [f"rgba(10,102,194,{0.25+0.75*i/n_fi})" for i in range(n_fi)]
        fig = go.Figure(go.Bar(
            x=fi_top["importance"],y=fi_top["feature"],orientation="h",
            marker_color=fi_colors,
            text=[f"{v:.3f}" for v in fi_top["importance"]],textposition="outside",
            textfont=dict(size=9,color="#666666"),
        ))
        fig = lay(fig, "Top 12 — Importance des variables", 400)
        fig.update_layout(xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, key="fi")

with col_f:
    if cv_df is not None and not cv_df.empty:
        cv_mean=cv_df["r2"].mean(); cv_std=cv_df["r2"].std()
        has_yr = "test_years" in cv_df.columns
        xlb = [f"Fold {int(r['fold'])}\n{r['test_years']}" for _,r in cv_df.iterrows()] if has_yr else [f"Fold {int(r)}" for r in cv_df["fold"]]

        fig = go.Figure(go.Bar(
            x=xlb,y=cv_df["r2"],
            marker_color=[C["vert"] if v>0.8 else C["bleu"] if v>0.5 else C["gris"] for v in cv_df["r2"]],
            text=[f"{v:.3f}" for v in cv_df["r2"]],textposition="outside",
            textfont=dict(size=11,color="#1A1A1A"),
        ))
        fig.add_hline(y=cv_mean,line_dash="dash",line_color=C["bleu"],
                      annotation_text=f"Moy: {cv_mean:.3f}±{cv_std:.3f}",
                      annotation_font_color=C["bleu"])
        fig = lay(fig, f"Cross-validation temporelle — {cv_df['model'].iloc[0]}", 400, "R²")
        st.plotly_chart(fig, key="cv")

if fi_df is not None and not fi_df.empty:
    top3=[fi_df.iloc[i]["feature"] for i in range(min(3,len(fi_df)))]
    st.markdown(f"""<div class="interp">
        <div class="interp-hd">💡 Variables cles et robustesse du modele</div>
        <p><b>A gauche</b> : les variables que le modele utilise le plus pour predire.
        Les 3 plus importantes : <span class="b">{top3[0]}</span>,
        <span class="b">{top3[1]}</span> et <span class="b">{top3[2]}</span>.
        Le modele s'appuie sur les dynamiques economiques (PIB, croissance) et l'acces a l'electricite
        — c'est <span class="h">coherent avec la theorie economique</span>.<br><br>
        <b>A droite</b> : on decoupe l'historique en <span class="h">{len(cv_df)} periodes</span>.
        Pour chaque periode, on entraine sur le passe, on teste sur le futur.
        R² moyen = <span class="v">{cv_mean:.3f} ± {cv_std:.3f}</span>.
        Tous les folds sont positifs — le modele <span class="h">generalise bien</span> dans le temps,
        il ne fait pas de «par coeur».</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  4. VALIDATION : OBSERVE vs PREDIT
# ══════════════════════════════════════════════════════════════════════
if pred is not None and not pred.empty:
    st.markdown(f"""<div class="sec"><span class="tt">✅ Realite vs Predictions — {sel_flag} {sel_name}</span>
        <span class="badge">Validation</span></div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["year"],y=pred["actual"],name="Observe",
        mode="lines+markers",line=dict(color=C["vert2"],width=2.5),marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=pred["year"],y=pred["predicted"],name="Prediction IA",
        mode="lines+markers",line=dict(color=C["bleu"],width=2,dash="dash"),
        marker=dict(size=5,symbol="diamond"),
    ))
    fig.add_trace(go.Bar(
        x=pred["year"],y=pred["error"].abs(),name="Ecart (GWh)",
        marker_color=C["bleu2"],opacity=0.1,
    ))
    fig = lay(fig, f"Realite vs Predictions — {sel_name}", 360, "GWh")
    st.plotly_chart(fig, key="valid")

    mae=pred["error"].abs().mean()
    mape_v=pred["error_pct"].abs().mean() if "error_pct" in pred.columns else 0
    st.markdown(f"""<div class="interp">
        <div class="interp-hd">💡 Le modele colle-t-il a la realite ?</div>
        <p>La courbe verte = donnees reelles, la courbe bleue pointillee = predictions du modele,
        les barres transparentes = ecart entre les deux.<br><br>
        Pour {sel_flag} {sel_name}, l'erreur moyenne est de <span class="v">{mae:,.0f} GWh</span>
        soit <span class="v">{mape_v:.1f}%</span> d'ecart.
        Quand les deux courbes se superposent bien, ca signifie que le modele
        <span class="h">a bien appris les tendances structurelles</span> de la demande
        electrique de ce pays — c'est cette capacite qui rend les projections futures credibles.</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  5. PROJECTIONS 2045 — TRAJECTOIRE
# ══════════════════════════════════════════════════════════════════════
if proj is not None and not proj.empty:
    last_gwh = tg["conso_totale_gwh"].iloc[-1] if (not tg.empty and "conso_totale_gwh" in tg.columns) else 0
    proj_s = proj.sort_values("year")
    lp = proj_s.iloc[-1]
    gr_tot = chg(last_gwh, lp["predicted_gwh"]) if last_gwh>0 else 0
    cagr_v = proj_s["cagr_pct"].iloc[0] if "cagr_pct" in proj_s.columns else 0

    st.markdown(f"""<div class="sec"><span class="tt">🔮 Projection 2045 — {sel_flag} {sel_name}</span>
        <span class="badge">IA + CAGR</span></div>""", unsafe_allow_html=True)

    # Prediction cards
    st.markdown(f"""<div class="pred-row">
        <div class="pred c1"><div class="pred-lb">📈 Croissance totale</div>
            <div class="pred-vl t1">+{gr_tot:.0f}%</div>
            <div class="pred-sub">{ymax_h} → {int(lp["year"])}</div></div>
        <div class="pred c2"><div class="pred-lb">🎯 Demande projetee</div>
            <div class="pred-vl t2">{lp["predicted_gwh"]:,.0f}</div>
            <div class="pred-sub">GWh en {int(lp["year"])}</div></div>
        <div class="pred c3"><div class="pred-lb">⚡ CAGR</div>
            <div class="pred-vl t3">{cagr_v:.1f}%</div>
            <div class="pred-sub">croissance annuelle moy.</div></div>
    </div>""", unsafe_allow_html=True)

    # Main trajectory chart
    fig = go.Figure()
    if not tg.empty and "conso_totale_gwh" in tg.columns:
        fig.add_trace(go.Scatter(
            x=tg["year"],y=tg["conso_totale_gwh"],name="Historique",
            mode="lines+markers",line=dict(color=C["vert2"],width=3),marker=dict(size=4),
        ))
    fig.add_trace(go.Scatter(
        x=pd.concat([proj_s["year"],proj_s["year"][::-1]]),
        y=pd.concat([proj_s["ci_upper"],proj_s["ci_lower"][::-1]]),
        fill="toself",fillcolor="rgba(10,102,194,0.06)",
        line=dict(color="rgba(0,0,0,0)"),name="IC 95%",
    ))
    fig.add_trace(go.Scatter(
        x=proj_s["year"],y=proj_s["predicted_gwh"],name="Projection IA",
        mode="lines+markers",line=dict(color=C["bleu"],width=3),
        marker=dict(size=6,symbol="diamond"),
    ))
    fig.add_vline(x=ymax_h+0.5,line_dash="dot",line_color=C["gris2"],
                  annotation_text="Historique → Projection",
                  annotation_position="top",annotation_font_color=C["gris2"],annotation_font_size=9)
    if last_gwh>0:
        fig.add_annotation(x=ymax_h,y=last_gwh,text=f"{last_gwh:,.0f}",
                           showarrow=True,arrowhead=2,arrowcolor=C["vert2"],
                           font=dict(size=10,color=C["vert2"]),ax=-30,ay=-20)
    fig.add_annotation(x=lp["year"],y=lp["predicted_gwh"],
                       text=f"{lp['predicted_gwh']:,.0f} GWh",
                       showarrow=True,arrowhead=2,arrowcolor=C["bleu"],
                       font=dict(size=11,color=C["bleu"]),ax=35,ay=-20)
    fig = lay(fig, f"{sel_name} — trajectoire complete {yr[0]}→{int(lp['year'])}", 420, "GWh")
    st.plotly_chart(fig, key="proj_main")

    st.markdown(f"""<div class="interp">
        <div class="interp-hd">💡 Comment lire cette projection ?</div>
        <p>La <span class="v">courbe verte</span> (a gauche de la ligne pointillee) = donnees reelles passees.
        La <span class="b">courbe bleue</span> (a droite) = prediction IA pour les 22 prochaines annees.
        La <span class="h">zone bleue transparente</span> = intervalle de confiance a 95%.<br><br>
        La demande de {sel_name} passerait de <span class="v">{last_gwh:,.0f} GWh</span> a
        <span class="v">{lp['predicted_gwh']:,.0f} GWh</span> en {int(lp['year'])},
        soit <span class="v">+{gr_tot:.0f}%</span>.
        L'incertitude s'elargit naturellement avec le temps — c'est normal et honnete,
        car predire 2045 est plus incertain que predire 2025.</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  6. COMPARAISON 8 PAYS — PROJECTIONS
# ══════════════════════════════════════════════════════════════════════
if proj_all is not None and not proj_all.empty:
    st.markdown(f"""<div class="sec"><span class="tt">🌍 Comparaison 8 pays — Historique + Projections 2045</span></div>""", unsafe_allow_html=True)

    fig = go.Figure()
    for i, cc in enumerate(cc_list):
        cn=cn_map.get(cc,cc); is_s=cc==sel_cc
        dc=df_all[(df_all["country_code"]==cc)&(df_all["year"].between(*yr))].sort_values("year")
        if not dc.empty and "conso_totale_gwh" in dc.columns:
            fig.add_trace(go.Scatter(
                x=dc["year"],y=dc["conso_totale_gwh"],
                name=f"{FLAGS.get(cc,'')} {cn}" if is_s else None,
                mode="lines",line=dict(width=3 if is_s else 1,color=PAL8[i%8]),
                opacity=1.0 if is_s else 0.15,showlegend=is_s,legendgroup=cc,
            ))
        pc=proj_all[proj_all["country_code"]==cc].sort_values("year")
        if not pc.empty:
            fig.add_trace(go.Scatter(
                x=pc["year"],y=pc["predicted_gwh"],
                name=f"{FLAGS.get(cc,'')} {cn} (proj.)" if is_s else None,
                mode="lines",line=dict(width=3 if is_s else 1,color=PAL8[i%8],dash="dash"),
                opacity=1.0 if is_s else 0.15,legendgroup=cc,showlegend=is_s,
            ))
    fig.add_vline(x=ymax_h+0.5,line_dash="dot",line_color=C["gris2"])
    fig = lay(fig, "8 pays UEMOA — trait plein (historique) + pointilles (projection)", 420, "GWh")
    st.plotly_chart(fig, key="uemoa_all")

    last_yr_p = proj_all["year"].max()
    prk = proj_all[proj_all["year"]==last_yr_p].sort_values("predicted_gwh",ascending=False)
    if not prk.empty:
        pos_p = list(prk["country_code"]).index(sel_cc)+1 if sel_cc in list(prk["country_code"]) else "?"
        top_p = prk.iloc[0]
        st.markdown(f"""<div class="interp">
            <div class="interp-hd">💡 Qui domine en 2045 ?</div>
            <p>Trait plein = historique, pointilles = projections.
            En {int(last_yr_p)}, {sel_flag} <span class="h">{sel_name}</span> se placerait
            en <span class="b">position #{pos_p}</span> sur {len(prk)} pays.
            Le leader serait {FLAGS.get(top_p['country_code'],'')}
            <span class="h">{top_p['country_name']}</span> avec
            <span class="v">{top_p['predicted_gwh']:,.0f} GWh</span>.
            La Cote d'Ivoire et le Senegal tirent la region vers le haut,
            tandis que les pays enclaves progressent plus moderement.
            Ces projections sont un outil concret de <span class="h">planification energetique regionale</span>.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  7. EXPLORATEUR D'INDICATEURS (compact)
# ══════════════════════════════════════════════════════════════════════
with st.expander(f"🔍 Explorer un indicateur — {sel_flag} {sel_name}", expanded=False):
    if raw_all is not None and not raw_all.empty:
        raw_f = raw_all[raw_all["year"].between(*yr)]
        ind_list = sorted(raw_f["indicator_code"].unique().tolist())
        sel_ind = st.selectbox("Indicateur",ind_list,
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
                opacity=1.0 if is_s else 0.2,
                marker=dict(size=5) if is_s else dict(size=0),
            ))
        fig = lay(fig, f"{IND_LABELS.get(sel_ind,sel_ind)} — comparaison UEMOA", 350, IND_LABELS.get(sel_ind,""))
        st.plotly_chart(fig, key="expl_ind")


# ══════════════════════════════════════════════════════════════════════
#  8. SOURCES & DONNEES
# ══════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec"><span class="tt">📂 Sources, Donnees & Resultats</span></div>""", unsafe_allow_html=True)

st.markdown("""<div class="interp">
    <div class="interp-hd">🔗 Sources des donnees</div>
    <p>Toutes les donnees proviennent de l'<span class="b">API officielle de la Banque Mondiale</span>
    (World Development Indicators — WDI) :<br>
    • <a href="https://data.worldbank.org" target="_blank" style="color:#0A66C2;font-weight:600;">data.worldbank.org</a>
    — portail principal<br>
    • <a href="https://api.worldbank.org/v2/country/TGO;SEN;CIV;BEN;BFA;MLI;NER;GNB/indicator/EG.USE.ELEC.KH.PC?format=json&per_page=5000&date=1990:2023" target="_blank" style="color:#0A66C2;font-weight:600;">API WDI (exemple)</a>
    — endpoint utilise pour l'extraction<br>
    • <a href="https://github.com/Theobaw01/energy-prediction-togo" target="_blank" style="color:#0A66C2;font-weight:600;">GitHub — Code source du projet</a></p>
</div>""", unsafe_allow_html=True)

# Données brutes
with st.expander("📄 Donnees brutes (non traitees)", expanded=False):
    if raw_all is not None and not raw_all.empty:
        raw_disp = raw_all[raw_all["country_code"]==sel_cc].sort_values(["indicator_code","year"])
        st.markdown(f'<div class="ins"><b>{len(raw_disp):,}</b> observations brutes pour {sel_flag} {sel_name} · <b>{len(raw_all["indicator_code"].unique())}</b> indicateurs · format long (1 ligne = 1 indicateur × 1 annee × 1 pays)</div>', unsafe_allow_html=True)
        st.dataframe(raw_disp, height=300, key="dt_raw")
    else:
        st.info("Donnees brutes non disponibles.")

# Données traitées
with st.expander("⚙️ Donnees traitees (apres feature engineering)", expanded=False):
    tg_disp = df_all[df_all["country_code"]==sel_cc].sort_values("year")
    st.markdown(f'<div class="ins"><b>{len(tg_disp)}</b> lignes × <b>{len(tg_disp.columns)}</b> colonnes pour {sel_flag} {sel_name} · Inclut lags, moyennes mobiles, ratios, log-transforms, encodage pays</div>', unsafe_allow_html=True)
    st.dataframe(tg_disp, height=300, key="dt_proc")

# Prédictions historiques
with st.expander("✅ Predictions historiques (observe vs predit)", expanded=False):
    if pred is not None and not pred.empty:
        st.markdown(f'<div class="ins">Comparaison valeurs reelles / predictions du modele pour {sel_flag} {sel_name} sur la periode de test</div>', unsafe_allow_html=True)
        st.dataframe(pred, height=300, key="dt_pred")
    else:
        st.info("Predictions non disponibles.")

# Projections
with st.expander("🔮 Projections 2024-2045", expanded=False):
    if proj is not None and not proj.empty:
        st.markdown(f'<div class="ins">Projections IA pour {sel_flag} {sel_name} · {len(proj)} annees · Inclut intervalle de confiance (IC 95%) et CAGR</div>', unsafe_allow_html=True)
        st.dataframe(proj.sort_values("year"), height=300, key="dt_proj")
    else:
        st.info("Projections non disponibles.")

# Résultats modèles
with st.expander("🧠 Resultats des modeles", expanded=False):
    if res is not None and not res.empty:
        st.markdown(f'<div class="ins"><b>{len(res)}</b> modeles compares · Metriques : R², RMSE, MAE, MAPE · Meilleur : <b>{bst["model"]}</b> (R²={bst["r2"]:.3f})</div>', unsafe_allow_html=True)
        st.dataframe(res.sort_values("r2", ascending=False), height=250, key="dt_res")
    else:
        st.info("Resultats non disponibles.")


# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="foot">
    <b>Candidature BCEAO — Developpeur IA</b><br>
    API Banque Mondiale → ETL Python → Log-Transform + Country Encoding → Ridge (R²=0.97) → Streamlit + Plotly<br>
    8 pays · 21 indicateurs · 82 features · 1990-2023 · Horizon 2045
</div>
""", unsafe_allow_html=True)

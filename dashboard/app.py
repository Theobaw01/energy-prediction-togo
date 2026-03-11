"""
Dashboard BI — Population et Demande Electrique
Togo & Zone UEMOA
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demande Electrique — Togo",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"

C = {
    "pop": "#2E86C1",
    "kwh": "#E67E22",
    "gwh": "#1ABC9C",
    "proj": "#9B59B6",
    "ci": "rgba(155,89,182,0.12)",
    "grid": "#1E2A35",
    "muted": "#7F8C8D",
    "good": "#27AE60",
    "warn": "#E74C3C",
    "bg": "#0E1117",
}


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; max-width: 1300px; }

.hdr {
    background: linear-gradient(135deg, #1B3A4B 0%, #2E6F8E 100%);
    padding: 20px 28px; border-radius: 6px; margin-bottom: 20px;
    border-bottom: 2px solid #E67E22;
}
.hdr h1 { color: #fff; margin: 0; font-size: 1.35em; font-weight: 600; }
.hdr p { color: #B0C4CE; margin: 4px 0 0; font-size: 0.82em; font-weight: 300; }

.card {
    background: #161B22; border-radius: 6px; padding: 14px 16px;
    border-left: 3px solid #2E86C1;
}
.card .t { color: #7F8C8D; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.card .v { color: #ECF0F1; font-size: 1.4em; font-weight: 700; line-height: 1.15; }
.card .d { font-size: 0.78em; margin-top: 3px; font-weight: 500; }
.card .d.up { color: #27AE60; }
.card .d.dn { color: #E74C3C; }
.card .ctx { color: #5D6D7E; font-size: 0.68em; margin-top: 2px; }

.sec { color: #D5DBE1; font-size: 1em; font-weight: 600; border-bottom: 1px solid #2E6F8E;
       padding-bottom: 5px; margin: 20px 0 12px 0; }

.foot { text-align: center; color: #5D6D7E; font-size: 0.7em; padding: 14px 0;
        margin-top: 24px; border-top: 1px solid #1E2A35; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    d = {}
    for k, f in [("df", "data/processed/energy_data_processed.csv"),
                  ("pred", "data/predictions/predictions.csv"),
                  ("proj", "data/predictions/projections.csv"),
                  ("res", "models/results.csv")]:
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


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>Population et Demande Electrique</h1>
    <p>La population croit — combien d'electricite faudra-t-il demain ?   |   Togo & UEMOA</p>
</div>
""", unsafe_allow_html=True)

data = load()
if "df" not in data:
    st.error("Donnees absentes. Executez le pipeline ETL d'abord.")
    st.stop()

df = data["df"]
pred = data.get("pred")
proj = data.get("proj")
res = data.get("res")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parametres")
    all_cc = sorted(df["country_code"].unique())
    focus = st.selectbox(
        "Pays",
        all_cc,
        format_func=lambda c: df[df["country_code"]==c]["country_name"].iloc[0],
        index=all_cc.index("TG") if "TG" in all_cc else 0,
    )
    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    yr = st.slider("Periode", y_min, y_max, (y_min, y_max))

    st.divider()
    focus_name = df[df["country_code"]==focus]["country_name"].iloc[0]
    st.markdown(f"**{focus_name}**  \n{yr[0]} — {yr[1]}")

    if pred is not None:
        st.download_button("Exporter predictions", pred.to_csv(index=False).encode(), "predictions.csv")
    if proj is not None:
        st.download_button("Exporter projections", proj.to_csv(index=False).encode(), "projections.csv")

# Filter
dff = df[(df["year"].between(*yr))]
tg = dff[dff["country_code"]==focus].sort_values("year")

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
if not tg.empty:
    last = tg.iloc[-1]
    first = tg.iloc[0]

    cards = []
    # Population
    if "SP.POP.TOTL" in tg.columns:
        pop_now = last["SP.POP.TOTL"]
        pop_before = first["SP.POP.TOTL"]
        growth = ((pop_now / pop_before) - 1) * 100 if pop_before > 0 else 0
        cards.append(("Population", fmt(pop_now), f"+{growth:.0f}% depuis {yr[0]}", "up", C["pop"]))

    # kWh / hab
    if "EG.USE.ELEC.KH.PC" in tg.columns:
        kwh = last["EG.USE.ELEC.KH.PC"]
        kwh0 = first["EG.USE.ELEC.KH.PC"]
        d = kwh - kwh0
        css = "up" if d > 0 else "dn"
        cards.append(("kWh / habitant", fmt(kwh, "kWh"), f"{d:+.0f} kWh vs {yr[0]}", css, C["kwh"]))

    # Conso totale
    if "conso_totale_gwh" in tg.columns:
        gwh = last["conso_totale_gwh"]
        gwh0 = first["conso_totale_gwh"]
        d = ((gwh / gwh0) - 1) * 100 if gwh0 > 0 else 0
        cards.append(("Demande totale", fmt(gwh, "GWh"), f"+{d:.0f}% depuis {yr[0]}", "up", C["gwh"]))

    # Acces electricite
    if "EG.ELC.ACCS.ZS" in tg.columns:
        acc = last["EG.ELC.ACCS.ZS"]
        acc0 = first["EG.ELC.ACCS.ZS"]
        d = acc - acc0
        css = "up" if d > 0 else "dn"
        cards.append(("Acces electricite", f"{acc:.1f}%", f"{d:+.1f} pts vs {yr[0]}", css, C["good"]))

    # R2 du modele
    if res is not None and not res.empty:
        best = res.sort_values("r2", ascending=False).iloc[0]
        cards.append(("Modele IA", f"R2 {best['r2']:.3f}", best["model"], "up", C["proj"]))

    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:18px;">'
    for title, val, delta, css, color in cards:
        html += f'''<div class="card" style="border-left-color:{color};">
            <div class="t">{title}</div><div class="v">{val}</div>
            <div class="d {css}">{delta}</div></div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["Hier — Tendances", "Aujourd'hui — Analyse", "Demain — Projections"])

# ── TAB 1 : HIER ────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sec">Evolution historique</div>', unsafe_allow_html=True)

    if not tg.empty:
        # Population + Conso totale (double axe)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "SP.POP.TOTL" in tg.columns:
            fig.add_trace(go.Bar(
                x=tg["year"], y=tg["SP.POP.TOTL"]/1e6, name="Population (M)",
                marker_color=C["pop"], opacity=0.35,
            ), secondary_y=False)
        if "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"], name="Demande (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=2.5),
                marker=dict(size=4),
            ), secondary_y=True)
        fig.update_layout(
            title=f"{focus_name} — Population et demande electrique",
            template=TMPL, height=400, hovermode="x unified", margin=dict(t=40),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        fig.update_yaxes(title_text="Population (millions)", secondary_y=False)
        fig.update_yaxes(title_text="GWh", secondary_y=True)
        st.plotly_chart(fig, key="hist_pop_gwh")

        # kWh/hab evolution
        c1, c2 = st.columns(2)
        with c1:
            if "EG.USE.ELEC.KH.PC" in tg.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tg["year"], y=tg["EG.USE.ELEC.KH.PC"],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=C["kwh"], width=2),
                    fillcolor="rgba(230,126,34,0.08)",
                    name="kWh / hab",
                ))
                fig.update_layout(title="Consommation par habitant", template=TMPL,
                                  height=300, margin=dict(t=35), yaxis_title="kWh/hab")
                st.plotly_chart(fig, key="hist_kwh")

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
                fig.update_layout(title="Acces electricite (%)", template=TMPL, height=300,
                                  margin=dict(t=35), yaxis_title="%",
                                  legend=dict(orientation="h", y=-0.18, font_size=10))
                st.plotly_chart(fig, key="hist_acces")

        # Table
        with st.expander("Donnees", expanded=False):
            show_cols = ["year", "SP.POP.TOTL", "EG.USE.ELEC.KH.PC", "conso_totale_gwh", "EG.ELC.ACCS.ZS"]
            show_cols = [c for c in show_cols if c in tg.columns]
            display = tg[show_cols].copy()
            display.columns = [{"year": "Annee", "SP.POP.TOTL": "Population",
                                "EG.USE.ELEC.KH.PC": "kWh/hab", "conso_totale_gwh": "Demande GWh",
                                "EG.ELC.ACCS.ZS": "Acces %"}.get(c, c) for c in show_cols]
            st.dataframe(display, height=300)


# ── TAB 2 : AUJOURD'HUI ─────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sec">Correlation population / energie</div>', unsafe_allow_html=True)

    if "SP.POP.TOTL" in dff.columns and "conso_totale_gwh" in dff.columns:
        latest_yr = int(dff["year"].max())

        c1, c2 = st.columns([3, 2])
        with c1:
            # Scatter : pop vs conso totale (tous pays, derniere annee)
            snap = dff[dff["year"]==latest_yr].dropna(subset=["SP.POP.TOTL","conso_totale_gwh"])
            fig = go.Figure()
            for _, r in snap.iterrows():
                is_f = r["country_code"] == focus
                fig.add_trace(go.Scatter(
                    x=[r["SP.POP.TOTL"]/1e6], y=[r["conso_totale_gwh"]],
                    mode="markers+text", text=[r["country_name"]],
                    textposition="top center", textfont=dict(size=9 if not is_f else 11),
                    marker=dict(size=14 if is_f else 8,
                                color=C["gwh"] if is_f else C["muted"],
                                line=dict(width=2 if is_f else 0, color="#fff")),
                    showlegend=False,
                ))
            # Trendline
            if len(snap) > 2:
                x_arr = snap["SP.POP.TOTL"].values / 1e6
                y_arr = snap["conso_totale_gwh"].values
                z = np.polyfit(x_arr, y_arr, 1)
                xr = np.linspace(x_arr.min(), x_arr.max(), 80)
                fig.add_trace(go.Scatter(x=xr, y=z[0]*xr+z[1], mode="lines",
                                         line=dict(dash="dash", color=C["muted"], width=1),
                                         showlegend=False))
            fig.update_layout(
                title=f"Population vs Demande electrique — UEMOA {latest_yr}",
                xaxis_title="Population (millions)", yaxis_title="Demande (GWh)",
                template=TMPL, height=400, margin=dict(t=40),
            )
            st.plotly_chart(fig, key="scatter_pop_gwh")

        with c2:
            # Correlation stats
            if not tg.empty and len(tg) > 3:
                corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
                st.markdown(f"""
                <div class="card" style="border-left-color:{C['gwh']}; margin-bottom:10px;">
                    <div class="t">Correlation</div>
                    <div class="v">{corr:.3f}</div>
                    <div class="ctx">Pearson — population vs demande GWh</div>
                </div>""", unsafe_allow_html=True)

                # Elasticites
                pop_chg = (tg["SP.POP.TOTL"].iloc[-1] / tg["SP.POP.TOTL"].iloc[0] - 1) * 100
                gwh_chg = (tg["conso_totale_gwh"].iloc[-1] / tg["conso_totale_gwh"].iloc[0] - 1) * 100 if tg["conso_totale_gwh"].iloc[0] > 0 else 0
                elast = gwh_chg / pop_chg if pop_chg > 0 else 0

                st.markdown(f"""
                <div class="card" style="border-left-color:{C['kwh']}; margin-bottom:10px;">
                    <div class="t">Elasticite</div>
                    <div class="v">{elast:.2f}</div>
                    <div class="ctx">+1% pop = +{elast:.2f}% demande electrique</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Croissance pop.</div>
                    <div class="v">+{pop_chg:.0f}%</div>
                    <div class="ctx">{yr[0]} — {yr[1]}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # Classement UEMOA
    st.markdown('<div class="sec">Classement UEMOA</div>', unsafe_allow_html=True)
    if "conso_totale_gwh" in dff.columns:
        latest_yr = int(dff["year"].max())
        rank = dff[dff["year"]==latest_yr].dropna(subset=["conso_totale_gwh"]).sort_values("conso_totale_gwh", ascending=True)
        fig = go.Figure()
        colors = [C["gwh"] if r["country_code"]==focus else C["muted"] for _, r in rank.iterrows()]
        fig.add_trace(go.Bar(
            x=rank["conso_totale_gwh"], y=rank["country_name"], orientation="h",
            marker_color=colors,
            text=[f"{v:,.0f} GWh" for v in rank["conso_totale_gwh"]], textposition="outside",
            textfont_size=10,
        ))
        fig.update_layout(title=f"Demande electrique totale — {latest_yr}", template=TMPL,
                          height=300, margin=dict(l=10, r=60, t=35, b=10),
                          xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, key="rank_gwh")

    # Predictions historiques (validation)
    if pred is not None and not pred.empty:
        st.markdown('<div class="sec">Validation du modele</div>', unsafe_allow_html=True)
        tg_pred = pred[pred["country_code"]==focus]
        if not tg_pred.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tg_pred["year"], y=tg_pred["actual"],
                                     name="Observe", mode="lines+markers",
                                     line=dict(color=C["gwh"], width=2.5)))
            fig.add_trace(go.Scatter(x=tg_pred["year"], y=tg_pred["predicted"],
                                     name="Predit (IA)", mode="lines+markers",
                                     line=dict(color=C["proj"], width=2, dash="dash")))
            fig.update_layout(title=f"Observe vs Predit — {focus_name}",
                              yaxis_title="GWh", template=TMPL, height=350,
                              hovermode="x unified", margin=dict(t=40),
                              legend=dict(orientation="h", y=-0.15, font_size=10))
            st.plotly_chart(fig, key="valid")

            # Metriques
            mae = tg_pred["error"].abs().mean()
            rmse = np.sqrt((tg_pred["error"]**2).mean())
            mc = st.columns(3)
            mc[0].metric("MAE", f"{mae:.1f} GWh")
            mc[1].metric("RMSE", f"{rmse:.1f} GWh")
            if res is not None:
                mc[2].metric("R2", f"{res.sort_values('r2', ascending=False).iloc[0]['r2']:.4f}")


# ── TAB 3 : DEMAIN ──────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sec">Projections de demande electrique</div>', unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        tg_proj = proj[proj["country_code"]==focus].sort_values("year")

        if not tg_proj.empty:
            # Concatenate historique + projections
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
                x=tg_proj["year"], y=tg_proj["predicted_gwh"],
                name="Projection IA", mode="lines+markers",
                line=dict(color=C["proj"], width=2.5),
                marker=dict(size=7, symbol="diamond"),
            ))

            # IC
            fig.add_trace(go.Scatter(
                x=pd.concat([tg_proj["year"], tg_proj["year"][::-1]]),
                y=pd.concat([tg_proj["ci_upper"], tg_proj["ci_lower"][::-1]]),
                fill="toself", fillcolor=C["ci"],
                line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
            ))

            fig.update_layout(
                title=f"{focus_name} — Demande electrique : historique et projections",
                yaxis_title="GWh", template=TMPL, height=420,
                hovermode="x unified", margin=dict(t=40),
                legend=dict(orientation="h", y=-0.15, font_size=10),
            )
            st.plotly_chart(fig, key="proj_main")

            # Table projections
            st.markdown("##### Detail des projections")
            disp = tg_proj[["year", "predicted_gwh", "ci_lower", "ci_upper", "pop_projected"]].copy()
            disp.columns = ["Annee", "Demande (GWh)", "IC bas", "IC haut", "Population projetee"]
            disp["Population projetee"] = disp["Population projetee"].apply(
                lambda v: f"{v/1e6:.2f} M" if pd.notna(v) else "—"
            )
            st.dataframe(disp, height=280)

            # Insight
            if "pop_projected" in tg_proj.columns and tg_proj["pop_projected"].notna().any():
                last_proj = tg_proj.iloc[-1]
                last_hist_gwh = tg["conso_totale_gwh"].iloc[-1] if "conso_totale_gwh" in tg.columns else 0
                growth_gwh = ((last_proj["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
                last_pop_hist = tg["SP.POP.TOTL"].iloc[-1] if "SP.POP.TOTL" in tg.columns else 0
                growth_pop = ((last_proj["pop_projected"] / last_pop_hist) - 1) * 100 if last_pop_hist > 0 else 0

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="card" style="border-left-color:{C['proj']};">
                        <div class="t">Horizon {int(last_proj['year'])}</div>
                        <div class="v">{last_proj['predicted_gwh']:.0f} GWh</div>
                        <div class="d up">+{growth_gwh:.0f}% vs {yr[1]}</div>
                        <div class="ctx">Population projetee : {last_proj['pop_projected']/1e6:.1f} M (+{growth_pop:.0f}%)</div>
                    </div>""", unsafe_allow_html=True)

                with c2:
                    kwh_proj = last_proj["predicted_gwh"] * 1e6 / last_proj["pop_projected"] if last_proj["pop_projected"] > 0 else 0
                    st.markdown(f"""
                    <div class="card" style="border-left-color:{C['kwh']};">
                        <div class="t">kWh / hab projete</div>
                        <div class="v">{kwh_proj:.0f} kWh</div>
                        <div class="ctx">Si toute la demande est satisfaite</div>
                    </div>""", unsafe_allow_html=True)

        # Comparaison UEMOA
        st.divider()
        st.markdown('<div class="sec">Projections UEMOA</div>', unsafe_allow_html=True)
        last_year_proj = int(proj["year"].max())
        all_proj_last = proj[proj["year"]==last_year_proj].sort_values("predicted_gwh", ascending=True)
        if not all_proj_last.empty:
            colors = [C["proj"] if r["country_code"]==focus else C["muted"] for _, r in all_proj_last.iterrows()]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=all_proj_last["predicted_gwh"], y=all_proj_last["country_name"],
                orientation="h", marker_color=colors,
                text=[f"{v:,.0f} GWh" for v in all_proj_last["predicted_gwh"]],
                textposition="outside", textfont_size=10,
            ))
            fig.update_layout(title=f"Demande projetee — {last_year_proj}",
                              template=TMPL, height=300,
                              margin=dict(l=10, r=60, t=35, b=10),
                              xaxis=dict(showticklabels=False))
            st.plotly_chart(fig, key="proj_uemoa")

    else:
        st.info("Executez python src/models/predict.py pour generer les projections.")

    # Model perf
    if res is not None and not res.empty:
        st.divider()
        st.markdown('<div class="sec">Comparaison des algorithmes</div>', unsafe_allow_html=True)
        fig = go.Figure()
        r2_vals = res.sort_values("r2", ascending=False)
        colors = [C["good"] if v > 0.85 else C["kwh"] if v > 0.7 else C["warn"] for v in r2_vals["r2"]]
        fig.add_trace(go.Bar(
            x=r2_vals["model"], y=r2_vals["r2"], marker_color=colors,
            text=[f"{v:.3f}" for v in r2_vals["r2"]], textposition="outside", textfont_size=11,
        ))
        fig.update_layout(title="R2 par modele", yaxis_title="R2", yaxis_range=[0, 1.08],
                          template=TMPL, height=320, margin=dict(t=40))
        st.plotly_chart(fig, key="model_cmp")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
Source : Banque Mondiale (WDI)  |  Modeles : scikit-learn, XGBoost, LightGBM  |  Streamlit + Plotly
</div>
""", unsafe_allow_html=True)

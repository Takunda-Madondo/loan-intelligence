"""app/pages/overview.py - Portfolio Overview"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT3 = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT3))
sys.path.append(str(ROOT3 / "pipelines"))

from app.data_loader import (load_kiva_predictions, load_sector_performance,
                              load_model_metrics, load_shap_importance, AFRICAN_COUNTRIES)
from app.components import (kpi, section, card, card_end, insight, apply_layout, toolbar,
                             C_BLUE, C_GREEN, C_RED, C_AMBER, C_TEAL, ch)

LABEL_FONT = dict(size=12, color="#111318", family="DM Sans")


def render():
    f       = toolbar("ov")
    preds   = load_kiva_predictions()
    sectors = load_sector_performance()
    metrics = load_model_metrics()
    shap    = load_shap_importance()

    african = preds[preds["country_code"].isin(AFRICAN_COUNTRIES)]
    df      = african if f["region"] == "Africa Only" else preds
    if f["risk"] != "All":
        df = df[df["risk_band"] == f["risk"]]

    if f["region"] != "Africa Only":
        non_african = preds[~preds["country_code"].isin(AFRICAN_COUNTRIES)]
        if len(non_african) > 0:
            n_countries = non_african["country_code"].nunique()
            insight(f"<b>All Regions view</b> includes {n_countries} non-African markets "
                    f"(Philippines, Cambodia, India, Peru, Bolivia and others - "
                    f"{len(non_african):,} loans). These are scored using the same relative risk model "
                    f"but the 8% calibration target reflects African MFI norms. "
                    f"Switch to <b>Africa Only</b> for investment-grade comparisons.")

    section("PORTFOLIO HEALTH")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    hi  = df["risk_band"].isin(["High","Very High"]).sum()
    low = (df["risk_band"] == "Low").sum()
    with k1: kpi("Total Loans",      f"{len(df):,}",                                  "in selection",               "neu", "#6378FF")
    with k2: kpi("Avg Risk Score",     f"{df['default_probability'].mean()*100:.1f}%", "calibrated to MFI norms",     "neg" if df['default_probability'].mean()>0.2 else "neu", C_AMBER)
    with k3: kpi("Elevated Risk Loans", f"{hi:,}",                                       f"{hi/max(len(df),1)*100:.0f}% of portfolio", "neg", C_RED)
    with k4: kpi("Low Risk Loans",    f"{low:,}",                                      f"{low/max(len(df),1)*100:.0f}% investable",  "pos", C_GREEN)
    with k5: kpi("Model AUC-ROC",     f"{metrics['auc_roc']}",                         "XGBoost accuracy",           "pos", C_BLUE)
    with k6: kpi("Countries",         f"{df['country_code'].nunique()}",              "markets tracked",             "neu", C_TEAL)

    section("RISK DISTRIBUTION")
    r1c1, r1c2, r1c3 = st.columns([1, 1, 1])

    with r1c1:
        band = (preds["risk_band"]
                .value_counts()
                .reindex(["Low","Medium","High","Very High"])
                .fillna(0)
                .reset_index())
        band.columns = ["Band","Count"]
        cmap = {"Low":C_GREEN,"Medium":C_AMBER,"High":C_RED,"Very High":"#9D174D"}
        fig  = px.bar(band, x="Band", y="Count", color="Band",
                      color_discrete_map=cmap,
                      text=band["Count"].apply(lambda x: f"{int(x):,}"))
        fig.update_traces(textposition="outside", marker_line_width=0, textfont=LABEL_FONT)
        apply_layout(fig, 240)
        fig.update_layout(yaxis_title="", xaxis_title="", showlegend=False,
                          yaxis=dict(showticklabels=False, showgrid=False))
        card("Risk Band Distribution", f"{len(preds):,} total loans")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with r1c2:
        sec_agg = (sectors.groupby("sector")["default_rate"]
                          .mean().reset_index()
                          .sort_values("default_rate"))
        fig = px.bar(sec_agg, x="default_rate", y="sector", orientation="h",
                     color="default_rate",
                     color_continuous_scale=["#BBF7D0", C_AMBER, C_RED],
                     text=sec_agg["default_rate"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside", marker_line_width=0, textfont=LABEL_FONT)
        apply_layout(fig, 240)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="",
                          xaxis_title="Default Rate (%)",
                          xaxis=dict(range=[0, sec_agg["default_rate"].max()*1.3]))
        card("Sector Default Rates", "Lending Club benchmark")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with r1c3:
        af_band = african["risk_band"].value_counts().reset_index()
        af_band.columns = ["Band","Count"]
        fig = px.pie(af_band, names="Band", values="Count", color="Band",
                     color_discrete_map={"Low":C_GREEN,"Medium":C_AMBER,
                                         "High":C_RED,"Very High":"#9D174D"},
                     hole=0.55)
        fig.update_traces(
            textfont=dict(size=12, color="#111318", family="DM Sans"),
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=2))
        )
        fig.update_layout(paper_bgcolor="white",
                          font=dict(family="DM Sans", size=12, color="#111318"),
                          margin=dict(t=10,b=30,l=0,r=0), height=240,
                          showlegend=False)
        card("African Portfolio Risk Split", f"{len(african):,} African loans")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    section("MARKET INTELLIGENCE & MODEL INSIGHTS")
    r2c1, r2c2 = st.columns([1.1, 1])

    with r2c1:
        crisk = (african.groupby("country")
                        .agg(risk=("default_probability","mean"), loans=("loan_id","count"))
                        .reset_index().sort_values("risk"))
        crisk["pct"]    = (crisk["risk"]*100).round(1)
        crisk["colour"] = crisk["pct"].apply(
            lambda x: C_GREEN if x < 5 else C_AMBER if x < 12 else C_RED)
        fig = go.Figure()
        for _, row in crisk.iterrows():
            fig.add_trace(go.Bar(
                x=[row["pct"]], y=[row["country"]],
                orientation="h", marker_color=row["colour"], marker_line_width=0,
                text=f"{row['pct']:.1f}%", textposition="outside",
                textfont=dict(size=11, color="#111318", family="DM Sans"),
                showlegend=False,
                hovertemplate=f"{row['country']}<br>Default: {row['pct']:.1f}%<br>Loans: {int(row['loans']):,}<extra></extra>",
            ))
        apply_layout(fig, 320)
        fig.update_layout(barmode="overlay", yaxis_title="",
                          xaxis_title="Avg Risk Score (%)",
                          xaxis=dict(range=[0, crisk["pct"].max()*1.3],
                                     tickfont=dict(size=11, color="#374151")),
                          yaxis=dict(tickfont=dict(size=11, color="#374151")))
        card("Country Risk Score", "Africa - sorted safest to riskiest")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with r2c2:
        top_shap = shap.head(10).copy()
        top_shap["feature_label"] = top_shap["feature"].apply(ch)
        top_shap = top_shap.sort_values("shap_importance")
        fig = px.bar(top_shap, x="shap_importance", y="feature_label", orientation="h",
                     color="shap_importance",
                     color_continuous_scale=["#C7D2FE", C_BLUE, "#312E81"],
                     text=top_shap["shap_importance"].apply(lambda x: f"{x:.3f}"))
        fig.update_traces(textposition="outside", marker_line_width=0, textfont=LABEL_FONT)
        apply_layout(fig, 320)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="",
                          xaxis_title="Mean |SHAP| Value",
                          xaxis=dict(range=[0, top_shap["shap_importance"].max()*1.3],
                                     tickfont=dict(size=11, color="#374151")),
                          yaxis=dict(tickfont=dict(size=11, color="#374151")))
        card("Default Prediction Drivers", "What the model weighs most heavily")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    insight("<b>About these risk scores:</b> The model was trained on 1.3M Lending Club loans "
            "(US consumer credit, 20% default rate) and applied via transfer learning to Kiva microfinance loans. "
            "Raw scores are recalibrated so the portfolio mean aligns with the African MFI industry benchmark "
            "(PAR30 ~8%). Scores should be read as <b>relative risk rankings</b> within this portfolio - "
            "not as absolute default probabilities. Low = &lt;5% | Medium = 5-12% | High = 12-20% | Very High = &gt;20%.")

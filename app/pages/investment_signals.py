"""app/pages/investment_signals.py - Business-focused: where to invest / avoid"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT3 = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT3))
sys.path.append(str(ROOT3 / "pipelines"))

from app.data_loader import load_sector_performance, load_kiva_predictions, AFRICAN_COUNTRIES
from app.components import (kpi, section, card, card_end, insight, signal, apply_layout,
                             toolbar, C_BLUE, C_GREEN, C_RED, C_AMBER, C_TEAL)

SECTOR_C = {
    "Agriculture & Farming":"#16A34A","Food & Beverage":"#D97706",
    "Retail & Trade":"#6378FF","Services":"#7C3AED",
    "Housing & Construction":"#0891B2","Education":"#DB2777",
    "Transport & Logistics":"#0D9488","Manufacturing":"#DC2626",
    "Health & Wellness":"#65A30D","Personal Use":"#9333EA",
    "Clothing & Textiles":"#EA580C","Arts & Crafts":"#CA8A04",
}


def render():
    f       = toolbar("is")
    sectors = load_sector_performance()
    preds   = load_kiva_predictions()

    df = sectors.copy()
    if f["region"] == "Africa Only":
        df = df[df["country_code"].isin(AFRICAN_COUNTRIES)]
    min_map = {"Any":0,"50+":50,"100+":100,"500+":500}
    df = df[df["total_loans"] >= min_map.get(f["min_vol"],0)]

    agg = df.groupby("sector").agg(
        roi=("roi_score","mean"), default=("default_rate","mean"),
        loans=("total_loans","sum"), roi_sd=("roi_score","std")
    ).reset_index().round(3)

    invest = agg[(agg["roi"] >= 0.90) & (agg["default"] < 8.0)].sort_values("roi", ascending=False)
    avoid  = agg[agg["default"] >= 10.0].sort_values("default", ascending=False)

    # KPIs
    section("INVESTMENT SUMMARY")
    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Investable Sectors",  str(len(invest)),         "ROI ≥ 0.90, risk < 8%",  "pos", C_GREEN)
    with k2: kpi("High-Risk Sectors",   str(len(avoid)),          "risk score >= 10% - avoid","neg", C_RED)
    with k3:
        best = agg.loc[agg["roi"].idxmax()]
        kpi("Top ROI Sector", best["sector"][:16], f"ROI {best['roi']:.3f}", "pos", C_BLUE)
    with k4:
        worst = agg.loc[agg["default"].idxmax()]
        kpi("Riskiest Sector", worst["sector"][:16], f"{worst['default']:.1f}% default", "neg", C_RED)

    # Signals column + Bubble chart
    section("OPPORTUNITY MAP")
    sc1, sc2 = st.columns([1, 1.6])

    with sc1:
        st.markdown('<div style="margin-bottom:0.4rem"><span class="badge badge-safe">INVEST</span></div>', unsafe_allow_html=True)
        for _, row in invest.head(5).iterrows():
            signal(
                row["sector"],
                f"ROI score <b>{row['roi']:.3f}</b> · Risk score <b>{row['default']:.1f}%</b> · {int(row['loans']):,} loans",
                invest=True
            )
        st.markdown('<div style="margin:0.6rem 0 0.4rem"><span class="badge badge-risk">AVOID</span></div>', unsafe_allow_html=True)
        for _, row in avoid.head(3).iterrows():
            signal(
                row["sector"],
                f"Risk score <b>{row['default']:.1f}%</b> - above 15% threshold | ROI {row['roi']:.3f}",
                invest=False
            )

    with sc2:
        fig = px.scatter(agg, x="default", y="roi",
                         size="loans", color="sector",
                         color_discrete_map=SECTOR_C,
                         hover_name="sector",
                         size_max=52,
                         labels={"default":"Risk Score (%)","roi":"ROI Score","loans":"Loans"})
        # Invest zone shading
        fig.add_hrect(y0=0.90, y1=agg["roi"].max()*1.1,
                      x0=0, x1=8.0,
                      fillcolor="rgba(22,163,74,0.05)",
                      line=dict(color="rgba(22,163,74,0.3)", dash="dot", width=1))
        fig.add_vline(x=8.0, line_dash="dot", line_color="#D97706", line_width=1)
        fig.add_hline(y=0.90, line_dash="dot", line_color="#16A34A", line_width=1)
        # Annotations
        fig.add_annotation(x=3.0, y=agg["roi"].max()*1.05,
                           text="Invest Zone", showarrow=False,
                           font=dict(size=9, color="#16A34A"), bgcolor="rgba(240,253,244,0.9)")
        fig.add_annotation(x=13.0, y=agg["roi"].min()*0.9,
                           text="Avoid Zone", showarrow=False,
                           font=dict(size=9, color="#DC2626"), bgcolor="rgba(254,242,242,0.9)")
        apply_layout(fig, 340, legend=True)
        fig.update_layout(legend=dict(font_size=11, x=1.01),
                          xaxis=dict(range=[-2, agg["default"].max()*1.2]),
                          yaxis=dict(range=[agg["roi"].min()*0.9, agg["roi"].max()*1.12]))
        card("ROI vs Risk Score - Opportunity Matrix",
             "Green zone = invest | Right of line = avoid | Bubble size = loan volume")
        st.plotly_chart(fig, width='stretch')
        card_end()

    # Country-level top opportunities
    section("TOP COUNTRY-SECTOR OPPORTUNITIES")
    top_ops = (df[df["country_code"].isin(AFRICAN_COUNTRIES)]
               .sort_values("roi_score", ascending=False)
               .head(20)[["sector","country_code","total_loans","avg_loan_amount",
                           "default_rate","roi_score","risk_tier","female_borrower_pct"]].copy())
    top_ops["roi_score"]        = top_ops["roi_score"].round(4)
    top_ops["default_rate"]     = top_ops["default_rate"].apply(lambda x: f"{x:.1f}%")
    top_ops["female_borrower_pct"] = top_ops["female_borrower_pct"].apply(lambda x: f"{x:.0f}%")
    top_ops["avg_loan_amount"]  = top_ops["avg_loan_amount"].apply(lambda x: f"${x:,.0f}")

    def colour_risk(val):
        return {"Low":"background-color:#F0FDF4","Medium":"background-color:#FFFBEB",
                "High":"background-color:#FEF2F2"}.get(val,"")

    card("African Investment Opportunities", "Ranked by ROI score - showing top 20")
    st.dataframe(
        top_ops.rename(columns={
            "sector":"Sector","country_code":"Country","total_loans":"Loans",
            "avg_loan_amount":"Avg Loan","default_rate":"Risk Score",
            "roi_score":"ROI Score","risk_tier":"Risk Tier","female_borrower_pct":"% Female"
        }).style.map(colour_risk, subset=["Risk Tier"]),
        width='stretch', hide_index=True, height=280
    )
    card_end()

    # Sector deep dive
    section("SECTOR DEEP DIVE")
    d1, d2 = st.columns(2)
    with d1:
        sel = st.selectbox("Select sector", sorted(df["sector"].unique()), key="is_sel")
    sdf = df[df["sector"] == sel]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sdf.sort_values("roi_score", ascending=False).head(12),
                     x="country_code", y="roi_score",
                     color="risk_tier",
                     color_discrete_map={"Low":C_GREEN,"Medium":C_AMBER,"High":C_RED},
                     labels={"country_code":"","roi_score":"ROI Score"})
        fig.update_traces(marker_line_width=0)
        apply_layout(fig, 220, legend=True)
        fig.update_layout(legend=dict(font_size=11, orientation="h", y=-0.2))
        card(f"{sel} - ROI by Country")
        st.plotly_chart(fig, width='stretch')
        card_end()

    with col2:
        fig = px.scatter(sdf, x="avg_gdp_growth", y="default_rate",
                         size="total_loans", color="country_code",
                         hover_name="country_code",
                         labels={"avg_gdp_growth":"GDP Growth (%)","default_rate":"Default Rate (%)"})
        apply_layout(fig, 220, legend=False)
        card(f"{sel} - Macro Risk vs Default Rate")
        st.plotly_chart(fig, width='stretch')
        card_end()

"""app/pages/country_risk.py - Default Risk Map"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT3 = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT3))
sys.path.append(str(ROOT3 / "pipelines"))

from app.data_loader import (load_kiva_predictions, load_kiva_features,
                              AFRICAN_COUNTRIES, ISO3_MAP)
from app.components import (kpi, section, card, card_end, insight, apply_layout,
                             toolbar, C_GREEN, C_RED, C_AMBER, C_BLUE, C_TEAL)


def render():
    f     = toolbar("cr")
    preds = load_kiva_predictions()
    kiva  = load_kiva_features()

    src = preds[preds["country_code"].isin(AFRICAN_COUNTRIES)] if f["region"] == "Africa Only" else preds
    agg = (src.groupby(["country","country_code"]).agg(
        def_prob =("default_probability","mean"),
        loans    =("loan_id","count"),
        hi       =("predicted_default","sum"),
        vol      =("loan_amount","sum"),
    ).reset_index())
    agg["iso3"]    = agg["country_code"].map(ISO3_MAP)
    agg["def_pct"] = (agg["def_prob"]*100).round(2)
    agg["hi_pct"]  = (agg["hi"]/agg["loans"]*100).round(1)
    agg["avg_loan"]= (agg["vol"]/agg["loans"]).round(0)
    agg = agg.dropna(subset=["iso3"])

    # KPIs
    section("MARKET OVERVIEW")
    k1,k2,k3,k4,k5 = st.columns(5)
    safest   = agg.loc[agg["def_pct"].idxmin()]
    riskiest = agg.loc[agg["def_pct"].idxmax()]
    with k1: kpi("Markets Tracked",  str(len(agg)),                 "",                           "neu", C_BLUE)
    with k2: kpi("Total Loans",      f"{agg['loans'].sum():,}",     "",                           "neu", C_TEAL)
    with k3: kpi("Avg Default Risk", f"{agg['def_pct'].mean():.1f}%", "across all markets",       "neu", C_AMBER)
    with k4: kpi("Safest Market",    safest["country"],              f"{safest['def_pct']:.1f}% default", "pos", C_GREEN)
    with k5: kpi("Riskiest Market",  riskiest["country"],           f"{riskiest['def_pct']:.1f}% default","neg", C_RED)

    # Map + table
    section("GEOGRAPHIC RISK")
    mc1, mc2 = st.columns([1.8, 1])

    with mc1:
        scope_arg = "africa" if f["region"] == "Africa Only" else "world"
        fig = px.choropleth(
            agg, locations="iso3", color="def_pct",
            hover_name="country",
            hover_data={"def_pct":":.2f","loans":":,","hi_pct":":.1f","iso3":False},
            color_continuous_scale=["#DCFCE7","#FEF9C3","#FCA5A5","#7F1D1D"],
            scope=scope_arg,
            labels={"def_pct":"Default %","loans":"Loans","hi_pct":"High Risk %"},
        )
        fig.update_layout(
            margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="white",
            geo=dict(showframe=False, showcoastlines=True,
                     landcolor="#F9FAFB", bgcolor="white",
                     showocean=True, oceancolor="#EFF6FF",
                     showlakes=False, showcountries=True,
                     countrycolor="#E5E7EB"),
            coloraxis_colorbar=dict(len=0.5, thickness=10,
                                    title=dict(text="Default %", font_size=11)),
            font=dict(family="DM Sans", size=10), height=340,
        )
        card("Default Probability Map", "Colour = avg default probability")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with mc2:
        disp = agg.sort_values("def_pct")[["country","loans","avg_loan","def_pct","hi_pct"]].copy()
        disp.columns = ["Country","Loans","Avg Loan ($)","Default %","High Risk %"]
        disp["Avg Loan ($)"] = disp["Avg Loan ($)"].apply(lambda x: f"${x:,.0f}")
        card("Country Risk Table", "Sorted safest to riskiest")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=340)
        card_end()

    # Country drill-down
    section("COUNTRY DRILL-DOWN")
    african_names = sorted(preds[preds["country_code"].isin(AFRICAN_COUNTRIES)]["country"].unique())

    d1, _ = st.columns([2, 3])
    with d1:
        sel = st.selectbox("Select country", african_names, key="cr_sel")

    cc = preds[preds["country"] == sel]
    ck = kiva[kiva["country"] == sel]

    col1, col2, col3 = st.columns(3)
    with col1: kpi("Loans in Market", f"{len(cc):,}", "", "neu", C_BLUE)
    with col2: kpi("Avg Risk Score", f"{cc['default_probability'].mean()*100:.1f}%", "calibrated to MFI norms", "neu", C_AMBER)
    with col3: kpi("High Risk Count", f"{cc['predicted_default'].sum():,}", f"{cc['predicted_default'].mean()*100:.1f}% of market", "neg", C_RED)

    st.markdown("<br>", unsafe_allow_html=True)
    dr1, dr2 = st.columns(2)

    with dr1:
        sec_d = (cc.groupby("sector").agg(loans=("loan_id","count"),
                                           risk=("default_probability","mean"))
                   .reset_index().sort_values("risk", ascending=False))
        fig = px.bar(sec_d, x="loans", y="sector", orientation="h",
                     color="risk", color_continuous_scale=["#DCFCE7",C_AMBER,C_RED],
                     labels={"loans":"Loans","sector":"","risk":"Default Prob"})
        fig.update_traces(marker_line_width=0)
        apply_layout(fig, 240)
        fig.update_layout(coloraxis_showscale=False)
        card(f"{sel} - Loans by Sector", "Colour = default probability")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with dr2:
        rd = cc["risk_band"].value_counts().reset_index()
        rd.columns = ["Band","Count"]
        fig = px.pie(rd, names="Band", values="Count", color="Band",
                     color_discrete_map={"Low":C_GREEN,"Medium":C_AMBER,
                                         "High":C_RED,"Very High":"#9D174D"},
                     hole=0.55)
        fig.update_traces(textfont=dict(size=12, color="#111318", family="DM Sans"),
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(paper_bgcolor="white", font=dict(family="DM Sans", size=10),
                          margin=dict(t=10,b=10,l=0,r=0), height=240,
                          legend=dict(orientation="h", y=-0.12, font_size=11))
        card(f"{sel} - Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    # Trend line if years available
    if "disbursed_year" in cc.columns:
        yr = (cc.dropna(subset=["disbursed_year"])
                .groupby("disbursed_year")
                .agg(loans=("loan_id","count"), risk=("default_probability","mean"))
                .reset_index())
        if len(yr) > 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yr["disbursed_year"], y=yr["risk"]*100,
                mode="lines+markers",
                line=dict(color=C_BLUE, width=2),
                marker=dict(size=5, color=C_BLUE),
                fill="tozeroy", fillcolor="rgba(99,120,255,0.06)",
                name="Avg Default %"
            ))
            apply_layout(fig, 180)
            fig.update_layout(yaxis_title="Default Probability (%)", xaxis_title="")
            card(f"{sel} - Default Risk Trend Over Time")
            st.plotly_chart(fig, use_container_width=True)
            card_end()

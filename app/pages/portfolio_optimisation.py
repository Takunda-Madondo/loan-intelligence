"""
app/pages/portfolio_optimisation.py
Portfolio Optimisation — Sector-level LP capital allocation across three lender profiles.

Shows:
  - Budget + risk tolerance controls
  - Side-by-side profile comparison (Conservative / Balanced / Aggressive)
  - Allocation breakdown per profile
  - Expected return vs risk trade-off chart
  - Loan-level next steps section
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "pipelines"))

from app.data_loader import load_sector_performance, AFRICAN_COUNTRIES
from app.components import (kpi, section, card, card_end, insight,
                             apply_layout, toolbar,
                             C_BLUE, C_GREEN, C_RED, C_AMBER, C_TEAL)

# ── Constants ─────────────────────────────────────────────────────────────────

PROFILE_META = {
    "Conservative": {
        "colour":              "#0891B2",
        "risk_ceiling":        0.065,
        "max_high_risk_alloc": 0.08,
        "min_sector_alloc":    0.01,
        "max_sector_alloc":    0.35,
        "high_risk_threshold": 0.09,
        "tagline":             "Tight covenants · Low PAR30 · Capital preservation",
    },
    "Balanced": {
        "colour":              "#6378FF",
        "risk_ceiling":        0.082,
        "max_high_risk_alloc": 0.18,
        "min_sector_alloc":    0.02,
        "max_sector_alloc":    0.30,
        "high_risk_threshold": 0.09,
        "tagline":             "Mission + return · African MFI benchmark · Diversified",
    },
    "Aggressive": {
        "colour":              "#D97706",
        "risk_ceiling":        0.105,
        "max_high_risk_alloc": 0.32,
        "min_sector_alloc":    0.02,
        "max_sector_alloc":    0.28,
        "high_risk_threshold": 0.09,
        "tagline":             "Growth-oriented · Deeper market penetration · Higher PAR30",
    },
}

SECTOR_COLOURS = {
    "Agriculture & Farming":  "#16A34A",
    "Food & Beverage":        "#D97706",
    "Retail & Trade":         "#6378FF",
    "Services":               "#7C3AED",
    "Housing & Construction": "#0891B2",
    "Education":              "#DB2777",
    "Transport & Logistics":  "#0D9488",
    "Manufacturing":          "#DC2626",
    "Health & Wellness":      "#65A30D",
    "Personal Use":           "#9333EA",
    "Clothing & Textiles":    "#EA580C",
    "Arts & Crafts":          "#CA8A04",
}

BASE_RATE        = 0.12
RISK_PREMIUM_MULT = 8.0
LGD              = 0.60     # Loss Given Default — microfinance ~60% (group lending recovers ~40%)


# ── LP solver (pure Python / numpy — no scipy import required at module level) ─

def _solve_lp(sectors_df: pd.DataFrame, profile: dict, budget: float) -> dict | None:
    """
    Thin wrapper that imports scipy only when called, so the page
    loads even if scipy isn't installed on Streamlit Cloud yet.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    n  = len(sectors_df)
    er = sectors_df["expected_return"].values
    dr = sectors_df["default_rate_dec"].values
    hi = (dr >= profile["high_risk_threshold"]).astype(float)

    # Inequality: A_ub @ x <= b_ub
    A_ub = np.array([dr, hi])
    b_ub = np.array([profile["risk_ceiling"], profile["max_high_risk_alloc"]])

    # Equality: sum(x) = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    bounds = [(profile["min_sector_alloc"], profile["max_sector_alloc"])] * n

    res = linprog(-er, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        # Relax and retry once
        b_ub_relax = np.array([profile["risk_ceiling"] * 1.25,
                                profile["max_high_risk_alloc"] * 1.30])
        bounds_relax = [(0.0, profile["max_sector_alloc"])] * n
        res = linprog(-er, A_ub=A_ub, b_ub=b_ub_relax, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds_relax, method="highs")
        if not res.success:
            return None

    x = np.clip(res.x, 0, 1)
    x = x / x.sum()

    return {
        "fractions":                  x,
        "allocations_usd":            x * budget,
        "portfolio_default_rate":     float(np.dot(dr, x)),
        "portfolio_expected_return":  float(np.dot(er, x)),
        "total_expected_profit_usd":  float(np.dot(er, x) * budget),
    }


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def _prepare_sectors(region_filter: str, min_loans: int,
                     country_filter: str | None = None) -> pd.DataFrame:
    """Aggregate sector_performance to one row per sector with derived metrics.
    Always Africa-scoped. If country_filter is a 2-letter code, restrict to that country.
    """
    raw = load_sector_performance()

    # Always restrict to Africa
    raw = raw[raw["country_code"].isin(AFRICAN_COUNTRIES)]

    # Optionally drill down to a single country
    if country_filter is not None:
        raw = raw[raw["country_code"] == country_filter]

    raw = raw[raw["total_loans"] >= min_loans]

    # Guard: if filter leaves us with no rows, fall back to all Africa
    if raw.empty:
        raw = load_sector_performance()
        raw = raw[raw["country_code"].isin(AFRICAN_COUNTRIES)]
        raw = raw[raw["total_loans"] >= min_loans]

    # default_rate stored as percentage → convert to decimal
    raw["default_rate_dec"] = raw["default_rate"] / 100.0

    agg = raw.groupby("sector").agg(
        roi_score           = ("roi_score",           "mean"),
        default_rate_dec    = ("default_rate_dec",    "mean"),
        default_rate_pct    = ("default_rate",        "mean"),
        total_loans         = ("total_loans",         "sum"),
        avg_loan_amount     = ("avg_loan_amount",     "mean"),
        female_borrower_pct = ("female_borrower_pct", "mean"),
        avg_gdp_growth      = ("avg_gdp_growth",      "mean"),
    ).reset_index()

    agg["implied_interest_rate"] = (
        BASE_RATE + RISK_PREMIUM_MULT * agg["default_rate_dec"]
    ).round(4)

    agg["expected_return"] = (
        agg["implied_interest_rate"] * (1 - agg["default_rate_dec"])
        - agg["default_rate_dec"] * LGD
    ).round(6)

    agg["risk_tier"] = pd.cut(
        agg["default_rate_dec"],
        bins=[0, 0.07, 0.09, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    return agg.reset_index(drop=True)


def _run_all_profiles(sectors: pd.DataFrame, budget: float) -> dict:
    """Run LP for all three profiles and return a dict of results."""
    results = {}
    for name, meta in PROFILE_META.items():
        r = _solve_lp(sectors, meta, budget)
        if r is not None:
            results[name] = r
    return results


# ── Chart builders ────────────────────────────────────────────────────────────

def _allocation_donut(sectors: pd.DataFrame, fractions: np.ndarray,
                      profile_name: str, colour: str) -> go.Figure:
    labels  = sectors["sector"].tolist()
    values  = (fractions * 100).round(1).tolist()
    colours = [SECTOR_COLOURS.get(s, "#94A3B8") for s in labels]

    # Only show text on slices large enough to avoid crowding (>= 6%)
    # Smaller slices are readable via the legend and hover tooltip
    text_labels = [
        f"{v:.0f}%" if v >= 6.0 else ""
        for v in values
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=colours, line=dict(color="#FFFFFF", width=2)),
        text=text_labels,
        textinfo="text",
        textposition="inside",
        textfont=dict(size=10, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}% of capital<extra></extra>",
        sort=True,
        direction="clockwise",
        showlegend=True,
    ))

    # Short sector names for the legend to avoid overflow
    SHORT = {
        "Agriculture & Farming":  "Agriculture",
        "Transport & Logistics":  "Transport",
        "Housing & Construction": "Housing",
        "Health & Wellness":      "Health",
        "Clothing & Textiles":    "Clothing",
        "Food & Beverage":        "Food & Bev",
        "Retail & Trade":         "Retail",
        "Arts & Crafts":          "Arts",
        "Personal Use":           "Personal",
        "Manufacturing":          "Manufacturing",
        "Services":               "Services",
        "Education":              "Education",
    }
    fig.update_traces(
        legendgrouptitle_text="",
        customdata=[SHORT.get(l, l) for l in labels],
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=9, color="#374151"),
            itemwidth=30,
            tracegroupgap=2,
        ),
        margin=dict(t=16, b=16, l=16, r=100),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{profile_name[:12]}</b>",
            x=0.38, y=0.5,
            font=dict(size=12, color=colour),
            showarrow=False,
            xref="paper", yref="paper",
        )],
    )
    return fig


def _comparison_bar(sectors: pd.DataFrame, all_results: dict) -> go.Figure:
    """Grouped bar chart: allocation % by sector across the three profiles."""
    fig = go.Figure()

    for profile_name, result in all_results.items():
        colour = PROFILE_META[profile_name]["colour"]
        fig.add_trace(go.Bar(
            name=profile_name,
            x=sectors["sector"].tolist(),
            y=(result["fractions"] * 100).round(1).tolist(),
            marker_color=colour,
            marker_line_width=0,
            hovertemplate=(
                f"<b>{profile_name}</b><br>"
                "%{x}<br>"
                "Allocation: %{y:.1f}%<extra></extra>"
            ),
        ))

    apply_layout(fig, 300, legend=True)
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title="Allocation (%)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.08, font_size=11),
    )
    return fig


def _return_risk_scatter(all_results: dict) -> go.Figure:
    """Scatter: expected return vs portfolio default rate per profile."""
    fig = go.Figure()

    for name, result in all_results.items():
        colour = PROFILE_META[name]["colour"]
        fig.add_trace(go.Scatter(
            x=[result["portfolio_default_rate"] * 100],
            y=[result["portfolio_expected_return"] * 100],
            mode="markers+text",
            name=name,
            marker=dict(size=22, color=colour,
                        line=dict(color="#FFFFFF", width=2)),
            text=[name],
            textposition="top center",
            textfont=dict(size=11, color=colour),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Portfolio Risk: %{x:.2f}%<br>"
                "Expected Return: %{y:.2f}%<extra></extra>"
            ),
        ))

    # Efficiency frontier shading
    if all_results:
        xs = [r["portfolio_default_rate"] * 100 for r in all_results.values()]
        ys = [r["portfolio_expected_return"] * 100 for r in all_results.values()]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color="rgba(99,120,255,0.3)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    apply_layout(fig, 280, legend=False)
    fig.update_layout(
        xaxis=dict(title="Portfolio Default Rate (%)", ticksuffix="%"),
        yaxis=dict(title="Expected Return (%)", ticksuffix="%"),
    )
    fig.add_annotation(
        x=min(xs) if all_results else 6,
        y=max(ys) * 1.05 if all_results else 10,
        text="← Lower risk, lower return",
        showarrow=False,
        font=dict(size=9, color="#9CA3AF"),
    )
    return fig


def _waterfall_profit(all_results: dict, budget: float) -> go.Figure:
    """Horizontal bar showing expected profit per profile."""
    profiles = list(all_results.keys())
    profits  = [all_results[p]["total_expected_profit_usd"] for p in profiles]
    colours  = [PROFILE_META[p]["colour"] for p in profiles]

    fig = go.Figure(go.Bar(
        y=profiles,
        x=profits,
        orientation="h",
        marker_color=colours,
        marker_line_width=0,
        text=[f"${p:,.0f}" for p in profits],
        textposition="outside",
        textfont=dict(size=11, color="#374151"),
        hovertemplate="<b>%{y}</b><br>Expected Profit: $%{x:,.0f}<extra></extra>",
    ))

    apply_layout(fig, 180, legend=False)
    fig.update_layout(
        xaxis=dict(title="Expected Profit (USD)", tickprefix="$"),
        yaxis=dict(title=""),
        margin=dict(r=80),
    )
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────

PROFILE_CSS = """
<style>
.profile-header {
    border-left: 4px solid var(--c);
    padding: 0.55rem 0.75rem;
    background: var(--bg);
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.5rem;
}
.profile-header .ph-icon  { font-size: 1.2rem; margin-right: 0.4rem; }
.profile-header .ph-name  { font-size: 0.95rem; font-weight: 700; color: #111318; }
.profile-header .ph-tag   { font-size: 0.72rem; color: #6B7280; margin-top: 0.1rem; }
.profile-header .ph-stat  {
    display: inline-block; margin-top: 0.35rem;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
    padding: 2px 7px; border-radius: 4px;
    background: var(--badge-bg); color: var(--badge-c);
}
.alloc-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.22rem 0; border-bottom: 1px solid #F3F4F6;
    font-size: 0.78rem; color: #374151;
}
.alloc-row:last-child { border-bottom: none; }
.alloc-bar-wrap {
    width: 80px; background: #F3F4F6; border-radius: 2px; height: 6px;
    display: inline-block; margin: 0 0.5rem;
}
.alloc-bar { height: 6px; border-radius: 2px; }
.comparison-kpi {
    background: #F9FAFB; border: 1px solid #E5E7EB;
    border-radius: 8px; padding: 0.7rem 1rem;
    text-align: center; margin-bottom: 0.5rem;
}
.comparison-kpi .ck-label { font-size: 0.68rem; color: #9CA3AF;
    text-transform: uppercase; letter-spacing: 0.06em; }
.comparison-kpi .ck-val   { font-size: 1.35rem; font-weight: 700; color: #111318; }
.comparison-kpi .ck-sub   { font-size: 0.7rem; color: #6B7280; }
.next-steps-box {
    background: linear-gradient(135deg, #F8FAFF 0%, #EFF2FF 100%);
    border: 1px solid #C7D2FE; border-radius: 10px;
    padding: 1.2rem 1.5rem; margin-top: 0.5rem;
}
.ns-title { font-size: 0.85rem; font-weight: 700; color: #3730A3; margin-bottom: 0.6rem; }
.ns-item  { font-size: 0.78rem; color: #374151; margin: 0.3rem 0; padding-left: 0.4rem; }
.ns-item::before { content: "→ "; color: #6378FF; font-weight: 600; }
.complexity-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 10px; margin-left: 0.4rem;
    letter-spacing: 0.05em;
}
.cb-hard   { background: #FEE2E2; color: #991B1B; }
.cb-medium { background: #FEF3C7; color: #92400E; }
.cb-easy   { background: #D1FAE5; color: #065F46; }
</style>
"""


# ── Render helpers ────────────────────────────────────────────────────────────

def _profile_card(name: str, meta: dict, result: dict,
                  sectors: pd.DataFrame, fractions: np.ndarray):
    """
    Renders a complete profile card as a single st.markdown HTML block.
    No open/close div split — Streamlit requires the full HTML in one call.
    """
    c      = meta["colour"]
    bg     = c + "12"
    profit = f"${result['total_expected_profit_usd']:,.0f}"
    rdr    = f"{result['portfolio_default_rate']*100:.2f}%"
    er     = f"{result['portfolio_expected_return']*100:.2f}%"

    # Build top-5 sector rows as HTML string
    sorted_idx = np.argsort(-fractions)
    top5       = sorted_idx[:5]
    max_pct    = fractions[top5[0]] * 100 if len(top5) else 35.0

    rows_html = ""
    for i in top5:
        sector = sectors.iloc[i]["sector"]
        pct    = fractions[i] * 100
        bar_w  = max(2, int(pct / max_pct * 68))
        tier   = sectors.iloc[i]["risk_tier"]
        tier_c = {"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}.get(tier, "#6B7280")
        label  = sector if len(sector) <= 22 else sector[:20] + "..."

        rows_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.25rem 0; border-bottom:1px solid #F3F4F6;">
                <span style="flex:1; font-size:0.73rem; color:#374151;
                             overflow:hidden; text-overflow:ellipsis;
                             white-space:nowrap; padding-right:0.4rem;">{label}</span>
                <div style="display:flex; align-items:center; gap:0.3rem; flex-shrink:0;">
                    <div style="width:68px; background:#F3F4F6; border-radius:2px; height:5px;">
                        <div style="width:{bar_w}px; height:5px; border-radius:2px;
                                    background:{c};"></div>
                    </div>
                    <span style="width:34px; text-align:right; font-size:0.73rem;
                                 font-weight:600; color:#111318;">{pct:.1f}%</span>
                    <span style="width:7px; height:7px; border-radius:50%;
                                 background:{tier_c}; display:inline-block;"></span>
                </div>
            </div>"""

    remaining = len(sorted_idx) - 5
    footer = (f'<div style="font-size:0.66rem; color:#9CA3AF; margin-top:0.35rem;">'
              f'+ {remaining} more in chart below</div>') if remaining > 0 else ""

    # Build the full card as a plain string — no f-string interpolation of rows_html
    # inside another f-string, which can cause Streamlit to escape the inner HTML.
    card_top = (
        '<div style="background:#ffffff; border:1px solid #E8EAED;'
        f' border-top:3px solid {c}; border-radius:10px;'
        ' padding:0.9rem 1rem; width:100%; box-sizing:border-box; overflow:hidden;">'
        f'<div style="font-size:0.9rem; font-weight:700; color:#111318; margin-bottom:0.15rem;">{name}</div>'
        f'<div style="font-size:0.71rem; color:#6B7280; margin-bottom:0.55rem; line-height:1.4;">{meta["tagline"]}</div>'
        '<div style="display:flex; flex-wrap:wrap; gap:0.3rem; margin-bottom:0.7rem;">'
        f'<span style="font-size:0.68rem; font-weight:600; padding:2px 7px; border-radius:4px; background:{bg}; color:{c};">Risk {rdr}</span>'
        f'<span style="font-size:0.68rem; font-weight:600; padding:2px 7px; border-radius:4px; background:{bg}; color:{c};">Return {er}</span>'
        f'<span style="font-size:0.68rem; font-weight:600; padding:2px 7px; border-radius:4px; background:{bg}; color:{c};">Profit {profit}</span>'
        '</div>'
        '<div style="font-size:0.66rem; font-weight:700; color:#9CA3AF;'
        ' text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.35rem;">Top 5 Sectors</div>'
    )
    card_bottom = footer + '</div>'
    st.markdown(card_top + rows_html + card_bottom, unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown(PROFILE_CSS, unsafe_allow_html=True)

    # ── Page-specific controls (no generic toolbar) ─────────────────────────
    section("PORTFOLIO PARAMETERS")

    # Load all African countries available in the data for the country filter
    _all_sector_data = load_sector_performance()
    _african_available = sorted(
        _all_sector_data[_all_sector_data["country_code"].isin(AFRICAN_COUNTRIES)]["country_code"]
        .dropna().unique().tolist()
    )
    # Map code → readable name where possible
    _CODE_NAMES = {
        "KE":"Kenya","UG":"Uganda","TZ":"Tanzania","RW":"Rwanda","ET":"Ethiopia",
        "GH":"Ghana","NG":"Nigeria","SN":"Senegal","ML":"Mali","MZ":"Mozambique",
        "ZM":"Zambia","ZW":"Zimbabwe","MW":"Malawi","MG":"Madagascar","CM":"Cameroon",
        "BF":"Burkina Faso","TG":"Togo","BJ":"Benin","GN":"Guinea",
    }
    _country_options = ["All Africa"] + [
        f"{_CODE_NAMES.get(c, c)} ({c})" for c in _african_available
    ]

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1.8, 1.8])

    with ctrl1:
        budget_label = st.select_slider(
            "Capital Budget",
            options=["$100K", "$250K", "$500K", "$1M", "$2M", "$5M", "$10M"],
            value="$1M",
            key="po_budget",
        )
        budget_map = {
            "$100K": 100_000, "$250K": 250_000, "$500K": 500_000,
            "$1M": 1_000_000, "$2M": 2_000_000, "$5M": 5_000_000,
            "$10M": 10_000_000,
        }
        budget = budget_map[budget_label]

    with ctrl2:
        country_sel = st.selectbox(
            "Country (Africa)",
            options=_country_options,
            key="po_country",
        )
        # Resolve to a country_code filter or None (= all Africa)
        if country_sel == "All Africa":
            country_filter = None   # use full African dataset
        else:
            country_filter = country_sel.split("(")[-1].rstrip(")")

    with ctrl3:
        min_vol_label = st.selectbox(
            "Min Loans per Sector",
            ["Any", "50+", "100+", "500+"],
            index=2,
            key="po_minvol",
        )
        min_vol_map = {"Any": 0, "50+": 50, "100+": 100, "500+": 500}
        min_loans = min_vol_map[min_vol_label]

    # ── Solve ─────────────────────────────────────────────────────────────────
    # Always Africa-scoped; optionally filtered to a single country
    sectors = _prepare_sectors("Africa Only", min_loans, country_filter)

    try:
        from scipy.optimize import linprog as _test_import  # noqa
        solver_ok = True
    except ImportError:
        solver_ok = False

    if not solver_ok:
        st.error(
            "**scipy** is required for portfolio optimisation. "
            "Add `scipy` to requirements.txt and redeploy, or run "
            "`pip install scipy` locally."
        )
        return

    all_results = _run_all_profiles(sectors, budget)

    if not all_results:
        st.warning("LP solver returned no feasible solutions. Try relaxing the budget or region filters.")
        return

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    section("PROFILE COMPARISON")

    kpi_cols = st.columns(len(all_results))
    for col, (name, result) in zip(kpi_cols, all_results.items()):
        meta   = PROFILE_META[name]
        profit = result["total_expected_profit_usd"]
        rdr    = result["portfolio_default_rate"] * 100
        er     = result["portfolio_expected_return"] * 100

        with col:
            st.markdown(
                f"""<div class="comparison-kpi" style="border-top: 3px solid {meta['colour']}">
                    <div class="ck-label">{name}</div>
                    <div class="ck-val" style="color:{meta['colour']}">${profit:,.0f}</div>
                    <div class="ck-sub">Expected Profit</div>
                    <div style="margin-top:0.4rem">
                        <span class="ph-stat" style="
                            display:inline-block; font-size:0.68rem; font-weight:600;
                            padding:2px 7px; border-radius:4px;
                            background:{meta['colour']}18; color:{meta['colour']}">
                            Risk {rdr:.1f}% · Return {er:.1f}%
                        </span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Side-by-side allocation columns ──────────────────────────────────────
    # Each profile card is rendered as a single HTML block (no split open/close divs).
    # Donut chart sits below the card as a separate plotly call per column.
    section("OPTIMAL CAPITAL ALLOCATION")
    profile_names = list(all_results.keys())
    alloc_cols    = st.columns(len(profile_names))

    for col, name in zip(alloc_cols, profile_names):
        result = all_results[name]
        meta   = PROFILE_META[name]
        with col:
            _profile_card(name, meta, result, sectors, result["fractions"])
            fig = _allocation_donut(sectors, result["fractions"], name, meta["colour"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Comparison charts ─────────────────────────────────────────────────────
    section("RISK-RETURN ANALYSIS")
    ch1, ch2 = st.columns([1.6, 1])

    with ch1:
        card("Sector Allocation by Profile",
             "How each risk philosophy distributes capital across sectors")
        st.plotly_chart(_comparison_bar(sectors, all_results),
                        use_container_width=True, config={"displayModeBar": False})
        card_end()

    with ch2:
        card("Risk-Return Trade-off",
             "Portfolio default rate vs expected return per profile")
        st.plotly_chart(_return_risk_scatter(all_results),
                        use_container_width=True, config={"displayModeBar": False})
        card_end()

    # Expected profit bar
    card("Expected Profit by Profile", f"Based on ${budget:,.0f} capital deployment")
    st.plotly_chart(_waterfall_profit(all_results, budget),
                    use_container_width=True, config={"displayModeBar": False})
    card_end()

    # ── Sector detail table ───────────────────────────────────────────────────
    section("SECTOR DETAILS")

    rows = []
    for name, result in all_results.items():
        for i, (_, srow) in enumerate(sectors.iterrows()):
            rows.append({
                "Profile":     name,
                "Sector":      srow["sector"],
                "Alloc %":     f"{result['fractions'][i]*100:.1f}%",
                "Amount":      f"${result['allocations_usd'][i]:,.0f}",
                "Risk Score":  f"{srow['default_rate_pct']:.1f}%",
                "E[Return]":   f"{srow['expected_return']*100:.2f}%",
                "ROI Score":   f"{srow['roi_score']:.3f}",
                "Risk Tier":   srow["risk_tier"],
                "Loans":       f"{int(srow['total_loans']):,}",
            })

    detail_df = pd.DataFrame(rows)

    profile_filter = st.selectbox(
        "Filter by profile",
        ["All Profiles"] + profile_names,
        key="po_profile_filter",
    )
    if profile_filter != "All Profiles":
        detail_df = detail_df[detail_df["Profile"] == profile_filter]

    def _colour_tier(val):
        return {
            "Low":    "background-color:#F0FDF4",
            "Medium": "background-color:#FFFBEB",
            "High":   "background-color:#FEF2F2",
        }.get(val, "")

    card("Full Allocation Table", "All sectors and profiles")
    st.dataframe(
        detail_df.style.applymap(_colour_tier, subset=["Risk Tier"]),
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    card_end()

    # ── Methodology note ─────────────────────────────────────────────────────
    section("METHODOLOGY & ASSUMPTIONS")
    insight(
        "Methodology — Each profile solves a Linear Programme maximising expected portfolio return "
        "under three constraints: (1) weighted average portfolio default rate ≤ risk ceiling, "
        "(2) capital in high-risk sectors (default > 9%) ≤ profile cap, "
        "(3) no single sector exceeds the concentration limit. "
        "Expected return per sector = interest_rate × (1 − default_rate) − default_rate × LGD, "
        "where interest rate = 12% base + 8× sector default rate (MFI risk pricing) "
        "and Loss Given Default = 0.60 (group lending recovers ~40%). "
        "All sector default rates use African MFI PAR30 benchmarks."
    )

    # ── Next steps — loan-level ───────────────────────────────────────────────
    section("NEXT STEPS — LOAN-LEVEL OPTIMISATION")
    st.markdown(
        """<div class="next-steps-box">
            <div class="ns-title">Extending to Individual Loan Selection</div>
            <p style="font-size:0.78rem;color:#374151;margin-bottom:0.6rem">
                The current implementation allocates capital at the <b>sector level</b> — the
                optimal approach for investment committee decisions. The natural extension
                is loan-level selection: choosing specific borrowers within each sector's
                allocation. This introduces a Binary Integer Programme (BIP) — more complex
                but more granular.
            </p>
            <div class="ns-item">
                <b>Two-stage approach</b> — use this LP output to set sector budgets,
                then within each sector's allocation, rank individual loans by
                expected_return / risk_score (Sharpe ratio proxy) and greedily select
                until the sector budget is exhausted.
                <span class="complexity-badge cb-easy">Low complexity</span>
            </div>
            <div class="ns-item">
                <b>Pre-filtered BIP</b> — restrict candidate loans to those with
                risk_score &lt; 0.08 and funded_ratio &gt; 0.90, reducing the 671K-loan
                problem to ~30–50K candidates, then solve as a Binary Integer Programme
                using PuLP with the CBC solver.
                <span class="complexity-badge cb-medium">Medium complexity</span>
            </div>
            <div class="ns-item">
                <b>Full loan-level LP (continuous relaxation)</b> — treat each loan as a
                fractional allocation variable (0 to max_loan_fraction), solve a continuous
                LP across all 671K loans. Computationally tractable with HiGHS solver;
                results rounded to binary for implementation.
                <span class="complexity-badge cb-medium">Medium complexity</span>
            </div>
            <div class="ns-item">
                <b>Behavioural features</b> — integrating mobile money transaction frequency,
                group cohesion scores, and repayment history would significantly improve
                per-loan expected return estimation, pushing AUC above 0.85 and making
                loan-level selection materially more accurate.
                <span class="complexity-badge cb-hard">Requires new data</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

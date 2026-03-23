"""app/components.py — Shared components. Senior UX design."""
import os, sys
import streamlit as st
from pathlib import Path

ROOT3 = Path(__file__).parent.parent
sys.path.append(str(ROOT3))
sys.path.append(str(ROOT3 / "pipelines"))

# ── Palette ─────────────────────────────────────────────────────────────────
C_BLUE  = "#6378FF"
C_GREEN = "#16A34A"
C_RED   = "#DC2626"
C_AMBER = "#D97706"
C_TEAL  = "#0D9488"
C_GRAY  = "#9CA3AF"

CHART_CFG = dict(
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    font          = dict(family="DM Sans", size=12, color="#111318"),
    margin        = dict(t=10, b=8, l=2, r=2),
    xaxis         = dict(gridcolor="#F3F4F6", linecolor="#D1D5DB",
                         tickfont=dict(size=11, color="#374151"),
                         title_font=dict(size=11, color="#374151"),
                         zeroline=False),
    yaxis         = dict(gridcolor="#F3F4F6", linecolor="#D1D5DB",
                         tickfont=dict(size=11, color="#374151"),
                         title_font=dict(size=11, color="#374151"),
                         zeroline=False),
)

SHAP_NAMES = {
    "sub_grade_enc":          "Credit Sub-Grade",
    "grade_x_term":           "Grade × Loan Term",
    "int_rate":               "Interest Rate",
    "revol_bal":              "Revolving Balance",
    "avg_cur_bal":            "Avg Current Balance",
    "home_enc":               "Home Ownership",
    "revol_util":             "Revolving Utilisation",
    "dti_x_int_rate":         "DTI × Interest Rate",
    "loan_to_income":         "Loan-to-Income Ratio",
    "num_bc_sats":            "Satisfactory Accounts",
    "term_months":            "Loan Term",
    "mort_acc":               "Mortgage Accounts",
    "loan_amnt":              "Loan Amount",
    "inc_x_dti":              "Income ÷ DTI",
    "open_acc":               "Open Accounts",
    "dti":                    "Debt-to-Income Ratio",
    "annual_inc":             "Annual Income",
    "inq_last_6mths":         "Credit Inquiries (6mo)",
    "installment":            "Monthly Instalment",
    "total_acc":              "Total Accounts",
    "delinq_2yrs":            "Delinquencies (2yr)",
    "pct_tl_nvr_dlq":         "% Accounts Never Delinquent",
    "grade_enc":              "Credit Grade",
    "verification_enc":       "Income Verification",
    "pub_rec":                "Public Records",
    "emp_length":             "Employment Length",
    "bc_util":                "Bankcard Utilisation",
    "sector_enc":             "Loan Sector",
    "funded_ratio":           "Funding Ratio",
    "num_rev_accts":          "Revolving Accounts",
    "tot_cur_bal":            "Total Current Balance",
    "num_tl_90g_dpd_24m":     "Severely Delinquent Accounts",
    "inq_x_delinq":           "Inquiries × Delinquency",
    "revol_util_x_bal":       "Utilisation × Balance",
    "payment_to_income":      "Payment-to-Income Ratio",
    "macro_risk_score":       "Country Macro Risk",
    "sector_default_rate":    "Sector Default Rate",
    "lender_trust_score":     "Lender Trust Score",
    "grade_enc":              "Credit Grade",
}


def ch(n):
    """Rename SHAP feature to human-readable label."""
    return SHAP_NAMES.get(n, n.replace("_", " ").title())


def apply_layout(fig, height=260, legend=False):
    fig.update_layout(**CHART_CFG, height=height, showlegend=legend)
    return fig


def kpi(label, value, delta="", delta_type="neu", accent="#6378FF"):
    arrow = {"pos": "↑ ", "neg": "↓ ", "neu": ""}[delta_type]
    st.markdown(f"""
    <div class="kpi" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_type}">{arrow}{delta}</div>
    </div>""", unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="sec-div">{label}</div>', unsafe_allow_html=True)


def card(title, meta=""):
    m = f'<span class="card-meta">{meta}</span>' if meta else ""
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">{title}</span>{m}
        </div>""", unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


def signal(title, body, invest=True):
    cls = "signal-invest" if invest else "signal-avoid"
    icon = "●" if invest else "●"
    st.markdown(f"""
    <div class="signal-card {cls}">
        <div class="signal-title">{"▲ " if invest else "▼ "}{title}</div>
        <div class="signal-body">{body}</div>
    </div>""", unsafe_allow_html=True)


# ── Toolbar: filters + AI in one row ─────────────────────────────────────────
def toolbar(page_key=""):
    """Returns filter values dict. Renders filter + AI bar inline."""

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.4])
    with f1:
        st.markdown('<div class="toolbar-label">Region</div>', unsafe_allow_html=True)
        region = st.selectbox("Region", ["All Markets", "Africa Only"],
                              key=f"{page_key}_reg", label_visibility="collapsed")
    with f2:
        st.markdown('<div class="toolbar-label">Risk Level</div>', unsafe_allow_html=True)
        risk = st.selectbox("Risk Level", ["All", "Low", "Medium", "High", "Very High"],
                            key=f"{page_key}_risk", label_visibility="collapsed")
    with f3:
        st.markdown('<div class="toolbar-label">Min Loan Volume</div>', unsafe_allow_html=True)
        min_vol = st.selectbox("Min Volume", ["Any", "50+", "100+", "500+"],
                               key=f"{page_key}_vol", label_visibility="collapsed")
    with f4:
        ai_bar(page_key)

    return {"region": region, "risk": risk, "min_vol": min_vol}


# ── AI bar (compact, inline) ──────────────────────────────────────────────────
def ai_bar(page_key=""):
    from config import ANTHROPIC_KEY
    import streamlit as st
    api_key = (ANTHROPIC_KEY
               or os.getenv("ANTHROPIC_API_KEY")
               or st.secrets.get("ANTHROPIC_API_KEY", ""))

    st.markdown('<div class="toolbar-label">Ask AI</div>', unsafe_allow_html=True)
    with st.expander("✦  Ask about your data", expanded=False):
        if not api_key:
            st.markdown(
                '<div style="font-size:0.75rem;color:#DC2626">Add ANTHROPIC_API_KEY to .env</div>',
                unsafe_allow_html=True)
            return

        for msg in st.session_state.get("chat_history", [])[-4:]:
            align = "flex-end" if msg["role"] == "user" else "flex-start"
            cls   = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
            st.markdown(
                f'<div style="display:flex;justify-content:{align};margin:0.15rem 0">'
                f'<div class="{cls}">{msg["content"]}</div></div>',
                unsafe_allow_html=True)

        user_input = st.chat_input("e.g. Which market has the lowest default risk?",
                                   key=f"ai_{page_key}")

        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            _call_claude(user_input)
            st.rerun()

        if st.session_state.get("chat_history"):
            if st.button("Clear chat", key=f"clr_{page_key}"):
                st.session_state["chat_history"] = []
                st.rerun()


def _call_claude(user_input):
    from anthropic import Anthropic
    from app.data_loader import (load_kiva_predictions, load_sector_performance,
                                  load_model_metrics, AFRICAN_COUNTRIES)
    from config import ANTHROPIC_KEY
    import os, streamlit as st

    api_key = (ANTHROPIC_KEY
               or os.getenv("ANTHROPIC_API_KEY")
               or st.secrets.get("ANTHROPIC_API_KEY", ""))
    preds   = load_kiva_predictions()
    sectors = load_sector_performance()
    metrics = load_model_metrics()
    african = preds[preds["country_code"].isin(AFRICAN_COUNTRIES)]

    cr = (african.groupby("country")
                 .agg(loans=("loan_id","count"), risk=("default_probability","mean"))
                 .reset_index().sort_values("risk").round(3))
    sr = (sectors[sectors["country_code"].isin(AFRICAN_COUNTRIES)]
          .groupby("sector")
          .agg(loans=("total_loans","sum"), default=("default_rate","mean"),
               roi=("roi_score","mean"))
          .reset_index().sort_values("default").round(3))

    system = f"""You are a senior microfinance portfolio analyst. Be direct and specific.
Answer in 2–3 concise sentences max. Always cite numbers.

Portfolio: {len(preds):,} loans | Africa: {len(african):,} | Avg default prob: {preds['default_probability'].mean()*100:.1f}%
Model: XGBoost AUC-ROC {metrics['auc_roc']} | Decision threshold: {metrics['optimal_threshold']}

COUNTRY RISK (low→high):
{cr.to_string(index=False)}

SECTOR PERFORMANCE:
{sr.to_string(index=False)}

Investment rule: ROI score > 0.75 and default rate < 20% = invest. Default > 25% = avoid."""

    client   = Anthropic(api_key=api_key)
    messages = [{"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_history"][-6:]]
    try:
        resp   = client.messages.create(model="claude-sonnet-4-20250514",
                                        max_tokens=350, system=system, messages=messages)
        answer = resp.content[0].text
    except Exception as e:
        answer = f"Error: {e}"
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

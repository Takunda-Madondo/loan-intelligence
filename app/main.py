"""
app/main.py - Loan Intelligence Dashboard
Run: streamlit run app/main.py
"""
import streamlit as st
import sys
from pathlib import Path

ROOT3 = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT3))
sys.path.insert(0, str(APP_DIR))
sys.path.append(str(ROOT3 / "pipelines"))

st.set_page_config(
    page_title="Loan Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

/* Hide collapse button and resize handle — sidebar is always visible */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* ─── MAIN CONTENT ───────────────────────────────────────────────────────── */
.main .block-container { padding: 0 !important; max-width: 100% !important; }

.pg-header {
    background: #ffffff;
    border-bottom: 1px solid #E8EAED;
    padding: 0.85rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.pg-title { font-size: 1.05rem; font-weight: 700; color: #111318; letter-spacing: -0.02em; }
.pg-subtitle { font-size: 0.75rem; color: #9CA3AF; margin-top: 0.1rem; }
.pg-body { padding: 1.2rem 1.8rem 1.8rem; }

.toolbar {
    background: #ffffff; border: 1px solid #E8EAED; border-radius: 10px;
    padding: 0.7rem 1.1rem; margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 1rem;
}
.toolbar-label {
    font-size: 0.68rem; font-weight: 600; color: #9CA3AF;
    text-transform: uppercase; letter-spacing: 0.07em; white-space: nowrap;
}

/* KPI Cards */
.kpi-grid { display: grid; gap: 0.75rem; margin-bottom: 1.1rem; }
.kpi {
    background: #ffffff; border: 1px solid #E8EAED; border-radius: 10px;
    padding: 0.85rem 1rem; position: relative; overflow: hidden;
}
.kpi::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent, #6378FF);
}
.kpi-label {
    font-size: 0.66rem; font-weight: 600; color: #9CA3AF; text-transform: uppercase;
    letter-spacing: 0.07em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kpi-value {
    font-size: 1.4rem; font-weight: 700; color: #111318; letter-spacing: -0.03em;
    white-space: nowrap; line-height: 1.25; margin: 0.2rem 0 0.1rem;
    font-family: 'DM Mono', monospace;
}
.kpi-delta { font-size: 0.7rem; font-weight: 500; white-space: nowrap; }
.kpi-delta.pos { color: #16A34A; }
.kpi-delta.neg { color: #DC2626; }
.kpi-delta.neu { color: #9CA3AF; }

.card {
    background: #ffffff; border: 1px solid #E8EAED; border-radius: 10px;
    padding: 1rem 1.1rem 0.6rem; margin-bottom: 0.75rem;
}
.card-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.6rem; }
.card-title { font-size: 0.78rem; font-weight: 600; color: #111318; letter-spacing: -0.01em; }
.card-meta { font-size: 0.67rem; color: #9CA3AF; }

.badge {
    display: inline-flex; align-items: center; padding: 0.2rem 0.65rem;
    border-radius: 999px; font-size: 0.68rem; font-weight: 600; gap: 0.3rem;
}
.badge-safe   { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
.badge-medium { background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; }
.badge-risk   { background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; }

.signal-card { border-radius:10px; padding:0.9rem 1.1rem; border:1px solid; margin-bottom:0.5rem; }
.signal-invest { background:#F0FDF4; border-color:#BBF7D0; }
.signal-avoid  { background:#FEF2F2; border-color:#FECACA; }
.signal-title  { font-size:0.75rem; font-weight:700; margin-bottom:0.3rem; }
.signal-invest .signal-title { color:#15803D; }
.signal-avoid  .signal-title { color:#B91C1C; }
.signal-body   { font-size:0.72rem; line-height:1.6; color:#374151; }

.ai-trigger { display:flex; align-items:center; gap:0.4rem; font-size:0.75rem; font-weight:600; color:#6378FF; cursor:pointer; white-space:nowrap; }
.ai-dot { width:6px; height:6px; background:#6378FF; border-radius:50%; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.chat-msg-user {
    background:#6378FF; color:#fff; border-radius:8px 8px 2px 8px;
    padding:0.5rem 0.8rem; font-size:0.78rem; line-height:1.5;
    display:inline-block; max-width:85%; margin-bottom:0.4rem;
}
.chat-msg-ai {
    background:#F3F4F6; color:#1F2937; border-radius:8px 8px 8px 2px;
    padding:0.5rem 0.8rem; font-size:0.78rem; line-height:1.6;
    display:inline-block; max-width:90%; margin-bottom:0.4rem;
}

.sec-div {
    font-size:0.64rem; font-weight:700; color:#9CA3AF; text-transform:uppercase;
    letter-spacing:0.1em; margin:0.9rem 0 0.5rem;
    display:flex; align-items:center; gap:0.6rem;
}
.sec-div::after { content:''; flex:1; height:1px; background:#E8EAED; }

.insight-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-left:3px solid #6378FF;
    border-radius:8px; padding:0.7rem 1rem; font-size:0.76rem; color:#1E3A8A;
    line-height:1.65; margin-top:0.5rem;
}

/* Streamlit overrides */
div[data-testid="stMetric"] { display: none !important; }
.stButton > button {
    background: #6378FF !important; color: #fff !important;
    border: none !important; border-radius: 7px !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    padding: 0.35rem 0.9rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #4F63E0 !important; }

.stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 0.72rem !important; color: #6B7280 !important;
    font-weight: 500 !important; margin-bottom: 0.15rem !important;
}
.stSelectbox > div > div {
    border-radius: 7px !important; border-color: #E8EAED !important;
    font-size: 0.8rem !important; min-height: 34px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; gap: 0 !important;
    border-bottom: 1px solid #E8EAED !important; margin-bottom: 0.8rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.78rem !important; font-weight: 500 !important; color: #6B7280 !important;
    padding: 0.45rem 1rem !important; background: transparent !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: #6378FF !important; border-bottom-color: #6378FF !important; font-weight: 600 !important;
}
[data-testid="stDataFrame"] { border-radius: 8px !important; }
[data-testid="stDataFrame"] table { font-size: 0.78rem !important; }
.stExpander {
    border: 1px solid #E8EAED !important; border-radius: 10px !important;
    background: #ffffff !important; margin-bottom: 0.75rem !important;
}
.stExpander summary {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: #374151 !important; padding: 0.6rem 1rem !important;
}
[data-testid="stFileUploadDropzone"] {
    border-radius: 8px !important; border-color: #E8EAED !important; background: #FAFAFA !important;
}
.js-plotly-plot { border-radius: 6px; }
[data-testid="stChatInput"] input { font-size: 0.8rem !important; border-radius: 8px !important; }
</style>

<script>
// Streamlit stores sidebar collapsed state in localStorage (key: stSidebarCollapsed-<hash>).
// If the user ever clicked collapse, it stays closed on every reload — ignoring
// initial_sidebar_state="expanded". This script wipes those keys on every page load.
(function() {
    try {
        Object.keys(localStorage).forEach(function(k) {
            if (k.indexOf('stSidebarCollapsed') === 0) localStorage.removeItem(k);
        });
    } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("page", "Portfolio Overview"), ("chat_history", []), ("ai_open", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

PAGES = {
    "Portfolio Overview":      "Overview of all key metrics and investment signals",
    "Default Risk Map":        "Geographic risk distribution across markets",
    "Investment Signals":      "Where to invest and where to avoid - by sector & country",
    "Portfolio Optimisation":  "LP-optimised capital allocation across Conservative, Balanced and Aggressive profiles",
    "Loan Predictor":          "Score individual loans for default probability",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Loan Intelligence")
    st.caption("Microfinance Analytics")
    st.divider()

    page = st.radio(
        "Navigate",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    if page != st.session_state["page"]:
        st.session_state["page"] = page
        st.rerun()

    st.divider()
    st.caption("Dataset: 1.3M loans  \nModel: XGBoost  \nAUC-ROC · 0.723  \nMarkets: 60+ countries")
    st.caption("Built by **Taku**")

# ── Page Header ────────────────────────────────────────────────────────────────
current_page = st.session_state["page"]
st.markdown(f"""
<div class="pg-header">
    <div>
        <div class="pg-title">{current_page}</div>
        <div class="pg-subtitle">{PAGES[current_page]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pg-body">', unsafe_allow_html=True)

# ── Render Page ────────────────────────────────────────────────────────────────
import importlib, importlib.util

PAGE_MAP = {
    "Portfolio Overview":     "overview",
    "Investment Signals":     "investment_signals",
    "Portfolio Optimisation": "portfolio_optimisation",
    "Default Risk Map":       "country_risk",
    "Loan Predictor":         "loan_predictor",
}

mod_name = PAGE_MAP[current_page]
mod_path = APP_DIR / "pages" / f"{mod_name}.py"
spec = importlib.util.spec_from_file_location(mod_name, mod_path)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.render()

st.markdown('</div>', unsafe_allow_html=True)

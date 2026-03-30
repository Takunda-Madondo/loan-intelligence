"""app/pages/loan_predictor.py - Loan Predictor with Lending Strategy Context"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT3 = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT3))
sys.path.append(str(ROOT3 / "pipelines"))

from app.data_loader import load_model, load_feature_list, load_model_metrics
from app.components import (kpi, section, card, card_end, insight, apply_layout,
                             toolbar, C_GREEN, C_RED, C_AMBER, C_BLUE)

# ── Encoding maps ─────────────────────────────────────────────────────────────
GM  = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
SGM = {f"{g}{n}":i+1 for i,(g,n) in enumerate([(g,n) for g in "ABCDEFG" for n in range(1,6)])}
HM  = {"RENT":0,"OWN":1,"MORTGAGE":2,"OTHER":3}
VM  = {"Not Verified":0,"Verified":1,"Source Verified":2}
SEM = {"Retail & Trade":0,"Personal Use":1,"Housing & Construction":2,
       "Transport & Logistics":3,"Health & Wellness":4,"Education":5,
       "Services":6,"Manufacturing":7}

# ── Lending strategy profiles ─────────────────────────────────────────────────
# Risk ceilings match the LP optimisation profiles exactly so both pages are
# consistent in their definition of acceptable risk per strategy.
STRATEGIES = {
    "Conservative": {
        "threshold":   0.065,
        "colour":      "#0891B2",
        "description": "Regulated deposit-taking MFI. Tight covenants, capital preservation.",
        "gauge_steps": [
            {"range": [0,    6.5],  "color": "#F0FDF4"},
            {"range": [6.5,  10.0], "color": "#FFFBEB"},
            {"range": [10.0, 20.0], "color": "#FEF2F2"},
            {"range": [20.0, 25.0], "color": "#FDF2F8"},
        ],
    },
    "Balanced": {
        "threshold":   0.082,
        "colour":      "#6378FF",
        "description": "Mid-size African MFI. Balances financial return with social mission.",
        "gauge_steps": [
            {"range": [0,    8.2],  "color": "#F0FDF4"},
            {"range": [8.2,  12.0], "color": "#FFFBEB"},
            {"range": [12.0, 20.0], "color": "#FEF2F2"},
            {"range": [20.0, 25.0], "color": "#FDF2F8"},
        ],
    },
    "Aggressive": {
        "threshold":   0.105,
        "colour":      "#D97706",
        "description": "Growth-oriented fintech MFI. Capturing underserved segments, higher PAR30 accepted.",
        "gauge_steps": [
            {"range": [0,    10.5], "color": "#F0FDF4"},
            {"range": [10.5, 15.0], "color": "#FFFBEB"},
            {"range": [15.0, 20.0], "color": "#FEF2F2"},
            {"range": [20.0, 25.0], "color": "#FDF2F8"},
        ],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_vector(inp, features):
    la   = float(inp.get("loan_amnt",   10000))
    ai   = float(inp.get("annual_inc",  65000))
    tm   = float(inp.get("term_months", 36))
    ir   = float(inp.get("int_rate",    13.5))
    dti  = float(inp.get("dti",         18))
    inst = la / max(tm, 1)
    ge   = GM.get(inp.get("grade", "C"), 4)
    se   = SGM.get(inp.get("sub_grade", "C3"), ge * 3)
    ru   = float(inp.get("revol_util", 45))
    rb   = float(inp.get("revol_bal",  la * 0.5))
    raw  = {
        "loan_amnt":              la,
        "funded_ratio":           1.0,
        "term_months":            tm,
        "int_rate":               ir,
        "installment":            inst,
        "grade_enc":              ge,
        "sub_grade_enc":          se,
        "emp_length":             float(inp.get("emp_length", 5)),
        "home_enc":               HM.get(inp.get("home_ownership", "RENT"), 0),
        "annual_inc":             ai,
        "verification_enc":       VM.get(inp.get("verification_status", "Not Verified"), 0),
        "dti":                    dti,
        "delinq_2yrs":            float(inp.get("delinq_2yrs", 0)),
        "inq_last_6mths":         float(inp.get("inq_last_6mths", 1)),
        "mths_since_last_delinq": 36.,
        "mths_since_last_record": 60.,
        "open_acc":               float(inp.get("open_acc", 8)),
        "pub_rec":                float(inp.get("pub_rec", 0)),
        "revol_bal":              rb,
        "revol_util":             ru,
        "total_acc":              float(inp.get("total_acc", 15)),
        "mort_acc":               float(inp.get("mort_acc", 0)),
        "num_bc_sats":            4.,
        "pct_tl_nvr_dlq":         85.,
        "num_tl_90g_dpd_24m":     0.,
        "avg_cur_bal":            la * 0.3,
        "bc_util":                40.,
        "num_rev_accts":          6.,
        "tot_cur_bal":            la * 2,
        "sector_enc":             SEM.get(inp.get("sector", "Services"), 6),
        "loan_to_income":         la / max(ai, 1),
        "payment_to_income":      inst / max(ai / 12, 1),
        "dti_x_int_rate":         dti * ir,
        "grade_x_term":           ge * tm,
        "revol_util_x_bal":       ru * np.log1p(rb),
        "inc_x_dti":              ai / max(dti, 1),
        "inq_x_delinq":           float(inp.get("inq_last_6mths", 1)) *
                                  (1 + float(inp.get("delinq_2yrs", 0))),
    }
    return np.array([raw.get(f, 0) for f in features], dtype=float).reshape(1, -1)


def gauge_colour(prob, threshold):
    if prob < threshold * 0.70:    return C_GREEN
    elif prob < threshold:         return C_AMBER
    elif prob < threshold * 1.50:  return C_RED
    else:                          return "#9D174D"


def risk_label(prob, threshold):
    if prob < threshold * 0.70:    return "LOW RISK"
    elif prob < threshold:         return "WITHIN THRESHOLD"
    elif prob < threshold * 1.50:  return "ABOVE THRESHOLD"
    else:                          return "HIGH RISK"


def risk_gauge(prob, strategy_name):
    """Strategy-aware gauge — zones and threshold line shift with the selected profile."""
    profile   = STRATEGIES[strategy_name]
    threshold = profile["threshold"]
    colour    = gauge_colour(prob, threshold)
    label     = risk_label(prob, threshold)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={
            "text": f"<b>{label}</b>",
            "font": {"size": 11, "family": "DM Sans", "color": "#374151"},
        },
        number={
            "suffix": "%",
            "font":   {"size": 28, "family": "DM Mono", "color": "#111318"},
        },
        gauge={
            "axis": {
                "range":    [0, 25],
                "tickwidth": 1,
                "tickcolor": "#E5E7EB",
                "tickfont":  {"size": 9, "family": "DM Sans"},
            },
            "bar":       {"color": colour, "thickness": 0.22},
            "bgcolor":   "white",
            "steps":     profile["gauge_steps"],
            "threshold": {
                "line":      {"color": profile["colour"], "width": 3},
                "thickness": 0.75,
                "value":     threshold * 100,
            },
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(t=20, b=0, l=10, r=10),
        paper_bgcolor="white",
        font=dict(family="DM Sans"),
    )
    return fig


def strategy_selector_bar(key_suffix=""):
    """Renders the strategy selector. Returns the selected strategy name."""
    st.markdown(
        '<div style="font-size:0.66rem; font-weight:700; color:#9CA3AF; '
        'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.3rem;">'
        'Lending Strategy</div>',
        unsafe_allow_html=True,
    )
    choice = st.radio(
        "Select lending strategy",
        options=list(STRATEGIES.keys()),
        index=1,            # Balanced as default
        horizontal=True,
        label_visibility="collapsed",
        key=f"strategy_radio_{key_suffix}",
    )

    # Visual description strip below the radio buttons
    profile = STRATEGIES[choice]
    st.markdown(
        f'<div style="font-size:0.72rem; color:{profile["colour"]}; '
        f'background:{profile["colour"]}10; border-radius:6px; '
        f'padding:0.35rem 0.7rem; margin-bottom:0.8rem; '
        f'border-left:3px solid {profile["colour"]};">'
        f'<b>{choice}</b> — approval threshold <b>{profile["threshold"]*100:.1f}%</b>. '
        f'{profile["description"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    return choice


def all_strategy_decisions(prob):
    """Returns approve/flag verdict for all three strategies."""
    return {
        name: "Approve" if prob < p["threshold"] else "Flag"
        for name, p in STRATEGIES.items()
    }


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    model    = load_model()
    features = load_feature_list()
    metrics  = load_model_metrics()

    tab1, tab2 = st.tabs(["Manual Entry", "CSV Bulk Scoring"])

    # ── Tab 1: Manual Entry ───────────────────────────────────────────────────
    with tab1:
        st.markdown(
            '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:0.8rem">'
            'Select your institution\'s lending strategy, then enter loan details. '
            'The approval threshold and gauge zones adjust automatically to the '
            'selected profile. All three strategy verdicts are shown on the result.'
            '</div>',
            unsafe_allow_html=True,
        )

        strategy  = strategy_selector_bar(key_suffix="manual")
        profile   = STRATEGIES[strategy]
        threshold = profile["threshold"]

        section("LOAN DETAILS")
        c1, c2 = st.columns(2)
        with c1:
            loan_amnt   = st.number_input("Loan Amount ($)", 500, 40000, 10000, 500, key="la")
            int_rate    = st.slider("Interest Rate (%)", 5.0, 36.0, 13.5, 0.5, key="ir")
            grade       = st.selectbox("Credit Grade", list("ABCDEFG"), index=2, key="g")
            sub_grade   = st.selectbox("Sub-Grade",
                [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)], index=12, key="sg")
            term_months = st.selectbox("Loan Term", [36, 60], key="tm",
                format_func=lambda x: f"{x} months")
        with c2:
            annual_inc   = st.number_input("Annual Income ($)", 10000, 500000, 65000, 1000, key="ai")
            dti          = st.slider("Debt-to-Income (%)", 0.0, 50.0, 18.0, 0.5, key="dti")
            home_own     = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"], key="ho")
            verification = st.selectbox("Income Verification",
                ["Not Verified","Verified","Source Verified"], key="iv")
            emp_length   = st.slider("Employment (years)", 0, 10, 5, key="el")

        section("CREDIT HISTORY")
        c3, c4 = st.columns(2)
        with c3:
            delinq     = st.number_input("Delinquencies (2yr)", 0, 20, 0, key="dq")
            inquiries  = st.number_input("Credit Inquiries (6mo)", 0, 20, 1, key="inq")
            open_acc   = st.number_input("Open Accounts", 1, 40, 8, key="oa")
            revol_util = st.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0, key="ru")
        with c4:
            revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 15000, 1000, key="rb")
            total_acc = st.number_input("Total Accounts", 1, 80, 15, key="ta")
            mort_acc  = st.number_input("Mortgage Accounts", 0, 10, 0, key="ma")
            pub_rec   = st.number_input("Public Records", 0, 5, 0, key="pr")

        sector = st.selectbox("Loan Purpose / Sector",
            ["Personal Use","Retail & Trade","Housing & Construction",
             "Transport & Logistics","Health & Wellness","Education",
             "Services","Manufacturing"], key="sec")

        predict = st.button("Calculate Risk Score", type="primary", width="stretch")

        if predict:
            inp = {
                "loan_amnt": loan_amnt, "int_rate": int_rate, "grade": grade,
                "sub_grade": sub_grade, "term_months": term_months,
                "annual_inc": annual_inc, "dti": dti, "home_ownership": home_own,
                "verification_status": verification, "emp_length": emp_length,
                "delinq_2yrs": delinq, "inq_last_6mths": inquiries,
                "open_acc": open_acc, "revol_util": revol_util, "revol_bal": revol_bal,
                "total_acc": total_acc, "mort_acc": mort_acc, "pub_rec": pub_rec,
                "sector": sector,
            }
            X    = build_vector(inp, features)
            prob = float(model.predict_proba(X)[0, 1])

            decisions  = all_strategy_decisions(prob)
            approved   = prob < threshold
            dec        = "Approve" if approved else "Flag for Review"
            bg         = "#F0FDF4" if approved else "#FEF2F2"
            txt        = "#166534" if approved else "#991B1B"
            bdr        = "#BBF7D0" if approved else "#FECACA"
            margin_pct = (threshold - prob) * 100

            res1, res2 = st.columns([1, 1.8])

            with res1:
                card("Risk Assessment", f"under {strategy} strategy")
                st.plotly_chart(risk_gauge(prob, strategy), width="stretch")

                st.markdown(
                    f"""<div style='background:{bg}; border:1px solid {bdr};
                                border-radius:8px; padding:0.75rem; text-align:center;
                                font-size:1rem; font-weight:700; color:{txt};
                                margin:0.5rem 0'>{dec}</div>
                    <div style='font-size:0.72rem; color:#9CA3AF;
                                margin-top:0.5rem; line-height:1.9'>
                        Risk score: <b style='color:#111318'>{prob*100:.2f}%</b><br>
                        Strategy threshold:
                        <b style='color:{profile["colour"]}'>{threshold*100:.1f}%</b><br>
                        Margin:
                        <b style='color:#111318'>
                            {"+" if margin_pct > 0 else ""}{margin_pct:.2f}%
                            {"headroom" if margin_pct > 0 else "over limit"}
                        </b><br>
                        Model: <b style='color:#111318'>XGBoost AUC {metrics["auc_roc"]}</b>
                    </div>""",
                    unsafe_allow_html=True,
                )
                card_end()

                # All three strategy verdicts
                st.markdown(
                    '<div style="font-size:0.66rem; font-weight:700; color:#9CA3AF; '
                    'text-transform:uppercase; letter-spacing:0.07em; '
                    'margin:0.6rem 0 0.3rem">All Strategy Verdicts</div>',
                    unsafe_allow_html=True,
                )
                vcols = st.columns(3)
                for vcol, (sname, verdict) in zip(vcols, decisions.items()):
                    sc   = STRATEGIES[sname]["colour"]
                    vbg  = "#F0FDF4" if verdict == "Approve" else "#FEF2F2"
                    vtxt = "#166534" if verdict == "Approve" else "#991B1B"
                    with vcol:
                        st.markdown(
                            f"""<div style="background:{vbg}; border-radius:6px;
                                            padding:0.4rem 0.3rem; text-align:center;">
                                <div style="font-size:0.64rem; color:{sc};
                                            font-weight:700; margin-bottom:0.1rem;">{sname}</div>
                                <div style="font-size:0.72rem; font-weight:700;
                                            color:{vtxt};">{verdict}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            with res2:
                risk_factors = []
                if int_rate > 18:         risk_factors.append(f"High interest rate ({int_rate}%)")
                if dti > 25:              risk_factors.append(f"High DTI ({dti}%)")
                if grade in list("EFG"):  risk_factors.append(f"Poor credit grade ({grade})")
                if delinq > 0:            risk_factors.append(f"Delinquency history ({delinq} events)")
                if revol_util > 75:       risk_factors.append(f"High revolving utilisation ({revol_util:.0f}%)")

                if approved:
                    base = (
                        f"<b>Approved under {strategy} strategy.</b> "
                        f"Risk score {prob*100:.2f}% is {margin_pct:.2f}% below the "
                        f"{threshold*100:.1f}% threshold. "
                    )
                    insight(base + (
                        "Contributing factors to note: " + ", ".join(risk_factors) + "."
                        if risk_factors else
                        "Credit grade, income level, and debt ratios are within acceptable limits."
                    ))
                else:
                    base = (
                        f"<b>Flagged under {strategy} strategy.</b> "
                        f"Risk score {prob*100:.2f}% exceeds the {threshold*100:.1f}% "
                        f"threshold by {abs(margin_pct):.2f}%. "
                    )
                    insight(base + (
                        "Contributing factors: " + ", ".join(risk_factors) +
                        ". Review income verification and revolving utilisation before proceeding."
                        if risk_factors else
                        "No single dominant factor — the combination of attributes places "
                        "this loan above the strategy limit."
                    ))

                # Show which other strategies would pass this loan
                other_approvals = [n for n, v in decisions.items()
                                   if v == "Approve" and n != strategy]
                if other_approvals and not approved:
                    names = " and ".join(other_approvals)
                    s     = "strategy" if len(other_approvals) == 1 else "strategies"
                    ac    = "accepts" if len(other_approvals) == 1 else "accept"
                    insight(
                        f"This loan would be approved under the <b>{names}</b> {s}, "
                        f"which {ac} a higher portfolio risk ceiling."
                    )

    # ── Tab 2: CSV Bulk Scoring ───────────────────────────────────────────────
    with tab2:
        section("BULK LOAN SCORING")

        bulk_strategy  = strategy_selector_bar(key_suffix="bulk")
        bulk_profile   = STRATEGIES[bulk_strategy]
        bulk_threshold = bulk_profile["threshold"]

        st.markdown(
            '<div style="font-size:0.8rem; color:#6B7280; margin-bottom:0.8rem">'
            'Upload a CSV with loan data to score multiple applications at once. '
            'The primary decision column reflects your selected strategy. '
            'All three strategy verdicts are included as separate columns in the download. '
            'Missing columns are filled with population medians.'
            '</div>',
            unsafe_allow_html=True,
        )

        tmpl = pd.DataFrame([{
            "loan_amnt": 10000, "int_rate": 13.5, "grade": "C", "sub_grade": "C3",
            "term_months": 36, "annual_inc": 65000, "dti": 18.0,
            "home_ownership": "RENT", "verification_status": "Not Verified",
            "emp_length": 5, "delinq_2yrs": 0, "inq_last_6mths": 1,
            "open_acc": 8, "revol_util": 45, "revol_bal": 15000,
            "total_acc": 15, "mort_acc": 0, "pub_rec": 0, "sector": "Personal Use",
        }])
        st.download_button("Download Template CSV", tmpl.to_csv(index=False),
                           "loan_template.csv", "text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(
                f'<div style="font-size:0.8rem; color:#374151; margin:0.4rem 0">'
                f'<b>{len(df):,} loans loaded</b></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(3), width="stretch")

            if st.button("Score All Loans", type="primary"):
                probs = []
                prog  = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        p = float(model.predict_proba(
                            build_vector(row.to_dict(), features))[0, 1])
                    except Exception:
                        p = 0.20
                    probs.append(p)
                    if i % max(1, len(df) // 20) == 0:
                        prog.progress(min(i / len(df), 1.0))
                prog.progress(1.0)

                df["risk_score"] = [round(p, 4) for p in probs]

                df["risk_band"] = pd.cut(
                    df["risk_score"],
                    bins=[0, 0.05, 0.12, 0.20, 1.0],
                    labels=["Low", "Medium", "High", "Very High"],
                    include_lowest=True,
                )

                # Primary decision column — reflects selected strategy
                df["decision"] = df["risk_score"].apply(
                    lambda p: "Approve" if p < bulk_threshold else "Flag"
                )

                # All three strategy verdict columns
                for sname, sprofile in STRATEGIES.items():
                    df[f"{sname.lower()}_decision"] = df["risk_score"].apply(
                        lambda p, t=sprofile["threshold"]: "Approve" if p < t else "Flag"
                    )

                section("RESULTS")
                rk1, rk2, rk3, rk4 = st.columns(4)
                n_approve = (df["decision"] == "Approve").sum()
                n_flag    = (df["decision"] == "Flag").sum()
                with rk1: kpi("Scored Loans",   f"{len(df):,}",                          "",                                    "neu", C_BLUE)
                with rk2: kpi("Approve",         f"{n_approve:,}",                        f"{n_approve/len(df)*100:.0f}%",       "pos", C_GREEN)
                with rk3: kpi("Flag for Review", f"{n_flag:,}",                           f"{n_flag/len(df)*100:.0f}%",          "neg", C_RED)
                with rk4: kpi("Avg Risk Score",  f"{df['risk_score'].mean()*100:.1f}%",   "",                                    "neu", C_AMBER)

                # Strategy approval rate comparison
                st.markdown(
                    '<div style="font-size:0.66rem; font-weight:700; color:#9CA3AF; '
                    'text-transform:uppercase; letter-spacing:0.07em; '
                    'margin:0.6rem 0 0.3rem">Approval Rate by Strategy</div>',
                    unsafe_allow_html=True,
                )
                acols = st.columns(3)
                for acol, (sname, sprofile) in zip(acols, STRATEGIES.items()):
                    col   = f"{sname.lower()}_decision"
                    n_app = (df[col] == "Approve").sum()
                    pct   = n_app / len(df) * 100
                    sc    = sprofile["colour"]
                    with acol:
                        st.markdown(
                            f"""<div style="background:{sc}10; border:1px solid {sc}30;
                                            border-radius:8px; padding:0.5rem 0.7rem;
                                            text-align:center;">
                                <div style="font-size:0.68rem; color:{sc};
                                            font-weight:700;">{sname}</div>
                                <div style="font-size:1.1rem; font-weight:700;
                                            color:#111318;">{n_app:,}</div>
                                <div style="font-size:0.68rem; color:#9CA3AF;">
                                    {pct:.0f}% approve rate</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.dataframe(df, width="stretch", height=300)
                st.download_button(
                    "Download Scored Loans",
                    df.to_csv(index=False),
                    "scored_loans.csv",
                    "text/csv",
                )

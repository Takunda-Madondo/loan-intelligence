"""app/pages/loan_predictor.py - Loan Predictor"""
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

GM  = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
SGM = {f"{g}{n}":i+1 for i,(g,n) in enumerate([(g,n) for g in "ABCDEFG" for n in range(1,6)])}
HM  = {"RENT":0,"OWN":1,"MORTGAGE":2,"OTHER":3}
VM  = {"Not Verified":0,"Verified":1,"Source Verified":2}
SEM = {"Retail & Trade":0,"Personal Use":1,"Housing & Construction":2,
       "Transport & Logistics":3,"Health & Wellness":4,"Education":5,"Services":6,"Manufacturing":7}


def build_vector(inp, features):
    la=float(inp.get("loan_amnt",10000)); ai=float(inp.get("annual_inc",65000))
    tm=float(inp.get("term_months",36)); ir=float(inp.get("int_rate",13.5))
    dti=float(inp.get("dti",18)); inst=la/max(tm,1)
    ge=GM.get(inp.get("grade","C"),4); se=SGM.get(inp.get("sub_grade","C3"),ge*3)
    ru=float(inp.get("revol_util",45)); rb=float(inp.get("revol_bal",la*0.5))
    raw={"loan_amnt":la,"funded_ratio":1.0,"term_months":tm,"int_rate":ir,"installment":inst,
         "grade_enc":ge,"sub_grade_enc":se,"emp_length":float(inp.get("emp_length",5)),
         "home_enc":HM.get(inp.get("home_ownership","RENT"),0),"annual_inc":ai,
         "verification_enc":VM.get(inp.get("verification_status","Not Verified"),0),
         "dti":dti,"delinq_2yrs":float(inp.get("delinq_2yrs",0)),
         "inq_last_6mths":float(inp.get("inq_last_6mths",1)),
         "mths_since_last_delinq":36.,"mths_since_last_record":60.,
         "open_acc":float(inp.get("open_acc",8)),"pub_rec":float(inp.get("pub_rec",0)),
         "revol_bal":rb,"revol_util":ru,"total_acc":float(inp.get("total_acc",15)),
         "mort_acc":float(inp.get("mort_acc",0)),"num_bc_sats":4.,"pct_tl_nvr_dlq":85.,
         "num_tl_90g_dpd_24m":0.,"avg_cur_bal":la*0.3,"bc_util":40.,"num_rev_accts":6.,
         "tot_cur_bal":la*2,"sector_enc":SEM.get(inp.get("sector","Services"),6),
         "loan_to_income":la/max(ai,1),"payment_to_income":inst/max(ai/12,1),
         "dti_x_int_rate":dti*ir,"grade_x_term":ge*tm,
         "revol_util_x_bal":ru*np.log1p(rb),"inc_x_dti":ai/max(dti,1),
         "inq_x_delinq":float(inp.get("inq_last_6mths",1))*(1+float(inp.get("delinq_2yrs",0)))}
    return np.array([raw.get(f,0) for f in features],dtype=float).reshape(1,-1)


def risk_gauge(prob):
    c = C_GREEN if prob<0.05 else C_AMBER if prob<0.12 else C_RED if prob<0.20 else "#9D174D"
    b = "LOW RISK" if prob<0.05 else "MEDIUM RISK" if prob<0.12 else "HIGH RISK" if prob<0.20 else "VERY HIGH RISK"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",value=prob*100,
        title={"text":f"<b>{b}</b>","font":{"size":12,"family":"DM Sans","color":"#374151"}},
        number={"suffix":"%","font":{"size":30,"family":"DM Mono","color":"#111318"}},
        gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#E5E7EB",
                       "tickfont":{"size":9,"family":"DM Sans"}},
               "bar":{"color":c,"thickness":0.22},"bgcolor":"white",
               "steps":[{"range":[0,5],"color":"#F0FDF4"},
                        {"range":[5,12],"color":"#FFFBEB"},
                        {"range":[12,20],"color":"#FEF2F2"},
                        {"range":[20,100],"color":"#FDF2F8"}],
               "threshold":{"line":{"color":"#1F2937","width":2},
                            "thickness":0.75,"value":prob*100}},
    ))
    fig.update_layout(height=210,margin=dict(t=20,b=0,l=10,r=10),
                      paper_bgcolor="white",font=dict(family="DM Sans"))
    return fig


def render():
    model=load_model(); features=load_feature_list(); metrics=load_model_metrics()
    threshold=metrics["optimal_threshold"]

    tab1, tab2 = st.tabs(["Manual Entry", "CSV Bulk Scoring"])

    # Tab 1: Manual
    with tab1:
        st.markdown("""
        <div style='font-size:0.8rem;color:#6B7280;margin-bottom:1rem'>
            Enter the loan details below to calculate the probability of default.
            All fields marked with a credit history section are optional - defaults will be applied.
        </div>""", unsafe_allow_html=True)

        form_col, result_col = st.columns([1.15, 1])

        section("LOAN DETAILS")
        c1,c2 = st.columns(2)
        with c1:
            loan_amnt   = st.number_input("Loan Amount ($)",500,40000,10000,500,key="la")
            int_rate    = st.slider("Interest Rate (%)",5.0,36.0,13.5,0.5,key="ir")
            grade       = st.selectbox("Credit Grade",list("ABCDEFG"),index=2,key="g")
            sub_grade   = st.selectbox("Sub-Grade",
                [f"{g}{n}" for g in "ABCDEFG" for n in range(1,6)],index=12,key="sg")
            term_months = st.selectbox("Loan Term",[36,60],key="tm",
                format_func=lambda x: f"{x} months")
        with c2:
            annual_inc  = st.number_input("Annual Income ($)",10000,500000,65000,1000,key="ai")
            dti         = st.slider("Debt-to-Income (%)",0.0,50.0,18.0,0.5,key="dti")
            home_own    = st.selectbox("Home Ownership",["RENT","MORTGAGE","OWN","OTHER"],key="ho")
            verification= st.selectbox("Income Verification",
                ["Not Verified","Verified","Source Verified"],key="iv")
            emp_length  = st.slider("Employment (years)",0,10,5,key="el")

        section("CREDIT HISTORY")
        c3,c4 = st.columns(2)
        with c3:
            delinq   = st.number_input("Delinquencies (2yr)",0,20,0,key="dq")
            inquiries= st.number_input("Credit Inquiries (6mo)",0,20,1,key="inq")
            open_acc = st.number_input("Open Accounts",1,40,8,key="oa")
            revol_util=st.slider("Revolving Utilisation (%)",0.0,100.0,45.0,key="ru")
        with c4:
            revol_bal= st.number_input("Revolving Balance ($)",0,200000,15000,1000,key="rb")
            total_acc= st.number_input("Total Accounts",1,80,15,key="ta")
            mort_acc = st.number_input("Mortgage Accounts",0,10,0,key="ma")
            pub_rec  = st.number_input("Public Records",0,5,0,key="pr")

        sector = st.selectbox("Loan Purpose / Sector",
            ["Personal Use","Retail & Trade","Housing & Construction",
             "Transport & Logistics","Health & Wellness","Education",
             "Services","Manufacturing"],key="sec")

        predict = st.button("Calculate Risk Score",
                             type="primary", use_container_width=True)

        if predict:
            inp = {"loan_amnt":loan_amnt,"int_rate":int_rate,"grade":grade,
                   "sub_grade":sub_grade,"term_months":term_months,"annual_inc":annual_inc,
                   "dti":dti,"home_ownership":home_own,"verification_status":verification,
                   "emp_length":emp_length,"delinq_2yrs":delinq,"inq_last_6mths":inquiries,
                   "open_acc":open_acc,"revol_util":revol_util,"revol_bal":revol_bal,
                   "total_acc":total_acc,"mort_acc":mort_acc,"pub_rec":pub_rec,"sector":sector}
            X    = build_vector(inp, features)
            prob = float(model.predict_proba(X)[0,1])

            res1, res2 = st.columns([1, 1.8])
            with res1:
                card("Risk Assessment", "calibrated to African MFI norms")
                st.plotly_chart(risk_gauge(prob), use_container_width=True)
                approved = prob < 0.12  # MFI-calibrated threshold
                bg  = "#F0FDF4" if approved else "#FEF2F2"
                txt = "#166534" if approved else "#991B1B"
                bdr = "#BBF7D0" if approved else "#FECACA"
                dec = "Approve" if approved else "Flag for Review"
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {bdr};border-radius:8px;
                            padding:0.75rem;text-align:center;font-size:1rem;
                            font-weight:700;color:{txt};margin:0.5rem 0'>{dec}</div>
                <div style='font-size:0.72rem;color:#9CA3AF;margin-top:0.5rem;line-height:1.8'>
                    Default probability: <b style='color:#111318'>{prob*100:.2f}%</b><br>
                    Threshold: <b style='color:#111318'>{threshold}</b><br>
                    Model: <b style='color:#111318'>XGBoost AUC {metrics['auc_roc']}</b>
                </div>""", unsafe_allow_html=True)
                card_end()

            with res2:
                risk_factors = []
                if int_rate > 18:       risk_factors.append(f"High interest rate ({int_rate}%)")
                if dti > 25:            risk_factors.append(f"High debt-to-income ratio ({dti}%)")
                if grade in list("EFG"): risk_factors.append(f"Poor credit grade ({grade})")
                if delinq > 0:          risk_factors.append(f"Delinquency history ({delinq} events)")
                if revol_util > 75:     risk_factors.append(f"High revolving utilisation ({revol_util:.0f}%)")
                if prob < 0.15:
                    insight("<b>Low risk profile.</b> This loan shows strong repayment indicators. "
                            "Credit grade, income level, and debt ratios are all within acceptable thresholds.")
                elif risk_factors:
                    insight("<b>Risk factors identified:</b><br>" +
                            "<br>".join(f"· {r}" for r in risk_factors))
                else:
                    insight(f"<b>Moderate risk.</b> Default probability of {prob*100:.1f}% "
                            f"is above the {threshold*100:.0f}% approval threshold. "
                            "Review income verification and revolving utilisation before proceeding.")

    # Tab 2: CSV
    with tab2:
        section("BULK LOAN SCORING")
        st.markdown('<div style="font-size:0.8rem;color:#6B7280;margin-bottom:0.8rem">Upload a CSV file with loan data to score multiple loans at once. Download the template below to see the required column format. Missing columns are filled with population medians.</div>', unsafe_allow_html=True)
        tmpl = pd.DataFrame([{"loan_amnt":10000,"int_rate":13.5,"grade":"C","sub_grade":"C3",
            "term_months":36,"annual_inc":65000,"dti":18.0,"home_ownership":"RENT",
            "verification_status":"Not Verified","emp_length":5,"delinq_2yrs":0,
            "inq_last_6mths":1,"open_acc":8,"revol_util":45,"revol_bal":15000,
            "total_acc":15,"mort_acc":0,"pub_rec":0,"sector":"Personal Use"}])
        st.download_button("Download Template CSV", tmpl.to_csv(index=False),
                               "loan_template.csv","text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f'<div style="font-size:0.8rem;color:#374151;margin:0.4rem 0"><b>{len(df):,} loans loaded</b></div>', unsafe_allow_html=True)
            st.dataframe(df.head(3), use_container_width=True)

            if st.button("Score All Loans", type="primary"):
                probs=[]; prog=st.progress(0)
                for i,row in df.iterrows():
                    try:    p=float(model.predict_proba(build_vector(row.to_dict(),features))[0,1])
                    except: p=0.20
                    probs.append(p)
                    if i%max(1,len(df)//20)==0: prog.progress(min(i/len(df),1.0))
                prog.progress(1.0)

                df["default_probability"]=[round(p,4) for p in probs]
                df["risk_band"]=pd.cut(df["default_probability"],
                    bins=[0,0.15,0.30,0.50,1.0],labels=["Low","Medium","High","Very High"])
                df["decision"]=df["default_probability"].apply(
                    lambda p:"Approve" if p<threshold else "Flag")

                section("RESULTS")
                rk1,rk2,rk3,rk4 = st.columns(4)
                with rk1: kpi("Scored Loans",   f"{len(df):,}",                                        "",         "neu", C_BLUE)
                with rk2: kpi("Approve",         f"{(df['decision']=='Approve').sum():,}",              f"{(df['decision']=='Approve').mean()*100:.0f}%", "pos", C_GREEN)
                with rk3: kpi("Flag for Review", f"{(df['decision']=='Flag').sum():,}",                 f"{(df['decision']=='Flag').mean()*100:.0f}%",    "neg", C_RED)
                with rk4: kpi("Avg Default Risk",f"{df['default_probability'].mean()*100:.1f}%",        "",         "neu", C_AMBER)

                st.dataframe(df, use_container_width=True, height=300)
                st.download_button("Download Scored Loans", df.to_csv(index=False),
                                   "scored_loans.csv","text/csv")

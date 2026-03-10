"""
app/data_loader.py
"""
import pickle
import json
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "pipelines"))

from config import DB_GOLD, DB_SILVER, MODELS_DIR
from db import get_conn


@st.cache_data
def load_kiva_predictions():
    with get_conn(DB_GOLD) as conn:
        return pd.read_sql_query("SELECT * FROM kiva_predictions", conn)


@st.cache_data
def load_sector_performance():
    with get_conn(DB_GOLD) as conn:
        return pd.read_sql_query("SELECT * FROM sector_performance", conn)


@st.cache_data
def load_kiva_features():
    with get_conn(DB_GOLD) as conn:
        return pd.read_sql_query(
            """SELECT id, loan_amount, funded_ratio, term_in_months,
               sector_standardised, country_code, country,
               is_female_borrower, lender_count, macro_risk_score,
               sector_default_rate, gdp_growth, inflation_rate,
               unemployment_rate, poverty_rate, disbursed_year,
               activity, region
               FROM kiva_features""",
            conn
        )


@st.cache_data
def load_lc_features_sample(n=50000):
    with get_conn(DB_GOLD) as conn:
        return pd.read_sql_query(f"SELECT * FROM lc_features LIMIT {n}", conn)


@st.cache_resource
def load_model():
    path = MODELS_DIR / "best_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_list():
    path = MODELS_DIR / "feature_list.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_model_metrics():
    path = MODELS_DIR / "model_metrics.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_shap_importance():
    path = MODELS_DIR / "shap_importance.json"
    with open(path) as f:
        return pd.read_json(f)


AFRICAN_COUNTRIES = [
    "KE","UG","TZ","RW","ET","GH","NG","SN","ML",
    "MZ","ZM","ZW","MW","MG","CM","BF","TG","BJ","GN"
]

ISO3_MAP = {
    "KE":"KEN","UG":"UGA","TZ":"TZA","RW":"RWA","ET":"ETH",
    "GH":"GHA","NG":"NGA","SN":"SEN","ML":"MLI","MZ":"MOZ",
    "ZM":"ZMB","ZW":"ZWE","MW":"MWI","MG":"MDG","CM":"CMR",
    "BF":"BFA","TG":"TGO","BJ":"BEN","GN":"GIN","SL":"SLE",
    "LR":"LBR","SD":"SDN","TN":"TUN","MA":"MAR","EG":"EGY",
    "PH":"PHL","IN":"IND","KH":"KHM","TJ":"TJK","PK":"PAK",
    "PE":"PER","BO":"BOL","CO":"COL","EC":"ECU","NP":"NPL",
    "LB":"LBN","PS":"PSE","BZ":"BLZ",
}

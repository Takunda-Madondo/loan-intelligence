"""
pipelines/05_feature_engineering.py
Stage 3 - Feature engineering into gold layer.

Run:
  python pipelines/05_feature_engineering.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_SILVER, DB_GOLD, MODELS_DIR, RANDOM_STATE
from db import get_conn, bulk_insert, table_info

# African MFI sector default rates (PAR30 benchmarks)
# Sources: MIX Market, CGAP, African MFI industry reports
# Agriculture benefits from group lending + seasonal repayment structures
LC_SECTOR_DEFAULT_RATES = {
    "Agriculture & Farming":  0.055,   # group lending, seasonal cycles - safest
    "Food & Beverage":        0.072,
    "Retail & Trade":         0.078,
    "Services":               0.080,
    "Housing & Construction": 0.090,
    "Personal Use":           0.095,   # higher — no productive asset backing
    "Clothing & Textiles":    0.085,
    "Education":              0.065,   # strong repayment motivation
    "Transport & Logistics":  0.070,
    "Arts & Crafts":          0.088,
    "Health & Wellness":      0.075,
    "Manufacturing":          0.110,   # highest - capital intensive, volatile
    "Entertainment":          0.100,
    "Unknown":                0.080,
}

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


# Helpers

def label_encode(series: pd.Series, name: str) -> tuple:
    categories = sorted(series.dropna().unique().tolist())
    mapping    = {cat: idx for idx, cat in enumerate(categories)}
    encoded    = series.map(mapping).fillna(-1).astype(int)
    return encoded, mapping


def compute_macro_risk_score(df: pd.DataFrame) -> pd.Series:
    score = (
        df.get("inflation_rate",    pd.Series(0, index=df.index)).fillna(0) * 0.4 +
        df.get("unemployment_rate", pd.Series(0, index=df.index)).fillna(0) * 0.3 +
        df.get("poverty_rate",      pd.Series(0, index=df.index)).fillna(0) * 0.3
    )
    min_s, max_s = score.min(), score.max()
    if max_s > min_s:
        score = (score - min_s) / (max_s - min_s)
    return score.round(4)


def parse_pct(val) -> float:
    """'13.56%' -> 13.56  or  13.56 -> 13.56"""
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return np.nan


def parse_term(val) -> float:
    """' 36 months' -> 36"""
    try:
        return float(str(val).strip().split()[0])
    except Exception:
        return np.nan


# 1. Lending Club Features

def build_lc_features() -> tuple:
    print("\nSTEP 1 - Building Lending Club Feature Table")

    with get_conn(DB_SILVER) as conn:
        df = pd.read_sql_query("SELECT * FROM lc_clean", conn)
    print(f"  Loaded {len(df):,} rows from lc_clean")

    # Diagnose nulls before doing anything
    print(f"\n  Null diagnosis on raw lc_clean columns:")
    key_cols = ["loan_amnt", "funded_amnt", "term_months", "int_rate",
                "installment", "grade", "emp_length", "annual_inc",
                "dti", "revol_util", "defaulted", "sector"]
    for col in key_cols:
        if col in df.columns:
            n   = df[col].isna().sum()
            pct = n / len(df) * 100
            print(f"    {col:<25} nulls: {n:>8,}  ({pct:.1f}%)")
        else:
            print(f"    {col:<25} *** NOT IN DATAFRAME ***")

    # Re-parse columns that may have survived as strings
    # int_rate and revol_util can come through as "13.5%" strings
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].apply(parse_pct)
        print(f"\n  Re-parsed int_rate from string")

    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].apply(parse_pct)
        print(f"  Re-parsed revol_util from string")

    # Re-parse term if it still has " months" text
    if "term_months" in df.columns and df["term_months"].dtype == object:
        df["term_months"] = df["term_months"].apply(parse_term)
        print(f"  Re-parsed term_months from string")
    elif "term_months" not in df.columns and "term" in df.columns:
        df["term_months"] = df["term"].apply(parse_term)
        print(f"  Created term_months from term column")

    # Encodings
    encoders = {}
    df["grade_enc"],        encoders["grade"]               = label_encode(df["grade"], "grade")
    df["sector_enc"],       encoders["lc_sector"]           = label_encode(df["sector"], "lc_sector")
    df["home_enc"],         encoders["home_ownership"]      = label_encode(df["home_ownership"], "home_ownership")
    df["verification_enc"], encoders["verification_status"] = label_encode(df["verification_status"], "verification_status")

    # Funded ratio (may already exist from silver)
    if "funded_ratio" not in df.columns:
        df["funded_ratio"] = (df["funded_amnt"] / df["loan_amnt"].clip(lower=1)).clip(0, 1)

    # Engineered features
    df["loan_to_income"]    = (df["loan_amnt"] / df["annual_inc"].clip(lower=1)).round(4)
    df["payment_to_income"] = (df["installment"] / (df["annual_inc"].clip(lower=1) / 12)).round(4)
    df["recovery_rate"]     = (df["recoveries"] / df["loan_amnt"].clip(lower=1)).round(4)
    df["repayment_rate"]    = (df["total_rec_prncp"] / df["loan_amnt"].clip(lower=1)).round(4)

    # Fill nulls with medians BEFORE dropping
    fill_with_median = [
        "emp_length", "revol_util", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "total_acc", "dti",
        "loan_to_income", "payment_to_income", "recovery_rate", "repayment_rate"
    ]
    for col in fill_with_median:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Select final feature columns
    feature_cols = [
        "id", "defaulted",
        "loan_amnt", "funded_ratio", "term_months", "int_rate",
        "installment", "grade_enc",
        "emp_length", "home_enc", "annual_inc", "verification_enc",
        "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_util", "total_acc", "sector_enc",
        "loan_to_income", "payment_to_income",
        "recovery_rate", "repayment_rate",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat      = df[feature_cols].copy()

    # Drop only rows where TARGET or key loan cols are null
    # Don't drop on every column - only the ones we can't impute
    must_have = ["defaulted", "loan_amnt", "int_rate", "annual_inc"]
    must_have = [c for c in must_have if c in df_feat.columns]
    before    = len(df_feat)
    df_feat   = df_feat.dropna(subset=must_have)
    dropped   = before - len(df_feat)

    # Fill any remaining nulls with column medians
    for col in df_feat.select_dtypes(include=[np.number]).columns:
        if df_feat[col].isna().any():
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    print(f"\n  Dropped {dropped:,} rows missing target/key columns")
    print(f"  Final: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")
    print(f"  Default rate: {df_feat['defaulted'].mean()*100:.1f}%")

    with get_conn(DB_GOLD) as conn:
        conn.execute("DROP TABLE IF EXISTS lc_features")
        bulk_insert(conn, "lc_features", df_feat, if_exists="replace")
    print(f"  Saved -> gold.db : lc_features")

    return df_feat, encoders


# 2. Kiva Features

def build_kiva_features(encoders: dict) -> pd.DataFrame:
    print("\nSTEP 2 - Building Kiva Feature Table")

    with get_conn(DB_SILVER) as conn:
        df = pd.read_sql_query("SELECT * FROM kiva_enriched", conn)
    print(f"  Loaded {len(df):,} rows from kiva_enriched")

    df["sector_enc"],  encoders["kiva_sector"] = label_encode(df["sector_standardised"], "kiva_sector")
    df["country_enc"], encoders["country"]      = label_encode(df["country_code"], "country")

    df["sector_default_rate"] = df["sector_standardised"].map(LC_SECTOR_DEFAULT_RATES).fillna(0.20)
    df["macro_risk_score"]    = compute_macro_risk_score(df)
    df["loan_per_month"]      = (df["loan_amount"] / df["term_in_months"].clip(lower=1)).round(2)
    df["lender_trust_score"]  = (df["lender_count"] / df["loan_amount"].clip(lower=1)).round(4)

    for col in ["gdp_growth", "inflation_rate", "unemployment_rate",
                "poverty_rate", "domestic_credit_pct"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    feature_cols = [
        "id", "loan_amount", "funded_ratio", "term_in_months",
        "lender_count", "is_female_borrower", "repayment_interval_enc",
        "sector_standardised", "sector_enc",
        "country", "country_code", "country_enc",
        "gdp_growth", "inflation_rate", "unemployment_rate",
        "poverty_rate", "domestic_credit_pct",
        "loan_per_month", "lender_trust_score",
        "macro_risk_score", "sector_default_rate",
        "activity", "region", "disbursed_year",
        "funded_time", "disbursed_time",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat      = df[feature_cols].copy()

    print(f"  Final: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")

    with get_conn(DB_GOLD) as conn:
        conn.execute("DROP TABLE IF EXISTS kiva_features")
        bulk_insert(conn, "kiva_features", df_feat, if_exists="replace")
    print(f"  Saved -> gold.db : kiva_features")

    return df_feat


# 3. Sector Performance

def build_sector_performance() -> pd.DataFrame:
    print("\nSTEP 3 - Building Sector Performance Table")

    with get_conn(DB_GOLD) as conn:
        kiva = pd.read_sql_query("SELECT * FROM kiva_features", conn)

    agg = kiva.groupby(["sector_standardised", "country_code"]).agg(
        total_loans          = ("loan_amount",         "count"),
        total_amount_usd     = ("loan_amount",         "sum"),
        avg_loan_amount      = ("loan_amount",         "mean"),
        avg_term_months      = ("term_in_months",      "mean"),
        female_borrower_pct  = ("is_female_borrower",  "mean"),
        avg_lender_count     = ("lender_count",        "mean"),
        avg_funded_ratio     = ("funded_ratio",        "mean"),
        avg_gdp_growth       = ("gdp_growth",          "mean"),
        avg_inflation        = ("inflation_rate",      "mean"),
        avg_macro_risk       = ("macro_risk_score",    "mean"),
        sector_default_rate  = ("sector_default_rate", "mean"),
    ).reset_index()

    agg["roi_score"] = (
        (1 - agg["sector_default_rate"]) *
        agg["avg_funded_ratio"] *
        (1 + agg["avg_gdp_growth"].clip(0, 10) / 100)
    ).round(4)

    agg["combined_risk"] = (
        agg["sector_default_rate"] * 0.6 +
        agg["avg_macro_risk"]      * 0.4
    )
    agg["risk_tier"] = pd.cut(
        agg["combined_risk"],
        bins=[0, 0.07, 0.10, 1.0],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    agg["avg_loan_amount"]     = agg["avg_loan_amount"].round(0)
    agg["total_amount_usd"]    = agg["total_amount_usd"].round(0)
    agg["female_borrower_pct"] = (agg["female_borrower_pct"] * 100).round(1)
    agg["avg_funded_ratio"]    = (agg["avg_funded_ratio"] * 100).round(1)
    agg["sector_default_rate"] = (agg["sector_default_rate"] * 100).round(1)
    agg["avg_gdp_growth"]      = agg["avg_gdp_growth"].round(2)
    agg["avg_inflation"]       = agg["avg_inflation"].round(2)
    agg["avg_lender_count"]    = agg["avg_lender_count"].round(1)
    agg = agg.rename(columns={
        "sector_standardised": "sector",
        "sector_default_rate": "default_rate",
    })

    print(f"  {len(agg):,} sector x country combinations")

    with get_conn(DB_GOLD) as conn:
        conn.execute("DROP TABLE IF EXISTS sector_performance")
        bulk_insert(conn, "sector_performance", agg, if_exists="replace")
    print(f"  Saved -> gold.db : sector_performance")

    print(f"\n  Top 10 Investment Opportunities (highest ROI, min 100 loans):")
    top = agg[agg["total_loans"] >= 100].nlargest(10, "roi_score")[
        ["sector", "country_code", "total_loans",
         "avg_loan_amount", "default_rate", "roi_score", "risk_tier"]
    ]
    print(top.to_string(index=False))

    print(f"\n  Top 10 Highest Risk Sectors:")
    risky = agg[agg["total_loans"] >= 100].nlargest(10, "default_rate")[
        ["sector", "country_code", "total_loans",
         "avg_loan_amount", "default_rate", "roi_score", "risk_tier"]
    ]
    print(risky.to_string(index=False))

    return agg


# 4. Save Encoders

def save_encoders(encoders: dict):
    path = MODELS_DIR / "label_encoders.json"
    with open(path, "w") as f:
        json.dump(encoders, f, indent=2)
    print(f"\n  Label encoders -> {path}")


# 5. Audit

def audit_gold():
    print("\nGOLD LAYER AUDIT")
    table_info(DB_GOLD)

    with get_conn(DB_GOLD) as conn:
        print(f"\nlc_features - Summary:")
        df = pd.read_sql_query(
            "SELECT ROUND(AVG(loan_amnt),0) as avg_loan, "
            "ROUND(AVG(int_rate),1) as avg_int_rate, "
            "ROUND(AVG(dti),1) as avg_dti, "
            "ROUND(AVG(defaulted)*100,1) as default_rate_pct, "
            "COUNT(*) as total FROM lc_features", conn
        )
        print(df.to_string(index=False))

        print(f"\nkiva_features - Summary:")
        df = pd.read_sql_query(
            "SELECT ROUND(AVG(loan_amount),0) as avg_loan, "
            "ROUND(AVG(macro_risk_score),3) as avg_macro_risk, "
            "ROUND(AVG(sector_default_rate)*100,1) as avg_sector_default_pct, "
            "COUNT(*) as total FROM kiva_features", conn
        )
        print(df.to_string(index=False))

        print(f"\nsector_performance - African Highlights:")
        df = pd.read_sql_query(
            "SELECT sector, country_code, total_loans, "
            "avg_loan_amount, default_rate, roi_score, risk_tier "
            "FROM sector_performance "
            "WHERE country_code IN "
            "('KE','UG','TZ','RW','ET','GH','NG','MZ','ZM','MW') "
            "ORDER BY roi_score DESC LIMIT 15", conn
        )
        print(df.to_string(index=False))


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 3: Feature Engineering -> Gold Layer\n")

    lc_features, encoders = build_lc_features()
    kiva_features          = build_kiva_features(encoders)
    sector_perf            = build_sector_performance()
    save_encoders(encoders)
    audit_gold()

    print(f"\n  Stage 3 complete!")
    print(f"    gold.db -> lc_features       ({len(lc_features):,} rows) - ML training data")
    print(f"    gold.db -> kiva_features      ({len(kiva_features):,} rows) - scoring data")
    print(f"    gold.db -> sector_performance ({len(sector_perf):,} combinations)")
    print(f"\n  Next: python pipelines/06_train_model.py\n")

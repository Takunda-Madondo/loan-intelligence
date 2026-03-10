"""
pipelines/04_etl_silver.py
Stage 2 - Clean, transform and join all datasets into silver.db.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_BRONZE, DB_SILVER, SILVER_SCHEMAS, SECTOR_MAP
from db import init_db, get_conn, bulk_insert, row_count, table_info

# Grade -> numeric (for sub_grade fallback)
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# Lending Club purpose -> unified sector map
# Maps LC's loan purpose values to the same sector labels as Kiva
LC_PURPOSE_TO_SECTOR = {
    "small_business":    "Retail & Trade",
    "debt_consolidation":"Personal Use",
    "credit_card":       "Personal Use",
    "home_improvement":  "Housing & Construction",
    "house":             "Housing & Construction",
    "major_purchase":    "Retail & Trade",
    "car":               "Transport & Logistics",
    "medical":           "Health & Wellness",
    "educational":       "Education",
    "vacation":          "Personal Use",
    "moving":            "Personal Use",
    "wedding":           "Personal Use",
    "renewable_energy":  "Manufacturing",
    "other":             "Services",
}

# Loan statuses that mean the loan is DONE and we know the outcome
LC_COMPLETED = ["Fully Paid", "Charged Off", "Default",
                "Does not meet the credit policy. Status:Fully Paid",
                "Does not meet the credit policy. Status:Charged Off"]

LC_DEFAULT = ["Charged Off", "Default",
              "Does not meet the credit policy. Status:Charged Off"]


# Helpers

def parse_int_rate(val) -> float:
    """'13.56%' → 13.56"""
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return np.nan

def parse_term(val) -> int:
    """' 36 months' → 36"""
    try:
        return int(str(val).strip().split()[0])
    except Exception:
        return np.nan

def parse_emp_length(val) -> float:
    """'10+ years' → 10,  '< 1 year' → 0,  'n/a' → NaN"""
    val = str(val).strip().lower()
    if val in ("n/a", "nan", "none", ""):
        return np.nan
    if "< 1" in val:
        return 0.0
    if "10+" in val:
        return 10.0
    try:
        return float(''.join(filter(lambda x: x.isdigit(), val)))
    except Exception:
        return np.nan

def parse_revol_util(val) -> float:
    """'62.4%' → 62.4"""
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return np.nan

def safe_date(val) -> str:
    """Parse any date string to YYYY-MM-DD, return None if unparseable."""
    try:
        return pd.to_datetime(val, utc=True).strftime("%Y-%m-%d")
    except Exception:
        return None

def extract_year(val) -> int:
    """Extract year from a date string."""
    try:
        return pd.to_datetime(val, utc=True).year
    except Exception:
        return None


# 1. Clean Lending Club

def clean_lending_club():
    print("\nSTEP 1 - Cleaning Lending Club")

    with get_conn(DB_BRONZE) as conn:
        print("  Loading from bronze.db...")
        df = pd.read_sql_query(
            f"SELECT * FROM lending_club_raw "
            f"WHERE loan_status IN ({','.join(['?']*len(LC_COMPLETED))})",
            conn,
            params=LC_COMPLETED
        )
    print(f"  Loaded {len(df):,} completed loans")

    # Target variable
    df["defaulted"] = df["loan_status"].apply(
        lambda x: 1 if x in LC_DEFAULT else 0
    )
    print(f"  Target: {df['defaulted'].sum():,} defaults  |  "
          f"{(df['defaulted']==0).sum():,} repaid  |  "
          f"rate: {df['defaulted'].mean()*100:.1f}%")

    # Parse numeric columns
    df["int_rate"]   = df["int_rate"].apply(parse_int_rate)
    df["term_months"]= df["term"].apply(parse_term)
    df["emp_length"] = df["emp_length"].apply(parse_emp_length)
    df["revol_util"] = df["revol_util"].apply(parse_revol_util)

    # Issue year
    df["issue_year"] = df["issue_d"].apply(extract_year)

    # Sector mapping
    df["sector"] = df["purpose"].map(LC_PURPOSE_TO_SECTOR).fillna("Services")

    # Funded ratio
    df["funded_ratio"] = (df["funded_amnt"] / df["loan_amnt"]).clip(0, 1)

    # Drop nulls on critical columns
    critical = ["loan_amnt", "term_months", "int_rate", "dti", "annual_inc"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"  Dropped {before - len(df):,} rows with nulls in critical columns")

    # Fill remaining nulls with medians
    fill_cols = ["emp_length", "revol_util", "delinq_2yrs",
                 "inq_last_6mths", "open_acc", "pub_rec", "total_acc"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Sub-grade encoding (A1=1 ... G5=35)
    # Much finer signal than grade alone - biggest single AUC gain
    sub_grade_order = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
    sub_grade_map   = {sg: i+1 for i, sg in enumerate(sub_grade_order)}
    if "sub_grade" in df.columns:
        df["sub_grade_enc"] = df["sub_grade"].map(sub_grade_map).fillna(
            df["grade"].map(GRADE_MAP).fillna(4) * 3  # fallback
        ).astype(int)
    else:
        df["sub_grade_enc"] = df["grade"].map(GRADE_MAP).fillna(4).astype(int)

    # Fill new nullable columns
    for col in ["mths_since_last_delinq", "mths_since_last_record",
                "mort_acc", "num_bc_sats", "pct_tl_nvr_dlq",
                "num_tl_90g_dpd_24m", "avg_cur_bal", "bc_util",
                "num_rev_accts", "tot_cur_bal", "revol_bal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Select final columns
    final_cols = [
        "id", "loan_amnt", "funded_amnt", "funded_ratio",
        "term_months", "int_rate", "installment", "grade",
        "sub_grade_enc", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "issue_year", "loan_status",
        "defaulted", "purpose", "sector", "dti", "delinq_2yrs",
        "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc", "num_bc_sats", "pct_tl_nvr_dlq", "num_tl_90g_dpd_24m",
        "avg_cur_bal", "bc_util", "num_rev_accts", "tot_cur_bal",
        "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
        "recoveries",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    print(f"  📐  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Save to silver ─────────────────────────────────────────
    with get_conn(DB_SILVER) as conn:
        conn.execute("DROP TABLE IF EXISTS lc_clean")
        bulk_insert(conn, "lc_clean", df, if_exists="replace")
    print(f"  Saved -> silver.db : lc_clean")
    return df


# 2. Clean Kiva

def clean_kiva():
    print("\nSTEP 2 - Cleaning Kiva")

    with get_conn(DB_BRONZE) as conn:
        print("  Loading from bronze.db...")
        df = pd.read_sql_query("SELECT * FROM kiva_loans_raw", conn)
    print(f"  Loaded {len(df):,} rows")

    # Parse dates
    for col in ["posted_time", "funded_time", "disbursed_time"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_date)

    # Disbursed year - needed for World Bank join
    df["disbursed_year"] = df["disbursed_time"].apply(
        lambda x: int(x[:4]) if x and len(str(x)) >= 4 else None
    )

    # Funded ratio
    df["funded_ratio"] = (
        df["funded_amount"] / df["loan_amount"]
    ).clip(0, 1)

    # Female borrower flag
    df["is_female_borrower"] = df["borrower_genders"].apply(
        lambda x: 1 if isinstance(x, str) and "female" in x.lower() else 0
    )

    # Standardise sectors
    df["sector_standardised"] = df["sector"].map(SECTOR_MAP).fillna(
        df["sector"].fillna("Unknown")
    )

    # Repayment interval encoding
    interval_map = {"monthly": 0, "bullet": 1, "irregular": 2}
    df["repayment_interval_enc"] = (
        df["repayment_interval"].str.lower()
                                .map(interval_map)
                                .fillna(2)
                                .astype(int)
    )

    # Drop rows with no loan amount
    before = len(df)
    df = df[df["loan_amount"].notna() & (df["loan_amount"] > 0)]
    print(f"  Dropped {before - len(df):,} rows with null/zero loan_amount")

    # Fill remaining nulls
    df["term_in_months"]  = df["term_in_months"].fillna(
        df["term_in_months"].median()
    )
    df["lender_count"]    = df["lender_count"].fillna(0).astype(int)
    df["sector_standardised"] = df["sector_standardised"].fillna("Unknown")
    df["country_code"]    = df["country_code"].fillna("XX")

    # Select final columns
    final_cols = [
        "id", "loan_amount", "funded_amount", "funded_ratio",
        "sector", "sector_standardised", "activity",
        "country", "country_code", "currency",
        "term_in_months", "lender_count", "is_female_borrower",
        "repayment_interval", "repayment_interval_enc",
        "posted_time", "funded_time", "disbursed_time", "disbursed_year",
        "use", "tags", "partner_id", "region",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    print(f"  📐  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Save to silver ─────────────────────────────────────────
    with get_conn(DB_SILVER) as conn:
        conn.execute("DROP TABLE IF EXISTS kiva_clean")
        bulk_insert(conn, "kiva_clean", df, if_exists="replace")
    print(f"  Saved -> silver.db : kiva_clean")
    return df


# 3. Clean World Bank

def clean_worldbank():
    print("\nSTEP 3 - Cleaning World Bank")

    with get_conn(DB_BRONZE) as conn:
        df = pd.read_sql_query("SELECT * FROM worldbank_raw", conn)
    print(f"  Loaded {len(df):,} rows")

    # Fill nulls with global median per indicator
    indicator_cols = ["gdp_growth", "inflation_rate", "unemployment_rate",
                      "poverty_rate", "domestic_credit_pct"]
    for col in indicator_cols:
        if col in df.columns:
            median = df[col].median()
            null_count = df[col].isna().sum()
            df[col] = df[col].fillna(median)
            if null_count:
                print(f"  {col}: filled {null_count} nulls with median ({median:.2f})")

    # Clip extreme outliers
    df["inflation_rate"]     = df["inflation_rate"].clip(-5, 100)
    df["gdp_growth"]         = df["gdp_growth"].clip(-20, 30)
    df["unemployment_rate"]  = df["unemployment_rate"].clip(0, 60)
    df["poverty_rate"]       = df["poverty_rate"].clip(0, 100)
    df["domestic_credit_pct"]= df["domestic_credit_pct"].clip(0, 200)

    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    with get_conn(DB_SILVER) as conn:
        conn.execute("DROP TABLE IF EXISTS macro_clean")
        bulk_insert(conn, "macro_clean", df, if_exists="replace")
    print(f"  Saved -> silver.db : macro_clean")
    return df


# 4. Join Kiva + World Bank -> kiva_enriched

def build_kiva_enriched():
    print("\nSTEP 4 - Joining Kiva + World Bank -> kiva_enriched")

    with get_conn(DB_SILVER) as conn:
        kiva  = pd.read_sql_query("SELECT * FROM kiva_clean", conn)
        macro = pd.read_sql_query("SELECT * FROM macro_clean", conn)

    print(f"  Kiva clean:  {len(kiva):,} rows")
    print(f"  Macro clean: {len(macro):,} rows")

    # Join on country_code + disbursed_year
    macro = macro.rename(columns={"_ingested_at": "_macro_ingested_at"})
    macro_cols = ["country_code", "year", "gdp_growth", "inflation_rate",
                  "unemployment_rate", "poverty_rate", "domestic_credit_pct"]
    macro = macro[[c for c in macro_cols if c in macro.columns]]

    enriched = kiva.merge(
        macro,
        left_on=["country_code", "disbursed_year"],
        right_on=["country_code", "year"],
        how="left"
    ).drop(columns=["year"], errors="ignore")

    # How many got macro data?
    matched = enriched["gdp_growth"].notna().sum()
    pct     = matched / len(enriched) * 100
    print(f"  Joined: {matched:,} / {len(enriched):,} loans got macro data ({pct:.1f}%)")

    # Fill unmatched with global medians
    for col in ["gdp_growth", "inflation_rate", "unemployment_rate",
                "poverty_rate", "domestic_credit_pct"]:
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna(enriched[col].median())

    print(f"  Final shape: {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")

    with get_conn(DB_SILVER) as conn:
        conn.execute("DROP TABLE IF EXISTS kiva_enriched")
        bulk_insert(conn, "kiva_enriched", enriched, if_exists="replace")
    print(f"  Saved -> silver.db : kiva_enriched")
    return enriched


# 5. Audit Silver Layer

def audit_silver():
    print("\nSILVER LAYER AUDIT")

    table_info(DB_SILVER)

    # Quick sector breakdown on kiva_enriched
    with get_conn(DB_SILVER) as conn:
        print(f"\nKiva Enriched - African Countries:")
        df = pd.read_sql_query(
            """SELECT country, country_code,
               COUNT(*) as loans,
               ROUND(AVG(loan_amount), 0) as avg_loan_usd,
               ROUND(AVG(gdp_growth), 1) as avg_gdp_growth
               FROM kiva_enriched
               WHERE country_code IN (
                 'KE','UG','TZ','RW','ET','GH','NG','SN','ML',
                 'MZ','ZM','ZW','MW','MG','CM','BF','TG','BJ'
               )
               GROUP BY country, country_code
               ORDER BY loans DESC""",
            conn
        )
        if len(df):
            print(df.to_string(index=False))
        else:
            print("  (no African country matches yet - will resolve in gold layer)")

        print(f"\nKiva Enriched - Sector Breakdown:")
        df = pd.read_sql_query(
            """SELECT sector_standardised,
               COUNT(*) as loans,
               ROUND(AVG(loan_amount), 0) as avg_loan_usd,
               ROUND(AVG(funded_ratio) * 100, 1) as avg_funded_pct
               FROM kiva_enriched
               WHERE sector_standardised IS NOT NULL
               GROUP BY sector_standardised
               ORDER BY loans DESC""",
            conn
        )
        print(df.to_string(index=False))

        print(f"\nLending Club Clean - Default Rate by Sector:")
        df = pd.read_sql_query(
            """SELECT sector,
               COUNT(*) as loans,
               ROUND(AVG(defaulted) * 100, 1) as default_rate_pct,
               ROUND(AVG(loan_amnt), 0) as avg_loan,
               ROUND(AVG(int_rate), 1) as avg_int_rate
               FROM lc_clean
               GROUP BY sector
               ORDER BY default_rate_pct DESC""",
            conn
        )
        print(df.to_string(index=False))


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 2: ETL -> Silver Layer\n")

    clean_lending_club()
    clean_kiva()
    clean_worldbank()
    build_kiva_enriched()
    audit_silver()

    print(f"\n  Stage 2 complete! Silver layer tables:")
    print(f"    silver.db -> lc_clean        (Lending Club, cleaned + target variable)")
    print(f"    silver.db -> kiva_clean       (Kiva, cleaned + engineered features)")
    print(f"    silver.db -> macro_clean      (World Bank, cleaned)")
    print(f"    silver.db -> kiva_enriched    (Kiva + World Bank joined)")
    print(f"\n  Next: python pipelines/05_feature_engineering.py\n")

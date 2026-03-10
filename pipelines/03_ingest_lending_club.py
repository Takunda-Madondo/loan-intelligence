"""
pipelines/03_ingest_lending_club.py
Stage 1 - Load Lending Club loan.csv into bronze.db.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_LC, DB_BRONZE, BRONZE_SCHEMAS
from db import init_db, get_conn, bulk_insert, row_count

CHUNK_SIZE = 50_000

# Keeps 15 columns
KEEP_COLS = [
    "id",
    "loan_amnt",               # loan amount
    "funded_amnt",             # amount funded
    "term",                    # 36 or 60 months
    "int_rate",                # interest rate
    "installment",             # monthly payment
    "grade",                   # LC risk grade A-G
    "sub_grade",               # finer grade A1-G5 — strongest single feature
    "emp_length",              # employment length
    "home_ownership",          # RENT / OWN / MORTGAGE
    "annual_inc",              # annual income
    "verification_status",     # income verified or not
    "issue_d",                 # loan issue date
    "loan_status",             # TARGET: Fully Paid / Charged Off / Default
    "purpose",                 # maps to Kiva sectors
    "dti",                     # debt-to-income ratio
    "delinq_2yrs",             # delinquencies in last 2 years
    "inq_last_6mths",          # credit inquiries last 6 months
    "mths_since_last_delinq",  # recency of last delinquency
    "mths_since_last_record",  # recency of public record
    "open_acc",                # number of open credit lines
    "pub_rec",                 # public derogatory records
    "revol_bal",               # total revolving balance
    "revol_util",              # revolving credit utilisation %
    "total_acc",               # total credit accounts
    "mort_acc",                # mortgage accounts
    "num_bc_sats",             # satisfactory bankcard accounts
    "pct_tl_nvr_dlq",          # % accounts never delinquent
    "num_tl_90g_dpd_24m",      # accounts 90+ days past due in 24m
    "avg_cur_bal",             # average current balance
    "bc_util",                 # bankcard utilisation
    "num_rev_accts",           # revolving account count
    "tot_cur_bal",             # total current balance
    "out_prncp",               # outstanding principal
    "total_pymnt",             # total payment received
    "total_rec_prncp",         # principal received
    "total_rec_int",           # interest received
    "recoveries",              # post charge-off recoveries
]


# 1. Add schema to Bronze

LC_SCHEMA = """
    CREATE TABLE IF NOT EXISTS lending_club_raw (
        id                  TEXT,
        loan_amnt           REAL,
        funded_amnt         REAL,
        term                TEXT,
        int_rate            TEXT,
        installment         REAL,
        grade               TEXT,
        emp_length          TEXT,
        home_ownership      TEXT,
        annual_inc          REAL,
        verification_status TEXT,
        issue_d             TEXT,
        loan_status         TEXT,
        purpose             TEXT,
        dti                 REAL,
        delinq_2yrs         REAL,
        inq_last_6mths      REAL,
        open_acc            REAL,
        pub_rec             REAL,
        revol_util          TEXT,
        total_acc           REAL,
        out_prncp           REAL,
        total_pymnt         REAL,
        total_rec_prncp     REAL,
        total_rec_int       REAL,
        recoveries          REAL,
        _ingested_at        TEXT DEFAULT (datetime('now'))
    )
"""


# 2. Load

def load_lending_club():
    csv_path = RAW_LC / "loan.csv"

    if not csv_path.exists():
        candidates = list(RAW_LC.glob("*.csv"))
        if not candidates:
            print(f"No CSV found in {RAW_LC}")
            print(f"    Download from: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv")
            return False
        csv_path = candidates[0]
        print(f"loan.csv not found, using {csv_path.name}")

    file_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"\nLoading: {csv_path.name}  ({file_mb:.0f} MB)")
    print(f"    Reading in chunks of {CHUNK_SIZE:,} rows...\n")

    # Create table
    with get_conn(DB_BRONZE) as conn:
        conn.execute("DROP TABLE IF EXISTS lending_club_raw")
        conn.execute(LC_SCHEMA)

    total_rows  = 0
    chunks_done = 0
    first_chunk = True
    skipped_rows = 0

    with get_conn(DB_BRONZE) as conn:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines="skip",    # LC file has some malformed rows
        ):
            # Keep only our target columns
            available = [c for c in KEEP_COLS if c in chunk.columns]
            chunk = chunk[available].copy()

            # Drop rows where loan_status is null — useless for modelling
            before = len(chunk)
            chunk = chunk[chunk["loan_status"].notna()]
            skipped_rows += before - len(chunk)

            if len(chunk) == 0:
                continue

            mode = "replace" if first_chunk else "append"
            bulk_insert(conn, "lending_club_raw", chunk, if_exists=mode)

            total_rows  += len(chunk)
            chunks_done += 1
            first_chunk  = False
            print(f"    Chunk {chunks_done:>3} -> {total_rows:>10,} rows loaded", end="\r")

    print(f"\n\nLoaded {total_rows:,} rows into bronze.db -> lending_club_raw")
    if skipped_rows:
        print(f"    (skipped {skipped_rows:,} rows with null loan_status)")
    return True


# 3. Audit

def audit_lending_club():
    print(f"\n{'='*60}")
    print(f"  DATA AUDIT - lending_club_raw")
    print(f"{'='*60}")

    with get_conn(DB_BRONZE) as conn:
        total = row_count(conn, "lending_club_raw")
        print(f"\nTotal rows: {total:,}")

        # Loan status
        print(f"\nLoan Status Distribution (our target variable):")
        df = pd.read_sql_query(
            "SELECT loan_status, "
            "COUNT(*) as loans, "
            f"ROUND(COUNT(*) * 100.0 / {total}, 1) as pct "
            "FROM lending_club_raw "
            "GROUP BY loan_status "
            "ORDER BY loans DESC",
            conn
        )
        print(df.to_string(index=False))

        # Flag what maps to default=1 vs default=0
        print(f"\n    Will label as defaulted=0 : 'Fully Paid'")
        print(f"    Will label as defaulted=1 : 'Charged Off', 'Default'")
        print(f"    Will EXCLUDE from training : everything else (active loans)")

        # Purpose breakdown - maps to Kiva sectors
        print(f"\nLoan Purpose (maps to Kiva sectors):")
        df = pd.read_sql_query(
            "SELECT purpose, "
            "COUNT(*) as loans, "
            "ROUND(AVG(CAST(loan_amnt AS REAL)), 0) as avg_usd, "
            "ROUND(SUM(CASE WHEN loan_status='Charged Off' THEN 1.0 ELSE 0 END) "
            f"   / COUNT(*) * 100, 1) as default_rate_pct "
            "FROM lending_club_raw "
            "WHERE purpose IS NOT NULL "
            "GROUP BY purpose "
            "ORDER BY loans DESC",
            conn
        )
        print(df.to_string(index=False))

        # Grade distribution
        print(f"\nRisk Grade Distribution:")
        df = pd.read_sql_query(
            "SELECT grade, COUNT(*) as loans, "
            "ROUND(SUM(CASE WHEN loan_status='Charged Off' THEN 1.0 ELSE 0 END) "
            f"   / COUNT(*) * 100, 1) as default_rate_pct "
            "FROM lending_club_raw "
            "WHERE grade IS NOT NULL "
            "GROUP BY grade ORDER BY grade",
            conn
        )
        print(df.to_string(index=False))

        # Key numeric stats
        print(f"\nLoan Amount & DTI Stats:")
        df = pd.read_sql_query(
            "SELECT "
            "ROUND(MIN(loan_amnt), 0)  as min_loan, "
            "ROUND(AVG(loan_amnt), 0)  as avg_loan, "
            "ROUND(MAX(loan_amnt), 0)  as max_loan, "
            "ROUND(AVG(dti), 1)        as avg_dti, "
            "ROUND(AVG(annual_inc), 0) as avg_income "
            "FROM lending_club_raw",
            conn
        )
        print(df.to_string(index=False))

        # Null check on columns we'll use for ML
        print(f"\nNull check on ML columns:")
        ml_cols = ["loan_amnt", "term", "int_rate", "grade", "dti",
                   "annual_inc", "purpose", "loan_status", "revol_util",
                   "inq_last_6mths", "delinq_2yrs"]
        for col in ml_cols:
            n = conn.execute(
                f"SELECT COUNT(*) FROM lending_club_raw WHERE {col} IS NULL"
            ).fetchone()[0]
            pct  = n / total * 100 if total else 0
            flag = "  " if pct > 20 else ""
            print(f"    {col:<25} {n:>8,}  ({pct:.1f}%){flag}")


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 1: Lending Club Ingestion -> SQLite\n")

    success = load_lending_club()

    if success:
        audit_lending_club()

    print("  Done!  bronze.db -> lending_club_raw")
    print("  All 3 source tables now loaded:")
    print("    bronze.db -> kiva_loans_raw      ({:,} rows)".format(671205))
    print("    bronze.db -> worldbank_raw        (312 rows)")
    print("    bronze.db -> lending_club_raw     (just loaded)")
    print("\n  Next: python pipelines/04_etl_silver.py  <- Stage 2 starts!")

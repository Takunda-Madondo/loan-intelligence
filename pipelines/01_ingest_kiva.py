"""
pipelines/01_ingest_kiva.py
Stage 1 - Download Kiva CSV from Kaggle and load into bronze.db.
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_KIVA, DB_BRONZE, BRONZE_SCHEMAS, KAGGLE_USER, KAGGLE_KEY
from db import init_db, get_conn, bulk_insert, row_count

os.environ["KAGGLE_USERNAME"] = KAGGLE_USER or ""
os.environ["KAGGLE_KEY"]      = KAGGLE_KEY or ""

# Columns we want — anything not in this list gets dropped
KEEP_COLS = [
    "id", "loan_amount", "funded_amount", "status", "sector",
    "activity", "country", "country_code", "currency",
    "term_in_months", "lender_count", "borrower_genders",
    "repayment_interval", "posted_time", "funded_time",
    "disbursed_time", "raised_time", "use", "tags",
    "partner_id", "region",
]

CHUNK_SIZE = 50_000


# 1. Download

def download_kiva() -> bool:
    print("Downloading Kiva dataset from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "kiva/data-science-for-good-kiva-crowdfunding",
            path=str(RAW_KIVA),
            unzip=True,
            quiet=False,
        )
        print(f"Downloaded to {RAW_KIVA}\n")
        return True
    except Exception as e:
        print(f"\nKaggle download failed: {e}")
        print("\nMANUAL DOWNLOAD:")
        print("    1. Go to: https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding")
        print("    2. Click Download -> extract the ZIP")
        print(f"    3. Place the CSV files in: {RAW_KIVA}/")
        print("    4. Re-run this script\n")
        return False


# 2. Load CSV under SQLite

def load_kiva_to_bronze() -> bool:
    """
    Read kiva_loans.csv in chunks and insert into bronze.db.
    """
    csv_path = RAW_KIVA / "kiva_loans.csv"

    if not csv_path.exists():
        candidates = list(RAW_KIVA.glob("*.csv"))
        if not candidates:
            print(f"No CSV found in {RAW_KIVA}. Download the data first.")
            return False
        csv_path = candidates[0]
        print(f"kiva_loans.csv not found, using {csv_path.name}")

    file_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"\nLoading: {csv_path.name}  ({file_mb:.0f} MB)")
    print(f"    Reading in chunks of {CHUNK_SIZE:,} rows...\n")

    # Always drop + recreate so broken previous runs don't leave stale schema
    with get_conn(DB_BRONZE) as conn:
        conn.execute("DROP TABLE IF EXISTS kiva_loans_raw")
    init_db(DB_BRONZE, BRONZE_SCHEMAS)

    total_rows  = 0
    chunks_done = 0
    first_chunk = True

    with get_conn(DB_BRONZE) as conn:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):

            # Keep only our target columns — ignore anything not in KEEP_COLS
            available = [c for c in KEEP_COLS if c in chunk.columns]
            chunk = chunk[available].copy()

            mode = "replace" if first_chunk else "append"
            bulk_insert(conn, "kiva_loans_raw", chunk, if_exists=mode)

            total_rows  += len(chunk)
            chunks_done += 1
            first_chunk  = False
            print(f"    Chunk {chunks_done:>3} -> {total_rows:>10,} rows loaded", end="\r")

    print(f"\n\nLoaded {total_rows:,} rows into bronze.db -> kiva_loans_raw")
    return True


# 3. Audit

def audit_bronze():
    print(f"\n{'='*60}")
    print(f"  DATA AUDIT - kiva_loans_raw")
    print(f"{'='*60}")

    with get_conn(DB_BRONZE) as conn:

        total = row_count(conn, "kiva_loans_raw")
        print(f"\nTotal rows: {total:,}")

        # Discover what columns actually landed
        cols_df   = pd.read_sql_query("PRAGMA table_info(kiva_loans_raw)", conn)
        col_names = cols_df["name"].tolist()
        print(f"\nColumns loaded ({len(col_names)}):")
        print("    " + ",  ".join(col_names))

        # Status distribution
        # Find the status column dynamically — could be "status" or "loan_status"
        status_col = next((c for c in col_names if "status" in c.lower()), None)
        if status_col:
            print(f"\nLoan Status ('{status_col}') - this becomes our target variable:")
            df = pd.read_sql_query(
                f"SELECT {status_col} as status, "
                f"COUNT(*) as loans, "
                f"ROUND(COUNT(*) * 100.0 / {total}, 1) as pct "
                f"FROM kiva_loans_raw "
                f"GROUP BY {status_col} "
                f"ORDER BY loans DESC",
                conn
            )
            print(df.to_string(index=False))
            print(f"\n    We will label: 'defaulted' -> 1,  'paid' -> 0,  others -> excluded")
        else:
            print(f"\nNo status column found. Columns present: {col_names}")

        # Sector breakdown
        if "sector" in col_names:
            print(f"\nLoans by Sector:")
            df = pd.read_sql_query(
                "SELECT sector, "
                "COUNT(*) as loans, "
                "ROUND(AVG(loan_amount), 0) as avg_usd "
                "FROM kiva_loans_raw "
                "WHERE sector IS NOT NULL "
                "GROUP BY sector "
                "ORDER BY loans DESC "
                "LIMIT 15",
                conn
            )
            print(df.to_string(index=False))

        # African country breakdown
        cc = next((c for c in col_names if "country_code" in c.lower()), None)
        cn = next((c for c in col_names if c.lower() == "country"), None)
        if cc:
            print(f"\nAfrican Countries in Dataset:")
            select_country = f"{cn} as country, " if cn else ""
            df = pd.read_sql_query(
                f"SELECT {select_country}{cc} as country_code, "
                f"COUNT(*) as loans, "
                f"ROUND(AVG(loan_amount), 0) as avg_loan_usd, "
                f"ROUND(SUM(loan_amount) / 1000.0, 0) as total_k_usd "
                f"FROM kiva_loans_raw "
                f"WHERE {cc} IN ("
                f"  'KE','UG','TZ','RW','ET','GH','NG','SN','ML',"
                f"  'MZ','ZM','ZW','MW','MG','CM','BF','TG','BJ','LR','SL'"
                f") "
                f"GROUP BY {cc} "
                f"ORDER BY loans DESC",
                conn
            )
            print(df.to_string(index=False))

        # Null check on key columns
        print(f"\nNull / missing check:")
        check = ["loan_amount", "sector", "country_code",
                 "term_in_months", "repayment_interval", "disbursed_time"]
        if status_col:
            check.insert(1, status_col)

        for col in check:
            if col not in col_names:
                print(f"    {col:<25}  -  not in dataset")
                continue
            n   = conn.execute(
                f"SELECT COUNT(*) FROM kiva_loans_raw WHERE {col} IS NULL"
            ).fetchone()[0]
            pct = n / total * 100 if total else 0
            flag = "  high nulls" if pct > 20 else ""
            print(f"    {col:<25}  {n:>8,}  ({pct:.1f}%){flag}")

        # Loan amount summary
        print(f"\nLoan Amount Summary (USD):")
        df = pd.read_sql_query(
            "SELECT "
            "ROUND(MIN(loan_amount), 0)  as min_usd, "
            "ROUND(AVG(loan_amount), 0)  as avg_usd, "
            "ROUND(MAX(loan_amount), 0)  as max_usd, "
            "ROUND(SUM(loan_amount) / 1000000.0, 2) as total_portfolio_M "
            "FROM kiva_loans_raw WHERE loan_amount IS NOT NULL",
            conn
        )
        print(df.to_string(index=False))


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 1: Kiva Ingestion -> SQLite\n")

    # Skip download if CSV already present
    if not list(RAW_KIVA.glob("*.csv")):
        download_kiva()
    else:
        print(f"CSV already in {RAW_KIVA} - skipping download")

    success = load_kiva_to_bronze()

    if success:
        audit_bronze()

    print("Done! bronze.db -> kiva_loans_raw")
    print("Tip: open data/db/bronze.db in DB Browser for SQLite to explore visually")
    print("Next: python pipelines/02_ingest_worldbank.py")

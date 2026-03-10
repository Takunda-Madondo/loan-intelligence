"""
pipelines/02_ingest_worldbank.py
Stage 1 - Pull World Bank macro indicators and load into bronze.db.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import (DB_BRONZE, BRONZE_SCHEMAS,
                    WB_INDICATORS, WB_COUNTRIES, WB_YEARS)
from db import init_db, get_conn, bulk_insert, row_count


# 1. Fetch from World Bank API

def fetch_worldbank() -> pd.DataFrame:
    """
    Pull all indicators for all target countries and years via wbgapi.
    Returns a wide DataFrame: country_code | year | gdp_growth | ...
    """
    try:
        import wbgapi as wb
    except ImportError:
        print("wbgapi not installed. Run: pip install wbgapi")
        return pd.DataFrame()

    print(f"Fetching World Bank indicators...")
    print(f"    {len(WB_COUNTRIES)} countries  |  {len(WB_INDICATORS)} indicators  "
          f"|  {WB_YEARS[0]}-{WB_YEARS[-1]}\n")

    all_frames = []

    for code, name in WB_INDICATORS.items():
        print(f"  {name}...", end=" ")
        try:
            raw = wb.data.DataFrame(
                code,
                economy=WB_COUNTRIES,
                time=WB_YEARS,
                labels=False,
            ).reset_index()

            # Melt year columns (YR2012, YR2013 …) to long format
            year_cols = [c for c in raw.columns
                         if str(c).startswith("YR") or str(c).isdigit()]
            id_cols   = [c for c in raw.columns if c not in year_cols]

            long = raw.melt(
                id_vars=id_cols,
                value_vars=year_cols,
                var_name="year_raw",
                value_name=name,
            )
            long["year"] = (long["year_raw"].astype(str)
                                            .str.replace("YR", "")
                                            .astype(int))

            # Standardise the country column name
            for alias in ["economy", "Economy", "index"]:
                if alias in long.columns:
                    long = long.rename(columns={alias: "country_code"})
                    break

            long[name] = pd.to_numeric(long[name], errors="coerce")
            all_frames.append(long[["country_code", "year", name]])
            print(f"{len(long):,} rows")

        except Exception as e:
            print(f"skipped ({e})")
            continue

    if not all_frames:
        return pd.DataFrame()

    # Merge all indicators on country_code + year
    df = all_frames[0]
    for frame in all_frames[1:]:
        df = df.merge(frame, on=["country_code", "year"], how="outer")

    print(f"\nFetched {len(df):,} rows total")
    return df


# 2. Fallback - Realistic Sample Data

def sample_worldbank_data() -> pd.DataFrame:
    """
    Generate realistic sample data when the API is unavailable.
    Values are based on actual World Bank ranges for Sub-Saharan Africa.
    Replace with real data as soon as you have internet access.
    """
    print("\nGenerating sample World Bank data (API unavailable)...")
    np.random.seed(42)

    # (mean, std) based on real WB data for SSA
    ranges = {
        "gdp_growth":          (4.5, 3.0),
        "inflation_rate":      (8.0, 6.0),
        "unemployment_rate":   (7.0, 5.0),
        "poverty_rate":        (35.0, 20.0),
        "domestic_credit_pct": (20.0, 15.0),
    }

    rows = []
    for country in WB_COUNTRIES:
        for year in WB_YEARS:
            row = {"country_code": country, "year": year}
            for col, (mean, std) in ranges.items():
                row[col] = round(np.random.normal(mean, std), 2)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"    {len(df):,} sample rows created")
    print(f"    SAMPLE DATA - run this script online to get real values\n")
    return df


# 3. Load to SQLite

def load_worldbank_to_bronze(df: pd.DataFrame):
    """Insert World Bank data into bronze.db → worldbank_raw."""
    init_db(DB_BRONZE, BRONZE_SCHEMAS)

    # Ensure column order matches schema
    cols = ["country_code", "year", "gdp_growth", "inflation_rate",
            "unemployment_rate", "poverty_rate", "domestic_credit_pct"]
    df = df[[c for c in cols if c in df.columns]].copy()

    with get_conn(DB_BRONZE) as conn:
        # Clear previous run then insert fresh
        conn.execute("DELETE FROM worldbank_raw")
        bulk_insert(conn, "worldbank_raw", df, if_exists="append")

    print(f"{len(df):,} rows inserted into bronze.db : worldbank_raw")


# 4. Audit

def audit_worldbank():
    print(f"\n{'='*60}")
    print(f"  DATA AUDIT - worldbank_raw")
    print(f"{'='*60}")

    with get_conn(DB_BRONZE) as conn:
        total = row_count(conn, "worldbank_raw")
        print(f"\nTotal rows: {total:,}")

        print(f"\nYear range:")
        yr = pd.read_sql_query(
            "SELECT MIN(year) as from_year, MAX(year) as to_year FROM worldbank_raw",
            conn
        )
        print(f"    {yr['from_year'][0]} – {yr['to_year'][0]}")

        print(f"\nCountries loaded ({conn.execute('SELECT COUNT(DISTINCT country_code) FROM worldbank_raw').fetchone()[0]}):")
        countries = pd.read_sql_query(
            "SELECT country_code, COUNT(*) as years_of_data "
            "FROM worldbank_raw GROUP BY country_code ORDER BY country_code",
            conn
        )
        print(countries.to_string(index=False))

        print(f"\nIndicator Statistics:")
        stats = pd.read_sql_query(
            """SELECT
               ROUND(AVG(gdp_growth),2)          as avg_gdp_growth,
               ROUND(AVG(inflation_rate),2)       as avg_inflation,
               ROUND(AVG(unemployment_rate),2)    as avg_unemployment,
               ROUND(AVG(poverty_rate),2)         as avg_poverty,
               ROUND(AVG(domestic_credit_pct),2)  as avg_credit_pct
               FROM worldbank_raw""",
            conn
        )
        print(stats.to_string(index=False))

        print(f"\nNull counts:")
        for col in ["gdp_growth", "inflation_rate", "unemployment_rate",
                    "poverty_rate", "domestic_credit_pct"]:
            n = conn.execute(
                f"SELECT COUNT(*) FROM worldbank_raw WHERE {col} IS NULL"
            ).fetchone()[0]
            pct = n / total * 100 if total else 0
            print(f"    {col:<25} {n:>6,}  ({pct:.1f}%)")


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 1: World Bank Ingestion -> SQLite\n")

    df = fetch_worldbank()

    if df.empty:
        df = sample_worldbank_data()

    load_worldbank_to_bronze(df)
    audit_worldbank()

    print(f"  Done! Data is in:  data/db/bronze.db  ->  table: worldbank_raw")
    print(f"  Tip: Open bronze.db in 'DB Browser for SQLite' to inspect the data")
    print(f"  Next: python pipelines/03_etl_silver.py  <- Stage 2 starts here")

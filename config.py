"""
config.py - Central project configuration.
All paths, constants, DB settings, and schema definitions live here.
Import this in every script. Avoid hardcoding paths or table names.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Root directory
ROOT_DIR = Path(__file__).parent

# Raw Data Landing Zone
RAW_DIR  = ROOT_DIR / "data" / "raw"
RAW_KIVA = RAW_DIR / "kiva"
RAW_WB   = RAW_DIR / "worldbank"
RAW_LC   = RAW_DIR / "lending_club"

# SQLite Database Files
DB_DIR    = ROOT_DIR / "data" / "db"
DB_BRONZE = DB_DIR / "bronze.db"
DB_SILVER = DB_DIR / "silver.db"
DB_GOLD   = DB_DIR / "gold.db"

# Models
MODELS_DIR = ROOT_DIR / "models" / "artifacts"
MLFLOW_URI = str(ROOT_DIR / "models" / "mlflow")

# API Keys
KAGGLE_USER   = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY    = os.getenv("KAGGLE_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

# World Bank Settings

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG":    "inflation_rate",
    "SL.UEM.TOTL.ZS":    "unemployment_rate",
    "SI.POV.DDAY":        "poverty_rate",
    "FS.AST.CGOV.GD.ZS": "domestic_credit_pct",
}

WB_COUNTRIES = [
    "KE","UG","TZ","RW","ET",
    "GH","NG","SN","ML","BF","TG","BJ","GN","SL","LR",
    "MZ","ZM","ZW","MW","MG","CM",
    "PH","IN","KH","TJ","PK",
]

WB_YEARS = list(range(2012, 2024))

# Sector normalisation map
SECTOR_MAP = {
    "Agriculture":    "Agriculture & Farming",
    "Food":           "Food & Beverage",
    "Retail":         "Retail & Trade",
    "Wholesale":      "Retail & Trade",
    "Services":       "Services",
    "Health":         "Health & Wellness",
    "Education":      "Education",
    "Transportation": "Transport & Logistics",
    "Housing":        "Housing & Construction",
    "Construction":   "Housing & Construction",
    "Manufacturing":  "Manufacturing",
    "Arts":           "Arts & Crafts",
    "Clothing":       "Clothing & Textiles",
    "Entertainment":  "Entertainment",
    "Personal Use":   "Personal Use",
}

# ML settings
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
TARGET_COL   = "defaulted"

# Database Schemas

BRONZE_SCHEMAS = {

    "kiva_loans_raw": """
        CREATE TABLE IF NOT EXISTS kiva_loans_raw (
            id                  INTEGER,
            loan_amount         REAL,
            funded_amount       REAL,
            status              TEXT,
            sector              TEXT,
            activity            TEXT,
            country             TEXT,
            country_code        TEXT,
            currency            TEXT,
            term_in_months      INTEGER,
            lender_count        INTEGER,
            borrower_genders    TEXT,
            repayment_interval  TEXT,
            posted_time         TEXT,
            funded_time         TEXT,
            disbursed_time      TEXT,
            raised_time         TEXT,
            use                 TEXT,
            tags                TEXT,
            partner_id          INTEGER,
            region              TEXT,
            _ingested_at        TEXT DEFAULT (datetime('now'))
        )
    """,

    "worldbank_raw": """
        CREATE TABLE IF NOT EXISTS worldbank_raw (
            country_code        TEXT,
            year                INTEGER,
            gdp_growth          REAL,
            inflation_rate      REAL,
            unemployment_rate   REAL,
            poverty_rate        REAL,
            domestic_credit_pct REAL,
            _ingested_at        TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (country_code, year)
        )
    """,
}

SILVER_SCHEMAS = {

    "loans_clean": """
        CREATE TABLE IF NOT EXISTS loans_clean (
            loan_id              INTEGER PRIMARY KEY,
            loan_amount          REAL    NOT NULL,
            funded_amount        REAL,
            status               TEXT,
            defaulted            INTEGER,
            sector               TEXT,
            sector_standardised  TEXT,
            activity             TEXT,
            country              TEXT,
            country_code         TEXT,
            currency             TEXT,
            term_in_months       INTEGER,
            lender_count         INTEGER,
            is_female_borrower   INTEGER,
            repayment_interval   TEXT,
            posted_date          TEXT,
            funded_date          TEXT,
            disbursed_date       TEXT,
            raised_date          TEXT,
            loan_purpose         TEXT,
            partner_id           INTEGER,
            region               TEXT,
            _created_at          TEXT DEFAULT (datetime('now'))
        )
    """,

    "macro_indicators": """
        CREATE TABLE IF NOT EXISTS macro_indicators (
            country_code        TEXT    NOT NULL,
            year                INTEGER NOT NULL,
            gdp_growth          REAL,
            inflation_rate      REAL,
            unemployment_rate   REAL,
            poverty_rate        REAL,
            domestic_credit_pct REAL,
            PRIMARY KEY (country_code, year)
        )
    """,

    "loans_enriched": """
        CREATE TABLE IF NOT EXISTS loans_enriched (
            loan_id              INTEGER PRIMARY KEY,
            loan_amount          REAL,
            funded_amount        REAL,
            defaulted            INTEGER,
            sector_standardised  TEXT,
            activity             TEXT,
            country              TEXT,
            country_code         TEXT,
            term_in_months       INTEGER,
            lender_count         INTEGER,
            is_female_borrower   INTEGER,
            repayment_interval   TEXT,
            disbursed_year       INTEGER,
            loan_purpose         TEXT,
            gdp_growth           REAL,
            inflation_rate       REAL,
            unemployment_rate    REAL,
            poverty_rate         REAL,
            domestic_credit_pct  REAL,
            _created_at          TEXT DEFAULT (datetime('now'))
        )
    """,
}

GOLD_SCHEMAS = {

    "loan_features": """
        CREATE TABLE IF NOT EXISTS loan_features (
            loan_id                 INTEGER PRIMARY KEY,
            defaulted               INTEGER NOT NULL,
            loan_amount             REAL,
            term_in_months          INTEGER,
            lender_count            INTEGER,
            funded_ratio            REAL,
            is_female_borrower      INTEGER,
            repayment_interval_enc  INTEGER,
            sector_enc              INTEGER,
            country_enc             INTEGER,
            gdp_growth              REAL,
            inflation_rate          REAL,
            unemployment_rate       REAL,
            poverty_rate            REAL,
            domestic_credit_pct     REAL,
            loan_per_month          REAL,
            macro_risk_score        REAL,
            _created_at             TEXT DEFAULT (datetime('now'))
        )
    """,

    "sector_performance": """
        CREATE TABLE IF NOT EXISTS sector_performance (
            sector               TEXT NOT NULL,
            country_code         TEXT NOT NULL,
            total_loans          INTEGER,
            total_amount_usd     REAL,
            avg_loan_amount      REAL,
            default_rate         REAL,
            avg_term_months      REAL,
            female_borrower_pct  REAL,
            avg_lender_count     REAL,
            avg_gdp_growth       REAL,
            avg_inflation        REAL,
            roi_score            REAL,
            risk_tier            TEXT,
            PRIMARY KEY (sector, country_code)
        )
    """,
}

# Ensure all directories exist on import
for _d in [RAW_KIVA, RAW_WB, RAW_LC, DB_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

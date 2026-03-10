"""
pipelines/db.py - Shared SQLite database utilities.
Centralizes SQLite connection handling.
"""

import sqlite3
import logging
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


# Connection

@contextmanager
def get_conn(db_path: Path):
    """
    Context manager for SQLite connections.
    Enables WAL mode and foreign keys.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row   # rows accessible by column name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Schema Initialization

def init_db(db_path: Path, schemas: dict):
    """
    Create tables defined in a schemas dict.
    Safe to call multiple times.
    """
    with get_conn(db_path) as conn:
        for table_name, ddl in schemas.items():
            conn.execute(ddl)
            logger.info(f"Table ready: {table_name}")
    print(f"Database initialised: {db_path.name}")


# Read / Write

def run_query(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a DataFrame.
    """
    return pd.read_sql_query(sql, conn, params=params)


def bulk_insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame,
                if_exists: str = "append", chunksize: int = 10_000):
    """
    Insert a DataFrame into a SQLite table efficiently in chunks.
    Automatically caps chunksize for SQLite safety.
    """
    # SQLite max variables = 999. Cap rows per batch to stay safely under.
    n_cols         = len(df.columns)
    safe_chunksize = max(1, min(chunksize, 999 // n_cols))

    df.to_sql(
        name=table,
        con=conn,
        if_exists=if_exists,
        index=False,
        chunksize=safe_chunksize,
        method=None,   # row-by-row insert avoids the 999 variable limit on Windows
    )
    logger.info(f"Inserted {len(df):,} rows into {table}")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check whether a table exists in the database."""
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return result is not None


def row_count(conn: sqlite3.Connection, table: str) -> int:
    """Return the number of rows in a table."""
    result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return result[0] if result else 0


def table_info(db_path: Path):
    """
    Print a summary of all tables and row counts in a database.
    """
    with get_conn(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        if not tables:
            print(f"  (no tables in {db_path.name})")
            return

        print(f"\n{db_path.name}")
        print(f"  {'Table':<35} {'Rows':>10}")
        print(f"  {'-'*35} {'-'*10}")
        for (tbl,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:<35} {count:>10,}")

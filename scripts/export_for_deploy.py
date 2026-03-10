"""
scripts/export_for_deploy.py
-----------------------------
Exports the two dashboard tables from gold.db to compressed Parquet files
that are small enough to commit to GitHub.

Run once after every pipeline rerun:
    python scripts/export_for_deploy.py

Output:
    data/deploy/kiva_predictions.parquet   (~10–25MB)
    data/deploy/sector_performance.parquet (~50KB)
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DB_GOLD   = ROOT / "data" / "db" / "gold.db"
DEPLOY_DIR = ROOT / "data" / "deploy"

DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


def export_table(conn, table: str, out_path: Path):
    print(f"  Exporting {table}...")
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    print(f"       {len(df):,} rows x {len(df.columns)} cols  ->  {out_path.name}")
    df.to_parquet(out_path, index=False, compression="gzip")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"       Saved - {size_mb:.1f} MB")
    return size_mb


def main():
    print("=" * 55)
    print("  LOAN INTELLIGENCE - DEPLOY EXPORT")
    print("=" * 55)

    if not DB_GOLD.exists():
        print(f"\n  gold.db not found at {DB_GOLD}")
        print("      Run the pipelines first:\n")
        print("      python pipelines/05_feature_engineering.py")
        print("      python pipelines/06_train_model.py")
        return

    total_mb = 0
    with sqlite3.connect(DB_GOLD) as conn:
        total_mb += export_table(conn, "kiva_predictions",  DEPLOY_DIR / "kiva_predictions.parquet")
        total_mb += export_table(conn, "sector_performance", DEPLOY_DIR / "sector_performance.parquet")

    print(f"\n  Total deploy data size: {total_mb:.1f} MB")

    if total_mb > 90:
        print("  Warning: total exceeds 90MB - consider filtering kiva_predictions")
        print("       to African countries only before exporting.")
    elif total_mb > 50:
        print("  Heads up: over 50MB - Git LFS recommended.")
    else:
        print("  Safe to commit directly to GitHub.")

    print(f"\n  Files written to: {DEPLOY_DIR}")
    print("\n  Next steps:")
    print("  1. git add data/deploy/")
    print("  2. git add models/artifacts/")
    print("  3. git commit -m 'chore: export deploy data'")
    print("  4. git push")


if __name__ == "__main__":
    main()

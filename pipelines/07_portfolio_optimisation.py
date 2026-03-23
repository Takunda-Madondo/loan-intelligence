"""
pipelines/07_portfolio_optimisation.py
Stage 5 — Sector-Level Portfolio Optimisation using Linear Programming

Solves a capital allocation problem: given a budget, maximise expected return
across sectors while respecting risk constraints and concentration limits.

Three lender profiles modelled:
  - Conservative : tight risk ceiling, minimal high-risk exposure
  - Balanced     : moderate risk tolerance, mission + return
  - Aggressive   : higher risk accepted for growth and market penetration

Outputs written to gold.db -> portfolio_allocations table.

Run:
  python pipelines/07_portfolio_optimisation.py

Dependencies:
  pip install scipy
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_GOLD, MODELS_DIR
from db import get_conn, bulk_insert

try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except ImportError:
    print("  scipy not found. Run: pip install scipy")
    SCIPY_OK = False


# ── Lender profiles ───────────────────────────────────────────────────────────
# Each profile defines:
#   risk_ceiling         : max weighted average portfolio default rate
#   max_high_risk_alloc  : max fraction of capital to sectors with default > high_risk_threshold
#   min_sector_alloc     : floor per sector (social mission — no sector gets 0)
#   max_sector_alloc     : concentration cap per sector
#   high_risk_threshold  : default rate above which a sector is considered "high risk"
#   description          : narrative label

PROFILES = {
    "Conservative": {
        "risk_ceiling":        0.065,   # 6.5% max weighted portfolio default rate
        "max_high_risk_alloc": 0.08,    # at most 8% of capital in high-risk sectors
        "min_sector_alloc":    0.01,    # 1% floor — social mission, no exclusion
        "max_sector_alloc":    0.35,    # no single sector > 35%
        "high_risk_threshold": 0.09,    # sectors above 9% default rate = high risk
        "description": "Regulated deposit-taking MFI. Tight covenants, low risk tolerance.",
        "colour": "#0891B2",
    },
    "Balanced": {
        "risk_ceiling":        0.082,   # 8.2% — aligns with African MFI PAR30 benchmark
        "max_high_risk_alloc": 0.18,    # up to 18% in higher-risk sectors
        "min_sector_alloc":    0.02,    # 2% floor
        "max_sector_alloc":    0.30,    # 30% concentration cap
        "high_risk_threshold": 0.09,
        "description": "Mid-size African MFI. Balances financial sustainability with social mission.",
        "colour": "#6378FF",
    },
    "Aggressive": {
        "risk_ceiling":        0.105,   # 10.5% — higher losses accepted for market growth
        "max_high_risk_alloc": 0.32,    # up to 32% in higher-risk sectors
        "min_sector_alloc":    0.02,    # 2% floor
        "max_sector_alloc":    0.28,    # slightly tighter concentration cap (spread the risk)
        "high_risk_threshold": 0.09,
        "description": "Growth-oriented fintech MFI. Capturing underserved segments, accepting higher PAR30.",
        "colour": "#D97706",
    },
}


# ── Data preparation ──────────────────────────────────────────────────────────

def load_sector_inputs(african_only: bool = True) -> pd.DataFrame:
    """
    Aggregate sector_performance to one row per sector.
    Returns columns: sector, default_rate (0-1), roi_score, total_loans,
                     avg_loan_amount, risk_tier, female_borrower_pct
    """
    AFRICAN_COUNTRIES = [
        "KE","UG","TZ","RW","ET","GH","NG","SN","ML",
        "MZ","ZM","ZW","MW","MG","CM","BF","TG","BJ","GN",
    ]

    with get_conn(DB_GOLD) as conn:
        df = pd.read_sql_query("SELECT * FROM sector_performance", conn)

    if african_only:
        df = df[df["country_code"].isin(AFRICAN_COUNTRIES)]

    # default_rate is stored as percentage (e.g. 7.5), convert to decimal
    df["default_rate_dec"] = df["default_rate"] / 100.0

    agg = df.groupby("sector").agg(
        roi_score           = ("roi_score",           "mean"),
        default_rate        = ("default_rate_dec",    "mean"),
        total_loans         = ("total_loans",         "sum"),
        avg_loan_amount     = ("avg_loan_amount",     "mean"),
        female_borrower_pct = ("female_borrower_pct", "mean"),
        avg_gdp_growth      = ("avg_gdp_growth",      "mean"),
    ).reset_index()

    # Expected return per unit: interest earned minus expected default loss
    # Proxy interest rate from sector default risk (higher risk → higher rate charged)
    # Base rate 12% + risk premium of 8× the sector default rate (realistic MFI pricing)
    BASE_RATE = 0.12
    RISK_PREMIUM_MULT = 8.0
    agg["implied_interest_rate"] = (BASE_RATE + RISK_PREMIUM_MULT * agg["default_rate"]).round(4)

    # Expected return = interest earned - expected loss on defaulted portion
    # Simplified: E[R] = interest_rate × (1 - default_rate) - default_rate × loss_given_default
    # Loss given default in microfinance ≈ 0.60 (group lending recovers ~40%)
    LGD = 0.60
    agg["expected_return"] = (
        agg["implied_interest_rate"] * (1 - agg["default_rate"])
        - agg["default_rate"] * LGD
    ).round(6)

    # Risk tier labels (aligned with African MFI norms)
    agg["risk_tier"] = pd.cut(
        agg["default_rate"],
        bins=[0, 0.07, 0.09, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    ).astype(str)

    print(f"\n  Sector inputs ({len(agg)} sectors):")
    print(f"  {'Sector':<28} {'Default':>8} {'E[Return]':>10} {'ROI':>7} {'Tier':>8}")
    print(f"  {'-'*65}")
    for _, row in agg.sort_values("expected_return", ascending=False).iterrows():
        print(
            f"  {row['sector']:<28} "
            f"{row['default_rate']*100:>7.1f}%"
            f"{row['expected_return']*100:>10.2f}%"
            f"{row['roi_score']:>8.3f}"
            f"  {row['risk_tier']:>8}"
        )

    return agg.reset_index(drop=True)


# ── Linear Programme ──────────────────────────────────────────────────────────

def solve_lp(sectors: pd.DataFrame, profile: dict, budget: float) -> dict:
    """
    Maximise expected portfolio return subject to:
      1. Capital sums to 1 (full allocation)
      2. No sector exceeds max_sector_alloc
      3. No sector below min_sector_alloc
      4. Weighted average default rate ≤ risk_ceiling
      5. Total allocation to high-risk sectors ≤ max_high_risk_alloc

    Decision variable x_i = fraction of budget allocated to sector i.

    scipy.linprog minimises, so we negate expected_return to maximise.

    Returns dict with allocation fractions, allocated amounts, and metrics.
    """
    n = len(sectors)
    er = sectors["expected_return"].values       # expected return per unit
    dr = sectors["default_rate"].values          # default rate per sector
    hi = (dr >= profile["high_risk_threshold"]).astype(float)   # high-risk flag

    # Objective: minimise -expected_return (= maximise expected_return)
    c = -er

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []

    # Constraint 1: weighted default rate <= risk_ceiling
    #   dr @ x <= risk_ceiling
    A_ub.append(dr.tolist())
    b_ub.append(profile["risk_ceiling"])

    # Constraint 2: high-risk allocation <= max_high_risk_alloc
    #   hi @ x <= max_high_risk_alloc
    A_ub.append(hi.tolist())
    b_ub.append(profile["max_high_risk_alloc"])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Equality constraint: allocations sum to 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    # Bounds: min_alloc <= x_i <= max_alloc per sector
    bounds = [(profile["min_sector_alloc"], profile["max_sector_alloc"])] * n

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # Fallback: relax constraints slightly and retry
        print(f"  WARNING: LP infeasible — relaxing constraints and retrying...")
        relaxed_profile = profile.copy()
        relaxed_profile["risk_ceiling"]        *= 1.20
        relaxed_profile["max_high_risk_alloc"] *= 1.25
        relaxed_profile["min_sector_alloc"]     = 0.0

        A_ub[0] = dr.tolist()
        b_ub[0] = relaxed_profile["risk_ceiling"]
        A_ub[1] = hi.tolist()
        b_ub[1] = relaxed_profile["max_high_risk_alloc"]
        bounds   = [(0.0, relaxed_profile["max_sector_alloc"])] * n

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if not result.success:
            print(f"  ERROR: LP solver failed even with relaxed constraints: {result.message}")
            return None

    x = np.clip(result.x, 0, 1)
    x = x / x.sum()   # renormalise to ensure exact sum = 1

    allocations = x * budget
    portfolio_default_rate   = float(np.dot(dr, x))
    portfolio_expected_return = float(np.dot(er, x))
    total_expected_profit    = portfolio_expected_return * budget

    return {
        "fractions":                x,
        "allocations_usd":          allocations,
        "portfolio_default_rate":   round(portfolio_default_rate, 6),
        "portfolio_expected_return": round(portfolio_expected_return, 6),
        "total_expected_profit_usd": round(total_expected_profit, 2),
        "solver_status":            result.message,
    }


# ── Run all profiles ──────────────────────────────────────────────────────────

def run_optimisation(budget: float = 1_000_000.0,
                     african_only: bool = True) -> pd.DataFrame:
    """
    Runs LP for all three lender profiles and assembles results into
    a single DataFrame written to gold.db -> portfolio_allocations.
    """
    print("\n" + "=" * 65)
    print("  LOAN INTELLIGENCE — Stage 5: Portfolio Optimisation")
    print("=" * 65)
    print(f"\n  Budget:       ${budget:,.0f}")
    print(f"  Scope:        {'Africa Only' if african_only else 'All regions'}")

    sectors = load_sector_inputs(african_only=african_only)

    rows = []
    summary = []

    for profile_name, profile in PROFILES.items():
        print(f"\n  ── {profile_name} Profile ──")
        result = solve_lp(sectors, profile, budget)

        if result is None:
            print(f"  Skipping {profile_name} — solver failed.")
            continue

        print(f"  Portfolio default rate : {result['portfolio_default_rate']*100:.2f}%")
        print(f"  Expected return        : {result['portfolio_expected_return']*100:.2f}%")
        print(f"  Expected profit        : ${result['total_expected_profit_usd']:,.0f}")
        print(f"  {'Sector':<28} {'Alloc %':>8} {'Amount $':>12} {'E[Return]':>10}")
        print(f"  {'-'*62}")

        for i, (_, sector_row) in enumerate(sectors.iterrows()):
            frac   = result["fractions"][i]
            amount = result["allocations_usd"][i]
            er_val = sector_row["expected_return"]

            print(
                f"  {sector_row['sector']:<28} "
                f"{frac*100:>7.1f}%"
                f"  ${amount:>10,.0f}"
                f"  {er_val*100:>9.2f}%"
            )

            rows.append({
                "profile":              profile_name,
                "sector":               sector_row["sector"],
                "allocation_pct":       round(frac * 100, 2),
                "allocation_usd":       round(amount, 2),
                "expected_return_pct":  round(er_val * 100, 4),
                "default_rate_pct":     round(sector_row["default_rate"] * 100, 2),
                "roi_score":            round(sector_row["roi_score"], 4),
                "risk_tier":            sector_row["risk_tier"],
                "total_loans":          int(sector_row["total_loans"]),
                "avg_loan_amount":      round(sector_row["avg_loan_amount"], 2),
                "female_borrower_pct":  round(sector_row["female_borrower_pct"], 1),
                "budget":               budget,
            })

        summary.append({
            "profile":                  profile_name,
            "portfolio_default_rate_pct": round(result["portfolio_default_rate"] * 100, 2),
            "portfolio_expected_return_pct": round(result["portfolio_expected_return"] * 100, 2),
            "total_expected_profit_usd": result["total_expected_profit_usd"],
            "risk_ceiling_pct":         round(profile["risk_ceiling"] * 100, 2),
            "description":              profile["description"],
        })

    df_out = pd.DataFrame(rows)

    # Write to gold.db
    with get_conn(DB_GOLD) as conn:
        conn.execute("DROP TABLE IF EXISTS portfolio_allocations")
        bulk_insert(conn, "portfolio_allocations", df_out, if_exists="replace")
    print(f"\n  Saved -> gold.db : portfolio_allocations ({len(df_out)} rows)")

    # Save profile summary as JSON for dashboard
    summary_path = MODELS_DIR / "portfolio_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved -> models/artifacts/portfolio_summary.json")

    # Print comparison table
    print(f"\n  ── Profile Comparison ──")
    print(f"  {'Profile':<14} {'Risk':>8} {'E[Return]':>10} {'Profit':>14}")
    print(f"  {'-'*50}")
    for s in summary:
        print(
            f"  {s['profile']:<14}"
            f"  {s['portfolio_default_rate_pct']:>6.2f}%"
            f"  {s['portfolio_expected_return_pct']:>8.2f}%"
            f"  ${s['total_expected_profit_usd']:>12,.0f}"
        )

    return df_out


# ── Export for deployment ─────────────────────────────────────────────────────

def export_for_deploy():
    """
    Exports portfolio_allocations to parquet for Streamlit Cloud deployment.
    Run this after the optimisation to update the deploy artefacts.
    """
    deploy_dir = Path(__file__).parent.parent / "data" / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    with get_conn(DB_GOLD) as conn:
        df = pd.read_sql_query("SELECT * FROM portfolio_allocations", conn)

    out = deploy_dir / "portfolio_allocations.parquet"
    df.to_parquet(out, index=False)
    print(f"  Exported -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not SCIPY_OK:
        print("Install scipy first: pip install scipy")
        sys.exit(1)

    result_df = run_optimisation(budget=1_000_000.0, african_only=True)
    export_for_deploy()

    print(f"\n  Stage 5 Complete!")
    print(f"  gold.db -> portfolio_allocations ({len(result_df)} rows)")
    print(f"  data/deploy/ -> portfolio_allocations.parquet")
    print(f"\n  Next: restart the dashboard to see the Portfolio Optimisation page\n")

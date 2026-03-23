# Loan Intelligence & Portfolio Optimization

**African microfinance default prediction, investment intelligence, and LP-optimised capital allocation.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-intelligence-ag8omzawexviq5demp52ap.streamlit.app)

---

## What This Project Does

Most African microfinance borrowers have no formal credit history. This system solves three problems simultaneously:

1. **Predict** which loans are most likely to default using transfer learning from 1.3M labelled Lending Club loans applied to 671K Kiva microfinance loans
2. **Identify** which sectors and countries offer the best risk-adjusted investment opportunity
3. **Optimise** capital allocation across sectors using Linear Programming — given a budget, the system solves for the mathematically optimal distribution under Conservative, Balanced, or Aggressive risk constraints

---

## Live Demo

[loan-intelligence.streamlit.app](https://loan-intelligence-ag8omzawexviq5demp52ap.streamlit.app)

---

## Dashboard Pages

| Page | Business Question |
|---|---|
| Portfolio Overview | What is our overall portfolio health and model performance? |
| Investment Signals | Which sectors should we increase or reduce exposure to? |
| Portfolio Optimisation | Given our budget and risk philosophy, what is the optimal capital allocation? |
| Default Risk Map | Which African markets carry the most systemic risk? |
| Loan Predictor | What is the default risk on this specific loan application? |

Every page includes an **Ask AI** panel powered by the Anthropic Claude API for natural language querying of portfolio data.

---

## Architecture

```
Raw Data (3 sources)
    │
    ▼
Bronze Layer  ──  bronze.db
(raw ingestion)   kiva_loans_raw · worldbank_raw · lending_club_raw
    │
    ▼
Silver Layer  ──  silver.db
(clean + join)    lc_clean · kiva_clean · macro_clean · kiva_enriched
    │
    ▼
Gold Layer    ──  gold.db
(ML-ready)        lc_features · kiva_features · sector_performance
                  kiva_predictions · portfolio_allocations
    │
    ▼
Dashboard  ──  Streamlit (5 pages)
```

---

## Key Technical Decisions

**Transfer learning** — Kiva's public dataset has no repayment outcomes. The model trains on Lending Club's verified labels and scores Kiva loans using shared features (loan amount, term, sector, repayment interval).

**Probability recalibration** — Raw model predictions average 28.7% (US consumer credit norms). A linear scaling factor aligns the portfolio mean to ~8%, matching African MFI PAR30 industry benchmarks. The model is positioned as a relative risk ranker, not an absolute probability estimator.

**African MFI sector benchmarks** — Agriculture & Farming uses a 5.5% default rate (vs 18% in US Lending Club data) because group lending and seasonal repayment structures suppress defaults. All sector rates are sourced from MIX Market and CGAP microfinance literature.

**Sector-level LP over loan-level BIP** — Loan-level selection is a Binary Integer Programme with 671K variables (NP-hard at this scale). Sector-level LP solves globally optimally in milliseconds using SciPy's HiGHS solver and maps directly to how MFI investment committees think.

**Data leakage identified and resolved** — First model run produced AUC 0.9992 due to `repayment_rate` and `recovery_rate` being post-loan outcome features. Removing them corrected AUC to honest 0.7231.

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | XGBoost (beats LightGBM by 1.9 AUC points on this dataset) |
| AUC-ROC | 0.7231 |
| Training rows | 1,044,856 |
| Features | 37 (including interaction terms) |
| Calibration target | 8% mean (African MFI PAR30 benchmark) |
| Optimal threshold | 0.55 (maximises F1 on default class) |

AUC of 0.72 is consistent with published benchmarks for application-only credit scoring without credit bureau data. The ceiling would be pushed higher with mobile money transaction history, group cohesion scores, or utility payment records.

---

## Portfolio Optimisation

The LP is formulated as:

**Maximise:** `Σ E[R_i] × x_i`

**Subject to:**
- `Σ x_i = 1` — fully deploy capital
- `Σ default_rate_i × x_i ≤ risk_ceiling` — portfolio risk constraint
- `Σ x_i (high-risk sectors) ≤ max_high_risk_alloc` — concentration limit
- `min_alloc ≤ x_i ≤ max_alloc` — floor (social mission) and cap (diversification)

Expected return per sector = `interest_rate × (1 − default_rate) − default_rate × LGD`, where `LGD = 0.60` (group lending recovers ~40% of defaulted principal).

Three profiles — Conservative (6.5% risk ceiling), Balanced (8.2%), and Aggressive (10.5%) — each produce a distinct allocation from the same solver.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ETL & data | Pandas, SQLite, PyArrow / Parquet |
| Machine learning | XGBoost, LightGBM, SHAP, scikit-learn |
| Optimisation | SciPy (HiGHS LP solver) |
| Dashboard | Streamlit, Plotly |
| AI querying | Anthropic Claude API |
| Data sources | Kaggle API (Kiva + Lending Club), wbgapi (World Bank) |
| Deployment | Streamlit Community Cloud |

---

## Running Locally

```bash
git clone https://github.com/Takunda-Madondo/loan-intelligence.git
cd loan-intelligence

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
ANTHROPIC_API_KEY=sk-ant-...
```

Run the full pipeline:
```bash
python pipelines/01_ingest_kiva.py
python pipelines/02_ingest_worldbank.py
python pipelines/03_ingest_lending_club.py
python pipelines/04_etl_silver.py
python pipelines/05_feature_engineering.py
python pipelines/06_train_model.py
python pipelines/07_portfolio_optimisation.py

streamlit run app/main.py
```

---

## Data Sources

| Dataset | Source | Role |
|---|---|---|
| Kiva Microfinance Loans | [Kaggle](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) | 671K African microfinance loans — scored by model |
| Lending Club Loan Data | [Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) | 2.26M labelled US consumer loans — model training |
| World Bank Financial Inclusion | [databank.worldbank.org](https://databank.worldbank.org/source/global-financial-inclusion) | Macro enrichment — GDP, inflation, unemployment, poverty |

---

## Next Steps

- **Loan-level optimisation** — Two-stage approach: sector LP sets budgets, then greedy Sharpe-ratio selection within each sector. Alternatively, a pre-filtered Binary Integer Programme on ~30K candidate loans using PuLP with the CBC solver.
- **Richer features** — Mobile money transaction frequency, group cohesion scores, and utility payment history would push AUC above 0.85.
- **Tighter calibration** — Platt scaling or isotonic regression against real portfolio repayment data from a partner MFI would improve absolute probability accuracy.
- **Sector expansion** — The current 8-sector taxonomy could be extended with activity-level granularity from Kiva's `activity` field.

---

## Author

**Takunda Madondo**
Data Scientist / ML Engineer · Johannesburg, South Africa

[GitHub](https://github.com/Takunda-Madondo) · [LinkedIn](https://linkedin.com/in/takunda-madondo-649b66218/) · [Portfolio](https://takunda-madondo.github.io/Website)

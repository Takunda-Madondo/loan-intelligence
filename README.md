# 🏦 Loan Intelligence & Portfolio Optimization
### Predicting loan defaults and identifying high-return sectors in African microfinance markets

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## What This Project Does

Most African microfinance borrowers have no formal credit history. This system solves three problems simultaneously:

1. **Default Risk** — Which borrowers are likely to default, and why?
2. **Capital Allocation** — Which sectors (agriculture, retail, services) deliver the best return on lending?

This project builds a full data science solution — from raw data ingestion to a live dashboard — that answers both questions using real-world loan data.

**Target Users:** Credit risk teams, loan officers, MFI portfolio managers

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

## 📁 Project Structure

```
loan-intelligence/
├── data/
│   ├── raw/              # Original downloaded files (gitignored)
│   │   ├── kiva/
│   │   └── worldbank/
│   ├── bronze/           # Ingested, minimal processing
│   ├── silver/           # Cleaned, standardised, joined
│   └── gold/             # ML-ready feature tables
│
├── pipelines/            # Numbered scripts (run in order)
│   ├── 01_ingest_kiva.py
│   ├── 02_ingest_worldbank.py
│   ├── 03_etl_bronze.py
│   ├── 04_etl_silver.py
│   ├── 05_feature_engineering.py
│   ├── 06_train_default_model.py
│   └── 07_train_sector_model.py
│
├── notebooks/            # EDA and storytelling notebooks
│   ├── 01_eda_kiva.ipynb
│   ├── 02_eda_worldbank.ipynb
│   └── 03_model_explainability.ipynb
│
├── models/
│   ├── artifacts/        # Saved model files
│   └── mlflow/           # MLflow experiment logs
│
├── app/                  # Streamlit dashboard
│   ├── main.py
│   ├── pages/
│   └── components/
│
├── docs/                 # Audit reports, architecture diagrams
├── config.py             # Central config
├── requirements.txt
├── setup.sh
└── README.md
```

---

## 📈 Key Features

- **Default Prediction** — Upload any CSV of loan applicants, get default probabilities
- **Sector Dashboard** — Visual risk vs return across Agriculture, Retail, Services, etc.
- **Country Risk Map** — Choropleth of default rates enriched with World Bank macro data
- **NL Querying** — Ask plain-English questions: *"Which sector in Kenya has the lowest default rate?"*
- **SHAP Explainability** — Understand WHY a specific loan is flagged as high risk

---

## 🗺️ Development Stages

- [x] Stage 1 — Project setup & data ingestion
- [ ] Stage 2 — ETL pipeline (Bronze → Silver → Gold)
- [ ] Stage 3 — EDA & feature engineering
- [ ] Stage 4 — ML model training
- [ ] Stage 5 — Streamlit dashboard + NL querying
- [ ] Stage 6 — Deployment

---

*Built as a portfolio project demonstrating the full data science lifecycle.*

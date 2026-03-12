# 🏦 Loan Intelligence & Portfolio Optimization
### Predicting loan defaults and identifying high-return sectors in African microfinance markets

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-In%20Completed-green)

---

## 🎯 Business Problem

Microfinance institutions (MFIs) operating in Sub-Saharan Africa face two critical challenges:

1. **Default Risk** — Which borrowers are likely to default, and why?
2. **Capital Allocation** — Which sectors (agriculture, retail, services) deliver the best return on lending?

This project builds a full data science solution — from raw data ingestion to a live dashboard — that answers both questions using real-world loan data.

**Target Users:** Credit risk teams, loan officers, MFI portfolio managers

---

## 📊 Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Kiva Microfinance Loans | Kaggle | Core loan + repayment data |
| World Bank Financial Inclusion | World Bank API | Country-level macro enrichment |
| Lending Club (optional enrichment) | Kaggle | Feature engineering reference |

---

## 🏗️ Architecture

```
Raw Data (Kiva + World Bank)
        ↓
   Bronze Layer          ← Raw, ingested, minimal cleaning
        ↓
   Silver Layer          ← Cleaned, standardised, joined
        ↓
    Gold Layer           ← Feature-engineered, ML-ready
        ↓
   ML Models             ← Default classifier + Sector ROI scorer
        ↓
 Streamlit Dashboard     ← Risk/return insights + NL querying
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data Storage | DuckDB + Parquet |
| ETL | Python + Pandas/Polars |
| ML | XGBoost, scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit + Plotly |
| NL Querying | Anthropic Claude API |
| Hosting | Streamlit Community Cloud |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-intelligence.git
cd loan-intelligence

# 2. Run setup (creates venv, installs deps, copies .env)
bash setup.sh

# 3. Add your API keys to .env
# (Kaggle key from kaggle.com/settings, Anthropic key from console.anthropic.com)

# 4. Ingest data
python pipelines/01_ingest_kiva.py
python pipelines/02_ingest_worldbank.py

# 5. Run ETL pipeline
python pipelines/03_etl_bronze.py
python pipelines/04_etl_silver.py
python pipelines/05_feature_engineering.py

# 6. Train models
python pipelines/06_train_default_model.py
python pipelines/07_train_sector_model.py

# 7. Launch dashboard
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

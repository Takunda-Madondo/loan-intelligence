"""
pipelines/06_train_model.py
Stage 4 - Train improved default prediction model targeting AUC 0.85+

Improvements over baseline (0.716):
  1. Extended feature set - sub_grade (A1-G5), revol_bal, delinquency
     recency, bankcard utilisation, balance metrics
  2. Interaction features - dti*int_rate, grade*term, util*balance
  3. Hyperparameter tuning - RandomizedSearchCV over key XGBoost params
  4. LightGBM comparison - train both, keep the better one
  5. Threshold optimisation - find best decision threshold for business use

Run:
  python pipelines/06_train_model.py

Expected time: 15-25 minutes (tuning + two models on 1M+ rows)
"""

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_GOLD, DB_SILVER, MODELS_DIR, RANDOM_STATE, TEST_SIZE, TARGET_COL
from db import get_conn, bulk_insert

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import xgboost as xgb

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not installed - run: pip install lightgbm")

import shap

# Feature Definitions

# Base features from lc_features gold table
BASE_FEATURES = [
    "loan_amnt", "funded_ratio", "term_months", "int_rate",
    "installment", "grade_enc", "sub_grade_enc",
    "emp_length", "home_enc", "annual_inc", "verification_enc",
    "dti", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "num_bc_sats", "pct_tl_nvr_dlq",
    "num_tl_90g_dpd_24m", "avg_cur_bal", "bc_util",
    "num_rev_accts", "tot_cur_bal",
    "sector_enc", "loan_to_income", "payment_to_income",
]

# Interaction features - engineered during load
INTERACTION_FEATURES = [
    "dti_x_int_rate",       # high debt AND high rate = double risk
    "grade_x_term",         # risky grade AND long term = compounding risk
    "revol_util_x_bal",     # high utilisation AND high balance
    "inc_x_dti",            # income adjusted debt burden
    "inq_x_delinq",         # recent inquiries + past delinquency
]

ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES

# Kiva scoring columns -> mapped to model features
KIVA_FEATURE_COLS = [
    "loan_amount", "funded_ratio", "term_in_months",
    "sector_default_rate", "loan_per_month", "sector_enc",
    "lender_count", "country_enc", "macro_risk_score",
    "is_female_borrower", "gdp_growth", "inflation_rate",
    "unemployment_rate", "poverty_rate", "domestic_credit_pct",
    "lender_trust_score", "repayment_interval_enc",
]


# 1. Load & Build Features

def load_and_engineer():
    print("\nSTEP 1 - Loading & Engineering Features")

    # Pull from silver lc_clean - has all the new columns
    with get_conn(DB_SILVER) as conn:
        df = pd.read_sql_query("SELECT * FROM lc_clean", conn)
    print(f"  Loaded {len(df):,} rows from lc_clean")
    print(f"  Available columns: {len(df.columns)}")

    # Re-parse any string columns
    for col in ["int_rate", "revol_util", "bc_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%","").str.strip(),
                errors="coerce"
            )

    if "term_months" not in df.columns and "term" in df.columns:
        df["term_months"] = pd.to_numeric(
            df["term"].astype(str).str.strip().str.split().str[0],
            errors="coerce"
        )

    # Encodings
    GRADE_MAP = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    df["grade_enc"] = df["grade"].map(GRADE_MAP).fillna(4)

    # sub_grade A1=1 ... G5=35
    sub_grade_order = [f"{g}{n}" for g in "ABCDEFG" for n in range(1,6)]
    sub_grade_map   = {sg: i+1 for i, sg in enumerate(sub_grade_order)}
    if "sub_grade_enc" not in df.columns:
        if "sub_grade" in df.columns:
            df["sub_grade_enc"] = df["sub_grade"].map(sub_grade_map).fillna(
                df["grade_enc"] * 3
            )
        else:
            df["sub_grade_enc"] = df["grade_enc"] * 3

    cat_encode = lambda s: pd.factorize(s.fillna("Unknown"))[0]
    df["home_enc"]         = cat_encode(df["home_ownership"])
    df["verification_enc"] = cat_encode(df["verification_status"])
    df["sector_enc"]       = cat_encode(df["sector"])

    # Derived features
    df["funded_ratio"]      = (df["funded_amnt"] / df["loan_amnt"].clip(1)).clip(0,1)
    df["loan_to_income"]    = (df["loan_amnt"] / df["annual_inc"].clip(1)).round(4)
    df["payment_to_income"] = (df["installment"] / (df["annual_inc"].clip(1)/12)).round(4)

    # Interaction features
    df["dti_x_int_rate"]   = (df["dti"]       * df["int_rate"]).round(3)
    df["grade_x_term"]     = (df["grade_enc"]  * df["term_months"]).round(3)
    df["inc_x_dti"]        = (df["annual_inc"] / df["dti"].clip(1)).round(2)
    df["inq_x_delinq"]     = (df["inq_last_6mths"] * (df["delinq_2yrs"] + 1)).round(3)

    if "revol_bal" in df.columns and "revol_util" in df.columns:
        df["revol_util_x_bal"] = (df["revol_util"] * np.log1p(df["revol_bal"])).round(3)
    else:
        df["revol_util_x_bal"] = df.get("revol_util", 0) * 10

    # Fill nulls - medians only, no dropping
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  Features not in data (will skip): {missing}")

    for col in available_features:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop only where target is null
    before = len(df)
    df = df[df["defaulted"].notna() & df["loan_amnt"].notna()]
    print(f"  Removed {before-len(df):,} rows with null target")
    print(f"  Final: {len(df):,} rows  |  {len(available_features)} features")
    print(f"  Default rate: {df['defaulted'].mean()*100:.1f}%")

    return df, available_features


# 2. Train / Test Split

def split_data(df, features):
    X = df[features].values
    y = df["defaulted"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"  scale_pos_weight: {spw:.2f}")
    return X_train, X_test, y_train, y_test, spw


# 3. XGBoost with Hyperparameter Tuning

def train_xgboost_tuned(X_train, y_train, spw, features):
    print("\nSTEP 2 - XGBoost with Hyperparameter Tuning")

    param_grid = {
        "n_estimators":     [300, 500],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.2],
        "reg_alpha":        [0, 0.1, 0.5],
        "reg_lambda":       [1, 1.5, 2],
    }

    base_model = xgb.XGBClassifier(
        scale_pos_weight  = spw,
        random_state      = RANDOM_STATE,
        eval_metric       = "auc",
        n_jobs            = -1,
        verbosity         = 0,
    )

    print(f"  Running RandomizedSearchCV (20 iterations, 3-fold CV)...")
    print(f"      This takes ~10-15 minutes...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions = param_grid,
        n_iter              = 20,
        scoring             = "roc_auc",
        cv                  = cv,
        random_state        = RANDOM_STATE,
        n_jobs              = 1,      # XGBoost already uses all cores internally
        verbose             = 2,
        refit               = True,
    )

    # Use a sample for tuning speed - 300K rows is enough to find good params
    sample_size = min(300_000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    search.fit(X_train[idx], y_train[idx])

    print(f"\n  Best CV AUC: {search.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in search.best_params_.items():
        print(f"      {k}: {v}")

    # Retrain best model on full training set
    print(f"\n  Retraining best config on full {len(X_train):,} rows...")
    best_params = search.best_params_.copy()
    best_params.update({
        "scale_pos_weight": spw,
        "random_state":     RANDOM_STATE,
        "eval_metric":      "auc",
        "n_jobs":           -1,
        "verbosity":        1,
        "early_stopping_rounds": 20,
    })

    final_xgb = xgb.XGBClassifier(**best_params)
    # Split off 10% of train as internal validation for early stopping
    Xt, Xv, yt, yv = train_test_split(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train
    )
    final_xgb.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=50)
    print(f"  Best iteration: {final_xgb.best_iteration}")

    return final_xgb, search.best_params_


# 4. LightGBM

def train_lightgbm(X_train, y_train, spw):
    if not LGBM_AVAILABLE:
        print("\n  Skipping LightGBM - not installed")
        return None

    print("\nSTEP 3 - LightGBM")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        is_unbalance      = True,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )

    print(f"  Training LightGBM on {len(X_train):,} rows...")
    Xt, Xv, yt, yv = train_test_split(
        X_train, y_train, test_size=0.1,
        random_state=RANDOM_STATE, stratify=y_train
    )
    callbacks = [lgb.early_stopping(20, verbose=False),
                 lgb.log_evaluation(50)]
    model.fit(
        Xt, yt,
        eval_set=[(Xv, yv)],
        callbacks=callbacks,
    )
    print(f"  LightGBM training complete")
    return model


# 5. Evaluate & Compare

def evaluate(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)

    # Find optimal threshold (maximise F1 for default class)
    thresholds = np.arange(0.2, 0.7, 0.05)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

    y_pred = (y_prob >= best_thresh).astype(int)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"\n  {name} Results:")
    print(f"      AUC-ROC:             {auc:.4f}")
    print(f"      Avg Precision:       {ap:.4f}")
    print(f"      Optimal threshold:   {best_thresh:.2f}")
    print(f"      False Positive Rate: {fpr*100:.1f}%")
    print(f"      False Negative Rate: {fnr*100:.1f}%")
    print(f"\n  {classification_report(y_test, y_pred, target_names=['Repaid','Default'])}")

    return {
        "model_name":          name,
        "auc_roc":             round(float(auc), 4),
        "avg_precision":       round(float(ap), 4),
        "optimal_threshold":   round(float(best_thresh), 2),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
    }, y_prob


# 6. SHAP

def compute_shap(model, X_test, features):
    print("\nSTEP 5 - SHAP Feature Importance")
    print("  Computing SHAP on 5,000 samples...")

    idx         = np.random.choice(len(X_test), size=5000, replace=False)
    X_sample    = X_test[idx]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM returns list for binary - take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap  = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature":          features[:len(mean_shap)],
        "shap_importance":  mean_shap.round(4),
    }).sort_values("shap_importance", ascending=False)

    print(f"\n  Top 15 Features:")
    print(f"  {'Feature':<28} SHAP Importance")
    print(f"  {'-'*28} {'-'*15}")
    for _, row in importance.head(15).iterrows():
        bar = "=" * int(row["shap_importance"] * 60)
        print(f"  {row['feature']:<28} {row['shap_importance']:.4f}  {bar}")

    path = MODELS_DIR / "shap_importance.json"
    importance.to_json(path, orient="records", indent=2)
    print(f"\n  Saved -> {path}")
    return importance


# 7. Score Kiva

def score_kiva_loans(model, threshold, features):
    print("\nSTEP 6 - Scoring 671K Kiva Loans")

    with get_conn(DB_GOLD) as conn:
        kiva = pd.read_sql_query("SELECT * FROM kiva_features", conn)
    # Drop duplicate columns (loan_amount appears in both feature cols and display cols)
    kiva = kiva.loc[:, ~kiva.columns.duplicated()]
    kiva = kiva.reset_index(drop=True)
    print(f"  Loaded {len(kiva):,} Kiva loans")

    n = len(kiva)

    # Extract all columns as numpy arrays FIRST - prevents any pandas
    # index alignment errors when combining columns
    loan_amount       = kiva["loan_amount"].to_numpy(dtype=float)
    funded_ratio      = kiva["funded_ratio"].to_numpy(dtype=float)
    term_months       = kiva["term_in_months"].to_numpy(dtype=float)
    sector_dr         = kiva["sector_default_rate"].to_numpy(dtype=float)
    loan_per_month    = kiva["loan_per_month"].to_numpy(dtype=float)
    lender_count      = kiva["lender_count"].to_numpy(dtype=float)
    country_enc       = kiva["country_enc"].to_numpy(dtype=float)
    macro_risk        = kiva["macro_risk_score"].to_numpy(dtype=float)
    is_female         = kiva["is_female_borrower"].to_numpy(dtype=float)
    inflation         = kiva["inflation_rate"].to_numpy(dtype=float)
    unemployment      = kiva["unemployment_rate"].to_numpy(dtype=float)
    poverty           = kiva["poverty_rate"].to_numpy(dtype=float)
    sector_enc_arr    = kiva["sector_enc"].to_numpy(dtype=float)

    raw = {
        "loan_amnt":               loan_amount,
        "funded_ratio":            funded_ratio,
        "term_months":             term_months,
        "int_rate":                sector_dr * 100,
        "installment":             loan_per_month,
        "grade_enc":               np.clip(sector_dr * 7, 1, 7),
        "sub_grade_enc":           np.clip(sector_dr * 28, 1, 35),
        "emp_length":              np.clip(lender_count, 0, 10),
        "home_enc":                country_enc % 4,
        "annual_inc":              loan_amount * 12,
        "verification_enc":        is_female,
        "dti":                     macro_risk * 40,
        "delinq_2yrs":             (inflation > 10).astype(float),
        "inq_last_6mths":          np.clip(unemployment / 5, 0, 5).round(0),
        "mths_since_last_delinq":  poverty * 2,
        "mths_since_last_record":  poverty * 3,
        "open_acc":                np.clip(lender_count, 0, 30),
        "pub_rec":                 (poverty > 40).astype(float),
        "revol_bal":               loan_amount,
        "revol_util":              macro_risk * 80,
        "total_acc":               np.clip(lender_count, 1, 50),
        "mort_acc":                np.zeros(n),
        "num_bc_sats":             np.clip(lender_count, 0, 20),
        "pct_tl_nvr_dlq":         (1 - sector_dr) * 100,
        "num_tl_90g_dpd_24m":      (inflation > 15).astype(float),
        "avg_cur_bal":             loan_amount,
        "bc_util":                 macro_risk * 60,
        "num_rev_accts":           np.clip(lender_count, 0, 20),
        "tot_cur_bal":             loan_amount,
        "sector_enc":              sector_enc_arr,
        "loan_to_income":          np.full(n, 1/12),
        "payment_to_income":       loan_per_month / np.clip(loan_amount, 1, None),
        # Interactions - all pure numpy, no pandas alignment
        "dti_x_int_rate":          macro_risk * 40 * sector_dr * 100,
        "grade_x_term":            sector_dr * 7 * term_months,
        "revol_util_x_bal":        macro_risk * 80 * np.log1p(loan_amount),
        "inc_x_dti":               loan_amount * 12 / np.clip(macro_risk * 40, 1, None),
        "inq_x_delinq":            np.clip(unemployment / 5, 0, 5) * (inflation > 10).astype(float),
    }

    X_kiva = pd.DataFrame({f: raw[f] for f in features if f in raw}).fillna(0)

    # Ensure column order matches training
    X_kiva = X_kiva.reindex(columns=features, fill_value=0)

    print(f"  Predicting...")
    proba_raw = model.predict_proba(X_kiva.values)[:, 1]

    # Calibration: rescale to African MFI benchmark
    # Raw model trained on Lending Club (20% default rate).
    # African microfinance PAR30 benchmark: ~8%.
    # We rescale so the portfolio mean aligns with 8%, preserving
    # relative rank ordering (used as risk ranker, not absolute estimator).
    TARGET_MEAN = 0.08
    raw_mean    = proba_raw.mean()
    scale       = TARGET_MEAN / raw_mean
    proba       = (proba_raw * scale).clip(0, 1).round(4)
    print(f"  Calibration: raw mean {raw_mean:.4f} -> scaled mean {proba.mean():.4f} (target {TARGET_MEAN})")

    # Risk bands aligned to African MFI norms
    # Low: < 5% (healthy PAR30)  Medium: 5-12%  High: 12-20%  Very High: > 20%
    results = pd.DataFrame({
        "loan_id":             kiva["id"].values,
        "raw_score":           proba_raw.round(4),
        "risk_score":          proba,
        "default_probability": proba,          # kept for backward compat
        "predicted_default":   (proba >= 0.12).astype(int),
        "risk_band":           pd.cut(
            proba,
            bins=[0, 0.05, 0.12, 0.20, 1.0],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True
        ).astype(str),
        "sector":              kiva["sector_standardised"].values,
        "country_code":        kiva["country_code"].values,
        "country":             kiva["country"].values,
        "loan_amount":         kiva["loan_amount"].values,
        "activity":            kiva["activity"].values,
        "disbursed_year":      kiva["disbursed_year"].values,
    })

    print(f"\n  Risk Band Distribution (calibrated to African MFI norms):")
    for band, cnt in results["risk_band"].value_counts().sort_index().items():
        pct = cnt / len(results) * 100
        bar = "=" * int(pct / 2)
        print(f"    {band:<12} {cnt:>8,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Avg Risk Score - African Countries:")
    african = results[results["country_code"].isin([
        "KE","UG","TZ","RW","ET","GH","NG","MZ","ZM","MW","MG","CM","SN"
    ])]
    risk = african.groupby("country")["risk_score"].mean().sort_values().round(3)
    for country, prob in risk.items():
        bar = "=" * int(prob * 100)
        print(f"    {country:<20} {prob:.3f}  {bar}")

    with get_conn(DB_GOLD) as conn:
        conn.execute("DROP TABLE IF EXISTS kiva_predictions")
        bulk_insert(conn, "kiva_predictions", results, if_exists="replace")
    print(f"\n  Saved -> gold.db : kiva_predictions ({len(results):,} rows)")
    return results


# 8. Save

def save_all(model, metrics, features, best_params=None):
    # Model
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Try native save
    try:
        model.save_model(str(MODELS_DIR / "best_model.json"))
    except Exception:
        pass

    # Feature list - critical for dashboard to build correct input
    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump(features, f, indent=2)

    # Metrics
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if best_params:
        with open(MODELS_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

    print(f"\n  Saved -> models/artifacts/best_model.pkl")
    print(f"  Saved -> models/artifacts/feature_list.json")
    print(f"  Saved -> models/artifacts/model_metrics.json")


# Main

if __name__ == "__main__":
    print("\nLOAN INTELLIGENCE - Stage 4: Improved Model Training\n")

    # 1. Load and engineer features
    df, features = load_and_engineer()
    X_train, X_test, y_train, y_test, spw = split_data(df, features)

    # 2. Train XGBoost with tuning
    xgb_model, best_params = train_xgboost_tuned(X_train, y_train, spw, features)
    xgb_metrics, xgb_prob  = evaluate(xgb_model, X_test, y_test, "XGBoost (Tuned)")

    # 3. Train LightGBM
    lgb_model = train_lightgbm(X_train, y_train, spw)
    if lgb_model:
        lgb_metrics, lgb_prob = evaluate(lgb_model, X_test, y_test, "LightGBM")
    else:
        lgb_metrics = {"auc_roc": 0}

    # 4. Pick winner
    print("\nSTEP 4 - Model Comparison")
    print(f"  XGBoost AUC:  {xgb_metrics['auc_roc']}")
    print(f"  LightGBM AUC: {lgb_metrics['auc_roc']}")

    if lgb_model and lgb_metrics["auc_roc"] > xgb_metrics["auc_roc"]:
        best_model   = lgb_model
        best_metrics = lgb_metrics
        print(f"  Winner: LightGBM ({lgb_metrics['auc_roc']})")
    else:
        best_model   = xgb_model
        best_metrics = xgb_metrics
        print(f"  Winner: XGBoost ({xgb_metrics['auc_roc']})")

    # 5. SHAP
    shap_imp = compute_shap(best_model, X_test, features)

    # 6. Score Kiva
    threshold   = best_metrics["optimal_threshold"]
    predictions = score_kiva_loans(best_model, threshold, features)

    # 7. Save everything
    save_all(best_model, best_metrics, features, best_params)

    print(f"\n  Stage 4 Complete!")
    print(f"  Best Model AUC-ROC:  {best_metrics['auc_roc']}")
    print(f"  Avg Precision:       {best_metrics['avg_precision']}")
    print(f"  Optimal Threshold:   {best_metrics['optimal_threshold']}")
    print(f"\n  Artifacts saved to models/artifacts/")
    print(f"  Kiva predictions -> gold.db : kiva_predictions")
    print(f"\n  Next: python app/main.py  <- Dashboard!\n")

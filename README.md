# 💳 EMIPredict AI — Intelligent Financial Risk Assessment Platform

> Dual-stage ML platform that predicts EMI eligibility and calculates maximum safe loan amounts in real time.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

Millions of loan applicants face rejection or default not because they are dishonest, but because financial institutions lack fast, accurate tools to assess real repayment capacity. Manual underwriting is slow, inconsistent, and expensive.

This project builds **EMIPredict AI** — a production-ready FinTech platform that solves two interconnected problems simultaneously:

- **Should this applicant get a loan?** → Classification: Eligible / High Risk / Not Eligible
- **How much EMI can they safely afford?** → Regression: Maximum monthly EMI in ₹

Both answers are delivered instantly using trained XGBoost models through an interactive Streamlit web application.

---

## 🎯 Problem Statement

Banks and lending platforms need a system that can:

- Automate eligibility screening across thousands of applications per day
- Calculate safe EMI limits based on a borrower's complete financial profile
- Reduce manual underwriting time without sacrificing accuracy
- Support 5 real-world EMI scenarios: E-commerce, Home Appliances, Vehicle, Personal Loan, Education

---

## 🏆 Final Model Results

### Stage 1 — Classification: EMI Eligibility (The Gatekeeper)

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 85.58% | — |
| Random Forest | 95.27% | — |
| **XGBoost ★ (Selected)** | **97.99%** | **0.9963** |

**Why XGBoost won:** XGBoost builds trees sequentially — each new tree corrects the residual errors left by the previous one. This boosting mechanism handles class imbalance natively by adjusting sample weights tree-by-tree, which is critical when "Not Eligible" cases are rare but consequential. Its ROC-AUC of 0.9963 means near-perfect separation between safe and risky applicants across all decision thresholds.

### Stage 2 — Regression: Maximum EMI Amount (The Financial Advisor)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | Baseline | — | — |
| Random Forest Regressor | — | — | — |
| **XGBoost Regressor ★ (Selected)** | **0.9899** | **₹780.73** | **₹365.00** |

**Why XGBoost won:** Real EMI capacity is non-linear — a small income increase for someone with no dependents matters far more than the same increase for someone with 4 dependents and an existing home loan. XGBoost captures these non-linear interactions without requiring manual transformation, producing an R² of 0.9899 (explains 99% of variance in EMI limits).

**Why RMSE was prioritized over MAE:** RMSE squares errors before averaging, so a single ₹5,000 overestimation hurts the score far more than ten ₹300 errors. This forces the model to be conservative — critical in lending, where overestimating EMI capacity directly causes loan defaults. Our RMSE of ₹780.73 confirms the model avoids dangerous large errors.

---

## 🗂️ Project Structure

```
EMI PROJECT/
│
├── app/
│   ├── pages/
│   │   ├── 1_Prediction.py       # Live eligibility + EMI prediction interface
│   │   └── 2_Data_Explorer.py    # Dataset distribution and analytics
│   └── app.py                    # Main Streamlit dashboard entry point
│
├── models/
│   ├── classification_model.pkl  # Trained XGBoost classifier
│   ├── label_encoder.pkl         # Serialized LabelEncoder for target classes
│   └── regression_model.pkl      # Trained XGBoost regressor
│
├── src/
│   ├── data_cleaning.py          # Missing value handling and type validation
│   ├── feature_engineering.py    # Winsorization and ratio extraction
│   ├── mlflow_tracking.py        # Experiment logging infrastructure
│   └── model_training.py         # Hyperparameter tuning and model training
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Name:** EMI_dataset  
**Source:** Synthetically generated from realistic financial profiles across 5 Indian lending categories

| Property | Detail |
|----------|--------|
| Total Records | 400,000 financial profiles |
| Input Features | 22 variables |
| Target Variables | 2 (Classification + Regression) |
| EMI Scenarios | 5 lending categories |

### EMI Scenario Distribution

| Scenario | Records | Loan Range | Tenure |
|----------|---------|------------|--------|
| E-commerce Shopping | 80,000 | ₹10K–₹2L | 3–24 months |
| Home Appliances | 80,000 | ₹20K–₹3L | 6–36 months |
| Vehicle | 80,000 | ₹80K–₹15L | 12–84 months |
| Personal Loan | 80,000 | ₹50K–₹10L | 12–60 months |
| Education | 80,000 | ₹50K–₹5L | 6–48 months |

---

## 🔢 Input Features (22 Variables)

**Personal Demographics:** `age`, `gender`, `marital_status`, `education`

**Employment & Income:** `monthly_salary`, `employment_type`, `years_of_employment`, `company_type`

**Housing & Family:** `house_type`, `monthly_rent`, `family_size`, `dependents`

**Monthly Obligations:** `school_fees`, `college_fees`, `travel_expenses`, `groceries_utilities`, `other_monthly_expenses`

**Financial Status:** `existing_loans`, `current_emi_amount`, `credit_score`, `bank_balance`, `emergency_fund`

**Loan Application:** `emi_scenario`, `requested_amount`, `requested_tenure`

### Target Variables

| Task | Variable | Values |
|------|----------|--------|
| Classification | `emi_eligibility` | Eligible / High_Risk / Not_Eligible |
| Regression | `max_monthly_emi` | ₹500 – ₹50,000 (continuous) |

---

## ⚙️ Feature Engineering Decisions

### Why Winsorization instead of removing outliers?
Removing outliers discards real applicant records — a person earning ₹2L/month is extreme but valid. Winsorization caps extreme values at the 1st and 99th percentile tails, preserving the record while stopping anomalous values from distorting model convergence on the majority population.

### Why LabelEncoder instead of One-Hot Encoding?
One-Hot Encoding explodes high-cardinality categorical columns into dozens of binary columns, creating the **Curse of Dimensionality** — a sparse matrix mostly full of zeros that inflates memory, slows training, and causes tree models to overfit on rare noise patterns. LabelEncoder keeps the data dense (one column, integer values), which is exactly what gradient-boosted trees expect. The `label_encoder.pkl` is serialized separately to guarantee consistent category-to-integer translation when the model runs on new data in production.

### Derived Financial Ratios
Key engineered features include debt-to-income ratio, expense-to-income ratio, and affordability ratios — capturing the relationship between what an applicant earns and what they already owe, which raw features alone cannot express.

---

## 🛠️ Techniques Used

| Technique | Why It Was Used |
|-----------|----------------|
| **Winsorization** | Cap extreme outliers at 1st/99th percentile without discarding valid records |
| **LabelEncoder** | Compact integer encoding for categorical targets — avoids sparse matrix from One-Hot |
| **MLflow Tracking** | Logs parameters, metrics, and model artifacts for every run — enables reproducible comparison |
| **Hyperparameter Tuning** | `max_depth`, `learning_rate`, `n_estimators` optimized per model to prevent overfitting |
| **Train/Validation/Test Split** | Separate evaluation sets prevent data leakage and test true generalization |

---

## 📈 Evaluation Metrics

### Classification
| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correct prediction rate |
| Precision | Of applicants flagged eligible, how many actually are |
| Recall | Of truly eligible applicants, how many we correctly approved |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Separation ability across all decision thresholds |

### Regression
| Metric | What It Measures |
|--------|-----------------|
| R² Score | Proportion of EMI variance explained by the model |
| RMSE ⭐ | Penalizes large errors heavily — critical to avoid overestimating EMI capacity |
| MAE | Average absolute error — confirms typical day-to-day variance |

---

## 🖥️ Application Features

| Feature | Description |
|---------|-------------|
| EMI Eligibility Prediction | Instant Eligible / High Risk / Not Eligible classification |
| Max EMI Calculation | Safe monthly EMI limit with real-time regression output |
| Data Explorer | Distribution plots and analytics across all 5 EMI scenarios |
| Multi-page Dashboard | Clean Streamlit UI separating prediction from analytics |

---

## 🚀 Getting Started

**Step 1 — Clone the repository:**
```bash
git clone [YOUR_GITHUB_URL]
cd emi-predict-ai
```

**Step 2 — Create and activate virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Run data cleaning:**
```bash
python src/data_cleaning.py
```

**Step 5 — Run feature engineering:**
```bash
python src/feature_engineering.py
```

**Step 6 — Train models:**
```bash
python src/model_training.py
```

**Step 7 — Launch the app:**
```bash
streamlit run app/app.py
```

Open browser at `http://localhost:8501`

---

## 🧪 MLflow Experiment Tracking

All model runs are tracked with parameters, metrics, and saved artifacts:

```bash
mlflow ui
```

Open browser at `http://localhost:5000`

Tracked per run: model name, hyperparameters (`max_depth`, `learning_rate`, `n_estimators`), accuracy/R²/RMSE/ROC-AUC, and serialized model binaries.

---

## ☁️ Cloud Deployment

The application is deployed on **Streamlit Cloud**.

**Live URL:** `[YOUR_STREAMLIT_URL]`

Every push to the `main` branch triggers automatic redeployment via GitHub integration.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| ML Models | XGBoost, Scikit-learn (Random Forest, Logistic Regression) |
| Experiment Tracking | MLflow |
| Web Application | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |
| Data Processing | Pandas, NumPy |

---

## ⚠️ Limitations

- Dataset is synthetically generated — real-world performance may differ on live bank data
- 22 features cover financial profile well, but exclude behavioral signals (payment history patterns, spending categories) used in production credit scoring
- Classification uses LabelEncoder which assumes ordinal relationship between classes — a minor theoretical limitation for multi-class problems
- Model retraining is manual; no automated retraining pipeline on data drift

---

## 📋 Deliverables

- [x] Data Cleaning Pipeline
- [x] Feature Engineering Module
- [x] 3+ Classification Models with MLflow tracking
- [x] 3+ Regression Models with MLflow tracking
- [x] Best Model Selection with justification
- [x] Serialized Models (.pkl)
- [x] Multi-page Streamlit Application
- [x] Streamlit Cloud Deployment
- [x] README Documentation

---

## 👤 Author

**Gokul**  
GUVI x HCL Tech Certification Program — 2026  
Domain: FinTech and Banking / Financial Risk Assessment

---

> ⭐ If you found this project useful, give it a star on GitHub!

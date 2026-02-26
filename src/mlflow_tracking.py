# ==========================================================
# EMI_Predict_AI - MLflow Tracking Script
# Step 4 & Step 5 - Fully Completed Version
# ==========================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("data/featured_EMI_dataset.csv")
print("Dataset Shape:", df.shape)

# ----------------------------------------------------------
# 2. Encode Target for Classification
# ----------------------------------------------------------

label_encoder = LabelEncoder()
df["emi_eligibility_encoded"] = label_encoder.fit_transform(df["emi_eligibility"])

# ----------------------------------------------------------
# 3. Split Features and Targets
# ----------------------------------------------------------

X = df.drop(["emi_eligibility", "emi_eligibility_encoded", "max_monthly_emi"], axis=1)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

y_class = df["emi_eligibility_encoded"]
y_reg = df["max_monthly_emi"]

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 4. Start MLflow Experiment
# ----------------------------------------------------------

mlflow.set_experiment("EMI_Predict_AI")

with mlflow.start_run():

    # ======================================================
    # CLASSIFICATION MODELS
    # ======================================================

    print("\n--- Classification Models ---")

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_class_train)
    y_pred_log = log_model.predict(X_test)
    acc_log = accuracy_score(y_class_test, y_pred_log)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_class_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_class_test, y_pred_rf)

    # XGBoost
    xgb_model = XGBClassifier(eval_metric="mlogloss")
    xgb_model.fit(X_train, y_class_train)
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_class_test, y_pred_xgb)

    # ROC-AUC (Multi-class)
    y_proba = xgb_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_class_test, y_proba, multi_class="ovr")

    print("Logistic Accuracy:", acc_log)
    print("Random Forest Accuracy:", acc_rf)
    print("XGBoost Accuracy:", acc_xgb)
    print("ROC-AUC:", roc_auc)

    # Log classification metrics
    mlflow.log_metric("logistic_accuracy", acc_log)
    mlflow.log_metric("rf_accuracy", acc_rf)
    mlflow.log_metric("xgb_accuracy", acc_xgb)
    mlflow.log_metric("classification_roc_auc", roc_auc)

    # ======================================================
    # REGRESSION MODELS
    # ======================================================

    print("\n--- Regression Models ---")

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_reg_train)
    y_pred_lin = lin_model.predict(X_test)

    rmse_lin = np.sqrt(mean_squared_error(y_reg_test, y_pred_lin))
    mae_lin = mean_absolute_error(y_reg_test, y_pred_lin)
    r2_lin = r2_score(y_reg_test, y_pred_lin)

    # Random Forest
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_reg_train)
    y_pred_rf_reg = rf_reg.predict(X_test)

    rmse_rf = np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg))
    mae_rf = mean_absolute_error(y_reg_test, y_pred_rf_reg)
    r2_rf = r2_score(y_reg_test, y_pred_rf_reg)

    # XGBoost Regressor
    xgb_reg = XGBRegressor()
    xgb_reg.fit(X_train, y_reg_train)
    y_pred_xgb_reg = xgb_reg.predict(X_test)

    rmse_xgb = np.sqrt(mean_squared_error(y_reg_test, y_pred_xgb_reg))
    mae_xgb = mean_absolute_error(y_reg_test, y_pred_xgb_reg)
    r2_xgb = r2_score(y_reg_test, y_pred_xgb_reg)

    mape_xgb = np.mean(
        np.abs((y_reg_test - y_pred_xgb_reg) / y_reg_test)
    ) * 100

    print("Linear RMSE:", rmse_lin)
    print("RF RMSE:", rmse_rf)
    print("XGB RMSE:", rmse_xgb)
    print("MAPE:", mape_xgb)

    # Log regression metrics
    mlflow.log_metric("linear_rmse", rmse_lin)
    mlflow.log_metric("rf_rmse", rmse_rf)
    mlflow.log_metric("xgb_rmse", rmse_xgb)

    mlflow.log_metric("linear_r2", r2_lin)
    mlflow.log_metric("rf_r2", r2_rf)
    mlflow.log_metric("xgb_r2", r2_xgb)

    mlflow.log_metric("regression_mae", mae_xgb)
    mlflow.log_metric("regression_mape", mape_xgb)

    # ======================================================
    # MODEL REGISTRY (BEST MODELS)
    # ======================================================

    mlflow.sklearn.log_model(
        xgb_model,
        artifact_path="xgb_classifier_model",
        registered_model_name="EMI_Eligibility_Model"
    )

    mlflow.sklearn.log_model(
        xgb_reg,
        artifact_path="xgb_regressor_model",
        registered_model_name="EMI_Max_EMI_Model"
    )

print("\nMLflow Tracking Complete.")
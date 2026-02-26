# 3_Model_Monitoring.py

import streamlit as st                     # I am using streamlit for dashboard UI
import plotly.express as px                # I am using plotly for charts
import pandas as pd                        # I am using pandas to structure metric tables

st.title("Model Performance Monitoring Dashboard")  
# This page shows final selected model performance metrics

st.subheader("Selected Production Models")

st.write("Classification Model: XGBoost Classifier")  
st.write("Regression Model: XGBoost Regressor")

# -------------------------------------------------
# Classification Metrics (From Final Training)
# -------------------------------------------------

st.subheader("Classification Performance")

classification_metrics = {
    "Metric": ["Accuracy", "Macro F1 Score", "High Risk Recall", "ROC AUC"],
    "Value": [0.979, 0.91, 0.71, 0.996]
}
# These are final metrics obtained during model evaluation

df_class = pd.DataFrame(classification_metrics)  
# I am converting metrics to dataframe for display

st.dataframe(df_class)  
# I am showing classification metrics in table format

fig1 = px.bar(df_class, x="Metric", y="Value", color="Metric", text="Value")  
# I am creating bar chart for classification performance

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------
# Regression Metrics (From Final Training)
# -------------------------------------------------

st.subheader("Regression Performance")

regression_metrics = {
    "Metric": ["RMSE", "R2 Score"],
    "Value": [773.69, 0.9899]
}
# These are final regression metrics

df_reg = pd.DataFrame(regression_metrics)  
# I am converting regression metrics into dataframe

st.dataframe(df_reg)  
# I am showing regression metrics table

fig2 = px.bar(df_reg, x="Metric", y="Value", color="Metric", text="Value")  
# I am creating bar chart for regression metrics

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# Model Comparison Summary
# -------------------------------------------------

st.subheader("Why XGBoost Was Selected")

st.write("""
I compared Logistic Regression, Random Forest, and XGBoost for classification.
XGBoost achieved highest accuracy and significantly improved minority class recall.

For regression, XGBoost achieved the lowest RMSE and highest R2 value,
explaining nearly 99 percent of variance in maximum EMI prediction.

Therefore, XGBoost was selected as the final production model for both tasks.
""")
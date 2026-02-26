# 2_Data_Explorer.py

import streamlit as st                     # I am using streamlit for UI
import pandas as pd                        # I am using pandas to load and manage dataset
import plotly.express as px                # I am using plotly for interactive charts
import os                                  # I am using os for correct file path

st.title("Interactive Data Exploration Dashboard")   # Page title

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  
# I am getting project root folder

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "featured_EMI_dataset.csv")  
# I am defining dataset path

@st.cache_data                             # I am caching dataset to improve performance
def load_data():
    df = pd.read_csv(DATA_PATH)            # I am loading featured dataset
    return df

df = load_data()                           # I am calling dataset

st.subheader("Dataset Overview")

st.write("Dataset Shape:", df.shape)       # I am showing dataset size
st.dataframe(df.head())                    # I am showing first few records

# -------------------------------------------------
# Filter Section
# -------------------------------------------------

st.sidebar.header("Filter Data")

selected_scenario = st.sidebar.multiselect(
    "Select EMI Scenario",
    options=df["emi_scenario"].unique(),
    default=df["emi_scenario"].unique()
)
# I am allowing user to filter by EMI type

filtered_df = df[df["emi_scenario"].isin(selected_scenario)]
# I am applying filter to dataset

# -------------------------------------------------
# EMI Scenario Distribution
# -------------------------------------------------

st.subheader("EMI Scenario Distribution")

fig1 = px.histogram(filtered_df, x="emi_scenario", color="emi_scenario")
# I am plotting EMI scenario distribution

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------
# Eligibility Distribution
# -------------------------------------------------

st.subheader("Eligibility Distribution")

fig2 = px.histogram(filtered_df, x="emi_eligibility", color="emi_eligibility")
# I am plotting EMI eligibility class distribution

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# Credit Score Distribution
# -------------------------------------------------

st.subheader("Credit Score Distribution")

fig3 = px.histogram(filtered_df, x="credit_score", nbins=30)
# I am plotting credit score distribution

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# Salary vs Maximum EMI Relationship
# -------------------------------------------------

st.subheader("Salary vs Maximum EMI")

fig4 = px.scatter(filtered_df,
                  x="monthly_salary",
                  y="max_monthly_emi",
                  color="emi_eligibility")
# I am analyzing relationship between salary and max EMI

st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
# I am selecting only numeric columns

correlation_matrix = numeric_df.corr()
# I am computing correlation matrix

fig5 = px.imshow(correlation_matrix,
                 text_auto=True,
                 aspect="auto",
                 color_continuous_scale="RdBu_r")
# I am creating heatmap for correlation

st.plotly_chart(fig5, use_container_width=True)
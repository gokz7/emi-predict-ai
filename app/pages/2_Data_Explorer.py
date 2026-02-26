# 2_Data_Explorer.py

import streamlit as st                     # I am using streamlit for UI
import pandas as pd                        # I am using pandas to load and manage dataset
import plotly.express as px                # I am using plotly for interactive charts
import os                                  # I am using os for file handling
import gdown                               # I am using gdown to download large file from Google Drive


st.title("Interactive Data Exploration Dashboard")   # Page title


# -------------------------------------------------
# Google Drive File Configuration
# -------------------------------------------------

FILE_ID = "1WvcuKXSXpN_oOGJlpqVNZyeEQ3PdqF-n"   # This is my Google Drive file ID
DOWNLOAD_PATH = "temp_dataset.csv"             # I am saving downloaded file locally


# -------------------------------------------------
# Load Dataset (with gdown support)
# -------------------------------------------------

@st.cache_data
def load_data():

    # I am checking if file already downloaded
    if not os.path.exists(DOWNLOAD_PATH):

        url = f"https://drive.google.com/uc?id={FILE_ID}"   # I am forming download URL

        gdown.download(url, DOWNLOAD_PATH, quiet=False)     # I am downloading dataset

    df = pd.read_csv(DOWNLOAD_PATH)                         # I am reading CSV file
    return df


df = load_data()                                            # I am calling dataset


# -------------------------------------------------
# Dataset Overview
# -------------------------------------------------

st.subheader("Dataset Overview")

st.write("Dataset Shape:", df.shape)        # I am showing dataset size
st.dataframe(df.head())                     # I am showing first few records


# -------------------------------------------------
# Filter Section
# -------------------------------------------------

st.sidebar.header("Filter Data")

selected_scenario = st.sidebar.multiselect(
    "Select EMI Scenario",
    options=df["emi_scenario"].unique(),
    default=df["emi_scenario"].unique()
)

filtered_df = df[df["emi_scenario"].isin(selected_scenario)]


# -------------------------------------------------
# EMI Scenario Distribution
# -------------------------------------------------

st.subheader("EMI Scenario Distribution")

fig1 = px.histogram(filtered_df, x="emi_scenario", color="emi_scenario")

st.plotly_chart(fig1, use_container_width=True)


# -------------------------------------------------
# Eligibility Distribution
# -------------------------------------------------

st.subheader("Eligibility Distribution")

fig2 = px.histogram(filtered_df, x="emi_eligibility", color="emi_eligibility")

st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------
# Credit Score Distribution
# -------------------------------------------------

st.subheader("Credit Score Distribution")

fig3 = px.histogram(filtered_df, x="credit_score", nbins=30)

st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------------------
# Salary vs Maximum EMI Relationship
# -------------------------------------------------

st.subheader("Salary vs Maximum EMI")

fig4 = px.scatter(
    filtered_df,
    x="monthly_salary",
    y="max_monthly_emi",
    color="emi_eligibility"
)

st.plotly_chart(fig4, use_container_width=True)


# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])

correlation_matrix = numeric_df.corr()

fig5 = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig5, use_container_width=True)
# 4_Admin.py

# 4_Admin.py

import streamlit as st
import pandas as pd
import os
import gdown


st.title("Administrative Data Management Panel")


# -------------------------------------------------
# Google Drive Dataset Configuration
# -------------------------------------------------

FILE_ID = "1WvcuKXSXpN_oOGJlpqVNZyeEQ3PdqF-n"
DOWNLOAD_PATH = "temp_dataset.csv"


@st.cache_data
def load_data():

    if not os.path.exists(DOWNLOAD_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, DOWNLOAD_PATH, quiet=False, fuzzy=True)

    df = pd.read_csv(DOWNLOAD_PATH)
    return df


# -------------------------------------------------
# Initialize Session Dataset
# -------------------------------------------------

if "admin_df" not in st.session_state:
    st.session_state.admin_df = load_data()


df = st.session_state.admin_df


# -------------------------------------------------
# Show Dataset
# -------------------------------------------------

st.subheader("Current Dataset Records")

st.write("Dataset Shape:", df.shape)
st.dataframe(df.head(20))


# -------------------------------------------------
# Add New Record
# -------------------------------------------------

st.subheader("Add New Record")

with st.form("add_record_form"):

    age = st.number_input("Age", 18, 80)
    monthly_salary = st.number_input("Monthly Salary", 10000, 1000000)
    credit_score = st.number_input("Credit Score", 300, 850)
    requested_amount = st.number_input("Requested Loan Amount", 10000, 5000000)
    requested_tenure = st.number_input("Requested Tenure (Months)", 3, 240)

    submit_button = st.form_submit_button("Add Record")

    if submit_button:

        new_row = {
            "age": age,
            "monthly_salary": monthly_salary,
            "credit_score": credit_score,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure
        }

        new_df = pd.DataFrame([new_row])

        st.session_state.admin_df = pd.concat(
            [st.session_state.admin_df, new_df],
            ignore_index=True
        )

        st.success("New record added successfully.")


# -------------------------------------------------
# Delete Record
# -------------------------------------------------

st.subheader("Delete Record")

if len(st.session_state.admin_df) > 0:

    delete_index = st.number_input(
        "Enter Row Index to Delete",
        0,
        len(st.session_state.admin_df) - 1
    )

    if st.button("Delete Record"):

        st.session_state.admin_df = st.session_state.admin_df.drop(
            index=delete_index
        ).reset_index(drop=True)

        st.success("Record deleted successfully.")
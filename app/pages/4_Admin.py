# 4_Admin.py

import streamlit as st                          # I am using streamlit for admin UI
import pandas as pd                             # I am using pandas to manage dataset
import os                                       # I am using os to handle file paths

st.title("Administrative Data Management Panel")  
# This page allows managing dataset records

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  
# I am getting project root path

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "featured_EMI_dataset.csv")  
# I am defining dataset path

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)                 # I am loading dataset
    return df

df = load_data()

st.subheader("Current Dataset Records")

st.write("Dataset Shape:", df.shape)           # I am showing current dataset size
st.dataframe(df.head(20))                      # I am showing first 20 rows

# -------------------------------------------------
# Add New Record
# -------------------------------------------------

st.subheader("Add New Record")

with st.form("add_record_form"):

    age = st.number_input("Age", 25, 60)
    monthly_salary = st.number_input("Monthly Salary", 15000, 300000)
    credit_score = st.number_input("Credit Score", 300, 850)
    requested_amount = st.number_input("Requested Loan Amount", 10000, 2000000)
    requested_tenure = st.number_input("Requested Tenure", 3, 120)

    submit_button = st.form_submit_button("Add Record")

    if submit_button:

        new_row = {
            "age": age,
            "monthly_salary": monthly_salary,
            "credit_score": credit_score,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure
        }
        # I am creating a new row dictionary

        new_df = pd.DataFrame([new_row])        # I am converting it to dataframe

        df_updated = pd.concat([df, new_df], ignore_index=True)
        # I am appending new record to existing dataset

        df_updated.to_csv(DATA_PATH, index=False)
        # I am saving updated dataset back to CSV file

        st.success("New record added successfully.")
        st.cache_data.clear()                   # I am clearing cache to refresh dataset

# -------------------------------------------------
# Delete Record
# -------------------------------------------------

st.subheader("Delete Record")

delete_index = st.number_input("Enter Row Index to Delete", 0, len(df)-1)

if st.button("Delete Record"):

    df_dropped = df.drop(index=delete_index)
    # I am removing selected row

    df_dropped.to_csv(DATA_PATH, index=False)
    # I am saving updated dataset

    st.success("Record deleted successfully.")
    st.cache_data.clear()                      # I am refreshing dataset
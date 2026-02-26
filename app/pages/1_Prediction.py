# 1_Prediction.py

import streamlit as st                     # I am using streamlit to build my web application interface
import pandas as pd                        # I am using pandas to create dataframe for prediction
import joblib                              # I am using joblib to load my saved models
import os                                  # I am using os to correctly handle file paths

st.title("EMI Eligibility & Maximum EMI Prediction")   # This is my prediction page title

# -------------------------------------------------
# Load Saved Models and Label Encoder
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  
# Here I am getting my main project root directory

MODEL_PATH = os.path.join(PROJECT_ROOT, "models")  
# Here I am defining the models folder path

@st.cache_resource                          # I am caching models so they load only once
def load_models():
    clf_model = joblib.load(os.path.join(MODEL_PATH, "classification_model.pkl"))  
    # I am loading my saved XGBoost classification pipeline

    reg_model = joblib.load(os.path.join(MODEL_PATH, "regression_model.pkl"))      
    # I am loading my saved XGBoost regression pipeline

    label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))     
    # I am loading label encoder to decode classification output

    return clf_model, reg_model, label_encoder

clf_model, reg_model, label_encoder = load_models()  
# Here I am loading all required components

st.subheader("Enter Financial Details")     # This section collects user input

# -------------------------------------------------
# Personal Information
# -------------------------------------------------

age = st.number_input("Age", 25, 60)  
# I am collecting age between 25 and 60

gender = st.selectbox("Gender", ["Male", "Female"])  
# I am collecting gender

marital_status = st.selectbox("Marital Status", ["Single", "Married"])  
# I am collecting marital status

education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])  
# I am collecting education level

# -------------------------------------------------
# Employment & Income
# -------------------------------------------------

monthly_salary = st.number_input("Monthly Salary", 15000, 300000)  
# I am collecting monthly income

employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])  
# I am collecting employment category

years_of_employment = st.number_input("Years of Employment", 0, 40)  
# I am collecting work experience

company_type = st.selectbox("Company Type", ["Small", "Medium", "Large"])  
# I am collecting company type

# -------------------------------------------------
# Housing & Family
# -------------------------------------------------

house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])  
# I am collecting housing type

monthly_rent = st.number_input("Monthly Rent", 0, 100000)  
# I am collecting rent amount

family_size = st.number_input("Family Size", 1, 10)  
# I am collecting total family members

dependents = st.number_input("Dependents", 0, 10)  
# I am collecting number of dependents

# -------------------------------------------------
# Monthly Expenses
# -------------------------------------------------

school_fees = st.number_input("School Fees", 0, 50000)  
college_fees = st.number_input("College Fees", 0, 100000)  
travel_expenses = st.number_input("Travel Expenses", 0, 50000)  
groceries_utilities = st.number_input("Groceries & Utilities", 0, 100000)  
other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 100000)  
# I am collecting all expense components

# -------------------------------------------------
# Financial Status
# -------------------------------------------------

existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])  
# I am checking whether customer has existing loans

current_emi_amount = st.number_input("Current EMI Amount", 0, 50000)  
# I am collecting current EMI burden

credit_score = st.number_input("Credit Score", 300, 850)  
# I am collecting credit score

bank_balance = st.number_input("Bank Balance", 0, 10000000)  
# I am collecting bank balance

emergency_fund = st.number_input("Emergency Fund", 0, 1000000)  
# I am collecting emergency savings

# -------------------------------------------------
# Loan Application Details
# -------------------------------------------------

emi_scenario = st.selectbox("EMI Scenario",
    ["E-commerce Shopping EMI", "Home Appliances EMI",
     "Vehicle EMI", "Personal Loan EMI", "Education EMI"])  
# I am selecting EMI category

requested_amount = st.number_input("Requested Loan Amount", 10000, 2000000)  
# I am collecting requested loan amount

requested_tenure = st.number_input("Requested Tenure (Months)", 3, 120)  
# I am collecting loan tenure in months

# -------------------------------------------------
# Feature Engineering (Same Logic Used During Training)
# -------------------------------------------------

total_monthly_expenses = (
    monthly_rent + school_fees + college_fees +
    travel_expenses + groceries_utilities + other_monthly_expenses
)  
# I am calculating total monthly expenses

debt_to_income_ratio = current_emi_amount / monthly_salary if monthly_salary > 0 else 0  
# I am calculating debt to income ratio

expense_to_income_ratio = total_monthly_expenses / monthly_salary if monthly_salary > 0 else 0  
# I am calculating expense to income ratio

emi_burden_ratio = (current_emi_amount + requested_amount / requested_tenure) / monthly_salary if monthly_salary > 0 else 0  
# I am calculating total EMI burden ratio

savings_ratio = emergency_fund / monthly_salary if monthly_salary > 0 else 0  
# I am calculating savings ratio

financial_stability_score = (credit_score / 850) * (1 - debt_to_income_ratio)  
# I am calculating financial stability score

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------

if st.button("Predict"):  
    # When user clicks predict, I prepare complete dataframe

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "total_monthly_expenses": total_monthly_expenses,
        "debt_to_income_ratio": debt_to_income_ratio,
        "expense_to_income_ratio": expense_to_income_ratio,
        "emi_burden_ratio": emi_burden_ratio,
        "savings_ratio": savings_ratio,
        "financial_stability_score": financial_stability_score
    }])

    eligibility_encoded = clf_model.predict(input_data)[0]  
    # I am getting numeric prediction

    eligibility = label_encoder.inverse_transform([eligibility_encoded])[0]  
    # I am converting numeric output back to original class label

    max_emi = reg_model.predict(input_data)[0]  
    # I am predicting maximum safe EMI

    st.success(f"EMI Eligibility: {eligibility}")  
    # I am displaying final eligibility result

    st.info(f"Maximum Safe EMI: ₹ {round(max_emi, 2)}")  
    # I am displaying predicted EMI amount
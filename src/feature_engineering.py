import pandas as pd  # I am using pandas to handle dataset

# ----------------------------
# 1. Load Cleaned Dataset
# ----------------------------
df = pd.read_csv("data/cleaned_EMI_dataset.csv")  
# I am loading the cleaned dataset prepared in Phase 1

print("Loaded Shape:", df.shape)


# ----------------------------
# 2. Create Total Monthly Expenses
# ----------------------------
df['total_monthly_expenses'] = (
    df['monthly_rent'] +
    df['school_fees'] +
    df['college_fees'] +
    df['travel_expenses'] +
    df['groceries_utilities'] +
    df['other_monthly_expenses'] +
    df['current_emi_amount']
)
# I am calculating total financial outflow per month


# ----------------------------
# 3. Debt to Income Ratio
# ----------------------------
df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary']
# This shows what percentage of income is already going toward EMI


# ----------------------------
# 4. Expense to Income Ratio
# ----------------------------
df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary']
# This shows total financial burden compared to income


# ----------------------------
# 5. EMI Burden Ratio
# ----------------------------
df['emi_burden_ratio'] = df['requested_amount'] / (df['monthly_salary'] * df['requested_tenure'])
# This estimates pressure of requested EMI relative to income


# ----------------------------
# 6. Savings Ratio
# ----------------------------
df['savings_ratio'] = df['bank_balance'] / df['monthly_salary']
# This measures liquidity strength relative to monthly income


# ----------------------------
# 7. Financial Stability Score
# ----------------------------
df['financial_stability_score'] = (
    df['credit_score'] * 0.4 +
    (1 - df['debt_to_income_ratio']) * 0.3 +
    df['savings_ratio'] * 0.3
)
# I am creating a composite stability score combining credit strength, low debt, and savings


print("New Shape After Feature Engineering:", df.shape)

# 8. Outlier Capping (Winsorization)
# ----------------------------

ratio_columns = [
    'debt_to_income_ratio',
    'expense_to_income_ratio',
    'emi_burden_ratio',
    'savings_ratio'
]
# These ratio features may contain extreme outliers

for col in ratio_columns:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
# I am capping extreme 1% lower and upper values to reduce outlier distortion


# ----------------------------
# 8. Outlier Capping (Winsorization)
# ----------------------------

ratio_columns = [
    'debt_to_income_ratio',
    'expense_to_income_ratio',
    'emi_burden_ratio',
    'savings_ratio'
]
# These ratio features may contain extreme outliers

for col in ratio_columns:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
# I am capping extreme 1% lower and upper values to reduce outlier distortion



# ----------------------------
# 8. Save Engineered Dataset
# ----------------------------
df.to_csv("data/featured_EMI_dataset.csv", index=False)
# I am saving dataset with new engineered features

print("Feature engineered dataset saved successfully.")


#
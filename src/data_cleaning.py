import pandas as pd                                  # I am using pandas for handling dataset
from sklearn.model_selection import train_test_split # I need this to split the dataset for training

# ----------------------------
# 1. Load Dataset Properly
# ----------------------------
df = pd.read_csv("data/EMI_dataset.csv", low_memory=False)  
# I am loading the dataset and setting low_memory=False to avoid mixed datatype warnings

print("Original Shape:", df.shape)  
# This shows how many rows and columns are present before cleaning


# ----------------------------
# 2. Convert Numeric Columns
# ----------------------------
numeric_cols = [
    'age',
    'monthly_salary',
    'years_of_employment',
    'monthly_rent',
    'family_size',
    'dependents',
    'school_fees',
    'college_fees',
    'travel_expenses',
    'groceries_utilities',
    'other_monthly_expenses',
    'current_emi_amount',
    'credit_score',
    'bank_balance',
    'emergency_fund',
    'requested_amount',
    'requested_tenure',
    'max_monthly_emi'
]
# These columns should be numeric but some may have been stored as text

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  
# I am converting all these columns to numeric, and if any invalid values exist,
# they will automatically become NaN so I can handle them properly later

print("\nNumeric Conversion Done")


# ----------------------------
# 3. Remove Duplicates
# ----------------------------
df = df.drop_duplicates()  
# I am removing duplicate rows to ensure clean data without repetition


# ----------------------------
# 4. Handle Missing Values
# ----------------------------

df['education'] = df['education'].fillna(df['education'].mode()[0])  
# If education is missing, I am filling it with the most common category (mode)

df['monthly_rent'] = df['monthly_rent'].fillna(0)  
# If monthly rent is missing, I assume the person may own a house, so I fill 0

df['monthly_salary'] = df['monthly_salary'].fillna(df['monthly_salary'].median())  
# Salary is critical, so I fill missing values with median to avoid bias

df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())  
# Credit score missing values are replaced with median for stability

df['bank_balance'] = df['bank_balance'].fillna(df['bank_balance'].median())  
# Bank balance missing values are filled with median to maintain distribution

df['emergency_fund'] = df['emergency_fund'].fillna(df['emergency_fund'].median())  
# Emergency fund missing values are also replaced using median

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())  
# If any numeric column still has missing values, I fill it with its median

print("\nMissing After Cleaning:\n", df.isnull().sum())  
# This confirms that all missing values are handled


# ----------------------------
# 5. Validate Ranges
# ----------------------------
df = df[(df['age'] >= 25) & (df['age'] <= 60)]  
# I am filtering valid age range based on project specification

df = df[(df['credit_score'] >= 300) & (df['credit_score'] <= 850)]  
# I am ensuring credit score stays within realistic financial limits

df = df[df['monthly_salary'] > 0]  
# Salary must be positive, so I remove invalid entries

print("\nAfter Range Validation Shape:", df.shape)  
# This shows how many rows remain after validation

#Standardize Categorical Columns
# ----------------------------

df['gender'] = df['gender'].str.strip().str.lower()  
# I am removing extra spaces and converting gender to lowercase for consistency

df['gender'] = df['gender'].replace({
    'f': 'female',
    'm': 'male'
})  
# I am standardizing short forms into full labels to avoid duplicate categories


# ----------------------------
# 6. Train / Validation / Test Split
# ----------------------------

X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])  
# I separate features from target variables

y_class = df['emi_eligibility']  
# This is my classification target

y_reg = df['max_monthly_emi']  
# This is my regression target

X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_class, y_reg,
    test_size=0.3,
    random_state=42,
    stratify=y_class
)
# I am splitting 70% training and 30% temporary data
# I use stratify=y_class to maintain class distribution because dataset is imbalanced

X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_class_temp
)
# Now I split the remaining 30% equally into validation and test sets (15% each)

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)
# This confirms proper dataset splitting


# ----------------------------
# 7. Save Cleaned Dataset
# ----------------------------
df.to_csv("data/cleaned_EMI_dataset.csv", index=False)  
# I am saving the cleaned dataset so I can use it for EDA and modeling

print("\nCleaned dataset saved successfully.")
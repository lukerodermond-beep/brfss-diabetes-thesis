"""
Create final modeling dataset for Master Thesis
BRFSS 2023 – Diabetes classification

This script:
- Loads raw BRFSS data
- Recodes the target variable
- Recodes all predictors
- Handles missing values
- Saves ONE final modeling dataset

Output:
- modeling_dataset.csv
"""

import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_FILE = "LLCP2023.XPT"

print("Loading BRFSS data...")
df = pd.read_sas(DATA_FILE, format="xport")
df.columns = [str(c).upper() for c in df.columns]

print("Initial shape:", df.shape)

# --------------------------------------------------
# 2. TARGET VARIABLE (DIABETE4)
# --------------------------------------------------
# 1 = Diabetes
# 3 = No diabetes
# Other categories excluded

df["TARGET_BIN"] = df["DIABETE4"].replace({1: 1, 3: 0})
df["TARGET_BIN"] = df["TARGET_BIN"].where(df["TARGET_BIN"].isin([0, 1]))

# Drop rows without valid target
df = df.dropna(subset=["TARGET_BIN"])

print("After target filtering:", df.shape)

# --------------------------------------------------
# 3. PREDICTOR RECODING
# --------------------------------------------------

# AGE (_AGE80) – already numeric
df["AGE"] = df["_AGE80"]

# SEX (SEXVAR)
df["SEX"] = df["SEXVAR"].map({1: "Male", 2: "Female"})

# EDUCATION (EDUCA) – keep categorical
df["EDUCATION"] = df["EDUCA"]

# INCOME (INCOME3) – DK/Refused → Missing
df["INCOME"] = df["INCOME3"].replace({77: np.nan, 99: np.nan})

# SMOKING STATUS (_SMOKER3)
df["SMOKING"] = df["_SMOKER3"].replace({9: np.nan})

# PHYSICAL ACTIVITY (_TOTINDA)
df["PHYS_ACTIVITY"] = df["_TOTINDA"].replace({9: np.nan})

# --------------------------------------------------
# BMI RECODING (_BMI5 → BMI categories)
# --------------------------------------------------

# Convert BMI back to real values
df["BMI"] = df["_BMI5"].replace({9999: np.nan}) / 100

# Create BMI categories (WHO-style, very standard)
def bmi_category(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["BMI_CAT"] = df["BMI"].apply(bmi_category)


# HYPERTENSION (BPHIGH6)
# 2 = pregnancy only → exclude
df["HYPERTENSION"] = df["BPHIGH6"].replace({
    2: np.nan,
    7: np.nan,
    9: np.nan
})

# CHOLESTEROL (TOLDHI3)
df["CHOLESTEROL"] = df["TOLDHI3"].replace({
    7: np.nan,
    9: np.nan
})

# --------------------------------------------------
# 4. ALCOHOL RECODING (ALCDAY4)
# --------------------------------------------------

def recode_alcohol(x):
    if pd.isna(x):
        return np.nan
    if x in [777, 888, 999]:
        return np.nan
    if x == 0:
        return "0 days"
    if 100 <= x <= 199:
        return "Monthly"
    if 200 <= x <= 299:
        return "Weekly"
    return "Other"

df["ALCOHOL_FREQ"] = df["ALCDAY4"].apply(recode_alcohol)

# --------------------------------------------------
# 5. FINAL MODELING DATASET
# --------------------------------------------------

FINAL_VARS = [
    "TARGET_BIN",
    "AGE",
    "SEX",
    "EDUCATION",
    "INCOME",
    "SMOKING",
    "PHYS_ACTIVITY",
    "BMI_CAT",
    "HYPERTENSION",
    "CHOLESTEROL",
    "ALCOHOL_FREQ"
]


model_df = df[FINAL_VARS].copy()

# --------------------------------------------------
# 6. HANDLE MISSING VALUES
# --------------------------------------------------
# Simple & defensible choice: drop rows with missing predictors

model_df = model_df.dropna()
print("\nMissing values per column AFTER dropna():")
print(model_df.isna().sum())


print("Final modeling dataset shape:", model_df.shape)
print("\nTarget distribution:")
print(model_df["TARGET_BIN"].value_counts())

# --------------------------------------------------
# 7. SAVE DATASET
# --------------------------------------------------

model_df.to_csv("modeling_dataset.csv", index=False)


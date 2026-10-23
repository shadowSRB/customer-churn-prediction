import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Basic inspection
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nChurn distribution:")
print(df["Churn"].value_counts(dropna=False))

print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Data quality audit

print("\n--- DATA QUALITY AUDIT ---")

# 1. Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Check duplicate customer IDs
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# 3. Check blank strings in TotalCharges
blank_charges = df["TotalCharges"].str.strip().eq("").sum()
print("Blank TotalCharges values:", blank_charges)

# 4. Convert TotalCharges temporarily to numeric
charges_numeric = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

print(
    "TotalCharges values that cannot be converted:",
    charges_numeric.isna().sum()
)

# 5. Check target values
print("\nUnique Churn values:", df["Churn"].unique())

# 6. Check categorical columns
categorical_cols = df.select_dtypes(
    include=["object", "str"]
).columns

print("\nUnique values in categorical columns:")

for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# Identify rows with blank TotalCharges
blank_mask = df["TotalCharges"].str.strip().eq("")

print("\nCustomers with blank TotalCharges:")
print(
    df.loc[
        blank_mask,
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\nTenure values for these customers:")
print(df.loc[blank_mask, "tenure"].value_counts(dropna=False))

# Convert TotalCharges into a numeric column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df["TotalCharges"].isna().sum())

# Verify the condition before filling
blank_rows = df["TotalCharges"].isna()

if df.loc[blank_rows, "tenure"].eq(0).all():
    df.loc[blank_rows, "TotalCharges"] = 0
else:
    print("Some rows need further investigation.")

print("\nFinal TotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nRemaining missing TotalCharges:")
print(df["TotalCharges"].isna().sum())

from pathlib import Path

# Create a separate output folder
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Convert blank charges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype("string").str.strip(),
    errors="coerce"
)

# Fill only confirmed zero-tenure cases
mask = df["TotalCharges"].isna()

if df.loc[mask, "tenure"].eq(0).all():
    df.loc[mask, "TotalCharges"] = 0.0
else:
    raise ValueError(
        "Some missing TotalCharges values need investigation."
    )

# Validate
assert df["TotalCharges"].isna().sum() == 0

# Save cleaned data
df.to_csv("data/processed/telco_churn_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Shape:", df.shape)
print("TotalCharges type:", df["TotalCharges"].dtype)
print("Remaining missing values:", df.isna().sum().sum())

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_clean_data():
    """Load the raw dataset and clean TotalCharges."""
    data_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(data_path)

    # Convert blank strings to numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip(),
        errors="coerce",
    )

    # In this dataset, missing TotalCharges values belong
    # to customers whose tenure is zero.
    missing = df["TotalCharges"].isna()

    if not df.loc[missing, "tenure"].eq(0).all():
        raise ValueError(
            "Missing TotalCharges values need investigation."
        )

    df.loc[missing, "TotalCharges"] = 0.0

    return df


def prepare_features(df):
    """Separate model inputs from the churn target."""
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in Churn.")

    return X, y


def build_preprocessor(X):
    """Build transformations for numeric and categorical columns."""
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return preprocessor

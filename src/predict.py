
from pathlib import Path

import joblib
import pandas as pd


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the saved model
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """
    Predict whether a customer is likely to churn.

    customer_data: dictionary containing the customer's features.
    """
    customer_df = pd.DataFrame([customer_data])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0, 1])

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    # Example customer; values are illustrative, not a real customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2,
    }

    result = predict_churn(sample_customer)

    print("Customer Churn Prediction")
    print("-------------------------")
    print("Prediction:", result["prediction"])
    print("Churn probability:", result["churn_probability"])

# Customer Churn Prediction

## Project Overview
This project uses machine learning to predict customer churn for a telecommunications company. It explores customer behavior, compares classification models, and provides a reusable Python script for making churn predictions.

## Objectives
- Analyze customer churn patterns.
- Explore the relationships between churn, contract type, tenure, and monthly charges.
- Train and evaluate Logistic Regression and Random Forest classifiers.
- Save a trained model and use it for individual predictions.

## Technology Stack
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Machine Learning Workflow
1. Load and inspect the dataset.
2. Clean `TotalCharges` and handle blank values.
3. Perform exploratory data analysis.
4. Separate features and target.
5. Split the dataset into training and testing sets.
6. Encode categorical variables and scale numeric features using a preprocessing pipeline.
7. Train and compare classification models.
8. Save the selected model.
9. Predict churn using `src/predict.py`.

## Models Evaluated
- Logistic Regression
- Random Forest Classifier

## Initial Evaluation Results
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 |
| Random Forest | 0.770 | 0.559 | 0.634 | 0.594 | 0.823 |

These are results from the current train/test split and classification threshold. Model selection should also consider the business cost of missed churners and unnecessary retention offers.

## Project Structure
```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
└── src/
    └── predict.py
```

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run a Prediction
From the project root:

```bash
python src/predict.py
```

## Dataset
The project uses the IBM Telco Customer Churn sample dataset. See the original dataset source and license/usage information before redistributing it.

## Limitations
- Predictions are estimates and are not guarantees of future behavior.
- Results depend on the dataset, features, split, and model settings.
- Further validation and threshold tuning are required before real business use.

## Future Improvements
- Tune model hyperparameters.
- Optimize the classification threshold.
- Add a web interface or API.
- Monitor performance on new customer data.

## Disclaimer
This is an educational machine learning project, not a production-validated churn prediction system.

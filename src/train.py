
from pathlib import Path
import json

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import (
    load_clean_data,
    prepare_features,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(model, X_test, y_test):
    """Calculate classification evaluation metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "F1 Score": round(
            f1_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "ROC-AUC": round(
            roc_auc_score(y_test, probabilities), 4
        ),
    }


def main():
    df = load_clean_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = {}
    best_model = None
    best_score = -1

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        print(f"\n{name}")
        print("-" * len(name))
        for metric, score in metrics.items():
            print(f"{metric}: {score}")

        # Select using ROC-AUC for this initial comparison.
        if metrics["ROC-AUC"] > best_score:
            best_score = metrics["ROC-AUC"]
            best_model = pipeline

    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_model,
        models_dir / "churn_model.joblib",
    )

    with open(
        reports_dir / "model_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\nTraining completed successfully!")
    print("Selected model: highest ROC-AUC on the test set.")
    print("Saved model:", models_dir / "churn_model.joblib")
    print(
        "Saved metrics:",
        reports_dir / "model_comparison.json",
    )


if __name__ == "__main__":
    main()

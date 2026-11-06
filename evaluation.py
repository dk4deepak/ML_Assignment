import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "default payment next month"
FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0"]
    + [f"PAY_{month}" for month in range(2, 7)]
    + [f"BILL_AMT{month}" for month in range(1, 7)]
    + [f"PAY_AMT{month}" for month in range(1, 7)]
)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def validate_data(frame):
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = [column for column in FEATURES + [TARGET] if column not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    try:
        numeric = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("All features and the target must contain numeric values.") from error
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing and infinite values are not allowed.")
    if not numeric[TARGET].isin([0, 1]).all():
        raise ValueError("The target must be 0 (no default) or 1 (default).")
    return numeric[FEATURES], numeric[TARGET].astype(int)


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(features)[:, positive_index]
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(target, scores)) if target.nunique() == 2 else None,
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1": float(f1_score(target, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    return metrics, predictions, scores
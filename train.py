import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import MODEL_FILES, TARGET, evaluate_model, validate_data

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def update_readme(metadata):
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return
    results = metadata["metrics"]
    columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    table = ["| ML Model Name | " + " | ".join(columns) + " |", "|---|---|---|---|---|---|---|"]
    for name, metrics in results.items():
        table.append("| " + name + " | " + " | ".join(f"{metrics[column]:.4f}" for column in columns) + " |")
    winner = max(results, key=lambda name: results[name]["MCC"])
    explanations = {
        "Logistic Regression": "Scaled linear baseline; may miss nonlinear payment-history relationships.",
        "Decision Tree": "Depth-limited nonlinear rules; leaf-size regularization reduces overfitting.",
        "KNN": "Scaled distance-based classifier; neighborhood voting may favor the majority class.",
        "Naive Bayes": "Gaussian conditional-independence assumptions are restrictive for correlated financial variables.",
        "Support Vector Machine": "Scaled RBF kernel captures nonlinear relationships; fitting and inference are more expensive.",
    }
    observations = ["| ML Model Name | Observation about model performance |", "|---|---|"]
    for name, metrics in results.items():
        observations.append(
            f"| {name} | {explanations[name]} On this split: AUC {metrics['AUC']:.4f}, "
            f"default recall {metrics['Recall']:.4f}, MCC {metrics['MCC']:.4f}. |"
        )
    observations.append(
        f"| Overall winner for this dataset | **{winner}**, using highest held-out MCC "
        f"({results[winner]['MCC']:.4f}) as the predeclared criterion. This is a split-specific "
        "comparison, not proof of statistical superiority. |"
    )
    readme = readme_path.read_text()
    for section, content in [("METRICS", "\n".join(table)), ("OBSERVATIONS", "\n".join(observations))]:
        start = f"<!-- {section}:START -->"
        end = f"<!-- {section}:END -->"
        before, remainder = readme.split(start, 1)
        _, after = remainder.split(end, 1)
        readme = before + start + "\n" + content + "\n" + end + after
    readme_path.write_text(readme)


def load_dataset(dataset_path=None):
    local_workbook = ROOT / "default of credit card clients.xls"
    if dataset_path is None and local_workbook.exists():
        dataset_path = local_workbook
    if dataset_path:
        return pd.read_excel(dataset_path, header=1, engine="xlrd")
    with urlopen(DATA_URL, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        filename = next(name for name in archive.namelist() if name.endswith(".xls"))
        return pd.read_excel(io.BytesIO(archive.read(filename)), header=1, engine="xlrd")


def train(dataset_path=None):
    frame = load_dataset(dataset_path)
    features, target = validate_data(frame)
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=42)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=2)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=42, cache_size=512)
        ),
    }
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    test_frame = test_features.copy()
    test_frame[TARGET] = test_target
    test_path = ROOT / "test_data.csv"
    test_frame.to_csv(test_path, index=False)
    results = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(train_features, train_target)
        results[name], _, _ = evaluate_model(model, test_features, test_target)
        joblib.dump(model, model_dir / MODEL_FILES[name], compress=3)
        print(results[name], flush=True)
    metadata = {
        "dataset_url": DATA_URL,
        "random_state": 42,
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "default_rate": float(target.mean()),
        "sklearn_version": sklearn.__version__,
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "metrics": results,
    }
    (model_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    update_readme(metadata)
    print(pd.DataFrame(results).T.round(4).to_string())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the five credit-default classifiers.")
    parser.add_argument("--dataset", type=Path, help="Local original UCI .xls file (optional)")
    arguments = parser.parse_args()
    train(arguments.dataset)
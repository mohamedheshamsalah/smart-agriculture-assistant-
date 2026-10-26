"""Train and compare simple models for crop and soil-fertility prediction.

Beginner commands:
    python src/train_tabular.py --audit
    python src/train_tabular.py --task crop
    python src/train_tabular.py --task fertility
    python src/train_tabular.py --task all
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Each item contains the CSV path and its true label column.
TASKS = {
    "crop": (ROOT / "data/crop/Crop_recommendation.csv", "label"),
    "fertility": (ROOT / "data/fertility/soil_sight.csv", "Fertility_Level"),
}


def load_data(task):
    """Read one CSV and make column names consistent."""
    csv_path, target_name = TASKS[task]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing file: {csv_path}\nRun `python src/download_data.py` first."
        )
    data = pd.read_csv(csv_path)
    data.columns = [re.sub(r"\s+", "_", str(name).strip()) for name in data.columns]
    target_matches = [name for name in data.columns if name.lower() == target_name.lower()]
    if not target_matches:
        raise ValueError(f"Target '{target_name}' not found. Columns: {list(data.columns)}")
    return data, target_matches[0]


def audit_data(data, target, task):
    """Print a quick check before training."""
    print(f"\n--- {task.title()} data audit ---")
    print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
    print("Missing values:\n", data.isna().sum().to_string())
    print(f"Duplicate rows: {data.duplicated().sum():,}")
    print("Target counts:\n", data[target].value_counts(dropna=False).to_string())

    # Vegetation index values are usually between -1 and 1. Warn if a source
    # file has unusual values; do not silently change or delete real source data.
    for name in ["NDVI", "EVI", "SAVI"]:
        if name in data.columns:
            values = pd.to_numeric(data[name], errors="coerce")
            outside = int((~values.between(-1, 1)).sum())
            print(f"{name} values outside [-1, 1]: {outside:,}")
            if outside:
                print("  Check the source units before interpreting these values.")


def split_rows(data, target, task):
    """Make training, validation, and final test row numbers."""
    row_numbers = np.arange(len(data))

    # Nearby soil locations can be similar. Keep coordinate blocks together.
    column_names = {name.lower(): name for name in data.columns}
    latitude = column_names.get("latitude")
    longitude = column_names.get("longitude")
    if task == "fertility" and latitude and longitude:
        location_blocks = (
            data[latitude].round(3).astype(str) + "_" +
            data[longitude].round(3).astype(str)
        )
        first_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        training_and_validation, test_rows = next(
            first_split.split(row_numbers, data[target], groups=location_blocks)
        )
        second_split = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)
        train_part, validation_part = next(
            second_split.split(
                training_and_validation,
                data.iloc[training_and_validation][target],
                groups=location_blocks.iloc[training_and_validation],
            )
        )
        train_rows = training_and_validation[train_part]
        validation_rows = training_and_validation[validation_part]
        return train_rows, validation_rows, test_rows

    # Crop data is not grouped by farm, so keep each class represented in each split.
    train_and_validation, test_rows = train_test_split(
        row_numbers, test_size=0.20, random_state=SEED, stratify=data[target]
    )
    train_rows, validation_rows = train_test_split(
        train_and_validation, test_size=0.125, random_state=SEED,
        stratify=data.iloc[train_and_validation][target],
    )
    return train_rows, validation_rows, test_rows


def make_preprocessor(features, scale_numbers):
    """Fill missing values, encode text columns, and optionally scale numbers."""
    numbers = features.select_dtypes(include="number").columns.tolist()
    categories = [name for name in features.columns if name not in numbers]
    parts = []

    if numbers:
        number_steps = [("fill", SimpleImputer(strategy="median"))]
        if scale_numbers:
            number_steps.append(("scale", StandardScaler()))
        parts.append(("numbers", Pipeline(number_steps), numbers))

    if categories:
        category_steps = [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
        parts.append(("categories", Pipeline(category_steps), categories))

    return ColumnTransformer(parts)


def make_model(name, features, class_count):
    """Build one model and its matching preprocessing."""
    if name == "Logistic Regression":
        algorithm = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        scale_numbers = True
    elif name == "SVM":
        # LinearSVC is a fast SVM choice for the large fertility dataset.
        algorithm = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED)
        scale_numbers = True
    else:
        from xgboost import XGBClassifier
        algorithm = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=class_count,
            eval_metric="mlogloss", random_state=SEED,
            n_jobs=-1, tree_method="hist",
        )
        scale_numbers = False

    return Pipeline([
        ("prepare", make_preprocessor(features, scale_numbers)),
        ("model", algorithm),
    ])


def make_feature_form_info(features):
    """Save real input types and ranges for the Streamlit form."""
    information = {}
    for name in features.columns:
        if pd.api.types.is_numeric_dtype(features[name]):
            values = pd.to_numeric(features[name], errors="coerce")
            information[name] = {
                "kind": "numeric",
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
            }
        else:
            information[name] = {
                "kind": "categorical",
                "categories": sorted(features[name].dropna().astype(str).unique().tolist())[:100],
            }
    return information


def save_confusion_matrix(matrix, class_names, output_path, title):
    """Save a simple colored confusion-matrix image."""
    figure, axis = plt.subplots(figsize=(max(7, len(class_names) * 0.8), 6))
    axis.imshow(matrix, cmap="Greens")
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train_task(task):
    """Compare all algorithms, save the best one, then test it once."""
    data, target = load_data(task)
    data = data.dropna(subset=[target]).reset_index(drop=True)
    audit_data(data, target, task)

    # Convert text labels to numbers because the algorithms learn numeric targets.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[target].astype(str))
    class_names = label_encoder.classes_.tolist()
    features = data.drop(columns=[target]).copy()

    # Treat blank text cells as missing values for the imputer.
    for name in features.select_dtypes(include="object").columns:
        features[name] = features[name].replace(r"^\s*$", np.nan, regex=True)

    train_rows, validation_rows, test_rows = split_rows(data, target, task)
    if len(np.unique(y[train_rows])) != len(class_names):
        raise ValueError("A class is missing from the training split. Review the dataset and split.")

    algorithms = ["Logistic Regression", "SVM", "XGBoost"]
    results = []
    candidates = {}
    for name in algorithms:
        try:
            model = make_model(name, features, len(class_names))
            model.fit(features.iloc[train_rows], y[train_rows])
            validation_prediction = model.predict(features.iloc[validation_rows])
            score = f1_score(y[validation_rows], validation_prediction,
                             average="macro", zero_division=0)
            results.append({"model": name, "validation_macro_f1": float(score)})
            candidates[name] = model
            print(f"{name} validation Macro F1: {score:.4f}")
        except ImportError:
            print("XGBoost is not installed; install it with `pip install xgboost`.")

    if not results:
        raise RuntimeError("No models trained. Check the data and installed packages.")

    # Pick the algorithm using validation results, not the final test results.
    results.sort(key=lambda item: item["validation_macro_f1"], reverse=True)
    best_name = results[0]["model"]
    print(f"\nSelected by validation data: {best_name}")

    # Give the selected model both training and validation rows, but not test rows.
    training_rows = np.concatenate([train_rows, validation_rows])
    best_model = make_model(best_name, features, len(class_names))
    best_model.fit(features.iloc[training_rows], y[training_rows])
    test_prediction = best_model.predict(features.iloc[test_rows])

    accuracy = accuracy_score(y[test_rows], test_prediction)
    macro_f1 = f1_score(y[test_rows], test_prediction, average="macro", zero_division=0)
    class_numbers = list(range(len(class_names)))
    report = classification_report(
        y[test_rows], test_prediction, labels=class_numbers,
        target_names=class_names, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y[test_rows], test_prediction, labels=class_numbers)

    model_folder = ROOT / "models"
    report_folder = ROOT / "artifacts/evaluation"
    model_folder.mkdir(parents=True, exist_ok=True)
    report_folder.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": best_model,
        "feature_columns": features.columns.tolist(),
        "feature_metadata": make_feature_form_info(features),
        "classes": class_names,
        "target": target,
        "selected_model": best_name,
        "task": task,
    }
    joblib.dump(bundle, model_folder / f"{task}_model.joblib")

    metrics = {
        "task": task,
        "target": target,
        "rows": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in data[target].value_counts().items()},
        "validation_comparison": results,
        "selected_model": best_name,
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_classification_report": report,
        "split_sizes": {
            "train": int(len(train_rows)), "validation": int(len(validation_rows)),
            "test": int(len(test_rows)),
        },
        "split_note": "Fertility rows are grouped by rounded coordinates. Crop rows use stratified splits.",
    }
    (report_folder / f"{task}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        report_folder / f"{task}_confusion_matrix.csv"
    )
    save_confusion_matrix(
        matrix, class_names, report_folder / f"{task}_confusion_matrix.png",
        f"{task.title()} test results — {best_name}",
    )

    print(f"\nFinal test accuracy: {accuracy:.4f}")
    print(f"Final test Macro F1: {macro_f1:.4f}")
    print(f"Saved model: models/{task}_model.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train beginner-friendly agriculture ML baselines.")
    parser.add_argument("--task", choices=["crop", "fertility", "all"], default="all")
    parser.add_argument("--audit", action="store_true", help="Check the CSV files without training.")
    options = parser.parse_args()

    tasks = list(TASKS) if options.task == "all" else [options.task]
    for task in tasks:
        data, target = load_data(task)
        if options.audit:
            audit_data(data, target, task)
        else:
            train_task(task)


if __name__ == "__main__":
    main()

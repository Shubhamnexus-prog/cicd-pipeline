# ============================================================
# IRIS CLASSIFICATION - ML CI/CD PROJECT
# Logistic Regression vs Random Forest
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# Configuration
# ============================================================

sns.set_style("whitegrid")

RANDOM_STATE = 44

TARGET_NAMES = [
    "setosa",
    "versicolor",
    "virginica"
]


# ============================================================
# Create Output Directory
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = pd.read_csv("iris.csv")

print("\nDataset Shape:", dataset.shape)

print("\nOriginal Columns:")
print(dataset.columns.tolist())


# ============================================================
# Clean Column Names
# ============================================================

dataset.columns = [
    col.strip()
       .lower()
       .replace(" ", "_")
       .replace("(cm)", "")
       .replace("__", "_")
       .strip("_")
    for col in dataset.columns
]

print("\nCleaned Columns:")
print(dataset.columns.tolist())


# ============================================================
# Validate Required Columns
# ============================================================

required_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target"
]

missing_columns = [
    col for col in required_columns
    if col not in dataset.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Check Missing Values
# ============================================================

print("\nMissing Values:")
print(dataset.isnull().sum())

if dataset.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values."
    )


# ============================================================
# Feature Engineering
# ============================================================

dataset["sepal_length_width_ratio"] = (
    dataset["sepal_length"] /
    dataset["sepal_width"]
)

dataset["petal_length_width_ratio"] = (
    dataset["petal_length"] /
    dataset["petal_width"]
)


# ============================================================
# Features
# ============================================================

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "sepal_length_width_ratio",
    "petal_length_width_ratio"
]

target = "target"

X = dataset[features]
y = dataset[target]


# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logreg = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000
)

logreg.fit(X_train, y_train)

lr_train_predictions = logreg.predict(X_train)
lr_test_predictions = logreg.predict(X_test)


# ============================================================
# Logistic Regression Metrics
# ============================================================

lr_train_accuracy = accuracy_score(
    y_train,
    lr_train_predictions
)

lr_test_accuracy = accuracy_score(
    y_test,
    lr_test_predictions
)

lr_precision = precision_score(
    y_test,
    lr_test_predictions,
    average="weighted"
)

lr_recall = recall_score(
    y_test,
    lr_test_predictions,
    average="weighted"
)

lr_f1 = f1_score(
    y_test,
    lr_test_predictions,
    average="weighted"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE
)

rf_model.fit(X_train, y_train)

rf_train_predictions = rf_model.predict(X_train)
rf_test_predictions = rf_model.predict(X_test)


# ============================================================
# Random Forest Metrics
# ============================================================

rf_train_accuracy = accuracy_score(
    y_train,
    rf_train_predictions
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_predictions
)

rf_precision = precision_score(
    y_test,
    rf_test_predictions,
    average="weighted"
)

rf_recall = recall_score(
    y_test,
    rf_test_predictions,
    average="weighted"
)

rf_f1 = f1_score(
    y_test,
    rf_test_predictions,
    average="weighted"
)


# ============================================================
# Print Results
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nLOGISTIC REGRESSION")
print("-" * 40)

print(
    f"Train Accuracy : {lr_train_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy  : {lr_test_accuracy * 100:.2f}%"
)

print(
    f"Precision      : {lr_precision * 100:.2f}%"
)

print(
    f"Recall         : {lr_recall * 100:.2f}%"
)

print(
    f"F1 Score       : {lr_f1 * 100:.2f}%"
)


print("\nRANDOM FOREST")
print("-" * 40)

print(
    f"Train Accuracy : {rf_train_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy  : {rf_test_accuracy * 100:.2f}%"
)

print(
    f"Precision      : {rf_precision * 100:.2f}%"
)

print(
    f"Recall         : {rf_recall * 100:.2f}%"
)

print(
    f"F1 Score       : {rf_f1 * 100:.2f}%"
)


# ============================================================
# Classification Reports
# ============================================================

lr_report = classification_report(
    y_test,
    lr_test_predictions,
    target_names=TARGET_NAMES
)

rf_report = classification_report(
    y_test,
    rf_test_predictions,
    target_names=TARGET_NAMES
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION REPORT")
print("=" * 60)

print(lr_report)

print("\n" + "=" * 60)
print("RANDOM FOREST REPORT")
print("=" * 60)

print(rf_report)


# ============================================================
# Confusion Matrix Function
# ============================================================

def create_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Create Confusion Matrices
# ============================================================

create_confusion_matrix(
    y_test,
    lr_test_predictions,
    "Confusion Matrix - Logistic Regression",
    "outputs/ConfusionMatrix_LogisticRegression.png"
)

create_confusion_matrix(
    y_test,
    rf_test_predictions,
    "Confusion Matrix - Random Forest",
    "outputs/ConfusionMatrix_RandomForest.png"
)


# ============================================================
# Random Forest Feature Importance
# ============================================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)


# ============================================================
# Feature Importance Plot
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="importance",
    y="feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/FeatureImportance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Model Comparison
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Train Accuracy": [
        lr_train_accuracy,
        rf_train_accuracy
    ],

    "Test Accuracy": [
        lr_test_accuracy,
        rf_test_accuracy
    ],

    "Precision": [
        lr_precision,
        rf_precision
    ],

    "Recall": [
        lr_recall,
        rf_recall
    ],

    "F1 Score": [
        lr_f1,
        rf_f1
    ]
})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)


# ============================================================
# Model Comparison Plot
# ============================================================

comparison_plot = comparison.set_index(
    "Model"
)

comparison_plot[
    [
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Logistic Regression vs Random Forest"
)

plt.ylabel("Score")

plt.ylim(
    0,
    1.1
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "outputs/ModelComparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Save Scores
# ============================================================

with open(
    "outputs/scores.txt",
    "w"
) as score:

    score.write(
        "IRIS CLASSIFICATION MODEL RESULTS\n"
    )

    score.write(
        "=" * 50 + "\n\n"
    )

    score.write(
        "LOGISTIC REGRESSION\n"
    )

    score.write(
        "-" * 40 + "\n"
    )

    score.write(
        f"Train Accuracy : {lr_train_accuracy * 100:.2f}%\n"
    )

    score.write(
        f"Test Accuracy  : {lr_test_accuracy * 100:.2f}%\n"
    )

    score.write(
        f"Precision      : {lr_precision * 100:.2f}%\n"
    )

    score.write(
        f"Recall         : {lr_recall * 100:.2f}%\n"
    )

    score.write(
        f"F1 Score       : {lr_f1 * 100:.2f}%\n\n"
    )

    score.write(
        "RANDOM FOREST\n"
    )

    score.write(
        "-" * 40 + "\n"
    )

    score.write(
        f"Train Accuracy : {rf_train_accuracy * 100:.2f}%\n"
    )

    score.write(
        f"Test Accuracy  : {rf_test_accuracy * 100:.2f}%\n"
    )

    score.write(
        f"Precision      : {rf_precision * 100:.2f}%\n"
    )

    score.write(
        f"Recall         : {rf_recall * 100:.2f}%\n"
    )

    score.write(
        f"F1 Score       : {rf_f1 * 100:.2f}%\n\n"
    )

    score.write(
        "CLASSIFICATION REPORT - LOGISTIC REGRESSION\n"
    )

    score.write(
        "=" * 50 + "\n"
    )

    score.write(
        lr_report
    )

    score.write(
        "\n\n"
    )

    score.write(
        "CLASSIFICATION REPORT - RANDOM FOREST\n"
    )

    score.write(
        "=" * 50 + "\n"
    )

    score.write(
        rf_report
    )


# ============================================================
# Final Output
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

for file in os.listdir("outputs"):
    print(
        "✓",
        f"outputs/{file}"
    )

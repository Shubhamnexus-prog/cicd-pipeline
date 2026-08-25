# Iris Classification CI/CD

Machine Learning CI/CD pipeline using GitHub Actions.

## Models

- Logistic Regression
- Random Forest Classifier

## Feature Engineering

Additional features:

- Sepal Length / Sepal Width
- Petal Length / Petal Width

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Feature Importance

## CI/CD

GitHub Actions automatically:

1. Checkout repository
2. Setup Python 3.12
3. Install dependencies
4. Train ML models
5. Generate evaluation metrics
6. Generate visualization reports
7. Upload reports as GitHub Actions artifacts

## Project Structure

```text
cicd-pipeline/
│
├── .github/
│   └── workflows/
│       └── test.yaml
│
├── iris.csv
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore

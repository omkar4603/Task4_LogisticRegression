# main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from plots import plot_confusion, plot_roc, plot_sigmoid


def main():
    print("=== Task 4: Logistic Regression Classification ===")

    # 1️ Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")

    # 2️ Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3️ Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4️ Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 5️ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, "results/confusion.png")

    # 6️ ROC Curve
    auc = roc_auc_score(y_test, y_prob)
    plot_roc(y_test, y_prob, "results/roc.png")

    # 7️ Sigmoid Function Plot
    plot_sigmoid("results/sigmoid.png")

    # 8️ Metrics
    report = classification_report(y_test, y_pred)
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    with open("results/metrics.txt", "w") as f:
        f.write("=== Logistic Regression Evaluation ===\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")

    print("\n Done! Check the 'results/' folder for your outputs.")


if __name__ == "__main__":
    main()

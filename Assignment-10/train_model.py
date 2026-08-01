"""
train_model.py

Assignment 10: End-to-End ML Model Deployment using GitHub and Render
Task 1 (Data Understanding & Preprocessing) + Task 2 (Model Development)

Loads heart.csv, preprocesses it, trains a classifier to predict heart
disease risk, evaluates accuracy, and saves the trained model with Joblib.

Author: Palak Narang
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "heart.csv"
MODEL_PATH = "model.pkl"


def load_and_explore():
    df = pd.read_csv(DATA_PATH)

    print("First 5 records:")
    print(df.head())

    target_col = "target"
    numerical_features = [c for c in df.columns if c != target_col]
    print(f"\nNumerical features: {numerical_features}")
    print(f"Target variable: {target_col}")

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df, target_col


def preprocess_and_split(df, target_col):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda v: 1 if v > 0 else 0)  # binarize: disease present/absent

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc


def main():
    df, target_col = load_and_explore()
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df, target_col)
    model, acc = train_and_evaluate(X_train, X_test, y_train, y_test)

    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

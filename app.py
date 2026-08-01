"""
app.py

Assignment 10: End-to-End ML Model Deployment using GitHub and Render
Task 3: Flask REST API

Loads the trained model and exposes a /predict endpoint that accepts
patient clinical parameters as JSON and returns a heart disease
risk prediction as JSON.

Author: Palak Narang
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

bundle = joblib.load("model.pkl")
model = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = pd.DataFrame([{f: data[f] for f in FEATURE_NAMES}])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    result = {
        "prediction": "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
        "risk_probability": round(float(proba), 4),
    }
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)

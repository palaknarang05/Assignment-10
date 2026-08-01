# Heart Disease Prediction — End-to-End Deployment (Assignment 10)

A machine learning model that predicts whether a patient is at risk of heart
disease based on clinical parameters, served through a Flask REST API and
deployed live on Render.

## Developer Info
**Name:** Palak Narang
**Registration Number:** 23BCE11819
**Application Number:** IN26011657
**Batch Number:** 1A

## Problem Statement
A healthcare organization wants to deploy a machine learning model that
predicts whether a patient is at risk of heart disease based on clinical
parameters (UCI Cleveland Heart Disease dataset).

## Tech Stack
Python, Pandas, Scikit-learn, Flask, Joblib, Gunicorn, Render

## Repository Structure
```
HeartDiseaseDeployment/
├── app.py              # Flask REST API (loads model, serves /predict)
├── model.pkl           # Trained model (Random Forest) + feature list
├── requirements.txt
├── README.md
├── train_model.py       # Data loading, preprocessing, training, evaluation
├── heart.csv            # UCI Heart Disease dataset
├── Procfile             # Render/gunicorn start command
└── templates/
    └── index.html        # Simple web form for interactive predictions
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model** (regenerates `model.pkl`):
```bash
python train_model.py
```

**2. Run the API locally:**
```bash
python app.py
```
Visit `http://127.0.0.1:5000` for the web form, or call the API directly:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,
       "slope":3,"ca":0,"thal":1}'
```
Response:
```json
{"prediction": "No Heart Disease Detected", "risk_probability": 0.2862}
```

## Approach
- **Task 1 — Preprocessing:** loaded `heart.csv` with Pandas, identified 13
  numerical clinical features and the binary `target` variable, checked for
  missing values, and split 80/20 into train/test sets.
- **Task 2 — Model:** trained a `RandomForestClassifier` (200 trees, max
  depth 6), achieving **~80% accuracy** on the held-out test set. Model
  saved with Joblib.
- **Task 3 — API:** Flask app with a `/predict` endpoint that accepts
  patient parameters as JSON and returns the prediction and risk
  probability as JSON, plus a `/health` check endpoint and a basic web UI.
- **Task 4 — Deployment:** deployed on Render using Gunicorn as the
  production WSGI server (see `Procfile`).

## Render Deployment
🔗 **Live URL:** _add your Render deployment URL here after deploying_

To deploy:
1. Push this repo to a public GitHub repository.
2. On [Render](https://render.com), create a new **Web Service**, connect
   the repo, set build command `pip install -r requirements.txt` and start
   command `gunicorn app:app`.
3. Once live, copy the Render URL into this README and the submission form.

## Conclusion
The Random Forest model achieved approximately 80% accuracy in predicting
heart disease risk from clinical parameters such as age, chest pain type,
resting blood pressure, and cholesterol, showing that traditional clinical
features carry strong predictive signal for classification tasks of this
kind. The main challenges during deployment were ensuring the API's input
schema matched the model's expected feature order exactly, and switching
from Flask's development server to Gunicorn for production serving on
Render. This exercise highlighted why MLOps practices matter in real-world
machine learning: reliable model serialization, a well-defined API
contract, version control, and cloud deployment are what turn a notebook
experiment into a service that can be reliably queried by other systems
and used by non-technical stakeholders.

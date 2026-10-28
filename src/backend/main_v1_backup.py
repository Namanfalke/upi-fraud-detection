from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. INITIALIZE FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="UPI Fraud Detection API",
    description="API for detecting unusual UPI transactions",
    version="1.0.0"
)


# --------------------------------------------------
# 2. LOAD YOUR EXISTING MODEL BUNDLE
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "upi_isolation_forest_bundle.pkl"
)

ALERT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "api_alerts.jsonl"
)

try:
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]

except Exception as exc:
    raise RuntimeError(
        f"Could not load model bundle from {MODEL_PATH}. "
        "Check the path and bundle contents."
    ) from exc


# These are the numeric features used in training.
# They must match your existing training pipeline.

REQUIRED_NUMERIC_FEATURES = [
    "Amount",
    "Hour",
    "Day of Week",
    "Day of Month",
    "Month",
    "Is Weekend",
    "Is Night",
    "Time Since Previous Txn (minutes)",
    "Previous 5m Txn Count",
    "Previous 5m Total Amount",
    "Previous Hour Txn Count",
    "Previous Hour Total Amount",
    "Previous 24h Txn Count",
    "Previous 24h Total Amount",
    "Historical Amount Mean",
    "Amount Deviation Score",
    "Hourly Txn Count",
    "Historical spend pattern" 
]


# --------------------------------------------------
# 3. DEFINE REQUEST FORMAT
# --------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)

    # Features must already be engineered and encoded
    # in the same format used during model training.
    features: dict[str, float]


# --------------------------------------------------
# 4. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "UPI Fraud Detection API is running",
        "model_loaded": True
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 5. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_transaction(txn: TransactionRequest):

    # Check required numeric features
    missing = [
        col for col in REQUIRED_NUMERIC_FEATURES
        if col not in txn.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required engineered features",
                "missing_features": missing
            }
        )

    # Create one row in the exact training column order.
    # Missing one-hot category columns are set to zero.
    row = {
        col: txn.features.get(col, 0.0)
        for col in feature_columns
    }

    X = pd.DataFrame([row], columns=feature_columns)

    # Reject invalid values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Numeric missing values are handled by the
    # imputer saved during training.
    try:
        X_ready = imputer.transform(X)

        prediction = int(model.predict(X_ready)[0])

        anomaly_score = float(
            -model.score_samples(X_ready)[0]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc

    is_suspicious = prediction == -1

    result = {
        "transaction_id": txn.transaction_id,
        "sender_id": txn.sender_id,
        "status": (
            "SUSPICIOUS"
            if is_suspicious
            else "NOT_FLAGGED"
        ),
        "is_anomaly": is_suspicious,
        "anomaly_score": anomaly_score,
        "checked_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    # Save suspicious results for review
    if is_suspicious:
        ALERT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            ALERT_PATH,
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(result) + "\n"
            )

    return result

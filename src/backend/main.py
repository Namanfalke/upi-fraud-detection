import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import case, extract, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.backend.features import build_features
from src.backend.signals import evaluate_signals
from src.database.database import ROOT, Base, SessionLocal, engine, get_db
from src.database.seed_db import seed_if_empty
from src.database.models import Transaction

MODEL_PATH = ROOT / "models" / "upi_isolation_forest_bundle.pkl"
THRESHOLDS_PATH = ROOT / "models" / "detection_thresholds.json"
IST = timezone(timedelta(hours=5, minutes=30))

CATEGORICAL = ["Merchant category", "Transaction type", "Location",
               "Device type", "UPI channel"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = joblib.load(MODEL_PATH)
    state["thresholds"] = json.loads(THRESHOLDS_PATH.read_text())
    Base.metadata.create_all(bind=engine)

    # On a fresh database (e.g. after a cloud restart) reload the historical data
    with SessionLocal() as db:
        inserted = seed_if_empty(db)
    if inserted:
        print(f"Seeded database with {inserted} historical transactions")

    yield
    state.clear()


app = FastAPI(
    title="UPI Fraud Detection API",
    description="Scores UPI transactions using an Isolation Forest, IQR and hourly-volume signals.",
    version="3.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Request format
# ---------------------------------------------------------------
class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    receiver_id: Optional[str] = None
    amount: float = Field(gt=0)
    timestamp: Optional[datetime] = Field(
        default=None, description="Defaults to current time (IST) if omitted")

    merchant_category: Literal[
        "Education", "Electronics", "Entertainment", "Food", "Fuel",
        "Grocery", "Healthcare", "Online Services", "Restaurants",
        "Shopping", "Travel", "Utilities"]
    transaction_type: Literal[
        "Bill Payment", "Online Purchase", "P2M", "P2P", "Recharge"]
    location: Literal[
        "Ahmedabad", "Bengaluru", "Bhopal", "Chennai", "Delhi", "Hyderabad",
        "Indore", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Nagpur",
        "Patna", "Pune", "Surat"]
    device_type: Literal["Android", "Web", "iOS"]
    upi_channel: Literal[
        "Collect Request", "Mobile Number", "Online", "QR", "UPI ID"]

    historical_spend_pattern: Optional[float] = Field(
        default=None,
        description="Sender profile value. Looked up automatically for known senders; "
                    "required for new senders.")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def normalize_timestamp(ts: Optional[datetime]) -> datetime:
    """Return a naive datetime in IST (the dataset uses naive timestamps)."""
    if ts is None:
        return datetime.now(IST).replace(tzinfo=None, microsecond=0)
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST).replace(tzinfo=None)
    return ts


def score_transaction(features: dict, categories: dict) -> tuple[bool, float]:
    bundle = state["bundle"]
    X = pd.DataFrame([{**features, **categories}])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)
    X = X.reindex(columns=bundle["feature_columns"], fill_value=0).astype(float)

    X_ready = bundle["imputer"].transform(X)
    prediction = int(bundle["model"].predict(X_ready)[0])       # -1 = anomaly
    score = float(-bundle["model"].score_samples(X_ready)[0])   # higher = more unusual
    return prediction == -1, score


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "UPI Fraud Detection API is running"}


@app.get("/health")
def health(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Transaction.id)))
    return {
        "status": "healthy",
        "model_loaded": "bundle" in state,
        "thresholds_loaded": "thresholds" in state,
        "transactions_in_db": total,
    }


@app.post("/predict")
def predict(txn: TransactionRequest, db: Session = Depends(get_db)):
    ts = normalize_timestamp(txn.timestamp)

    # 1. Reject duplicate transaction IDs
    exists = db.scalar(
        select(Transaction.id).where(Transaction.transaction_id == txn.transaction_id))
    if exists:
        raise HTTPException(409, f"Transaction {txn.transaction_id} already exists")

    # 2. Sender profile value: from request, else from sender's stored history
    spend = txn.historical_spend_pattern
    if spend is None:
        spend = db.scalar(
            select(Transaction.historical_spend_pattern)
            .where(Transaction.sender_id == txn.sender_id,
                   Transaction.historical_spend_pattern.is_not(None))
            .order_by(Transaction.txn_timestamp.desc())
            .limit(1))
    if spend is None:
        raise HTTPException(
            422, "New sender: please provide historical_spend_pattern")

    # 3. This sender's earlier transactions
    rows = db.execute(
        select(Transaction.txn_timestamp, Transaction.amount)
        .where(Transaction.sender_id == txn.sender_id,
               Transaction.txn_timestamp < ts,
               Transaction.amount.is_not(None))).all()
    sender_history = [(r[0], r[1]) for r in rows]

    # 4. Volume of ALL transactions already stored in the same clock hour
    hour_start = ts.replace(minute=0, second=0, microsecond=0)
    hourly_before = db.scalar(
        select(func.count(Transaction.id))
        .where(Transaction.txn_timestamp >= hour_start,
               Transaction.txn_timestamp < hour_start + timedelta(hours=1)))

    # 5. Features -> model
    features = build_features(
        amount=txn.amount,
        ts=ts,
        sender_history=sender_history,
        hourly_count_before=hourly_before,
        historical_spend_pattern=spend,
    )
    categories = {
        "Merchant category": txn.merchant_category,
        "Transaction type": txn.transaction_type,
        "Location": txn.location,
        "Device type": txn.device_type,
        "UPI channel": txn.upi_channel,
    }
    try:
        is_anomaly, score = score_transaction(features, categories)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc

    # 6. Combine the three signals (IQR + Isolation Forest + hourly volume)
    signals = evaluate_signals(
        amount=txn.amount,
        hourly_count=features["Hourly Txn Count"],
        if_anomaly=is_anomaly,
        t=state["thresholds"],
    )
    status = "REVIEW" if signals["review_flag"] else "NOT_FLAGGED"

    # 7. Save to the database
    db.add(Transaction(
        transaction_id=txn.transaction_id,
        sender_id=txn.sender_id,
        receiver_id=txn.receiver_id,
        amount=txn.amount,
        txn_timestamp=ts,
        merchant_category=txn.merchant_category,
        transaction_type=txn.transaction_type,
        location=txn.location,
        device_type=txn.device_type,
        upi_channel=txn.upi_channel,
        historical_spend_pattern=spend,
        features_json=json.dumps(features),
        is_anomaly=signals["if_anomaly"],
        anomaly_score=score,
        iqr_anomaly=signals["iqr_anomaly"],
        ts_anomaly=signals["ts_anomaly"],
        signal_count=signals["signal_count"],
        risk_score=signals["risk_score"],
        fraud_review_flag=signals["review_flag"],
        status=status,
        source="api",
    ))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"Transaction {txn.transaction_id} already exists")

    return {
        "transaction_id": txn.transaction_id,
        "sender_id": txn.sender_id,
        "status": status,
        "review_flag": signals["review_flag"],
        "risk_score": signals["risk_score"],
        "signals": {
            "iqr_amount_outlier": signals["iqr_anomaly"],
            "isolation_forest": signals["if_anomaly"],
            "hourly_spike": signals["ts_anomaly"],
        },
        "anomaly_score": round(score, 4),
        "features_used": features,
        "timestamp": ts.isoformat(),
    }


# ---------------------------------------------------------------
# Read endpoints (used by the Streamlit dashboard)
# ---------------------------------------------------------------
Source = Optional[Literal["api", "seed"]]


@app.get("/transactions")
def list_transactions(
    limit: int = 50,
    offset: int = 0,
    source: Source = None,
    status: Optional[Literal["REVIEW", "NOT_FLAGGED"]] = None,
    sender_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Stored transactions, newest first. Optional filters: source, status, sender_id."""
    query = (select(Transaction)
             .order_by(Transaction.id.desc())
             .limit(min(limit, 500))
             .offset(max(offset, 0)))
    if source:
        query = query.where(Transaction.source == source)
    if status:
        query = query.where(Transaction.status == status)
    if sender_id:
        query = query.where(Transaction.sender_id == sender_id)

    return [
        {
            "transaction_id": t.transaction_id,
            "sender_id": t.sender_id,
            "receiver_id": t.receiver_id,
            "amount": t.amount,
            "timestamp": t.txn_timestamp.isoformat() if t.txn_timestamp else None,
            "merchant_category": t.merchant_category,
            "transaction_type": t.transaction_type,
            "location": t.location,
            "device_type": t.device_type,
            "upi_channel": t.upi_channel,
            "status": t.status,
            "risk_score": t.risk_score,
            "anomaly_score": round(t.anomaly_score, 4),
            "iqr_amount_outlier": t.iqr_anomaly,
            "isolation_forest": t.is_anomaly,
            "hourly_spike": t.ts_anomaly,
            "source": t.source,
        }
        for t in db.scalars(query).all()
    ]


@app.get("/stats")
def stats(source: Source = None, db: Session = Depends(get_db)):
    """Headline numbers for the dashboard."""
    cond = [Transaction.source == source] if source else []

    def count(*extra):
        return db.scalar(select(func.count(Transaction.id)).where(*cond, *extra)) or 0

    total = count()
    flagged = count(Transaction.fraud_review_flag.is_(True))
    return {
        "total_transactions": total,
        "review_flagged": flagged,
        "review_rate_percent": round(100 * flagged / total, 2) if total else 0.0,
        "avg_risk_score": round(
            db.scalar(select(func.avg(Transaction.risk_score)).where(*cond)) or 0, 2),
        "total_amount": round(
            db.scalar(select(func.sum(Transaction.amount)).where(*cond)) or 0, 2),
        "flagged_amount": round(
            db.scalar(select(func.sum(Transaction.amount)).where(
                *cond, Transaction.fraud_review_flag.is_(True))) or 0, 2),
        "signal_counts": {
            "iqr_amount_outlier": count(Transaction.iqr_anomaly.is_(True)),
            "isolation_forest": count(Transaction.is_anomaly.is_(True)),
            "hourly_spike": count(Transaction.ts_anomaly.is_(True)),
        },
    }


@app.get("/stats/breakdown")
def breakdown(
    by: Literal["merchant_category", "transaction_type", "location",
                "device_type", "upi_channel", "hour"],
    source: Source = None,
    db: Session = Depends(get_db),
):
    """Total vs review-flagged transactions grouped by a column (or by hour of day)."""
    if by == "hour":
        col = extract("hour", Transaction.txn_timestamp)
    else:
        col = getattr(Transaction, by)

    query = (
        select(
            col.label("grp"),
            func.count(Transaction.id).label("total"),
            func.sum(case((Transaction.fraud_review_flag.is_(True), 1), else_=0)).label("flagged"),
        )
        .group_by(col)
        .order_by(col)
    )
    if source:
        query = query.where(Transaction.source == source)

    return [
        {"group": str(r.grp), "total": int(r.total), "flagged": int(r.flagged or 0)}
        for r in db.execute(query)
    ]
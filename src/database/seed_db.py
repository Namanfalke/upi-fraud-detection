import json

import pandas as pd
from sqlalchemy import func, insert
from sqlalchemy.orm import Session

from .database import DATABASE_URL, ROOT, SessionLocal, engine
from .models import Base, Transaction

CSV_PATH = ROOT / "data" / "processed" / "upi_fraud_processed.csv"
BATCH_SIZE = 5000

NUMERIC_FEATURES = [
    "Amount", "Hour", "Day of Week", "Day of Month", "Month",
    "Is Weekend", "Is Night", "Time Since Previous Txn (minutes)",
    "Previous 5m Txn Count", "Previous 5m Total Amount",
    "Previous Hour Txn Count", "Previous Hour Total Amount",
    "Previous 24h Txn Count", "Previous 24h Total Amount",
    "Historical Amount Mean", "Amount Deviation Score",
    "Hourly Txn Count", "Historical spend pattern",
]


def clean(value):
    """NaN -> None so it is valid JSON / NULL."""
    return None if pd.isna(value) else float(value)


def seed_if_empty(db: Session) -> int:
    """Load the historical transactions if the table is empty.

    Returns the number of rows inserted (0 if the table already had data).
    """
    if db.query(Transaction).count() > 0:
        return 0

    df = pd.read_csv(CSV_PATH, parse_dates=["Timestamp"])
    records = df.to_dict("records")
    inserted = 0

    for start in range(0, len(records), BATCH_SIZE):
        rows = []
        for r in records[start:start + BATCH_SIZE]:
            ts = r["Timestamp"].to_pydatetime()
            rows.append({
                "transaction_id": r["Transaction ID"],
                "sender_id": r["Sender ID"],
                "receiver_id": r["Receiver ID"],
                "amount": float(r["Amount"]),
                "txn_timestamp": ts,
                "merchant_category": r["Merchant category"],
                "transaction_type": r["Transaction type"],
                "location": r["Location"],
                "device_type": r["Device type"],
                "upi_channel": r["UPI channel"],
                "historical_spend_pattern": float(r["Historical spend pattern"]),
                "features_json": json.dumps({c: clean(r[c]) for c in NUMERIC_FEATURES}),
                "is_anomaly": bool(r["IF Anomaly"]),
                "anomaly_score": float(r["IF Anomaly Score"]),
                "iqr_anomaly": bool(r["IQR Anomaly"]),
                "ts_anomaly": bool(r["Time Series Anomaly"]),
                "signal_count": int(r["Anomaly Signal Count"]),
                "risk_score": int(r["Combined Risk Score"]),
                "fraud_review_flag": bool(r["Fraud Review Flag"]),
                "status": "REVIEW" if r["Fraud Review Flag"] == 1 else "NOT_FLAGGED",
                "source": "seed",
                "checked_at": ts,
            })
        db.execute(insert(Transaction), rows)
        db.commit()
        inserted += len(rows)

    return inserted


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        inserted = seed_if_empty(db)
        if inserted:
            print(f"Inserted {inserted} rows into {DATABASE_URL}")
        else:
            print("Table already has rows - nothing to do.")

        total = db.query(func.count(Transaction.id)).scalar()
        flagged = db.query(func.count(Transaction.id)).filter(
            Transaction.fraud_review_flag.is_(True)).scalar()
        print("Total rows:     ", total)
        print("Review-flagged: ", flagged)
    finally:
        db.close()


if __name__ == "__main__":
    main()
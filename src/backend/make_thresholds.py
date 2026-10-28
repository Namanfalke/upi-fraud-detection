"""Compute detection thresholds from the processed training data.

Run once from the project root:
    python -m src.backend.make_thresholds
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT_PATH = ROOT / "models" / "detection_thresholds.json"


def main():
    amounts = pd.read_csv(PROCESSED / "upi_fraud_processed.csv", usecols=["Amount"])["Amount"]
    hourly = pd.read_csv(PROCESSED / "hourly_anomaly_analysis.csv", index_col=0)["Hourly Txn Count"]

    # Same IQR rule as the notebook
    q1, q3 = amounts.quantile([0.25, 0.75])
    iqr = q3 - q1

    thresholds = {
        "iqr_lower": float(q1 - 1.5 * iqr),
        "iqr_upper": float(q3 + 1.5 * iqr),
        "hourly_mean": float(hourly.mean()),
        "hourly_std": float(hourly.std()),
        "hourly_sigma": 3.0,        # spike = count > mean + sigma * std
        "min_hourly_count": 2,      # ...and at least this many transactions
    }

    OUT_PATH.write_text(json.dumps(thresholds, indent=2))
    print("Saved", OUT_PATH)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
"""Build the model's engineered features for ONE new transaction.

Mirrors the feature engineering in notebooks/analysis.ipynb, using only
transactions that happened BEFORE the new one (no future leakage).
"""
from datetime import datetime, timedelta

import numpy as np


def build_features(
    amount: float,
    ts: datetime,
    sender_history: list[tuple[datetime, float]],
    hourly_count_before: int,
    historical_spend_pattern: float,
) -> dict:
    """
    amount:                   current transaction amount
    ts:                       current transaction timestamp (naive)
    sender_history:           (timestamp, amount) of this sender's EARLIER transactions
    hourly_count_before:      transactions from ALL senders already stored in the same clock hour
    historical_spend_pattern: sender profile value
    """
    history = sorted(sender_history, key=lambda h: h[0])
    amounts = [a for _, a in history]

    def window(delta: timedelta):
        # Notebook: rolling(window, closed="left") -> [ts - window, ts)
        # No transactions in the window -> NaN (later filled by the imputer)
        values = [a for t, a in history if t >= ts - delta]
        if not values:
            return None, None
        return float(len(values)), float(sum(values))

    c5, s5 = window(timedelta(minutes=5))
    c1h, s1h = window(timedelta(hours=1))
    c24, s24 = window(timedelta(hours=24))

    if history:
        time_since = (ts - history[-1][0]).total_seconds() / 60
        hist_mean = float(np.mean(amounts))
    else:
        time_since = -1.0
        hist_mean = 0.0

    hist_std = float(np.std(amounts, ddof=1)) if len(amounts) >= 2 else None
    deviation = (amount - hist_mean) / hist_std if hist_std else 0.0

    weekday = ts.weekday()  # Monday = 0

    return {
        "Amount": float(amount),
        "Hour": ts.hour,
        "Day of Week": weekday,
        "Day of Month": ts.day,
        "Month": ts.month,
        "Is Weekend": int(weekday >= 5),
        "Is Night": int(ts.hour < 5),
        "Time Since Previous Txn (minutes)": time_since,
        "Previous 5m Txn Count": c5,
        "Previous 5m Total Amount": s5,
        "Previous Hour Txn Count": c1h,
        "Previous Hour Total Amount": s1h,
        "Previous 24h Txn Count": c24,
        "Previous 24h Total Amount": s24,
        "Historical Amount Mean": hist_mean,
        "Amount Deviation Score": float(deviation),
        "Hourly Txn Count": hourly_count_before + 1,  # includes this transaction
        "Historical spend pattern": float(historical_spend_pattern),
    }
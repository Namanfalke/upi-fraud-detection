"""Combine the three detection signals used in the notebook."""

WEIGHTS = {"iqr": 30, "if": 40, "ts": 30}
REVIEW_MIN_SIGNALS = 2


def evaluate_signals(amount: float, hourly_count: float, if_anomaly: bool, t: dict) -> dict:
    iqr_anomaly = bool(amount < t["iqr_lower"] or amount > t["iqr_upper"])

    spike_limit = t["hourly_mean"] + t["hourly_sigma"] * t["hourly_std"]
    ts_anomaly = bool(hourly_count > spike_limit and hourly_count >= t["min_hourly_count"])

    if_anomaly = bool(if_anomaly)
    signal_count = int(iqr_anomaly) + int(if_anomaly) + int(ts_anomaly)
    risk_score = (
        WEIGHTS["iqr"] * int(iqr_anomaly)
        + WEIGHTS["if"] * int(if_anomaly)
        + WEIGHTS["ts"] * int(ts_anomaly)
    )
    return {
        "iqr_anomaly": iqr_anomaly,
        "if_anomaly": if_anomaly,
        "ts_anomaly": ts_anomaly,
        "signal_count": signal_count,
        "risk_score": risk_score,
        "review_flag": signal_count >= REVIEW_MIN_SIGNALS,
    }
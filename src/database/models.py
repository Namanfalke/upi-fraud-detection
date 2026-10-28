from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    sender_id: Mapped[str] = mapped_column(String(50), index=True)
    receiver_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Raw transaction details (used to build history features)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    txn_timestamp: Mapped[datetime | None] = mapped_column(DateTime, index=True, nullable=True)
    merchant_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    transaction_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    location: Mapped[str | None] = mapped_column(String(50), nullable=True)
    device_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    upi_channel: Mapped[str | None] = mapped_column(String(50), nullable=True)
    historical_spend_pattern: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Exact model input, kept for auditing
    features_json: Mapped[str] = mapped_column(Text)

    # Detection signals
    is_anomaly: Mapped[bool] = mapped_column(Boolean, index=True)   # Isolation Forest
    anomaly_score: Mapped[float] = mapped_column(Float)             # Isolation Forest score
    iqr_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    ts_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)
    risk_score: Mapped[int] = mapped_column(Integer, default=0)     # 0-100 combined
    fraud_review_flag: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    status: Mapped[str] = mapped_column(String(20), index=True)
    source: Mapped[str] = mapped_column(String(10), default="api")  # "seed" or "api"
    checked_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
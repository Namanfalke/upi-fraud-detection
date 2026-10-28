import json

from sqlalchemy import inspect

from .database import DB_PATH, SessionLocal, engine
from .models import Base, Transaction


def main():
    Base.metadata.create_all(bind=engine)
    print("Database file:", DB_PATH)

    for table in inspect(engine).get_table_names():
        print(f"\nTable: {table}")
        for col in inspect(engine).get_columns(table):
            print(f"  {col['name']:20} {col['type']}")

    # Quick write/read test
    db = SessionLocal()
    try:
        db.add(Transaction(
            transaction_id="TEST_DB_CHECK",
            sender_id="USER_TEST",
            amount=100.0,
            features_json=json.dumps({"Amount": 100}),
            status="NOT_FLAGGED",
            is_anomaly=False,
            anomaly_score=0.41,
        ))
        db.commit()

        saved = db.query(Transaction).filter_by(transaction_id="TEST_DB_CHECK").one()
        print("\nWrite/read test OK ->", saved.transaction_id, saved.status, saved.checked_at)

        db.delete(saved)
        db.commit()
        print("Test row cleaned up.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
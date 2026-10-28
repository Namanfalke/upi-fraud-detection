from pathlib import Path
import joblib
import pandas as pd

# Project root = two levels up from this file (src/backend/ -> root)
ROOT = Path(__file__).resolve().parents[2]

bundle = joblib.load(ROOT / "models" / "upi_isolation_forest_bundle.pkl")
model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["feature_columns"]

print("Features expected:", len(features))

df = pd.read_csv(ROOT / "data" / "processed" / "upi_fraud_processed.csv").head(10)

categorical = ["Merchant category", "Transaction type", "Location",
               "Device type", "UPI channel"]

# Same preprocessing as the notebook: one-hot encode, then match training columns
X = pd.get_dummies(df, columns=categorical, dtype=int)
X = X.reindex(columns=features, fill_value=0)

X_imputed = imputer.transform(X)
pred = model.predict(X_imputed)                 # -1 = anomaly, 1 = normal
score = -model.score_samples(X_imputed)         # higher = more unusual (as in notebook)

print(pd.DataFrame({
    "Amount": df["Amount"],
    "IF Anomaly (saved)": df["IF Anomaly"],
    "IF Anomaly (now)": (pred == -1).astype(int),
    "IF Score (saved)": df["IF Anomaly Score"].round(4),
    "IF Score (now)": score.round(4),
}))
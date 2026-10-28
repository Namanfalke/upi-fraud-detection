import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st


def get_api_url() -> str:
    try:
        return st.secrets["API_URL"]            # set in Streamlit Cloud
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")  # local default


API_URL = get_api_url().rstrip("/")
IST = timezone(timedelta(hours=5, minutes=30))

# Must match the options accepted by the API
MERCHANT_CATEGORIES = ["Education", "Electronics", "Entertainment", "Food", "Fuel",
                       "Grocery", "Healthcare", "Online Services", "Restaurants",
                       "Shopping", "Travel", "Utilities"]
TRANSACTION_TYPES = ["Bill Payment", "Online Purchase", "P2M", "P2P", "Recharge"]
LOCATIONS = ["Ahmedabad", "Bengaluru", "Bhopal", "Chennai", "Delhi", "Hyderabad",
             "Indore", "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Nagpur",
             "Patna", "Pune", "Surat"]
DEVICE_TYPES = ["Android", "Web", "iOS"]
UPI_CHANNELS = ["Collect Request", "Mobile Number", "Online", "QR", "UPI ID"]

st.set_page_config(page_title="UPI Fraud Monitor", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------
# API helper (cached for 30 seconds so the page stays fast)
# ---------------------------------------------------------------
@st.cache_data(ttl=30)
def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=90)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("🛡️ UPI Fraud Monitor")

source_label = st.sidebar.radio(
    "Data source",
    ["All", "Live (API)", "Historical (seed)"],
    help="Live = transactions sent through /predict. Historical = the 25,000 loaded from your CSV.",
)
SOURCE = {"All": None, "Live (API)": "api", "Historical (seed)": "seed"}[source_label]

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"API: {API_URL}")

# ---------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------
st.title("UPI Fraud Detection Dashboard")

try:
    stats = api_get("/stats", {"source": SOURCE})
except Exception as exc:
    st.error(f"Cannot reach the API at {API_URL}. Is the FastAPI server running?\n\n{exc}")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{stats['total_transactions']:,}")
c2.metric("Flagged for review", f"{stats['review_flagged']:,}")
c3.metric("Review rate", f"{stats['review_rate_percent']}%")
c4.metric("Avg risk score", stats["avg_risk_score"])
c5.metric("Flagged amount", f"₹{stats['flagged_amount']:,.0f}")

st.subheader("Detection signals triggered")
signals_df = pd.DataFrame(
    {
        "Signal": ["IQR amount outlier", "Isolation Forest", "Hourly spike"],
        "Transactions": [
            stats["signal_counts"]["iqr_amount_outlier"],
            stats["signal_counts"]["isolation_forest"],
            stats["signal_counts"]["hourly_spike"],
        ],
    }
).set_index("Signal")
st.bar_chart(signals_df)
st.caption("A transaction is flagged for review when at least 2 of these 3 signals fire.")

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_overview, tab_flagged, tab_score = st.tabs(
    ["📊 Overview", "🚩 Flagged transactions", "🔍 Score a transaction"]
)


def breakdown_chart(by: str, title: str):
    data = api_get("/stats/breakdown", {"by": by, "source": SOURCE})
    df = pd.DataFrame(data)
    st.subheader(title)
    if df.empty:
        st.info("No data for this selection.")
        return

    if by == "hour":
        # zero-pad so the chart sorts 00, 01, ... 23
        df["group"] = pd.to_numeric(df["group"]).astype(int).astype(str).str.zfill(2)

    df["flag rate %"] = (100 * df["flagged"] / df["total"]).round(2)
    st.bar_chart(df.set_index("group")["flagged"])
    with st.expander("Show table"):
        st.dataframe(
            df.rename(columns={"group": by.replace("_", " ")}),
            hide_index=True,
        )


with tab_overview:
    st.caption("Number of transactions flagged for review, grouped by each attribute.")
    left, right = st.columns(2)
    with left:
        breakdown_chart("location", "Flagged by city")
    with right:
        breakdown_chart("merchant_category", "Flagged by merchant category")

    left, right = st.columns(2)
    with left:
        breakdown_chart("hour", "Flagged by hour of day")
    with right:
        breakdown_chart("transaction_type", "Flagged by transaction type")

    left, right = st.columns(2)
    with left:
        breakdown_chart("device_type", "Flagged by device type")
    with right:
        breakdown_chart("upi_channel", "Flagged by UPI channel")

with tab_flagged:
    f1, f2 = st.columns([1, 2])
    rows = f1.slider("Rows to load", min_value=10, max_value=500, value=100, step=10)
    sender = f2.text_input("Filter by sender ID (optional)", placeholder="e.g. S00001")

    records = api_get(
        "/transactions",
        {
            "status": "REVIEW",
            "limit": rows,
            "source": SOURCE,
            "sender_id": sender.strip() or None,
        },
    )

    if not records:
        st.info("No flagged transactions match these filters.")
    else:
        df = pd.DataFrame(records)
        columns = [
            "transaction_id", "timestamp", "sender_id", "amount", "risk_score",
            "iqr_amount_outlier", "isolation_forest", "hourly_spike",
            "merchant_category", "transaction_type", "location",
            "device_type", "upi_channel", "source",
        ]
        df = df[columns]
        st.write(f"Showing **{len(df)}** flagged transactions (newest first)")
        st.dataframe(df, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            df.to_csv(index=False),
            file_name="flagged_transactions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------
# Score a transaction (calls POST /predict)
# ---------------------------------------------------------------
def new_transaction_id() -> str:
    return "TXN" + datetime.now(IST).strftime("%Y%m%d%H%M%S")


with tab_score:
    st.caption("Submit a transaction. It is scored, saved to the database, "
               "and will appear in the dashboard.")

    # Keep the default ID stable between reruns; refresh it after a successful submit
    if "default_txn_id" not in st.session_state:
        st.session_state.default_txn_id = new_transaction_id()

    now_ist = datetime.now(IST)

    with st.form("score_form"):
        a, b, c = st.columns(3)
        transaction_id = a.text_input("Transaction ID", value=st.session_state.default_txn_id)
        sender_id = b.text_input("Sender ID", value="S00001")
        amount = c.number_input("Amount (₹)", min_value=1.0, value=500.0, step=50.0)

        d, e, f = st.columns(3)
        merchant_category = d.selectbox("Merchant category", MERCHANT_CATEGORIES)
        transaction_type = e.selectbox("Transaction type", TRANSACTION_TYPES, index=2)
        location = f.selectbox("Location", LOCATIONS, index=1)

        g, h, i = st.columns(3)
        device_type = g.selectbox("Device type", DEVICE_TYPES)
        upi_channel = h.selectbox("UPI channel", UPI_CHANNELS, index=3)
        spend_pattern = i.number_input(
            "Historical spend pattern",
            min_value=0.0, value=0.0, step=10.0,
            help="Only needed for NEW senders. Leave 0 for senders already in the database.",
        )

        j, k, l = st.columns(3)
        use_now = j.checkbox("Use current time", value=True)
        txn_date = k.date_input("Date (if not using current time)", value=now_ist.date())
        txn_time = l.time_input("Time (if not using current time)",
                                value=now_ist.time().replace(second=0, microsecond=0))

        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        payload = {
            "transaction_id": transaction_id.strip(),
            "sender_id": sender_id.strip(),
            "amount": amount,
            "merchant_category": merchant_category,
            "transaction_type": transaction_type,
            "location": location,
            "device_type": device_type,
            "upi_channel": upi_channel,
        }
        if not use_now:
            payload["timestamp"] = datetime.combine(txn_date, txn_time).isoformat()
        if spend_pattern > 0:
            payload["historical_spend_pattern"] = spend_pattern

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        except Exception as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            if resp.status_code == 200:
                result = resp.json()
                if result["review_flag"]:
                    st.error(f"🚩 FLAGGED FOR REVIEW  -  risk score {result['risk_score']}/100")
                else:
                    st.success(f"✅ Not flagged  -  risk score {result['risk_score']}/100")

                m1, m2, m3 = st.columns(3)
                m1.metric("IQR amount outlier", "Yes" if result["signals"]["iqr_amount_outlier"] else "No")
                m2.metric("Isolation Forest", "Yes" if result["signals"]["isolation_forest"] else "No")
                m3.metric("Hourly spike", "Yes" if result["signals"]["hourly_spike"] else "No")
                st.caption(f"Isolation Forest anomaly score: {result['anomaly_score']}  |  "
                           f"Timestamp used: {result['timestamp']}")

                with st.expander("Features the model used"):
                    st.json(result["features_used"])

                st.session_state.default_txn_id = new_transaction_id()
                st.cache_data.clear()   # so the other tabs show the new transaction
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Request failed ({resp.status_code}): {detail}")

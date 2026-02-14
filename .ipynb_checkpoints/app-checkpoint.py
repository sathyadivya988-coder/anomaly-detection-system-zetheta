import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="AI Fraud Monitoring System", layout="wide")

st.title("🏦 AI Fraud Monitoring Dashboard")
st.markdown("Real-time Credit Card Fraud Risk Analysis using Random Forest")

st.divider()

# -------------------------------------------------
# Define Paths
# -------------------------------------------------
MODEL_PATH = os.path.join("models", "rf_fraud_model.pkl")
DATA_PATH = os.path.join("data", "raw", "creditcard.csv")

# -------------------------------------------------
# Load Model Safely
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found! Check models folder.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# Load Dataset Safely
# -------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset file not found! Check data/raw folder.")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# -------------------------------------------------
# Sidebar Input
# -------------------------------------------------
st.sidebar.header("🔍 Transaction Input")

user_amount = st.sidebar.number_input(
    "Enter Transaction Amount",
    min_value=0.0,
    step=1.0
)

predict_button = st.sidebar.button("Analyze Transaction")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if predict_button:

    # Find closest transaction by amount
    closest_index = (df["Amount"] - user_amount).abs().idxmin()
    matched_row = df.loc[closest_index]

    # Remove target column
    input_features = matched_row.drop("Class")

    # Convert to 2D array
    input_array = np.array([input_features.values])

    # Prediction
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.subheader("📊 Transaction Risk Report")

    col1, col2 = st.columns(2)

    col1.metric("Entered Amount", f"₹ {user_amount:,.2f}")
    col2.metric("Matched Historical Amount", f"₹ {matched_row['Amount']:,.2f}")

    st.divider()

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"### Fraud Probability: {probability * 100:.2f}%")

    # Risk Level Logic
    if probability > 0.8:
        st.warning("🔴 Risk Level: HIGH")
    elif probability > 0.4:
        st.info("🟡 Risk Level: MEDIUM")
    else:
        st.success("🟢 Risk Level: LOW")

    st.divider()

    st.markdown("### 🧠 System Explanation")
    st.write("""
    The system matches the entered transaction amount with the closest 
    historical transaction from the dataset. 

    The full feature profile (Time + V1–V28 + Amount) is then used 
    by the trained Random Forest model to determine fraud probability.
    """)


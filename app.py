import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="AI Fraud Monitoring Dashboard", layout="wide")

st.title("🚨 AI Fraud Monitoring Dashboard")
st.subheader("Real-time Credit Card Fraud Detection System (Random Forest + SMOTE)")

# -----------------------------
# Threshold (Calculated using F1-score)
# -----------------------------
THRESHOLD = 0.7

# -----------------------------
# Load Model
# -----------------------------
model_path = "models/rf_fraud_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found in models folder")
    st.stop()

model = joblib.load(model_path)

# -----------------------------
# Load Dataset (for realistic feature sampling)
# -----------------------------
data_path = "data/raw/creditcard.csv"

if not os.path.exists(data_path):
    st.error("❌ Dataset file not found!")
    st.stop()

df = pd.read_csv(data_path)

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("💳 Transaction Input")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

if st.sidebar.button("Check Transaction"):

    # Take random real transaction (without Class)
    sample = df.sample(1).drop("Class", axis=1)

    # Replace Amount with user input
    sample["Amount"] = amount

    input_data = sample.values

    # -----------------------------
    # Predict Probability
    # -----------------------------
    prob = model.predict_proba(input_data)[0][1]
    prob_percent = prob * 100

    st.markdown("## 🔍 Prediction Result")

    st.metric(label="Fraud Probability", value=f"{prob_percent:.2f}%")

    # -----------------------------
    # Risk Classification
    # -----------------------------
    if prob >= THRESHOLD:
        st.error("🚨 HIGH RISK - Fraudulent Transaction")
        risk_level = "HIGH"
    elif prob >= 0.4:
        st.warning("⚠️ MEDIUM RISK - Suspicious Transaction")
        risk_level = "MEDIUM"
    else:
        st.success("✅ LOW RISK - Normal Transaction")
        risk_level = "LOW"

    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Model Threshold Used:** {THRESHOLD}")

    # -----------------------------
    # Show Transaction Details
    # -----------------------------
    with st.expander("📄 View Transaction Details"):
        st.dataframe(sample)



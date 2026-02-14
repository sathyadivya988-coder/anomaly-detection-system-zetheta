import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="AI Fraud Detection", layout="centered")

st.title("🚨 AI Credit Card Fraud Detection")
st.write("Real-time fraud risk analysis system")

# Load model
MODEL_PATH = os.path.join("models", "rf_fraud_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.divider()

st.subheader("Enter Transaction Details")

# Simple demo inputs
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)

# Dummy vector (replace later with full features if needed)
input_data = np.zeros((1, 30))
input_data[0][-1] = amount  # Put amount in last column

if st.button("Analyze Transaction"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("🔍 Prediction Result")

    st.progress(probability)

    st.write(f"Fraud Probability: {probability*100:.2f}%")

    if probability > 0.8:
        st.error("🔴 HIGH RISK - Fraud Likely")
    elif probability > 0.4:
        st.warning("🟡 MEDIUM RISK - Suspicious")
    else:
        st.success("🟢 LOW RISK - Normal Transaction")


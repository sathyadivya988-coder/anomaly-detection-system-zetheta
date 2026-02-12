import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/rf_fraud_model.pkl")


st.title("🚨 Credit Card Fraud Detection System")

st.write("Enter transaction amount below:")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Predict"):
    # Create dummy input (Amount + 29 zero features)
    input_data = pd.DataFrame([[amount] + [0]*29])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")

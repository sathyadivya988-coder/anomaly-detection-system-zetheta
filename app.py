# ============================================
# 💳 AI FRAUD DETECTION SYSTEM - FINAL PRO ULTRA VERSION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# 🌈 YOUR ORIGINAL NEON THEME (UNCHANGED)
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0B0F1A,#111827);
    color:#E5E7EB;
}
.main-title {
    color:#00BFFF;
    font-size:42px;
    font-weight:900;
    text-shadow:0 0 20px #00BFFF;
}
.section-amount {
    color:#FF2E63;
    font-size:24px;
    font-weight:700;
    text-shadow:0 0 10px #FF2E63;
}
.section-pca {
    color:#39FF14;
    font-size:24px;
    font-weight:700;
    text-shadow:0 0 10px #39FF14;
}
.section-model {
    color:#FFD700;
    font-size:26px;
    font-weight:700;
    text-shadow:0 0 10px #FFD700;
}
label {
    color:#A020F0 !important;
    font-weight:600 !important;
}
.stButton>button {
    background: linear-gradient(90deg,#00BFFF,#39FF14);
    color:black;
    font-weight:bold;
    border-radius:10px;
}
.card {
    padding:25px;
    border-radius:15px;
    margin-top:20px;
    font-size:20px;
    font-weight:bold;
    text-align:center;
}
.success-card {
    background:#002b1f;
    color:#39FF14;
    box-shadow:0 0 20px #39FF14;
}
.danger-card {
    background:#2b0000;
    color:#FF2E63;
    box-shadow:0 0 20px #FF2E63;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FEATURE ORDER
# --------------------------------------------------

feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# --------------------------------------------------
# LOGIN
# --------------------------------------------------

def login():
    st.markdown("<div class='main-title'>🔐 Secure Login</div>", unsafe_allow_html=True)
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main_app():

    st.markdown("<div class='main-title'>💳 Credit Card Fraud Detection Dashboard</div>", unsafe_allow_html=True)

    model_path = "models/rf_fraud_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model not found.")
        return

    model = joblib.load(model_path)

    menu = st.sidebar.radio("Navigation",
                            ["Single Prediction",
                             "Bulk CSV Detection",
                             "Model Performance"])

    # ==========================================================
    # 🔍 SINGLE PREDICTION
    # ==========================================================

    if menu == "Single Prediction":

        st.markdown("<div class='section-amount'>💰 Transaction Details</div>", unsafe_allow_html=True)

        input_data = {}
        input_data["Time"] = st.number_input("Transaction Time", 0.0)
        input_data["Amount"] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=0.0,
            format="%.2f"
        )

        threshold = st.slider(
            "Select Fraud Detection Threshold",
            min_value=0.1,
            max_value=0.95,
            value=0.7,
            step=0.05
        )

        st.markdown("<div class='section-pca'>📊 PCA Features (V1 - V28)</div>", unsafe_allow_html=True)

        cols = st.columns(3)
        for i in range(1, 29):
            input_data[f"V{i}"] = cols[(i-1)//10].number_input(
                f"V{i}",
                value=0.0,
                format="%.6f"
            )

        df_input = pd.DataFrame([input_data])[feature_order]

        if st.button("🔍 Predict Fraud"):

            prob = model.predict_proba(df_input)[0][1]
            percent = prob * 100

            # 🔥 Animated Progress Bar
            st.markdown("### ⚡ Live Fraud Probability Monitor")
            progress_bar = st.progress(0)
            for i in range(int(percent) + 1):
                progress_bar.progress(i)

            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg,#FF2E63,#FF8C00);
                padding:10px;
                border-radius:10px;
                text-align:center;
                font-weight:bold;
                font-size:18px;
                box-shadow:0 0 15px #FF8C00;
            ">
                🚨 Live Fraud Probability: {percent:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # Risk Level
            if percent < 40:
                risk = "🟢 LOW RISK"
                risk_color = "#39FF14"
            elif percent < 70:
                risk = "🟡 MEDIUM RISK"
                risk_color = "#FFD700"
            else:
                risk = "🔴 HIGH RISK"
                risk_color = "#FF2E63"

            if prob >= threshold:
                text = "🚨 FRAUD DETECTED"
                css = "danger-card"
            else:
                text = "✅ LEGITIMATE TRANSACTION"
                css = "success-card"

            st.markdown(f"""
            <div class="card {css}">
            {text}<br><br>
            Fraud Probability: {percent:.2f}%<br>
            Risk Level: <span style="color:{risk_color}">{risk}</span><br>
            Threshold: {threshold}
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================
    # 📂 BULK CSV DETECTION (FIXED)
    # ==========================================================

    elif menu == "Bulk CSV Detection":

        st.markdown("<div class='section-amount'>📂 Upload CSV Dataset</div>", unsafe_allow_html=True)

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file is not None:

            df = pd.read_csv(file)

            if not all(col in df.columns for col in feature_order):
                st.error("❌ CSV format mismatch. Must contain correct features.")
                return

            df = df[feature_order]

            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]

            df["Fraud Probability"] = probs
            df["Prediction"] = preds

            st.success("Prediction Completed")

            st.dataframe(df.head())

            st.metric("🚨 Total Fraud Transactions", int(sum(preds)))

    # ==========================================================
    # 📊 MODEL PERFORMANCE
    # ==========================================================

    elif menu == "Model Performance":

        st.markdown("<div class='section-model'>📊 Model Performance Metrics</div>", unsafe_allow_html=True)

        dataset_path = "data/raw/creditcard.csv"

        if os.path.exists(dataset_path):

            df = pd.read_csv(dataset_path)

            X = df[feature_order]
            y = df["Class"]

            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:,1]

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1:.4f}")

            cm = confusion_matrix(y,y_pred)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                        xticklabels=["Legit","Fraud"],
                        yticklabels=["Legit","Fraud"])
            st.pyplot(fig_cm)

            # 🔥 Fraud Heatmap Dashboard
            st.markdown("## 🔥 Fraud Pattern Heatmap Dashboard")

            df["Time_bin"] = pd.cut(df["Time"], bins=24)
            df["Amount_bin"] = pd.cut(df["Amount"], bins=20)

            heatmap_data = pd.pivot_table(
                df,
                values="Class",
                index="Time_bin",
                columns="Amount_bin",
                aggfunc="sum"
            )

            fig_heat, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(heatmap_data, cmap="inferno", linewidths=0.1)
            ax.set_title("Fraud Density by Time and Transaction Amount")
            st.pyplot(fig_heat)

        else:
            st.warning("Dataset not found at data/raw/creditcard.csv")
            # --------------------------------------------------
# SESSION CONTROL
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login()
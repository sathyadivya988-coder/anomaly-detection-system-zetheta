## 💳 AI-Powered Financial Anomaly & Fraud Detection Dashboard
<p align="center">














</p>
🚀 Live Demo

🔗 https://anomaly-detection-system-zetheta.streamlit.app

## 📌 Project Overview

A production-grade financial anomaly detection system built to identify fraudulent credit card transactions in real time.

This system integrates:

Machine Learning (Random Forest)

Explainable AI (SHAP)

Interactive Fraud Dashboard

Adjustable Risk Threshold

Automated PDF Reporting

Designed to simulate a real-world fintech fraud monitoring platform.

## 🧠 Problem Statement

Financial fraud detection is challenging due to:

Severe class imbalance (fraud < 1%)

High cost of false negatives

Need for real-time detection

Requirement for model transparency

This system solves these challenges using threshold tuning, SHAP explainability, and risk-based classification.

## 🏗 System Architecture
                ┌────────────────────┐
                │  Credit Card Data  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Data Preprocessing │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Random Forest Model│
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
 Real-Time Prediction   SHAP Engine     Model Metrics
        │                 │                 │
        ▼                 ▼                 ▼
 Risk Classification  Feature Impact   ROC / PR Curves
        │
        ▼
 Streamlit Dashboard
        │
        ▼
 PDF Report Generation
## ✨ Key Features

🔐 Authentication

Secure login interface for controlled access.

🔍 Real-Time Fraud Prediction

Instant single transaction prediction

Adjustable probability threshold

Risk classification (Low / Medium / High)

📂 Bulk CSV Fraud Detection

Upload dataset

Detect fraud at scale

View suspicious transactions

Download full PDF report

🧠 Explainable AI (SHAP)

Waterfall plot (Single transaction)

Beeswarm summary plot (Bulk data)

Top 5 contributing features

Human-readable fraud reasoning

📊 Advanced Analytics Dashboard

Confusion Matrix

Fraud Density Heatmap

ROC Curve

Precision-Recall Curve

Live probability monitoring

📑 Automated Reporting

Single transaction PDF report

Bulk fraud summary report

Top 10 suspicious transactions

⚖️ Handling Imbalanced Data

Fraud datasets are highly imbalanced.

This system handles imbalance by:

Prioritizing Recall & F1 Score over Accuracy

Using ROC & Precision-Recall curves

Allowing dynamic threshold tuning

Visualizing fraud concentration patterns

Providing SHAP transparency for regulatory trust

## 🤖 Machine Learning Model

Algorithm: Random Forest Classifier

Why Random Forest?

Handles high dimensional data

Robust to noise

Performs well on imbalanced data

Provides feature importance

## 📊 Evaluation Metrics

Metric	Purpose

Accuracy	Overall correctness

Precision	Fraud prediction reliability

Recall	Ability to detect fraud

F1 Score	Balance between Precision & Recall

ROC Curve	Performance across thresholds

PR Curve	Imbalanced data evaluation

## 🛠 Tech Stack

Category	Technology

Programming	Python

ML Framework	Scikit-Learn

Explainability	SHAP

Dashboard	Streamlit

Data Processing	Pandas, NumPy

Visualization	Matplotlib, Seaborn

Report Generation	ReportLab

Model Serialization	Joblib

Version Control	Git & GitHub

## 📁 Project Structure

anomaly-detection-system/
│
├── app.py
├── src/
│   └── model_training.py
├── models/
├── data/
├── requirements.txt
├── README.md
└── .gitignore

## ▶️ Run Locally

git clone https://github.com/sathyadivya988-coder/anomaly-detection-system-zetheta.git

cd anomaly-detection-system-zetheta

pip install -r requirements.txt

streamlit run app.py

## 📈 Why This Project Stands Out


✅ End-to-End ML Pipeline

✅ Real-Time Deployment

✅ Explainable AI Integration

✅ Interactive Dashboard

✅ Production-Style Reporting

✅ Clean UI/UX Design

✅ Portfolio-Ready Architecture

This is not just a model —
This is a deployable fraud detection system.

## 🚀 Future Improvements

XGBoost model comparison

Auto anomaly detection (Isolation Forest)

Real-time streaming fraud detection

Database integration

Cloud deployment (AWS)

CI/CD integration

## 👨‍💻 Author

P. Dhivyasri
B.Tech – Information Technology

Machine Learning | Explainable AI | Deployment | Data Science

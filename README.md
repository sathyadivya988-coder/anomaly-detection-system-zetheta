## 💳 AI-Powered Financial Anomaly & Fraud Detection Dashboard

## 🚀 Project Overview

This project is an end-to-end machine learning system designed to detect fraudulent credit card transactions in real time.

It applies anomaly detection principles to identify rare and suspicious financial activities within a highly imbalanced dataset. The system integrates model training, evaluation, and an interactive dashboard to simulate a production-level fraud monitoring platform.

## 🧠 Problem Statement

Financial fraud is a critical challenge in digital payment systems. Fraudulent transactions represent a very small percentage of total transactions, making detection difficult due to severe class imbalance.

The objective of this project is to:

Detect fraudulent transactions accurately

Minimize false positives

Provide real-time prediction capability

Visualize fraud patterns interactively

## ✨ Key Features

🔐 Secure Login System

🔍 Real-Time Single Transaction Fraud Prediction

🎚 Adjustable Fraud Detection Threshold

📂 Bulk CSV Fraud Detection

📊 Model Performance Metrics (Accuracy, Precision, Recall, F1)

📈 Confusion Matrix Visualization

🔥 Fraud Pattern Heatmap Dashboard

🎨 Custom Neon-Themed Interactive UI

## 🏗 System Architecture

Data Preprocessing

Model Training (Random Forest Classifier)

Model Evaluation

Model Serialization using Joblib

Streamlit Dashboard Integration

Real-Time Prediction & Visualization

## ⚖️ Handling Imbalanced Data

Fraud detection datasets are highly imbalanced because fraudulent transactions are rare.

To address this:

Focus is placed on Precision, Recall, and F1 Score rather than Accuracy alone

ROC Curve and Precision-Recall Curve are used for deeper evaluation

Adjustable probability threshold allows tuning between false positives and false negatives

## 🤖 Machine Learning Model

Algorithm: Random Forest Classifier
Dataset: Credit Card Fraud Detection Dataset

## 📊 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC Curve

Precision-Recall Curve

## 📊 Model Performance Dashboard

The application includes:

Confusion Matrix Visualization

Fraud Density Heatmap (Time vs Transaction Amount)

Real-Time Probability Monitoring

Risk Level Classification (Low / Medium / High)

## 🛠 Tech Stack

Python

Streamlit

Scikit-learn

Pandas & NumPy

Matplotlib & Seaborn

Joblib

Git & GitHub

## 📁 Project Structure
anomaly-detection-system/
│
├── app.py                  # Streamlit dashboard application
├── src/
│   └── model_training.py   # Model training pipeline
├── models/                 # Saved ML models (excluded from Git)
├── data/                   # Dataset directory (excluded from Git)
├── .gitignore
└── README.md

## ▶️ How to Run Locally

git clone https://github.com/sathyadivya988-coder/ai-fraud-detection-dashboard.git

cd ai-fraud-detection-dashboard


pip install -r requirements.txt

streamlit run app.py

## 🚀 Future Improvements

Model comparison (Logistic Regression, XGBoost)

Explainable AI using SHAP

Real-time streaming data integration

Database logging for transaction history

Cloud deployment (AWS / Streamlit Cloud)

## 👨‍💻 Author

Built as a practical machine learning project to simulate real-world financial fraud detection systems and demonstrate end-to-end ML application development.

## 🚀 Live Demo
https://anomaly-detection-system-zetheta.streamlit.app

## 🔍 Features
- Secure Login
- Fraud Probability Prediction
- Adjustable Threshold Slider
- Risk Level Indicator
- Interactive Graphs

## 💳 AI-Powered Financial Anomaly & Fraud Detection Dashboard

## 🚀 Project Overview

This project is an end-to-end machine learning system designed to detect fraudulent credit card transactions in real time.

It applies anomaly detection principles to identify rare and suspicious financial activities in a highly imbalanced dataset. The system integrates model training, evaluation, and an interactive dashboard to simulate a production-level fraud monitoring platform.

## 🧠 Problem Statement

Financial fraud is a critical challenge in digital payment systems. Fraudulent transactions represent a very small percentage of total transactions, making detection difficult due to severe class imbalance.

Objectives:

Detect fraudulent transactions accurately

Minimize false positives

Provide real-time prediction capability

Visualize fraud patterns interactively

## ✨ Key Features

🔐 Secure Login System

🔍 Real-Time Single Transaction Fraud Prediction

🎚 Adjustable Fraud Detection Threshold

📂 Bulk CSV Fraud Detection

📊 Model Performance Metrics (Accuracy, Precision, Recall, F1 Score)

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

Focus on Precision, Recall, and F1 Score rather than Accuracy alone

Use ROC Curve and Precision-Recall Curve for deeper evaluation

Implement adjustable probability threshold to tune between false positives and false negatives

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

## 📊 Model Performance Dashboard Includes

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
├── app.py                     # Streamlit dashboard application
├── src/
│   └── model_training.py      # Model training pipeline
├── models/                    # Saved ML models
├── data/                      # Dataset directory
├── .gitignore
└── README.md

## ▶️ How to Run Locally
git clone https://github.com/sathyadivya988-coder/anomaly-detection-system-zetheta.git

cd anomaly-detection-system-zetheta

pip install -r requirements.txt

streamlit run app.py

## 🚀 Live Demo

🔗 https://anomaly-detection-system-zetheta.streamlit.app

## 🚀 Future Improvements

Model comparison (Logistic Regression, XGBoost)

Explainable AI using SHAP

Real-time streaming data integration

Database logging for transaction history

Cloud deployment (AWS)

## 👨‍💻 Author

P.Dhivyasri

B.Tech Information Technology

End-to-End Machine Learning & Deployment Project

# ================================
# FRAUD DETECTION MODEL TRAINING
# ================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. LOAD DATASET
# -------------------------------

print("Loading dataset...")
df = pd.read_csv("C:/Users/it272/Documents/anomaly-detection-system/data/raw/creditcard.csv")


X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# 2. TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 3. HANDLE IMBALANCE (SMOTE)
# -------------------------------

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# -------------------------------
# 4. TRAIN RANDOM FOREST MODEL
# -------------------------------

print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_bal, y_train_bal)

# -------------------------------
# 5. SAVE TRAINED MODEL
# -------------------------------

joblib.dump(rf_model, "../models/rf_fraud_model.pkl")
print("Model saved successfully!")

# -------------------------------
# 6. MODEL EVALUATION
# -------------------------------

print("Evaluating model...")

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# ----- Confusion Matrix -----

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("../models/confusion_matrix.png")
plt.close()

# ----- Classification Report -----

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----- ROC Curve -----

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.savefig("../models/roc_curve.png")
plt.close()

print(f"ROC-AUC Score: {roc_auc:.4f}")

# ----- Feature Importance -----

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 10 Important Features")
plt.savefig("../models/feature_importance.png")
plt.close()

print("Feature importance graph saved!")

print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")

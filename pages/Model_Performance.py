import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.header("📉 Model Performance")

# --- Load model, scaler, and columns
model = joblib.load("data/best_logistic_model.pkl")
scaler = joblib.load("data/scaler.pkl")
columns = joblib.load("data/columns.pkl")

# --- Load X_test and y_test
X_test = pd.read_pickle("data/X_test.pkl")
y_test = pd.read_pickle("data/y_test.pkl")

# --- Ensure correct column order
X_test = X_test[[col for col in columns if col != "Outcome"]]


# --- Scale X_test
X_test_scaled = scaler.transform(X_test)

# --- Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# --- Confusion Matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

# --- Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

# --- ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

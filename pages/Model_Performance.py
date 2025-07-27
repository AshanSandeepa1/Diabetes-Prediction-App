import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

df = pd.read_csv(os.path.join("data", "diabetes.csv"))
model = joblib.load("model.pkl")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

st.header("📉 Model Performance")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()
st.pyplot(fig2)

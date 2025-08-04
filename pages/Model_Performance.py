import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Model Performance", page_icon="📉")

st.header("📉 Model Performance")


# ------------------ Explanation ------------------ #
st.markdown("---")
st.subheader("💡 Explanation")

st.markdown("""
This page presents a detailed evaluation of the diabetes prediction models we have trained and tested. You will find:

- **Confusion Matrix:** Visualizes the performance of the classification model by showing true positives, true negatives, false positives, and false negatives.
- **Classification Report:** Detailed metrics including precision, recall, F1-score, and support for each class to understand how well the model performs.
- **ROC Curve:** Displays the trade-off between sensitivity (true positive rate) and specificity (false positive rate) for different thresholds. The Area Under the Curve (AUC) score summarizes the overall model performance.
- **Model Comparison Table:** Compares multiple models on key metrics to help you identify the best-performing model.

Use this information to understand the strengths and weaknesses of each model and make informed decisions for diabetes risk prediction.
""")

st.info(
    """
    ### 🔔 Selected Model for Detailed Analysis
    
    The detailed performance metrics and charts shown below correspond to the **Logistic Regression** model, which we selected as our primary model for diabetes prediction.
    """
)

st.markdown("---")

# --- Load model, scaler, and columns ---
model = joblib.load("data/best_logistic_model.pkl")
scaler = joblib.load("data/scaler.pkl")
columns = joblib.load("data/columns.pkl")

# --- Load X_test and y_test ---
X_test = pd.read_pickle("data/X_test.pkl")
y_test = pd.read_pickle("data/y_test.pkl")

# --- Ensure correct column order ---
X_test = X_test[[col for col in columns if col != "Outcome"]]

# --- Scale X_test ---
X_test_scaled = scaler.transform(X_test)

# --- Predict ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

st.markdown("## 🧮 Logistic Regression Model Performance")

# --- Confusion Matrix ---
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

# --- Classification Report ---
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format({
    "precision": "{:.2f}",
    "recall": "{:.2f}",
    "f1-score": "{:.2f}",
    "support": "{:.0f}"
}))

# --- ROC Curve ---
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


# ------------------ Model Comparison Section ------------------ #
st.markdown("---")
st.subheader("🔎 Model Comparison")

# Predefined model metrics from training results in notebook
model_metrics = {
    "Logistic Regression": {
        "Accuracy": 0.7907,
        "Precision": 0.6992,
        "Recall": 0.5658,
        "F1-Score": 0.6255
    },
    "Random Forest": {
        "Accuracy": 0.7785,
        "Precision": 0.6748,
        "Recall": 0.5461,
        "F1-Score": 0.6036
    },
    "SVM": {
        "Accuracy": 0.7744,
        "Precision": 0.6847,
        "Recall": 0.5000,
        "F1-Score": 0.5779
    }
}

# Convert to DataFrame for display and plotting
metrics_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})

# Sort by F1-Score descending
metrics_df = metrics_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

# Show metrics as table
st.markdown("### 📊 Performance Metrics Table")
st.dataframe(
    metrics_df.style.format(
        {col: "{:.4f}" for col in metrics_df.columns if col != "Model"}
    )
)


# Highlight best model based on F1-Score
best_model = metrics_df.iloc[0]
st.success(f"⭐ **Best Model: {best_model['Model']}** with F1-Score = {best_model['F1-Score']:.4f}")

# ------------------ Bar Chart ------------------ #
st.markdown("### 📉 Comparison of Metrics")

fig_bar = px.bar(
    metrics_df.melt(id_vars="Model"),
    x="Model",
    y="value",
    color="variable",
    barmode="group",
    labels={"value": "Score", "variable": "Metric"},
    title="Model Performance Metrics Comparison"
)
fig_bar.update_layout(yaxis=dict(range=[0,1]))
st.plotly_chart(fig_bar, use_container_width=True)

# ------------------ Radar Chart ------------------ #
st.markdown("### 🎯 Radar Chart for Model Metrics")

categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig_radar = go.Figure()

for _, row in metrics_df.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row[cat] for cat in categories],
        theta=categories,
        fill='toself',
        name=row["Model"]
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0,1])
    ),
    showlegend=True,
    title="Radar Chart of Model Performance Metrics"
)

st.plotly_chart(fig_radar, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Python @ 2025 Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)
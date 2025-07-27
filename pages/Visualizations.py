import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.express as px

df = pd.read_csv(os.path.join("data", "diabetes.csv"))

st.header("📈 Data Visualizations")

st.subheader("Glucose vs Outcome")
fig1 = px.histogram(df, x="Glucose", color="Outcome", barmode="overlay")
st.plotly_chart(fig1)

st.subheader("Correlation Heatmap")
fig2, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig2)

st.subheader("BMI Distribution")
fig3 = px.box(df, y="BMI", color="Outcome")
st.plotly_chart(fig3)

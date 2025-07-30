import streamlit as st
import pandas as pd
import os

# Load dataset
df = pd.read_csv(os.path.join("data", "diabetes.csv"))

# Fix: Convert nullable Int64 columns to native int64
for col in df.select_dtypes(include="Int64").columns:
    df[col] = df[col].astype("int64")

# Streamlit layout
st.header("📊 Data Exploration")
st.write("### Dataset Overview")
st.write(f"Shape: {df.shape}")
st.write(df.dtypes)

st.write("### Sample Data")
st.dataframe(df.sample(10))

st.write("### Filter Data")
glucose = st.slider("Glucose Level", int(df['Glucose'].min()), int(df['Glucose'].max()))
filtered_df = df[df['Glucose'] >= glucose]
st.write(f"Filtered rows: {filtered_df.shape[0]}")
st.dataframe(filtered_df)

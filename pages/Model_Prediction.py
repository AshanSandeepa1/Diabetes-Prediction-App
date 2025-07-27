import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and expected columns
model = joblib.load("model.pkl")
columns = joblib.load("data/columns.pkl")  # already fixed without 'Outcome'

st.header("🤖 Diabetes Prediction")
st.markdown("Provide patient data to predict the likelihood of diabetes.")

# 1. Collect base inputs
inputs = {}
base_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for feature in base_features:
    inputs[feature] = st.number_input(feature, min_value=0.0, step=1.0)

# 2. On predict: add engineered features
if st.button("Predict"):
    # Create DataFrame with 1 row
    df = pd.DataFrame([inputs])
    
    # Age Group
    age = df.at[0, "Age"]
    age_groups = {
        "Teen": (0, 19),
        "Young Adult": (20, 29),
        "Adult": (30, 39),
        "Middle-aged": (40, 49),
        "Senior": (50, 59),
        "Elderly": (60, 150),
    }
    for group, (low, high) in age_groups.items():
        df[f"AgeGroup_{group}"] = int(low <= age <= high)

    # BMI Group
    bmi = df.at[0, "BMI"]
    bmi_groups = {
        "Underweight": (0, 18.4),
        "Normal": (18.5, 24.9),
        "Overweight": (25.0, 29.9),
        "Obese": (30.0, 100),
    }
    for group, (low, high) in bmi_groups.items():
        df[f"BMIGroup_{group}"] = int(low <= bmi <= high)

    # Ensure column order matches model
    df = df[[col for col in columns if col in df.columns]]

    # 3. Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][prediction]

    st.success(f"Prediction: {'Diabetic' if prediction else 'Not Diabetic'}")
    st.info(f"Confidence: {probability:.2f}")

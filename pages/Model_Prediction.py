import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.header("🤖 Diabetes Prediction")

st.markdown("Provide patient data to predict the likelihood of diabetes.")

inputs = {}
for feature in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]:
    inputs[feature] = st.number_input(feature, min_value=0.0, step=1.0)

if st.button("Predict"):
    data = np.array([list(inputs.values())])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][prediction]
    st.success(f"Prediction: {'Diabetic' if prediction else 'Not Diabetic'}")
    st.info(f"Confidence: {probability:.2f}")

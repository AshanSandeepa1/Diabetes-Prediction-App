import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Model Prediction", page_icon="🤖")

# Load model, scaler, and expected columns
model = joblib.load("model.pkl")
scaler = joblib.load("data/scaler.pkl")
columns = joblib.load("data/columns.pkl")

st.header("🤖 Diabetes Prediction")
st.markdown("Provide patient data to predict the likelihood of diabetes.")

# Preset example inputs
examples = {
    "Clear Inputs": {
        "Pregnancies": 0,
        "Glucose": 0,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 0.0,
        "DiabetesPedigreeFunction": 0.0,
        "Age": 0,
    },
    "Likely Diabetic": {
        "Pregnancies": 5,
        "Glucose": 180,
        "BloodPressure": 90,
        "SkinThickness": 35,
        "Insulin": 150,
        "BMI": 35.0,
        "DiabetesPedigreeFunction": 0.8,
        "Age": 45,
    },
    "Likely Non-Diabetic": {
        "Pregnancies": 1,
        "Glucose": 90,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 22.0,
        "DiabetesPedigreeFunction": 0.2,
        "Age": 30,
    },
}

selected_example = st.selectbox("📋 Select an example input:", list(examples.keys()))
if "last_example" not in st.session_state:
    st.session_state.last_example = None

if selected_example != st.session_state.last_example:
    for feature, val in examples[selected_example].items():
        st.session_state[feature] = val
    st.session_state.last_example = selected_example

# Feature definitions with units and help text for tooltips
feature_info = {
    "Pregnancies": {
        "min": 0, "max": 20, "step": 1, "format": "%d", "unit": "(count)",
        "help": "Number of times the patient has been pregnant."
    },
    "Glucose": {
        "min": 0, "max": 200, "step": 1, "format": "%d", "unit": "(mg/dL)",
        "help": "Plasma glucose concentration after 2 hours in an oral glucose tolerance test."
    },
    "BloodPressure": {
        "min": 0, "max": 150, "step": 1, "format": "%d", "unit": "(mm Hg)",
        "help": "Diastolic blood pressure (mm Hg)."
    },
    "SkinThickness": {
        "min": 0, "max": 100, "step": 1, "format": "%d", "unit": "(mm)",
        "help": "Triceps skin fold thickness (mm)."
    },
    "Insulin": {
        "min": 0, "max": 900, "step": 1, "format": "%d", "unit": "(μU/mL)",
        "help": "2-Hour serum insulin (μU/mL)."
    },
    "BMI": {
        "min": 0.0, "max": 70.0, "step": 0.1, "format": "%.1f", "unit": "(kg/m²)",
        "help": "Body Mass Index, calculated as weight in kg divided by height in meters squared."
    },
    "DiabetesPedigreeFunction": {
        "min": 0.0, "max": 3.0, "step": 0.01, "format": "%.2f", "unit": "",
        "help": "Likelihood of diabetes based on family history (higher means greater risk)."
    },
    "Age": {
        "min": 0, "max": 120, "step": 1, "format": "%d", "unit": "(years)",
        "help": "Age of the patient in years."
    },
}

inputs = {}
cols = st.columns(2)

for idx, feature in enumerate(feature_info):
    info = feature_info[feature]
    default_val = st.session_state.get(feature, info["min"])
    inputs[feature] = cols[idx % 2].number_input(
        f"{feature} {info['unit']}",
        min_value=info["min"],
        max_value=info["max"],
        value=default_val,
        step=info["step"],
        format=info["format"],
        key=feature,
        help=info["help"]
    )

def validate_inputs(data):
    errors = []
    # Example validations:
    if data["Glucose"] == 0:
        errors.append("Glucose level cannot be zero.")
    if data["BMI"] == 0:
        errors.append("BMI cannot be zero.")
    if data["Age"] == 0:
        errors.append("Age cannot be zero.")
    if data["BloodPressure"] == 0:
        errors.append("Blood Pressure cannot be zero.")
    # Add more checks as needed
    return errors

if st.button("🔍 Predict"):
    errors = validate_inputs(inputs)
    if errors:
        for err in errors:
            st.error(err)
    else:
        with st.spinner("Predicting diabetes risk..."):
            df = pd.DataFrame([inputs])

            # Age Groups
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

            # BMI Groups
            bmi = df.at[0, "BMI"]
            bmi_groups = {
                "Underweight": (0, 18.4),
                "Normal": (18.5, 24.9),
                "Overweight": (25.0, 29.9),
                "Obese": (30.0, 100),
            }
            for group, (low, high) in bmi_groups.items():
                df[f"BMIGroup_{group}"] = int(low <= bmi <= high)

            # Final model-ready DataFrame
            df = df[[col for col in columns if col in df.columns]]
            df_scaled = scaler.transform(df)

            prediction = model.predict(df_scaled)[0]
            prob_positive = model.predict_proba(df_scaled)[0][1]

            # Result Message
            if prediction == 1:
                st.markdown(
                    f"""
                    <div style="background-color:#ff4d4d;padding:20px;border-radius:10px;">
                        <h3 style="color:white;">🚨 Alert: Diabetic Likely</h3>
                        <p style="color:white;">Confidence: {prob_positive:.2%}</p>
                        <p style="color:white;">Please consult a healthcare provider.</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown(
                    f"""
                    <div style="background-color:#28a745;padding:20px;border-radius:10px;">
                        <h3 style="color:white;">✅ Result: Not Diabetic</h3>
                        <p style="color:white;">Confidence: {prob_positive:.2%}</p>
                        <p style="color:white;">Keep up the healthy habits! 🎉</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()

            # Suggestion Logic

            st.markdown("### 🩺 Health Tips Based on Your Inputs")

            suggestions = []

            if bmi > 25:
                suggestions.append(f"- Your **BMI ({bmi:.1f})** is in the overweight/obese range. Aim for 18.5–24.9 with regular exercise and balanced meals.")
            elif bmi < 18.5:
                suggestions.append(f"- Your **BMI ({bmi:.1f})** is considered underweight. Consider consulting a dietician for healthy weight gain.")

            if inputs["Glucose"] > 140:
                suggestions.append("- **High glucose** levels detected. Consider limiting simple sugars and processed carbs.")
            elif 70 <= inputs["Glucose"] < 90:
                suggestions.append("- Your **glucose** is on the lower end. Ensure you eat regularly.")

            if inputs["BloodPressure"] > 130:
                suggestions.append("- **Blood Pressure** seems elevated. Reduce sodium intake and monitor regularly.")
            elif inputs["BloodPressure"] < 60:
                suggestions.append("- Very low **blood pressure** may cause fatigue. Stay hydrated.")

            if inputs["SkinThickness"] > 50:
                suggestions.append("- **Skin thickness** is high. This may reflect insulin resistance. Keep monitoring and consider dietary changes.")

            if not suggestions:
                st.success("🎉 All values appear within healthy ranges. Keep it up!")
            else:
                for s in suggestions:
                    st.markdown(s)

            # Show Input Table
            st.markdown("---")
            st.markdown("### 📋 Input Summary")
            st.table(df.T.rename(columns={0: "Value"}))

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Python @ 2025 Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)

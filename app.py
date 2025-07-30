# app.py
import streamlit as st
from streamlit_lottie import st_lottie
import json
import os

# ---------------------- Config ---------------------- #
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    page_icon="🧠"
)

# ---------------------- Load Local Lottie ---------------------- #
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_diabetes = load_lottie_file("assets/blood_pressure.json")

# ---------------------- Main Area ---------------------- #
st.title("🧠 Diabetes Prediction System")

st.markdown("""
Welcome to the **Diabetes Prediction System** powered by **Machine Learning**.  
This application helps you:
- 🔍 Explore the diabetes dataset
- 📊 Visualize key health metrics
- 🤖 Predict diabetes using trained ML models
- 📈 Review model performance

Use the sidebar to navigate through the app.
""")

# Display Lottie animation
if lottie_diabetes:
    st_lottie(lottie_diabetes, height=300)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit, Scikit-learn & Python @ 2025 Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)

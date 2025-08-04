import streamlit as st
from streamlit_lottie import st_lottie
import json

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------
# Load Lottie Animation
# ------------------------------
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_diabetes = load_lottie_file("assets/blood_pressure.json")

# ------------------------------
# Hero Section
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 3.5rem; font-weight: 700;'>Diabetes Prediction System</h1>
        <p style='font-size: 1.25rem; color: #888;'>Powered by Machine Learning • Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Centered animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if lottie_diabetes:
        st_lottie(lottie_diabetes, height=300)

st.markdown("---")

# ------------------------------
# What is This App?
# ------------------------------
st.markdown("## 🤖 What is This App?")
st.markdown("""
This web application helps users **predict the likelihood of diabetes** using a machine learning model trained on the **PIMA Indian Diabetes dataset**.

It combines **data analysis**, **interactive visualizations**, and **predictive modeling** to assist both patients and researchers in understanding diabetic risk factors.
""")

# ------------------------------
# How to Use This App
# ------------------------------
st.markdown("## 🧭 How to Use This App")

st.markdown("""
You can explore different sections from the **left sidebar**:

- **Home** — You're here! Learn about the app and its features.
- **Data Exploration** — View and filter patient data.
- **Data Visualization** — Visualize important health indicators.
- **Diabetes Prediction** — Enter your health stats and get a prediction.
- **Model Performance** — See how our ML models perform and compare.
""")

with st.expander("📘 Click to View Step-by-Step Guide", expanded=False):
    st.markdown("""
    This app is designed to guide you through understanding diabetes risk prediction. Here's how you can navigate and use each section:

    ### Model Data Exploration
    - Browse the dataset used to train the model.
    - Filter by medical parameters (e.g., age, glucose).
    - View records in a sortable and searchable table.

    ### Data Visualization
    - Graphical visualizations of health metrics.
    - Use filters and dropdowns to explore trends (e.g., glucose vs. outcome).
    - Understand the relationship between features and diabetes risk.

    ### Model Diabetes Prediction
    - Input your medical values (e.g., BMI, Glucose, Age, etc.).
    - Click "Predict" to get an instant result.
    - A friendly message, visual cue, and confidence level will be shown.
    - *Prediction result is color-coded for clarity (red = diabetic, green = non-diabetic).*

    ### Model Performance
    - Review how well each ML model performs.
    - Compare accuracy, precision, recall, and F1 score.
    - See confusion matrices and ROC curves.

    ### ℹ️ Tips
    - Use **sliders** or **number inputs** accurately to get the best prediction.
    - Hover over icons or charts for extra information.
    - App supports dark/light theme and works on desktop or mobile.
    """)

st.info("🔒 All data is anonymized and used solely for demonstration purposes.")

# ------------------------------
# Features
# ------------------------------
st.markdown("## 🌟 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - Trained ML models: Logistic Regression, SVM, Random Forest  
    - Based on real patient data  
    - Interactive predictions with confidence score  
    - Explore data and visualize relationships  
    - Model performance metrics
    """)

with col2:
    st.markdown("""
    - Clean & responsive UI  
    - Dark/Light theme support  
    - Error handling for user inputs  
    - Loading animations for long tasks  
    - Built with Python, Streamlit, and Scikit-learn
    """)

# ------------------------------
# Medical Disclaimer
# ------------------------------
with st.expander("⚠️ Medical Disclaimer"):
    st.write("""
    This app is intended for **educational and informational** purposes only.  
    It is **not a substitute** for professional medical advice, diagnosis, or treatment.  
    Always consult a qualified healthcare provider regarding medical concerns.
    """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Python • 2025 © Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)

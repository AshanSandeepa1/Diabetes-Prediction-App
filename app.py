# app.py
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ---------------------- Config ---------------------- #
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    page_icon="🧠"
)

# ---------------------- Sidebar ---------------------- #
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Blue_circle_for_diabetes.svg/768px-Blue_circle_for_diabetes.svg.png",
    width=100,
)
st.sidebar.title("Navigation")
st.sidebar.markdown("🔹 Use the pages on the left to explore:")

# ---------------------- Lottie Animation ---------------------- #
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_diabetes = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_yr6zz3wv.json")

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

# Add animation
if lottie_diabetes:
    st_lottie(lottie_diabetes, height=300)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit, Scikit-learn & Python</small></center>",
    unsafe_allow_html=True
)

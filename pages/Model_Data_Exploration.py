import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Data Exploration", layout="wide", page_icon="🔎")
df = pd.read_csv(os.path.join("data", "diabetes.csv"))

# Ensure correct data types
for col in df.select_dtypes(include="Int64").columns:
    df[col] = df[col].astype("int64")

# ------------------ Defaults for Filters ------------------ #
defaults = {
    "age_range": (int(df["Age"].min()), int(df["Age"].max())),
    "preg_range": (int(df["Pregnancies"].min()), int(df["Pregnancies"].max())),
    "glucose_range": (int(df["Glucose"].min()), int(df["Glucose"].max())),
    "bp_range": (int(df["BloodPressure"].min()), int(df["BloodPressure"].max())),
    "skin_range": (int(df["SkinThickness"].min()), int(df["SkinThickness"].max())),
    "insulin_range": (int(df["Insulin"].min()), int(df["Insulin"].max())),
    "bmi_range": (float(df["BMI"].min()), float(df["BMI"].max())),
    "dpf_range": (float(df["DiabetesPedigreeFunction"].min()), float(df["DiabetesPedigreeFunction"].max())),
    "outcome_filter": list(df["Outcome"].unique())
}

# ------------------ Initialize Session State ------------------ #
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------ Reset Filters ------------------ #
with st.sidebar:
    st.header("🔎 Filter Options")
    if st.button("🔄 Reset Filters"):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

    with st.expander("👶 Demographics", expanded=True):
        st.session_state.age_range = st.slider("Age", *defaults["age_range"], value=st.session_state.age_range)
        st.session_state.preg_range = st.slider("Pregnancies", *defaults["preg_range"], value=st.session_state.preg_range)

    with st.expander("🧪 Medical Measures", expanded=False):
        st.session_state.glucose_range = st.slider("Glucose", *defaults["glucose_range"], value=st.session_state.glucose_range)
        st.session_state.bp_range = st.slider("Blood Pressure", *defaults["bp_range"], value=st.session_state.bp_range)
        st.session_state.skin_range = st.slider("Skin Thickness", *defaults["skin_range"], value=st.session_state.skin_range)
        st.session_state.insulin_range = st.slider("Insulin", *defaults["insulin_range"], value=st.session_state.insulin_range)

    with st.expander("⚖️ Health Indicators", expanded=False):
        st.session_state.bmi_range = st.slider("BMI", *defaults["bmi_range"], value=st.session_state.bmi_range)
        st.session_state.dpf_range = st.slider("Diabetes Pedigree Function", *defaults["dpf_range"], value=st.session_state.dpf_range)

    st.session_state.outcome_filter = st.multiselect(
        "🧬 Diabetes Outcome",
        options=df["Outcome"].unique(),
        default=st.session_state.outcome_filter,
        format_func=lambda x: "Positive (1)" if x == 1 else "Negative (0)"
    )

# ---------------------- Apply Filters ---------------------- #
filtered_df = df[
    (df["Pregnancies"].between(*st.session_state.preg_range)) &
    (df["Glucose"].between(*st.session_state.glucose_range)) &
    (df["BloodPressure"].between(*st.session_state.bp_range)) &
    (df["SkinThickness"].between(*st.session_state.skin_range)) &
    (df["Insulin"].between(*st.session_state.insulin_range)) &
    (df["BMI"].between(*st.session_state.bmi_range)) &
    (df["DiabetesPedigreeFunction"].between(*st.session_state.dpf_range)) &
    (df["Age"].between(*st.session_state.age_range)) &
    (df["Outcome"].isin(st.session_state.outcome_filter))
]

# ---------------------- Page Body ---------------------- #
st.header("📊 Data Exploration")
st.markdown("Use the filter panel to explore and analyze the diabetes dataset interactively.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", df.shape[0])
    st.metric("Filtered Records", filtered_df.shape[0])
with col2:
    st.write("**Column Types:**")
    st.dataframe(df.dtypes.rename("Type"))

# -------------- Sample Section --------------- #
st.subheader("📌 Sample Records (10 Random Rows)")
if "sample_data" not in st.session_state:
    st.session_state.sample_data = df.sample(10, random_state=42)
if st.button("🔄 Shuffle Sample"):
    st.session_state.sample_data = df.sample(10)
st.dataframe(st.session_state.sample_data)

# -------------- Filtered Data Section --------------- #
st.subheader("🧮 Filtered Data View")
st.dataframe(filtered_df, use_container_width=True)

with st.expander("📊 Show Descriptive Statistics for Filtered Data"):
    st.dataframe(filtered_df.describe())

@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

csv = convert_df_to_csv(filtered_df)
st.download_button("📥 Download Filtered Data as CSV", data=csv, file_name="filtered_diabetes_data.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Python @ 2025 Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)
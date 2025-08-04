import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.express as px

# ---------------------- Page Config ---------------------- #
st.set_page_config(page_title="Visualizations", layout="wide", page_icon="📊")

# ---------------------- Load Dataset ---------------------- #
df = pd.read_csv(os.path.join("data", "diabetes.csv"))

# ---------------------- Sidebar Filters ---------------------- #
st.sidebar.header("📊 Visualization Filters")


outcome_filter = st.sidebar.multiselect(
    "Filter by Diabetes Outcome",
    options=df["Outcome"].unique(),
    default=df["Outcome"].unique(),
)

st.sidebar.markdown("### Outcome Legend")
st.sidebar.markdown("""
- `1`: 🔴 Diabetes
- `0`: 🟢 No Diabetes  
""")

# Apply filter
filtered_df = df[df["Outcome"].isin(outcome_filter)]

# ---------------------- Main Page ---------------------- #
st.header("📈 Data Visualizations")
st.markdown("""
Visualize key relationships and distributions within the diabetes dataset.
Use the sidebar to filter visualizations based on diabetes outcome.
""")

# ---------------------- Glucose Histogram ---------------------- #
with st.expander("🔍 Glucose Distribution by Outcome", expanded=True):
    fig1 = px.histogram(
        filtered_df,
        x="Glucose",
        color="Outcome",
        barmode="overlay",
        nbins=30,
        title="Distribution of Glucose Levels by Diabetes Outcome",
        labels={"Glucose": "Glucose Level", "Outcome": "Diabetes Outcome"},
    )
    st.plotly_chart(fig1, use_container_width=True)

# ---------------------- BMI Box Plot ---------------------- #
with st.expander("📦 BMI Distribution by Outcome", expanded=False):
    fig2 = px.box(
        filtered_df,
        y="BMI",
        color="Outcome",
        title="BMI Distribution by Diabetes Outcome",
        labels={"BMI": "Body Mass Index", "Outcome": "Diabetes Outcome"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------- Correlation Heatmap ---------------------- #
with st.expander("🧮 Correlation Heatmap (All Variables)", expanded=False):
    fig3, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig3)

# ---------------------- Optional Pair Plot ---------------------- #
with st.expander("🔗 Pairwise Relationships (Glucose, BMI, Age, Outcome)", expanded=False):
    selected_cols = ["Glucose", "BMI", "Age", "Outcome"]
    fig4 = px.scatter_matrix(
        filtered_df[selected_cols],
        dimensions=selected_cols[:-1],
        color=filtered_df["Outcome"].astype(str),
        title="Scatter Matrix of Selected Features by Outcome",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------- Dynamic Boxplot Selector ---------------------- #
with st.expander("📦 Select Feature for Boxplot by Outcome", expanded=True):
    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    selected_feature = st.selectbox("Select Numeric Feature", options=numeric_cols)

    fig5 = px.box(
        filtered_df,
        y=selected_feature,
        color="Outcome",
        points="all",
        title=f"Distribution of {selected_feature} by Diabetes Outcome",
        labels={selected_feature: selected_feature, "Outcome": "Diabetes Outcome"},
    )
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------- Notes ---------------------- #
st.markdown("---")
st.info("You can interact with charts by zooming, hovering, or filtering using the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit & Python @ 2025 Ashan Sandeepa</small></center>",
    unsafe_allow_html=True
)
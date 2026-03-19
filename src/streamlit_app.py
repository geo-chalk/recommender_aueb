import streamlit as st

st.set_page_config(page_title="Recommender System", layout="wide")

st.title("Welcome to the Restaurant Recommender")
st.markdown("""
### System Overview
This application is powered by a **CatBoost** model managed via **MLflow**.
- **Dashboard**: Check system health and see which model version is "Champion".
- **Predictor**: Upload data to get real-time personalized restaurant rankings.
""")

st.info("Select a page from the sidebar to begin.")
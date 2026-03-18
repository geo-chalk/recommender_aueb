import streamlit as st
import pandas as pd
import requests
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:8081")

st.title("🚀 Prediction Performance")
st.write("Upload a Parquet file to see the model in action.")

uploaded_file = st.file_uploader("Choose a file", type=["parquet"])

if uploaded_file:
    df = pd.read_parquet(uploaded_file)
    st.dataframe(df.head(5))

    if st.button("Generate Match Scores"):
        with st.spinner("Analyzing data..."):
            # Reset file pointer and send to FastAPI
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), "application/octet-stream")}

            response = requests.post(f"{API_URL}/api/v1/predict-parquet", files=files)

            if response.status_code == 200:
                predictions = response.json().get("predictions")
                df["Match_Score"] = predictions

                st.success("Analysis Complete")
                st.subheader("Top Recommendations")
                st.dataframe(df.sort_index())
            else:
                st.error("Prediction failed. Check API logs.")
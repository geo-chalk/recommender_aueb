import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv("API_URL", "http://recommender-api:8081")

st.set_page_config(layout="wide")
st.title("🚀 Real-Time Prediction Simulator")

st.write("Upload a Parquet file to see the model in action. The request will be sent in a JSON format.")

uploaded_file = st.file_uploader("Upload test data for simulation", type=["csv", "parquet"])

if uploaded_file:
    # Load data locally for simulation
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(5))

    # Simulation settings
    num_rows = st.slider("Number of rows to send (Simulate batch)", 1, 50, 5)

    if st.button("🚀 Run Real-Time Inference"):
        # Select the subset of data to simulate the request
        data_to_send = df.head(num_rows).to_dict(orient="records")

        # --- NEW SECTION: JSON Preview Box ---
        st.subheader("📤 Request Payload")
        with st.expander("View JSON being sent to API", expanded=False):
            st.json(data_to_send)
        # -------------------------------------

        with st.spinner(f"Sending {num_rows} records as JSON..."):
            try:
                # Send standard JSON POST request
                response = requests.post(f"{API_URL}/api/v1/predict", json=data_to_send)

                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Predictions received from Model: {results.get('model_used', 'Unknown')} (v{results.get('version', '?')})")

                    # Display results next to inputs
                    st.subheader("📥 Prediction Results")
                    output_df = df.head(num_rows).copy()
                    output_df["Prediction"] = results["predictions"]
                    st.table(output_df)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")
import streamlit as st
import requests
import time
import os

API_URL = os.getenv("API_URL", "http://localhost:8081")

st.title("🏥 System Health & API Info")

try:
    start_time = time.time()
    response = requests.get(f"{API_URL}/health")
    latency = (time.time() - start_time) * 1000

    if response.status_code == 200:
        data = response.json()
        st.success(f"API is Online (Latency: {latency:.2f}ms)")

        col1, col2 = st.columns(2)
        col1.metric("Champion Model", f"v{data['model_details']['version']}")
        col2.metric("Registry Alias", data['model_details']['alias'])

        st.divider()
        st.subheader("Raw API Response")
        st.json(data)
    else:
        st.warning("API reachable but reporting unhealthy status.")
except Exception as e:
    st.error(f"Connection failed: {e}")
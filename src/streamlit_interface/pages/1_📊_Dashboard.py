import streamlit as st
import requests
import time
import os
st.set_page_config(layout="centered")


# Internal network URL for Docker
API_URL = os.getenv("API_URL", "http://recommender-api:8081")


# --- 1. THE FIX: Callback Function ---
# This runs BEFORE the script reruns, ensuring the API is updated first.
def handle_model_switch():
    # Access values directly from session state via widget keys
    m_name = st.session_state.model_name_input
    m_alias = st.session_state.model_alias_input

    try:
        # Match the endpoint in your app.py
        response = requests.post(
            f"{API_URL}/api/v1/switch-model",
            params={"model_name": m_name, "alias": m_alias}
        )
        if response.status_code == 200:
            st.toast(f"Successfully switched to {m_name}!", icon="✅")
        else:
            st.error(f"Failed to switch: {response.text}")
    except Exception as e:
        st.error(f"Error during switch: {e}")


st.title("📊 System Performance")

# --- 2. Fetch Status ---
# Because of the callback, this will now catch the NEW data on the first pass.
try:
    start_time = time.time()
    res = requests.get(f"{API_URL}/health")
    latency = (time.time() - start_time) * 1000

    if res.status_code == 200:
        data = res.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", data["status"].upper())

        # Access nested model metadata
        model_ver = data.get("model_details", {}).get("version", "?")
        col2.metric("Model Version", f"v{model_ver}")
        col3.metric("Latency", f"{latency:.2f}ms")
    else:
        st.error("API is online but reporting unhealthy status.")
except Exception as e:
    st.error(f"Cannot connect to API: {e}")

st.divider()

# --- 3. UI Form ---
st.subheader("🔄 Change Active Model")
with st.form("load_model_form"):
    # 'key' links these inputs to st.session_state for the callback
    st.text_input("Registered Model Name", value="recommender_model", key="model_name_input")
    st.text_input("Alias", value="champion", key="model_alias_input")

    # Use 'on_click' to trigger the callback logic
    st.form_submit_button("Switch Model", on_click=handle_model_switch)
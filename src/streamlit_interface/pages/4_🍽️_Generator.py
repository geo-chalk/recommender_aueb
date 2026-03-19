import streamlit as st
import pandas as pd
import requests
import os
import plotly.graph_objects as go

# Configuration for Docker/Local networking
API_URL = os.getenv("API_URL", "http://recommender-api:8081")

st.set_page_config(layout="wide", page_title="Restaurant Generator")

st.title("🍽️ Synthetic Restaurant Generator")
st.write("Generate and visualize synthetic data using Plotly Graph Objects.")

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("Generation Parameters")
    rest_num = st.slider("Number of Restaurants", min_value=1, max_value=1000, value=100)

    st.subheader("Ratings & Delivery")
    rating_mean = st.slider("Average Rating Mean", 0.0, 5.0, 3.7)
    rating_std = st.slider("Average Rating Std", 0.1, 2.0, 1.0)
    del_mean = st.number_input("Avg Delivery Time (Mean)", value=30)
    del_std = st.number_input("Avg Delivery Time (Std)", value=7)

    st.subheader("Costs")
    min_cost_mean = st.number_input("Min Cost (Mean)", value=6)
    min_cost_std = st.number_input("Min Cost (Std)", value=3)

# --- 2. API Trigger ---
if st.button("🚀 Generate Restaurants"):
    # Construct the JSON payload matching RestaurantConfig
    payload = {
        "rest_num": rest_num,
        "max_cuisines": 3,
        "avg_rating": [rating_mean, rating_std],
        "avg_del_time": [del_mean, del_std],
        "min_cost": [min_cost_mean, min_cost_std],
        "cuisines": ['Pizza', 'Burger', 'Pasta', 'Souvlaki', 'Sushi', 'Chinese'],
        "price_samples": [0.5, 0.3, 0.15, 0.05],
        "has_extra_del_cost_prob": 0.2,
        "payment_methods": [["CASH", "CARD"], ['COUPON']]
    }

    with st.spinner("Generating..."):
        try:
            # POST request to modular endpoint
            response = requests.post(f"{API_URL}/api/v1/generate_restaurants/", json=payload)
            if response.status_code == 200:
                df = pd.DataFrame(response.json())
                st.session_state["generated_restaurants"] = df
                st.success(f"Generated {len(df)} restaurants.")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# --- 3. Visualization Section (Plotly Graph Objects) ---
if "generated_restaurants" in st.session_state:
    df = st.session_state["generated_restaurants"]
    st.divider()

    # Selection Controls
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        feat_1 = st.selectbox("Feature for Plot 1", df.columns, index=6)  # avg_rating
        type_1 = st.selectbox("Type 1", ["Histogram", "Box", "Scatter"], key="t1")
    with ctrl_col2:
        feat_2 = st.selectbox("Feature for Plot 2", df.columns, index=5)  # min_cost
        type_2 = st.selectbox("Type 2", ["Histogram", "Box", "Scatter"], key="t2")


    # Helper function to build GO figures
    def create_go_plot(plot_type, feature, color):
        fig = go.Figure()

        if plot_type == "Histogram":
            fig.add_trace(go.Histogram(x=df[feature], marker_color=color, name=feature))
            fig.update_layout(xaxis_title=feature, yaxis_title="Count")
        elif plot_type == "Box":
            fig.add_trace(go.Box(y=df[feature], marker_color=color, name=feature))
            fig.update_layout(yaxis_title=feature)
        else:  # Scatter
            fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='markers', marker_color=color, name=feature))
            fig.update_layout(xaxis_title="Index", yaxis_title=feature)

        fig.update_layout(title=f"{plot_type} of {feature}", template="plotly_white",
                          margin=dict(l=20, r=20, t=40, b=20))
        return fig


    # Render Side-by-Side
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.plotly_chart(create_go_plot(type_1, feat_1, "#636EFA"), use_container_width=True)
    with fig_col2:
        st.plotly_chart(create_go_plot(type_2, feat_2, "#EF553B"), use_container_width=True)

    with st.expander("View Raw Data"):
        st.dataframe(df)
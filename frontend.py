# Streamlit frontend for solar prediction (Enhanced UI)
import streamlit as st
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Sidebar ---
st.sidebar.image(
    "https://img.icons8.com/fluency/96/solar-panel.png",
    width=80,
)
st.sidebar.title("Solar Power Prediction")
st.sidebar.markdown(
    """
<span style='font-size: 14px;'>
Select your desired date, time, API source, and prediction models. Click <b>Predict for Next Day</b> to view the results.
</span>
""",
    unsafe_allow_html=True,
)

date = st.sidebar.date_input("Select Date", datetime.date.today())
time = st.sidebar.time_input("Select Time", datetime.datetime.now().time())

api_options = ["NASA", "Local Weather", "Custom API"]
api_choice = st.sidebar.selectbox("Select API Source", api_options)

model_folder = "save_model"
available_models = [
    ("LSTM", "lstm_model.h5"),
    ("CNN-LSTM", "cnn_lstm_model.h5"),
    ("Random Forest", "random_forest_model.pkl"),
    ("XGBoost", "xgboost_model.pkl"),
]
model_files = [
    m for m in available_models if os.path.exists(os.path.join(model_folder, m[1]))
]
if not model_files:
    model_files = available_models
model_names = [m[0] for m in model_files]
selected_models = st.sidebar.multiselect(
    "Select Models", model_names, default=model_names
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Date:** {date}")
st.sidebar.write(f"**Time:** {time}")
st.sidebar.write(f"**API:** {api_choice}")
st.sidebar.write(
    f"**Models:** {', '.join(selected_models) if selected_models else 'None'}"
)

# --- Main Content ---
st.markdown(
    """
<h1 style='text-align: center; color: #2c3e50;'>Solar Power Prediction Dashboard</h1>
<p style='text-align: center; color: #555;'>
Easily compare the performance of multiple models for next-day solar output prediction.<br>
<span style='font-size: 16px;'>Select your options from the sidebar and click <b>Predict for Next Day</b>.</span>
</p>
<hr>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Prediction Graph", "Instructions"])

with tab2:
    st.markdown(
        """
    ### How to Use
    1. **Select Date & Time:** Choose the date and time for which you want to base the prediction.
    2. **Select API Source:** Pick the data source for weather/solar data.
    3. **Select Models:** Choose one or more models to compare their predictions.
    4. **Click 'Predict for Next Day':** View the predicted solar output for the next day, hour by hour, for each selected model.
    5. **Compare Results:** The graph will show the predictions for each model in different colors.
    """
    )
    st.info(
        "Model files must be present in the 'save_model' folder. This demo uses simulated predictions."
    )

with tab1:
    st.header("Prediction and Model Performance")
    st.markdown(
        "<span style='color: #888;'>Prediction results for the next day (hourly):</span>",
        unsafe_allow_html=True,
    )
    if st.button("Predict for Next Day"):
        hours = np.arange(24)
        predictions = {}
        for model in selected_models:
            predictions[model] = np.random.uniform(0, 1, size=24)
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, preds in predictions.items():
            ax.plot(hours, preds, label=model, marker="o")
        ax.set_xticks(hours)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Predicted Solar Output", fontsize=12)
        ax.set_title("Next Day Solar Output Prediction", fontsize=14, color="#2c3e50")
        # Move legend to upper right outside the plot
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Select input and click 'Predict for Next Day' to see results.")

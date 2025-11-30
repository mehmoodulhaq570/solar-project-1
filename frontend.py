# Streamlit frontend for solar prediction
import streamlit as st
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Solar Power Prediction Dashboard")

# Sidebar for user input
st.sidebar.header("User Input")

# Date and time selection
date = st.sidebar.date_input("Select Date", datetime.date.today())
time = st.sidebar.time_input("Select Time", datetime.datetime.now().time())

# API dropdown (example options, update as needed)
api_options = ["NASA", "Local Weather", "Custom API"]
api_choice = st.sidebar.selectbox("Select API Source", api_options)

# Model selection (multiple selection allowed)
model_folder = "save_model"
available_models = [
    ("LSTM", "lstm_model.h5"),
    ("CNN-LSTM", "cnn_lstm_model.h5"),
    ("Random Forest", "random_forrest_model.pkl"),
    ("XGBoost", "xgboost_model.pkl"),
]
model_files = [
    m for m in available_models if os.path.exists(os.path.join(model_folder, m[1]))
]
if not model_files:
    model_files = available_models  # Show all if none found, for UI demo
model_names = [m[0] for m in model_files]
selected_models = st.sidebar.multiselect(
    "Select Models", model_names, default=model_names
)

# Placeholder for model performance and prediction
st.header("Prediction and Model Performance")

if st.button("Predict for Next Day"):
    # Placeholder: Simulate predictions for each selected model
    hours = np.arange(24)
    predictions = {}
    for model in selected_models:
        # Simulate random predictions
        predictions[model] = np.random.uniform(0, 1, size=24)

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, preds in predictions.items():
        ax.plot(hours, preds, label=model)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Solar Output")
    ax.set_title("Next Day Solar Output Prediction")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Select input and click 'Predict for Next Day' to see results.")

# Flexibility: Show selected options
st.sidebar.markdown("---")
st.sidebar.write(f"**Date:** {date}")
st.sidebar.write(f"**Time:** {time}")
st.sidebar.write(f"**API:** {api_choice}")
st.sidebar.write(
    f"**Models:** {', '.join(selected_models) if selected_models else 'None'}"
)

# Streamlit frontend for solar prediction (Enhanced UI)

import streamlit as st
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb


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
        # --- Load last 24 hours from lahore_hourly_filled.csv ---
        try:
            df = pd.read_csv("lahore_hourly_filled.csv", parse_dates=["datetime"])
            df = df.sort_values("datetime")
            df = df.apply(pd.to_numeric, errors="coerce")
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df.set_index("datetime", inplace=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Filter for daylight only (SolarZenith < 90)
        if "SolarZenith" in df.columns:
            df_day = df[df["SolarZenith"] < 90].copy()
        else:
            st.error("'SolarZenith' column not found in data.")
            st.stop()
        df_day.dropna(subset=["SolarRadiation"], inplace=True)
        df_day["hour"] = df_day.index.hour
        df_day["month"] = df_day.index.month

        # Get last 24 daylight records
        last_24 = df_day.iloc[-24:].copy()
        if len(last_24) < 24:
            st.error("Not enough daylight records for prediction (need 24).")
            st.stop()

        # Prepare input for tree models
        def make_lag_features(series_df, max_lag=24):
            cols_base = [
                "SolarRadiation",
                "hour",
                "month",
                "Temperature",
                "HumiditySpecific",
                "HumidityRelative",
                "Pressure",
                "WindSpeed",
                "WindDirection",
            ]
            df_feat = series_df[cols_base].copy()
            for lag in range(1, max_lag + 1):
                df_feat[f"lag_{lag}"] = df_feat["SolarRadiation"].shift(lag)
            return df_feat.dropna()

        # Prepare input for deep models
        features = [
            "hour",
            "month",
            "Temperature",
            "HumiditySpecific",
            "HumidityRelative",
            "Pressure",
            "WindSpeed",
            "WindDirection",
            "SolarRadiation",
        ]
        seq_input = last_24[features].dropna().copy()
        if len(seq_input) < 24:
            st.error("Not enough valid records for deep model input.")
            st.stop()

        # Load scalers
        try:
            scaler_X = joblib.load(os.path.join("save_model", "scaler_X.pkl"))
            scaler_y = joblib.load(os.path.join("save_model", "scaler_y.pkl"))
        except Exception as e:
            st.error(f"Error loading scalers: {e}")
            st.stop()

        # Prepare input for tree models (XGBoost, Random Forest)
        lag_input = make_lag_features(last_24, 24)
        if lag_input.empty:
            st.error("Not enough lag data for tree-based models.")
            st.stop()
        X_pred_tree = lag_input.drop(columns=["SolarRadiation"])

        # Prepare input for deep models (LSTM, CNN-LSTM)
        X_pred_seq = scaler_X.transform(seq_input.drop(columns=["SolarRadiation"]))
        X_pred_seq = X_pred_seq.reshape(1, 24, -1)

        # --- Predict next 24 hours ---
        predictions = {}
        hours = np.arange(24)
        # XGBoost
        if "XGBoost" in selected_models:
            try:
                xgb_model = joblib.load(os.path.join("save_model", "xgboost_model.pkl"))
                preds = []
                # For each next hour, roll the input and append prediction
                lag_data = last_24.copy()
                for i in range(24):
                    lag_feats = make_lag_features(lag_data, 24)
                    if lag_feats.empty:
                        preds.append(np.nan)
                        continue
                    X_pred = lag_feats.drop(columns=["SolarRadiation"]).iloc[[-1]]
                    pred = xgb_model.predict(X_pred)[0]
                    preds.append(pred)
                    # Append prediction for next lag
                    new_row = lag_data.iloc[-1].copy()
                    new_row["SolarRadiation"] = pred
                    new_row.name = lag_data.index[-1] + pd.Timedelta(hours=1)
                    lag_data = pd.concat([lag_data, pd.DataFrame([new_row])])
                predictions["XGBoost"] = preds
            except Exception as e:
                st.warning(f"XGBoost prediction error: {e}")

        # Random Forest
        if "Random Forest" in selected_models:
            try:
                rf_model = joblib.load(
                    os.path.join("save_model", "random_forest_model.pkl")
                )
                preds = []
                lag_data = last_24.copy()
                for i in range(24):
                    lag_feats = make_lag_features(lag_data, 24)
                    if lag_feats.empty:
                        preds.append(np.nan)
                        continue
                    X_pred = lag_feats.drop(columns=["SolarRadiation"]).iloc[[-1]]
                    pred = rf_model.predict(X_pred)[0]
                    preds.append(pred)
                    new_row = lag_data.iloc[-1].copy()
                    new_row["SolarRadiation"] = pred
                    new_row.name = lag_data.index[-1] + pd.Timedelta(hours=1)
                    lag_data = pd.concat([lag_data, pd.DataFrame([new_row])])
                predictions["Random Forest"] = preds
            except Exception as e:
                st.warning(f"Random Forest prediction error: {e}")

        # LSTM
        if "LSTM" in selected_models:
            try:
                lstm_model = load_model(os.path.join("save_model", "lstm_model.h5"))
                preds = []
                seq = X_pred_seq.copy()
                for i in range(24):
                    pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                    preds.append(pred)
                    # Roll sequence for next hour
                    next_features = (
                        seq_input.drop(columns=["SolarRadiation"]).iloc[-23:].values
                    )
                    next_row = np.append(next_features[-1], pred)
                    seq_input_next = np.vstack([next_features, next_row])
                    seq = scaler_X.transform(seq_input_next)
                    seq = seq.reshape(1, 24, -1)
                predictions["LSTM"] = preds
            except Exception as e:
                st.warning(f"LSTM prediction error: {e}")

        # CNN-LSTM
        if "CNN-LSTM" in selected_models:
            try:
                cnn_lstm_model = load_model(
                    os.path.join("save_model", "cnn_lstm_model.h5")
                )
                preds = []
                seq = X_pred_seq.copy()
                for i in range(24):
                    pred_scaled = cnn_lstm_model.predict(seq, verbose=0)[0][0]
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                    preds.append(pred)
                    next_features = (
                        seq_input.drop(columns=["SolarRadiation"]).iloc[-23:].values
                    )
                    next_row = np.append(next_features[-1], pred)
                    seq_input_next = np.vstack([next_features, next_row])
                    seq = scaler_X.transform(seq_input_next)
                    seq = seq.reshape(1, 24, -1)
                predictions["CNN-LSTM"] = preds
            except Exception as e:
                st.warning(f"CNN-LSTM prediction error: {e}")

        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, preds in predictions.items():
            ax.plot(hours, preds, label=model, marker="o")
        ax.set_xticks(hours)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Predicted Solar Output", fontsize=12)
        ax.set_title("Next Day Solar Output Prediction", fontsize=14, color="#2c3e50")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Select input and click 'Predict for Next Day' to see results.")

# ===============================================
# Streamlit frontend for solar prediction (Updated)
# Supports future date selection and rolling forecast
# ===============================================

import streamlit as st
import os, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/96/solar-panel.png", width=80)
st.sidebar.title("Solar Power Prediction")
st.sidebar.markdown("""
Select date, time, API source, and models. Click 'Predict' to view hourly predictions.
""")

# --- Date & time input ---
selected_date = st.sidebar.date_input("Forecast Date", datetime.date.today())
selected_time = st.sidebar.time_input("Forecast Time", datetime.datetime.now().time())

api_choice = st.sidebar.selectbox("API Source", ["NASA", "Local Weather", "Custom API"])

# --- Model selection ---
model_folder = "save_model"
available_models = [
    ("LSTM", "lstm_model.h5"),
    ("CNN-LSTM", "cnn_lstm_model.h5"),
    ("Random Forest", "random_forest_model.pkl"),
    ("XGBoost", "xgboost_model.pkl")
]
# Check if files exist
model_files = [m for m in available_models if os.path.exists(os.path.join(model_folder, m[1]))]
if not model_files: model_files = available_models
model_names = [m[0] for m in model_files]
selected_models = st.sidebar.multiselect("Select Models", model_names, default=model_names)

st.sidebar.markdown("---")
st.sidebar.write(f"**Date:** {selected_date}")
st.sidebar.write(f"**Time:** {selected_time}")
st.sidebar.write(f"**Models:** {', '.join(selected_models) if selected_models else 'None'}")

# --- Main ---
st.markdown("<h1 style='text-align:center;color:#2c3e50;'>Solar Power Prediction Dashboard</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Graph", "Instructions"])

with tab2:
    st.markdown("""
### How to Use
1. Select forecast date & time.
2. Pick API source for weather data.
3. Choose models.
4. Click 'Predict' to see hourly solar radiation forecast.
""")
    st.info("Ensure trained model files exist in 'save_model' folder.")

with tab1:
    st.header("Hourly Solar Forecast")

    if st.button("Predict"):

        # --- Load historical data ---
        try:
            df = pd.read_csv("NASA meteriological and solar radiaton data/lahore_hourly_filled.csv",
                             parse_dates=["datetime"])
            df = df.sort_values("datetime")
            df = df.apply(pd.to_numeric, errors="coerce")
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df.set_index("datetime", inplace=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        if "SolarZenith" not in df.columns:
            st.error("'SolarZenith' column missing in data.")
            st.stop()

        df_day = df[df["SolarZenith"] < 90].copy()
        df_day.dropna(subset=["SolarRadiation"], inplace=True)
        df_day["hour"] = df_day.index.hour
        df_day["month"] = df_day.index.month

        # --- Calculate how many hours ahead the selected date is ---
        last_time = df_day.index.max()
        target_datetime = pd.Timestamp(datetime.datetime.combine(selected_date, selected_time))
        hours_ahead = int((target_datetime - last_time).total_seconds() / 3600)

        if hours_ahead < 0:
            st.error("Selected date must be after last available historical data.")
            st.stop()
        elif hours_ahead > 30*24:
            st.warning("Forecast is far beyond training data. Accuracy may be low.")

        # --- Prepare scalers ---
        try:
            scaler_X = joblib.load(os.path.join(model_folder, "scaler_X.pkl"))
            scaler_y = joblib.load(os.path.join(model_folder, "scaler_y.pkl"))
        except Exception as e:
            st.error(f"Error loading scalers: {e}")
            st.stop()

        # --- Features ---
        features = ["hour","month","Temperature","HumiditySpecific","HumidityRelative",
                    "Pressure","WindSpeed","WindDirection","SolarRadiation"]

        def make_lag_features(series_df, max_lag=24):
            df_feat = series_df[features[:-1]].copy()
            df_feat["SolarRadiation"] = series_df["SolarRadiation"]
            for lag in range(1, max_lag+1):
                df_feat[f"lag_{lag}"] = df_feat["SolarRadiation"].shift(lag)
            return df_feat.dropna()

        # --- Recursive forecasting ---
        forecast_hours = 24
        total_steps = hours_ahead + forecast_hours
        future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=total_steps, freq="H")

        predictions = {model: [] for model in selected_models}

        rolling_df = df_day.tail(24).copy()  # last 24 hours as initial input

        for i, ts in enumerate(future_times):
            # Create lag features
            lag_input = make_lag_features(rolling_df, min(24, len(rolling_df)-1))
            if lag_input.empty:
                lag_input_row = None
            else:
                lag_input_row = lag_input.drop(columns=["SolarRadiation"]).iloc[[-1]]

            # --- Predict for each model ---
            for model in selected_models:
                pred = 0
                if model == "XGBoost" and XGBOOST_AVAILABLE and lag_input_row is not None:
                    xgb_model = joblib.load(os.path.join(model_folder, "xgboost_model.pkl"))
                    pred = xgb_model.predict(lag_input_row)[0]
                elif model == "Random Forest" and lag_input_row is not None:
                    rf_model = joblib.load(os.path.join(model_folder, "random_forest_model.pkl"))
                    pred = rf_model.predict(lag_input_row)[0]
                elif model == "LSTM" and TENSORFLOW_AVAILABLE:
                    lstm_model = load_model(os.path.join(model_folder, "lstm_model.h5"))
                    seq_input = rolling_df[features].tail(24)
                    X_seq = scaler_X.transform(seq_input.drop(columns=["SolarRadiation"])).reshape(1, 24, -1)
                    pred_scaled = lstm_model.predict(X_seq, verbose=0)[0][0]
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                elif model == "CNN-LSTM" and TENSORFLOW_AVAILABLE:
                    cnn_model = load_model(os.path.join(model_folder, "cnn_lstm_model.h5"))
                    seq_input = rolling_df[features].tail(24)
                    X_seq = scaler_X.transform(seq_input.drop(columns=["SolarRadiation"])).reshape(1, 24, -1)
                    pred_scaled = cnn_model.predict(X_seq, verbose=0)[0][0]
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

                pred = max(pred, 0)
                # Only daylight
                if ts.hour < 6 or ts.hour > 18:
                    pred = 0

                predictions[model].append(pred)

            # Append for next lag
            new_row = rolling_df.iloc[-1].copy()
            new_row.name = ts
            new_row["hour"] = ts.hour
            new_row["month"] = ts.month
            new_row["SolarRadiation"] = np.mean([predictions[m][-1] for m in selected_models])
            rolling_df = pd.concat([rolling_df, pd.DataFrame([new_row])])

        # --- Plot only selected date ---
        forecast_df = pd.DataFrame(predictions, index=future_times)
        selected_day_df = forecast_df[forecast_df.index.date == selected_date]

        if selected_day_df.empty:
            st.warning("Selected date not within forecast range.")
        else:
            fig, ax = plt.subplots(figsize=(11,5))
            for model in selected_models:
                ax.plot(selected_day_df.index.hour, selected_day_df[model], label=model, marker="o")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Solar Radiation")
            ax.set_title(f"Solar Forecast for {selected_date}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig, use_container_width=True)

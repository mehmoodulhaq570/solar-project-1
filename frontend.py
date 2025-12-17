# ===============================================
# Streamlit frontend for solar prediction (Updated)
# Using working model loading approach from trend.py
# ===============================================

import streamlit as st
import os, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
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
st.sidebar.markdown(
    """
Select date, time, API source, and models. Click 'Predict' to view hourly predictions.
"""
)

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
    ("XGBoost", "xgboost_model.pkl"),
]
# Check if files exist
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
st.sidebar.write(f"**Date:** {selected_date}")
st.sidebar.write(f"**Time:** {selected_time}")
st.sidebar.write(
    f"**Models:** {', '.join(selected_models) if selected_models else 'None'}"
)

# --- Main ---
st.markdown(
    "<h1 style='text-align:center;color:#2c3e50;'>Solar Power Prediction Dashboard</h1>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Prediction Graph", "Instructions"])

with tab2:
    st.markdown(
        """
### How to Use
1. Select forecast date & time.
2. Pick API source for weather data.
3. Choose models.
4. Click 'Predict' to see hourly solar radiation forecast.
"""
    )
    st.info("Ensure trained model files exist in 'save_model' folder.")

with tab1:
    st.header("Hourly Solar Forecast")

    if st.button("Predict"):

        if not selected_models:
            st.error("Please select at least one model.")
            st.stop()

        # ---------- 1. Load historical data ----------
        try:
            df = pd.read_csv(
                "NASA meteriological and solar radiaton data/lahore_hourly_filled.csv",
                parse_dates=["datetime"],
                dayfirst=True,
                index_col="datetime",
            )
            df = df.sort_index().apply(pd.to_numeric, errors="coerce")
            df.index = pd.to_datetime(df.index, dayfirst=True)

            # Use ALL hours (not just daylight) - model should know night = 0
            df_day = df.copy()
            df_day.dropna(subset=["SolarRadiation"], inplace=True)
            df_day["hour"] = df_day.index.hour
            df_day["month"] = df_day.index.month
            # Add cyclical encoding for hour and month
            df_day["hour_sin"] = np.sin(2 * np.pi * df_day["hour"] / 24)
            df_day["hour_cos"] = np.cos(2 * np.pi * df_day["hour"] / 24)
            df_day["month_sin"] = np.sin(2 * np.pi * df_day["month"] / 12)
            df_day["month_cos"] = np.cos(2 * np.pi * df_day["month"] / 12)

            target_col = "SolarRadiation"

            st.success(f"✅ Loaded {len(df)} records")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # ---------- 2. Load trained models ----------
        model_folder = "save_model"
        loaded_models = {}

        try:
            if "XGBoost" in selected_models and XGBOOST_AVAILABLE:
                loaded_models["XGBoost"] = joblib.load(
                    f"{model_folder}/xgboost_model.pkl"
                )
                st.success("✅ XGBoost loaded")
        except Exception as e:
            st.warning(f"⚠️ XGBoost: {e}")

        try:
            if "Random Forest" in selected_models:
                loaded_models["Random Forest"] = joblib.load(
                    f"{model_folder}/random_forest_model.pkl"
                )
                st.success("✅ Random Forest loaded")
        except Exception as e:
            st.warning(f"⚠️ Random Forest: {e}")

        try:
            if "LSTM" in selected_models and TENSORFLOW_AVAILABLE:
                loaded_models["LSTM"] = load_model(f"{model_folder}/lstm_model.h5")
                st.success("✅ LSTM loaded")
        except Exception as e:
            st.warning(f"⚠️ LSTM: {e}")

        try:
            if "CNN-LSTM" in selected_models and TENSORFLOW_AVAILABLE:
                loaded_models["CNN-LSTM"] = load_model(
                    f"{model_folder}/cnn_lstm_model.h5"
                )
                st.success("✅ CNN-LSTM loaded")
        except Exception as e:
            st.warning(f"⚠️ CNN-LSTM: {e}")

        if not loaded_models:
            st.error(
                "❌ No models could be loaded. Check model files and package versions."
            )
            st.stop()

        # Load scalers for deep learning models
        scaler_X = None
        scaler_y = None
        if "LSTM" in loaded_models or "CNN-LSTM" in loaded_models:
            try:
                scaler_X = joblib.load(f"{model_folder}/scaler_X.pkl")
                scaler_y = joblib.load(f"{model_folder}/scaler_y.pkl")
                st.success("✅ Scalers loaded")
            except Exception as e:
                st.warning(f"⚠️ Scalers: {e}. Deep learning predictions may fail.")

        # ---------- 3. Forecast settings ----------
        MAX_LAG = 24
        SEQ_LEN = 24

        # Extract date components
        year = selected_date.year
        month = selected_date.month
        day = selected_date.day

        # Generate hourly datetime index for the day
        date_index = pd.date_range(
            start=f"{year}-{month:02d}-{day:02d} 00:00",
            end=f"{year}-{month:02d}-{day:02d} 23:00",
            freq="h",
        )
        hours = date_index.hour

        # ---------- 4. Tree-based model predictions ----------
        tree_predictions = {"XGBoost": [], "Random Forest": []}

        if "XGBoost" in loaded_models or "Random Forest" in loaded_models:
            last_hist_values = df_day[target_col].values
            tree_series = last_hist_values[-MAX_LAG:].tolist()

            # Use last known weather values
            Temperature = df_day["Temperature"].iloc[-1]
            HumiditySpecific = df_day["HumiditySpecific"].iloc[-1]
            HumidityRelative = df_day["HumidityRelative"].iloc[-1]
            Pressure = df_day["Pressure"].iloc[-1]
            WindSpeed = df_day["WindSpeed"].iloc[-1]
            WindDirection = df_day["WindDirection"].iloc[-1]

            for h in range(24):
                hour = hours[h]
                # Compute cyclical features for current hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)

                lag_feats = tree_series[-MAX_LAG:]
                # Features order must match training: hour, month, hour_sin, hour_cos, month_sin, month_cos, weather, lags
                X_row = [
                    hour,
                    month,
                    hour_sin,
                    hour_cos,
                    month_sin,
                    month_cos,
                    Temperature,
                    HumiditySpecific,
                    HumidityRelative,
                    Pressure,
                    WindSpeed,
                    WindDirection,
                ] + lag_feats
                X_row = np.array(X_row).reshape(1, -1)

                if "XGBoost" in loaded_models:
                    xgb_pred = loaded_models["XGBoost"].predict(X_row)[0]
                    tree_predictions["XGBoost"].append(max(0, xgb_pred))
                    tree_series.append(xgb_pred)

                if "Random Forest" in loaded_models:
                    rf_pred = loaded_models["Random Forest"].predict(X_row)[0]
                    tree_predictions["Random Forest"].append(max(0, rf_pred))
                    if "XGBoost" not in loaded_models:
                        tree_series.append(rf_pred)

        # ---------- 5. Sequence-based model predictions ----------
        seq_predictions = {"LSTM": [], "CNN-LSTM": []}

        if (
            ("LSTM" in loaded_models or "CNN-LSTM" in loaded_models)
            and scaler_X
            and scaler_y
        ):
            # Features must match training order: hour, month, hour_sin, hour_cos, month_sin, month_cos, weather, target
            seq_features = [
                "hour",
                "month",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "Temperature",
                "HumiditySpecific",
                "HumidityRelative",
                "Pressure",
                "WindSpeed",
                "WindDirection",
                target_col,
            ]
            seq_data = df_day[seq_features].copy()

            # Use last known weather values
            Temperature = df_day["Temperature"].iloc[-1]
            HumiditySpecific = df_day["HumiditySpecific"].iloc[-1]
            HumidityRelative = df_day["HumidityRelative"].iloc[-1]
            Pressure = df_day["Pressure"].iloc[-1]
            WindSpeed = df_day["WindSpeed"].iloc[-1]
            WindDirection = df_day["WindDirection"].iloc[-1]

            # Scale features (excluding target which is last column)
            X_seq_scaled = scaler_X.transform(
                seq_data.drop(columns=[target_col])
            ).astype(np.float32)

            # Start with last SEQ_LEN rows
            seq_array = X_seq_scaled[-SEQ_LEN:].copy()

            for h in range(24):
                X_seq_input = seq_array[-SEQ_LEN:].reshape(
                    1, SEQ_LEN, seq_array.shape[1]
                )

                if "LSTM" in loaded_models:
                    lstm_pred_scaled = loaded_models["LSTM"].predict(
                        X_seq_input, verbose=0
                    )
                    lstm_pred = scaler_y.inverse_transform(
                        lstm_pred_scaled.reshape(-1, 1)
                    )[0, 0]
                    seq_predictions["LSTM"].append(max(0, lstm_pred))

                if "CNN-LSTM" in loaded_models:
                    cnn_pred_scaled = loaded_models["CNN-LSTM"].predict(
                        X_seq_input, verbose=0
                    )
                    cnn_pred = scaler_y.inverse_transform(
                        cnn_pred_scaled.reshape(-1, 1)
                    )[0, 0]
                    seq_predictions["CNN-LSTM"].append(max(0, cnn_pred))

                # Prepare next input row with UPDATED hour features for next hour
                next_hour = hours[(h + 1) % 24]  # Wrap around for next day
                next_hour_sin = np.sin(2 * np.pi * next_hour / 24)
                next_hour_cos = np.cos(2 * np.pi * next_hour / 24)
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)

                # Create next row with proper hour features (scaled)
                next_row_raw = np.array(
                    [
                        next_hour,
                        month,
                        next_hour_sin,
                        next_hour_cos,
                        month_sin,
                        month_cos,
                        Temperature,
                        HumiditySpecific,
                        HumidityRelative,
                        Pressure,
                        WindSpeed,
                        WindDirection,
                    ]
                ).reshape(1, -1)
                next_row = scaler_X.transform(next_row_raw).flatten()

                seq_array = np.vstack([seq_array, next_row])

        # ---------- 6. Combine results ----------
        results = {"datetime": date_index}
        for model_name in selected_models:
            if model_name in tree_predictions and tree_predictions[model_name]:
                results[model_name] = tree_predictions[model_name]
            elif model_name in seq_predictions and seq_predictions[model_name]:
                results[model_name] = seq_predictions[model_name]

        results_df = pd.DataFrame(results)

        # ---------- 7. Plot ----------
        if len(results_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            for model_name in selected_models:
                if model_name in results_df.columns:
                    ax.plot(
                        results_df["datetime"].dt.hour,
                        results_df[model_name],
                        label=model_name,
                        marker="o",
                    )

            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Predicted Solar Radiation (W/m²)")
            ax.set_title(f"Hourly Solar Forecast: {year}-{month:02d}-{day:02d}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

            # Display summary table
            st.subheader("Hourly Predictions")
            display_df = results_df.copy()
            display_df["Hour"] = display_df["datetime"].dt.strftime("%H:%00")
            display_df = display_df.drop(columns=["datetime"]).set_index("Hour")
            st.dataframe(display_df.style.format("{:.2f}"))
        else:
            st.error("No predictions generated.")

# Streamlit frontend for solar prediction (Enhanced UI)

import streamlit as st
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    tf_error = str(e)
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    xgb_error = str(e)


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
            df = pd.read_csv(
                "NASA meteriological and solar radiaton data/lahore_hourly_filled.csv",
                parse_dates=["datetime"],
            )
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

        # Always use the last 24 available daylight records for prediction
        last_24 = df_day.tail(24).copy()
        if len(last_24) < 24:
            st.warning(
                "Less than 24 daylight records found. Predictions will use all available records, but may be less accurate."
            )

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
        if len(seq_input) < 2:
            st.error("Not enough valid records for deep model input (need at least 2).")
            st.stop()
        elif len(seq_input) < 24:
            st.warning(
                f"Only {len(seq_input)} valid records for deep model input. Predictions may be less accurate."
            )

        # Load scalers
        try:
            scaler_X = joblib.load(os.path.join("save_model", "scaler_X.pkl"))
            scaler_y = joblib.load(os.path.join("save_model", "scaler_y.pkl"))
        except Exception as e:
            st.error(f"Error loading scalers: {e}")
            st.stop()

        # Prepare input for tree models (XGBoost, Random Forest)
        # Use as many lags as possible (minimum 1 row)
        lag_input = make_lag_features(last_24, min(24, len(last_24) - 1))
        if lag_input.empty:
            st.warning(
                "Not enough lag data for tree-based models. At least 2 daylight records are required for 1 prediction."
            )
        elif lag_input.shape[0] < 1 or lag_input.shape[1] < 10:
            st.warning("Very few lag records available. Prediction may be unreliable.")
        elif lag_input.shape[0] < 24:
            st.warning(
                f"Only {lag_input.shape[0]} lag records available for tree-based models. Predictions may be less accurate."
            )
        X_pred_tree = lag_input.drop(columns=["SolarRadiation"])

        # Prepare input for deep models (LSTM, CNN-LSTM)
        X_pred_seq = scaler_X.transform(seq_input.drop(columns=["SolarRadiation"]))
        X_pred_seq = X_pred_seq.reshape(1, 24, -1)

        # --- Predict next 24 hours ---
        predictions = {}
        start_time = last_24.index[-1] + pd.Timedelta(hours=1)
        future_times = pd.date_range(start=start_time, periods=24, freq="H")
        # XGBoost
        if "XGBoost" in selected_models:
            if not XGBOOST_AVAILABLE:
                st.warning("XGBoost is not available. Skipping XGBoost prediction.")
            else:
                try:
                    xgb_model = joblib.load(
                        os.path.join("save_model", "xgboost_model.pkl")
                    )
                    preds = []
                    lag_data = last_24.copy()
                    for i in range(24):
                        lag_feats = make_lag_features(lag_data, 24)
                        if lag_feats.empty:
                            preds.append(0)
                            continue
                        X_pred = lag_feats.drop(columns=["SolarRadiation"]).iloc[[-1]]
                        pred = xgb_model.predict(X_pred)[0]
                        pred = max(pred, 0)  # Clip negatives

                        # Daylight-aware prediction
                        next_time = future_times[i]
                        if next_time.hour < 6 or next_time.hour > 18:
                            pred = 0

                        preds.append(pred)

                        # Append prediction for next lag
                        new_row = lag_data.iloc[-1].copy()
                        new_row["SolarRadiation"] = pred
                        new_row["hour"] = next_time.hour
                        new_row["month"] = next_time.month
                        new_row.name = next_time
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
                        preds.append(0)
                        continue
                    X_pred = lag_feats.drop(columns=["SolarRadiation"]).iloc[[-1]]
                    pred = rf_model.predict(X_pred)[0]
                    pred = max(pred, 0)  # Clip negatives

                    # Daylight-aware prediction
                    next_time = future_times[i]
                    if next_time.hour < 6 or next_time.hour > 18:
                        pred = 0

                    preds.append(pred)

                    new_row = lag_data.iloc[-1].copy()
                    new_row["SolarRadiation"] = pred
                    new_row["hour"] = next_time.hour
                    new_row["month"] = next_time.month
                    new_row.name = next_time
                    lag_data = pd.concat([lag_data, pd.DataFrame([new_row])])
                predictions["Random Forest"] = preds
            except Exception as e:
                st.warning(f"Random Forest prediction error: {e}")

        # LSTM
        if "LSTM" in selected_models:
            if not TENSORFLOW_AVAILABLE:
                st.warning("TensorFlow is not available. Skipping LSTM prediction.")
            else:
                try:
                    lstm_model = load_model(os.path.join("save_model", "lstm_model.h5"))
                    preds = []
                    rolling_df = last_24.copy()

                    for i in range(24):
                        # Prepare sequence
                        seq_input_current = rolling_df[features].tail(24)
                        X_current = scaler_X.transform(
                            seq_input_current.drop(columns=["SolarRadiation"])
                        )
                        seq = X_current.reshape(1, 24, -1)

                        # Predict
                        pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
                        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                        pred = max(pred, 0)  # Clip negatives

                        # Daylight-aware prediction
                        next_time = future_times[i]
                        if next_time.hour < 6 or next_time.hour > 18:
                            pred = 0

                        preds.append(pred)

                        # Build next row
                        new_row = rolling_df.iloc[-1].copy()
                        new_row.name = next_time
                        new_row["SolarRadiation"] = pred
                        new_row["hour"] = next_time.hour
                        new_row["month"] = next_time.month
                        rolling_df = pd.concat([rolling_df, pd.DataFrame([new_row])])

                    predictions["LSTM"] = preds
                except Exception as e:
                    st.warning(f"LSTM prediction error: {e}")

        # CNN-LSTM
        if "CNN-LSTM" in selected_models:
            if not TENSORFLOW_AVAILABLE:
                st.warning("TensorFlow is not available. Skipping CNN-LSTM prediction.")
            else:
                try:
                    cnn_lstm_model = load_model(
                        os.path.join("save_model", "cnn_lstm_model.h5")
                    )
                    preds = []
                    rolling_df = last_24.copy()

                    for i in range(24):
                        # Prepare sequence
                        seq_input_current = rolling_df[features].tail(24)
                        X_current = scaler_X.transform(
                            seq_input_current.drop(columns=["SolarRadiation"])
                        )
                        seq = X_current.reshape(1, 24, -1)

                        # Predict
                        pred_scaled = cnn_lstm_model.predict(seq, verbose=0)[0][0]
                        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                        pred = max(pred, 0)  # Clip negatives

                        # Daylight-aware prediction
                        next_time = future_times[i]
                        if next_time.hour < 6 or next_time.hour > 18:
                            pred = 0

                        preds.append(pred)

                        # Build next row
                        new_row = rolling_df.iloc[-1].copy()
                        new_row.name = next_time
                        new_row["SolarRadiation"] = pred
                        new_row["hour"] = next_time.hour
                        new_row["month"] = next_time.month
                        rolling_df = pd.concat([rolling_df, pd.DataFrame([new_row])])

                    predictions["CNN-LSTM"] = preds
                except Exception as e:
                    st.warning(f"CNN-LSTM prediction error: {e}")

        # Plot predictions
        fig, ax = plt.subplots(figsize=(11, 5))
        for model, preds in predictions.items():
            ax.plot(future_times, preds, label=model, marker="o")
        ax.set_xlabel("Datetime", fontsize=12)
        ax.set_ylabel("Predicted Solar Radiation", fontsize=12)
        ax.set_title("Next 24 Hours Solar Power Forecast", fontsize=14, color="#2c3e50")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Select input and click 'Predict for Next Day' to see results.")

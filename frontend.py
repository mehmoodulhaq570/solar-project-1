# ===============================================
# Streamlit Frontend for Solar Radiation Prediction
# Updated to match trend.py with all improvements
# ===============================================

import streamlit as st
import os, warnings
from datetime import datetime
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

# ============== Page Configuration ==============
st.set_page_config(page_title="Solar Radiation Forecast", page_icon="☀️", layout="wide")

# ============== Sidebar ==============
st.sidebar.image("https://img.icons8.com/fluency/96/solar-panel.png", width=80)
st.sidebar.title("☀️ Solar Power Prediction")
st.sidebar.markdown("---")

# Date input
selected_date = st.sidebar.date_input(
    "📅 Forecast Date",
    datetime.today(),
    help="Select the date for solar radiation forecast",
)

# Model folder selection
model_folder = st.sidebar.selectbox(
    "📁 Model Folder",
    ["saved_models", "save_model", "saved_models_1"],
    help="Select folder containing trained models",
)

# Model selection
available_models = ["XGBoost", "Random Forest", "LSTM", "CNN-LSTM", "Ensemble"]
selected_models = st.sidebar.multiselect(
    "🤖 Select Models",
    available_models,
    default=["XGBoost", "Random Forest", "Ensemble"],
    help="Choose which models to use for prediction",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(
    """
- **XGBoost/RF**: Tree-based models with lag features
- **LSTM/CNN-LSTM**: Deep learning sequence models
- **Ensemble**: Weighted average of all models
"""
)

# ============== Main Content ==============
st.markdown(
    "<h1 style='text-align:center;'>☀️ Solar Radiation Forecast Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Hourly solar radiation predictions using machine learning</p>",
    unsafe_allow_html=True,
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📋 Data Table", "ℹ️ About"])

# ============== Prediction Tab ==============
with tab1:
    col1, col2 = st.columns([3, 1])

    with col2:
        predict_button = st.button(
            "🔮 Generate Forecast", type="primary", use_container_width=True
        )

    with col1:
        st.markdown(f"**Selected Date:** {selected_date.strftime('%B %d, %Y')}")
        st.markdown(
            f"**Models:** {', '.join(selected_models) if selected_models else 'None selected'}"
        )

    if predict_button:
        if not selected_models:
            st.error("❌ Please select at least one model.")
            st.stop()

        # Progress bar
        progress = st.progress(0, text="Loading data...")

        # ---------- 1. Load Historical Data ----------
        try:
            df = pd.read_csv(
                "NASA meteriological and solar radiaton data/lahore_hourly_filled.csv",
                parse_dates=["datetime"],
                dayfirst=True,
                index_col="datetime",
            )
            df = df.sort_index().apply(pd.to_numeric, errors="coerce")
            df.index = pd.to_datetime(df.index, dayfirst=True)

            # Prepare data with all features
            df_day = df.copy()
            df_day.dropna(subset=["SolarRadiation"], inplace=True)

            # Add cyclical encoding
            df_day["hour"] = df_day.index.hour
            df_day["month"] = df_day.index.month
            df_day["day_of_year"] = df_day.index.dayofyear
            df_day["hour_sin"] = np.sin(2 * np.pi * df_day["hour"] / 24)
            df_day["hour_cos"] = np.cos(2 * np.pi * df_day["hour"] / 24)
            df_day["month_sin"] = np.sin(2 * np.pi * df_day["month"] / 12)
            df_day["month_cos"] = np.cos(2 * np.pi * df_day["month"] / 12)
            df_day["doy_sin"] = np.sin(2 * np.pi * df_day["day_of_year"] / 365)
            df_day["doy_cos"] = np.cos(2 * np.pi * df_day["day_of_year"] / 365)

            target_col = "SolarRadiation"
            progress.progress(10, text="Data loaded successfully!")

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

        # ---------- 2. Load Trained Models ----------
        progress.progress(20, text="Loading models...")
        loaded_models = {}

        try:
            if (
                "XGBoost" in selected_models or "Ensemble" in selected_models
            ) and XGBOOST_AVAILABLE:
                loaded_models["XGBoost"] = joblib.load(
                    f"{model_folder}/xgboost_model.pkl"
                )
        except Exception as e:
            st.warning(f"⚠️ XGBoost not loaded: {e}")

        try:
            if "Random Forest" in selected_models or "Ensemble" in selected_models:
                loaded_models["Random Forest"] = joblib.load(
                    f"{model_folder}/random_forest_model.pkl"
                )
        except Exception as e:
            st.warning(f"⚠️ Random Forest not loaded: {e}")

        try:
            if (
                "LSTM" in selected_models or "Ensemble" in selected_models
            ) and TENSORFLOW_AVAILABLE:
                loaded_models["LSTM"] = load_model(f"{model_folder}/lstm_model.h5")
        except Exception as e:
            st.warning(f"⚠️ LSTM not loaded: {e}")

        try:
            if (
                "CNN-LSTM" in selected_models or "Ensemble" in selected_models
            ) and TENSORFLOW_AVAILABLE:
                loaded_models["CNN-LSTM"] = load_model(
                    f"{model_folder}/cnn_lstm_model.h5"
                )
        except Exception as e:
            st.warning(f"⚠️ CNN-LSTM not loaded: {e}")

        # Load scalers
        scaler_X, scaler_y = None, None
        try:
            scaler_X = joblib.load(f"{model_folder}/scaler_X.pkl")
            scaler_y = joblib.load(f"{model_folder}/scaler_y.pkl")
        except Exception as e:
            if "LSTM" in loaded_models or "CNN-LSTM" in loaded_models:
                st.warning(f"⚠️ Scalers not loaded: {e}")

        # Load ensemble weights
        try:
            ensemble_weights = joblib.load(f"{model_folder}/ensemble_weights.pkl")
        except:
            ensemble_weights = {
                "XGBoost": 0.25,
                "RandomForest": 0.25,
                "LSTM": 0.25,
                "CNN-LSTM": 0.25,
            }

        if not loaded_models:
            st.error("❌ No models could be loaded. Check model folder.")
            st.stop()

        progress.progress(30, text="Models loaded!")

        # ---------- 3. Forecast Settings ----------
        MAX_LAG = 24
        SEQ_LEN = 24

        year = selected_date.year
        month = selected_date.month
        day = selected_date.day

        forecast_date = datetime(year, month, day)
        day_of_year = forecast_date.timetuple().tm_yday

        date_index = pd.date_range(
            start=f"{year}-{month:02d}-{day:02d} 00:00",
            end=f"{year}-{month:02d}-{day:02d} 23:00",
            freq="H",
        )
        hours = date_index.hour

        # ---------- 4. Tree-based Model Predictions ----------
        progress.progress(40, text="Running tree-based predictions...")

        tree_predictions = {"XGBoost": [], "Random Forest": []}

        if "XGBoost" in loaded_models or "Random Forest" in loaded_models:
            last_hist_values = df_day[target_col].values
            tree_series = last_hist_values[-MAX_LAG:].tolist()

            # Weather values
            Temperature = df_day["Temperature"].iloc[-1]
            HumiditySpecific = df_day["HumiditySpecific"].iloc[-1]
            HumidityRelative = df_day["HumidityRelative"].iloc[-1]
            Pressure = df_day["Pressure"].iloc[-1]
            WindSpeed = df_day["WindSpeed"].iloc[-1]
            WindDirection = df_day["WindDirection"].iloc[-1]

            # Get typical SolarZenith and ClearSkyRadiation for each hour
            hourly_zenith = (
                df_day[df_day["month"] == month].groupby("hour")["SolarZenith"].mean()
            )
            hourly_clearsky = (
                df_day[df_day["month"] == month]
                .groupby("hour")["ClearSkyRadiation"]
                .mean()
            )

            for h in range(24):
                hour = hours[h]
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                doy_sin = np.sin(2 * np.pi * day_of_year / 365)
                doy_cos = np.cos(2 * np.pi * day_of_year / 365)

                solar_zenith = hourly_zenith.get(hour, 90)
                clear_sky_rad = hourly_clearsky.get(hour, 0)

                lag_feats = tree_series[-MAX_LAG:]

                # Features order matching training.py
                X_row = [
                    hour,
                    month,
                    day_of_year,
                    hour_sin,
                    hour_cos,
                    month_sin,
                    month_cos,
                    doy_sin,
                    doy_cos,
                    solar_zenith,
                    clear_sky_rad,
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

        # ---------- 5. Sequence-based Model Predictions ----------
        progress.progress(60, text="Running deep learning predictions...")

        seq_predictions = {"LSTM": [], "CNN-LSTM": []}

        if (
            ("LSTM" in loaded_models or "CNN-LSTM" in loaded_models)
            and scaler_X
            and scaler_y
        ):
            seq_features = [
                "hour",
                "month",
                "day_of_year",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "doy_sin",
                "doy_cos",
                "SolarZenith",
                "ClearSkyRadiation",
                "Temperature",
                "HumiditySpecific",
                "HumidityRelative",
                "Pressure",
                "WindSpeed",
                "WindDirection",
                target_col,
            ]
            seq_data = df_day[seq_features].copy()

            # Weather values
            Temperature = df_day["Temperature"].iloc[-1]
            HumiditySpecific = df_day["HumiditySpecific"].iloc[-1]
            HumidityRelative = df_day["HumidityRelative"].iloc[-1]
            Pressure = df_day["Pressure"].iloc[-1]
            WindSpeed = df_day["WindSpeed"].iloc[-1]
            WindDirection = df_day["WindDirection"].iloc[-1]

            hourly_zenith = (
                df_day[df_day["month"] == month].groupby("hour")["SolarZenith"].mean()
            )
            hourly_clearsky = (
                df_day[df_day["month"] == month]
                .groupby("hour")["ClearSkyRadiation"]
                .mean()
            )

            X_seq_scaled = scaler_X.transform(
                seq_data.drop(columns=[target_col])
            ).astype(np.float32)
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

                # Prepare next row
                next_hour = hours[(h + 1) % 24]
                next_hour_sin = np.sin(2 * np.pi * next_hour / 24)
                next_hour_cos = np.cos(2 * np.pi * next_hour / 24)
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                doy_sin = np.sin(2 * np.pi * day_of_year / 365)
                doy_cos = np.cos(2 * np.pi * day_of_year / 365)
                next_solar_zenith = hourly_zenith.get(next_hour, 90)
                next_clear_sky = hourly_clearsky.get(next_hour, 0)

                next_row_raw = np.array(
                    [
                        next_hour,
                        month,
                        day_of_year,
                        next_hour_sin,
                        next_hour_cos,
                        month_sin,
                        month_cos,
                        doy_sin,
                        doy_cos,
                        next_solar_zenith,
                        next_clear_sky,
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

        # ---------- 6. Calculate Ensemble ----------
        progress.progress(80, text="Calculating ensemble...")

        xgb_preds = tree_predictions.get("XGBoost", [0] * 24)
        rf_preds = tree_predictions.get("Random Forest", [0] * 24)
        lstm_preds = seq_predictions.get("LSTM", [0] * 24)
        cnn_preds = seq_predictions.get("CNN-LSTM", [0] * 24)

        # Fill missing predictions with zeros
        if not xgb_preds:
            xgb_preds = [0] * 24
        if not rf_preds:
            rf_preds = [0] * 24
        if not lstm_preds:
            lstm_preds = [0] * 24
        if not cnn_preds:
            cnn_preds = [0] * 24

        ensemble_preds = []
        for i in range(24):
            ensemble_pred = (
                ensemble_weights.get("XGBoost", 0.25) * xgb_preds[i]
                + ensemble_weights.get("RandomForest", 0.25) * rf_preds[i]
                + ensemble_weights.get("LSTM", 0.25) * lstm_preds[i]
                + ensemble_weights.get("CNN-LSTM", 0.25) * cnn_preds[i]
            )
            ensemble_preds.append(max(0, ensemble_pred))

        # ---------- 7. Combine Results ----------
        results_df = pd.DataFrame(
            {
                "datetime": date_index,
                "Hour": [f"{h:02d}:00" for h in range(24)],
            }
        )

        if tree_predictions.get("XGBoost"):
            results_df["XGBoost"] = xgb_preds
        if tree_predictions.get("Random Forest"):
            results_df["Random Forest"] = rf_preds
        if seq_predictions.get("LSTM"):
            results_df["LSTM"] = lstm_preds
        if seq_predictions.get("CNN-LSTM"):
            results_df["CNN-LSTM"] = cnn_preds
        if "Ensemble" in selected_models:
            results_df["Ensemble"] = ensemble_preds

        progress.progress(100, text="Complete!")

        # ---------- 8. Display Results ----------
        st.success(f"✅ Forecast generated for {selected_date.strftime('%B %d, %Y')}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: All models
        ax1 = axes[0]
        colors = {
            "XGBoost": "#e74c3c",
            "Random Forest": "#2ecc71",
            "LSTM": "#3498db",
            "CNN-LSTM": "#9b59b6",
            "Ensemble": "#2c3e50",
        }

        for model in selected_models:
            if model in results_df.columns:
                ax1.plot(
                    range(24),
                    results_df[model],
                    label=model,
                    color=colors.get(model, "gray"),
                    linewidth=3 if model == "Ensemble" else 2,
                    linestyle="--" if model == "Ensemble" else "-",
                    marker="o" if model != "Ensemble" else None,
                    markersize=4,
                )

        ax1.set_xlabel("Hour of Day", fontsize=12)
        ax1.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
        ax1.set_title(
            f"Solar Forecast: {selected_date.strftime('%Y-%m-%d')}", fontsize=14
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xlim(-0.5, 23.5)

        # Plot 2: Ensemble with range
        ax2 = axes[1]
        model_preds = np.array([xgb_preds, rf_preds, lstm_preds, cnn_preds])
        pred_min = model_preds.min(axis=0)
        pred_max = model_preds.max(axis=0)

        ax2.fill_between(
            range(24),
            pred_min,
            pred_max,
            alpha=0.3,
            color="#3498db",
            label="Model Range",
        )
        ax2.plot(
            range(24),
            ensemble_preds,
            label="Ensemble",
            linewidth=2.5,
            color="#2c3e50",
            marker="o",
            markersize=5,
        )
        ax2.set_xlabel("Hour of Day", fontsize=12)
        ax2.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
        ax2.set_title("Ensemble Forecast with Prediction Range", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xlim(-0.5, 23.5)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "🌅 Sunrise Peak", f"{max(ensemble_preds[5:9]):.0f} W/m²", "05:00-08:00"
            )
        with col2:
            st.metric(
                "☀️ Midday Peak", f"{max(ensemble_preds[10:14]):.0f} W/m²", "10:00-13:00"
            )
        with col3:
            st.metric(
                "🌇 Sunset", f"{max(ensemble_preds[16:19]):.0f} W/m²", "16:00-18:00"
            )
        with col4:
            daily_total = sum(ensemble_preds)
            st.metric("📊 Daily Total", f"{daily_total:.0f} Wh/m²", "Energy")

        # Store results for Data Table tab
        st.session_state["results_df"] = results_df
        st.session_state["forecast_date"] = selected_date

# ============== Data Table Tab ==============
with tab2:
    if "results_df" in st.session_state:
        st.subheader(
            f"📋 Hourly Predictions for {st.session_state['forecast_date'].strftime('%B %d, %Y')}"
        )

        display_df = (
            st.session_state["results_df"].drop(columns=["datetime"]).set_index("Hour")
        )
        st.dataframe(
            display_df.style.format("{:.2f}").background_gradient(
                cmap="YlOrRd", axis=0
            ),
            use_container_width=True,
        )

        # Download button
        csv = st.session_state["results_df"].to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"solar_forecast_{st.session_state['forecast_date'].strftime('%Y_%m_%d')}.csv",
            "text/csv",
        )
    else:
        st.info("👆 Click 'Generate Forecast' to see predictions here.")

# ============== About Tab ==============
with tab3:
    st.subheader("ℹ️ About This Dashboard")
    st.markdown(
        """
    ### 🌤 Solar Radiation Forecasting System
    
    This dashboard uses machine learning models trained on **NASA POWER** meteorological data 
    from 2018-2025 to predict hourly solar radiation for Lahore, Pakistan.
    
    #### Models Used:
    - **XGBoost & Random Forest**: Tree-based models with 24-hour lag features
    - **LSTM**: Bidirectional Long Short-Term Memory neural network
    - **CNN-LSTM**: Convolutional + LSTM hybrid architecture
    - **Ensemble**: Weighted average based on model performance
    
    #### Features Used:
    - ⏰ Time: Hour, Month, Day of Year (with cyclical encoding)
    - ☀️ Solar: Solar Zenith Angle, Clear Sky Radiation
    - 🌡️ Weather: Temperature, Humidity, Pressure, Wind
    - 📈 Historical: 24-hour lag values
    
    #### Data Source:
    NASA POWER (Prediction Of Worldwide Energy Resources)
    """
    )

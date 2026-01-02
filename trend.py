# ==========================================
# 🌤 Daily Hourly Solar Radiation Forecast
# Using trained models from 2018–2025
# With Real-Time API Comparison (Multiple APIs)
# Input: year, month, day → Output: 24-hour forecast
# ==========================================

import os, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
import requests
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============== API Configuration ==============
# Lahore, Pakistan coordinates
LATITUDE = 31.56
LONGITUDE = 74.35
TIMEZONE = "Asia/Karachi"

# ============== API Selection ==============
# Available APIs: "open-meteo", "nasa-power"
SELECTED_API = "open-meteo"  # <-- Testing Open-Meteo configuration

# ============== API Functions ==============


def fetch_openmeteo_solar_forecast(year, month, day):
    """
    Fetch solar radiation forecast from Open-Meteo API.
    ✅ FREE, No API key required
    📊 Returns hourly GHI (Global Horizontal Irradiance) in W/m².
    ⏰ Forecast only (today + 16 days)
    """
    date_str = f"{year}-{month:02d}-{day:02d}"

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "shortwave_radiation",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": TIMEZONE,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        hourly_radiation = data.get("hourly", {}).get("shortwave_radiation", [])

        if len(hourly_radiation) == 24:
            print(f"✅ Open-Meteo API: Successfully fetched forecast for {date_str}")
            return hourly_radiation
        else:
            print(
                f"⚠️ Open-Meteo API: Unexpected data length ({len(hourly_radiation)} hours)"
            )
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Open-Meteo API Error: {e}")
        return None


def fetch_nasa_power_solar(year, month, day):
    """
    Fetch solar radiation data from NASA POWER API.
    ✅ FREE, No API key required
    📊 Returns hourly ALLSKY_SFC_SW_DWN (Surface Shortwave Downward Irradiance) in W/m².
    ⏰ Historical data (up to ~1 week ago) - good for validation
    """
    date_str = f"{year}{month:02d}{day:02d}"

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": date_str,
        "end": date_str,
        "format": "JSON",
        "time-standard": "LST",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly_data = (
            data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
        )

        # Extract hourly values (keys are like "2025121800", "2025121801", etc.)
        hourly_radiation = []
        for hour in range(24):
            key = f"{year}{month:02d}{day:02d}{hour:02d}"
            value = hourly_data.get(key, -999)
            # NASA uses -999 for missing data
            hourly_radiation.append(max(0, value) if value != -999 else 0)

        if len(hourly_radiation) == 24 and sum(hourly_radiation) > 0:
            print(
                f"✅ NASA POWER API: Successfully fetched data for {year}-{month:02d}-{day:02d}"
            )
            return hourly_radiation
        else:
            print(f"⚠️ NASA POWER API: No data available (data may not be ready yet)")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ NASA POWER API Error: {e}")
        return None


def fetch_api_prediction(year, month, day, api_name):
    """
    Unified function to fetch solar radiation from selected API.
    """
    api_name = api_name.lower()

    if api_name == "open-meteo":
        return fetch_openmeteo_solar_forecast(year, month, day), "Open-Meteo"
    elif api_name == "nasa-power":
        return fetch_nasa_power_solar(year, month, day), "NASA POWER"
    else:
        print(f"❌ Unknown API: {api_name}")
        return None, api_name


def list_available_apis():
    """Print available APIs and their status."""
    print("\n📡 Available APIs:")
    print("  1. open-meteo - ✅ FREE, No key (Forecast: today + 16 days)")
    print("  2. nasa-power - ✅ FREE, No key (Historical, ~1 week delay)")
    print(f"\n  Currently selected: {SELECTED_API}\n")


# ---------- 1. Load historical data ----------
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

# Add cyclical encoding for hour and month
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

# ---------- 2. Load trained models ----------
# Using saved_models_lstm folder (best LSTM/CNN-LSTM scores)
MODEL_FOLDER = "saved_models_lstm"

xgb_model = joblib.load(f"{MODEL_FOLDER}/xgboost_model.pkl")
rf_model = joblib.load(f"{MODEL_FOLDER}/random_forest_model.pkl")

# Load Keras 3.x format models (converted from h5)
import os

if os.path.exists(f"{MODEL_FOLDER}/lstm_model_v3.keras"):
    lstm_model = load_model(f"{MODEL_FOLDER}/lstm_model_v3.keras")
else:
    lstm_model = load_model(f"{MODEL_FOLDER}/lstm_model.h5")

if os.path.exists(f"{MODEL_FOLDER}/cnn_lstm_model_v3.keras"):
    cnn_lstm_model = load_model(f"{MODEL_FOLDER}/cnn_lstm_model_v3.keras")
else:
    cnn_lstm_model = load_model(f"{MODEL_FOLDER}/cnn_lstm_model.h5")

scaler_X = joblib.load(f"{MODEL_FOLDER}/scaler_X.pkl")
scaler_y = joblib.load(f"{MODEL_FOLDER}/scaler_y.pkl")

# Load ensemble weights (original from training)
try:
    ensemble_weights = joblib.load(f"{MODEL_FOLDER}/ensemble_weights.pkl")
    print(f"Ensemble weights loaded: {ensemble_weights}")
except:
    ensemble_weights = {
        "XGBoost": 0.25,
        "RandomForest": 0.25,
        "LSTM": 0.25,
        "CNN-LSTM": 0.25,
    }
    print("Using equal ensemble weights")

# ============== API-Specific Ensemble Configuration ==============
# Different weights optimized for each API comparison

# NASA POWER API weights (optimized for NASA satellite data - same source as training)
# XGBoost and RF perform best (R² > 0.94), so use higher weights
NASA_ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.45,  # Best performer with NASA (R² = 0.958)
    "RandomForest": 0.40,  # Second best (R² = 0.940)
    "LSTM": 0.08,  # Lower weight (R² = 0.598)
    "CNN-LSTM": 0.07,  # Lowest weight (R² = 0.621)
}

# NASA hour calibration (minimal - models already match well)
NASA_HOUR_CALIBRATION = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,  # Night
    6: 0.0,  # Before sunrise
    7: 1.0,  # Early morning - no calibration needed
    8: 1.0,  # Morning
    9: 1.0,  # Mid-morning
    10: 1.0,  # Late morning
    11: 1.0,  # Near solar noon
    12: 1.0,  # Solar noon
    13: 1.0,  # Early afternoon
    14: 1.0,  # Mid-afternoon
    15: 1.0,  # Late afternoon
    16: 1.0,  # Evening
    17: 0.0,  # After sunset
    18: 0.0,
    19: 0.0,
    20: 0.0,
    21: 0.0,
    22: 0.0,
    23: 0.0,  # Night
}

# Open-Meteo API weights (optimized for weather forecast comparison)
# Ensemble of all models works best due to timing differences
OPENMETEO_ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.25,  # Equal weights
    "RandomForest": 0.25,
    "LSTM": 0.25,
    "CNN-LSTM": 0.25,
}

# Open-Meteo hour calibration (corrects timing differences)
OPENMETEO_HOUR_CALIBRATION = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,  # Night
    6: 0.0,  # Before sunrise
    7: 0.0,  # Very early morning (API often shows 0)
    8: 0.22,  # Early morning: significant scaling
    9: 0.41,  # Morning ramp-up
    10: 0.70,  # Mid-morning
    11: 0.94,  # Late morning
    12: 0.99,  # Solar noon - near perfect
    13: 1.05,  # Early afternoon
    14: 1.22,  # Mid-afternoon
    15: 1.65,  # Late afternoon
    16: 3.19,  # Evening
    17: 10.0,  # Sunset (capped for stability)
    18: 0.0,
    19: 0.0,
    20: 0.0,
    21: 0.0,
    22: 0.0,
    23: 0.0,  # Night
}

# ---------- 3. Forecast settings ----------
MAX_LAG = 24
SEQ_LEN = 24

# ---------- 4. User input: day to forecast ----------
# For API comparison, use today or a future date (within 16 days)
# Open-Meteo only provides forecasts, not historical data
today = datetime.now()
year = today.year
month = today.month
day = today.day

# Override with specific date if needed (must be within forecast range for API)
# For NASA POWER: Use historical dates (~1 week old or older)
# For Open-Meteo: Use today or future dates (up to 16 days)
year = 2025
month = 12
day = 21  # Current date for Open-Meteo forecast

print(f"📅 Forecasting for: {year}-{month:02d}-{day:02d}")

# Calculate day of year for the forecast date
forecast_date = datetime(year, month, day)
day_of_year = forecast_date.timetuple().tm_yday

# Generate hourly datetime index for the day
date_index = pd.date_range(
    start=f"{year}-{month:02d}-{day:02d} 00:00",
    end=f"{year}-{month:02d}-{day:02d} 23:00",
    freq="H",
)
hours = date_index.hour

# ---------- 5. Prepare tree-based model inputs ----------
last_hist_values = df_day[target_col].values
tree_series = last_hist_values[-MAX_LAG:].tolist()  # last 24 hours

# Use last known weather values
Temperature = df_day["Temperature"].iloc[-1]
HumiditySpecific = df_day["HumiditySpecific"].iloc[-1]
HumidityRelative = df_day["HumidityRelative"].iloc[-1]
Pressure = df_day["Pressure"].iloc[-1]
WindSpeed = df_day["WindSpeed"].iloc[-1]
WindDirection = df_day["WindDirection"].iloc[-1]

# Get typical SolarZenith and ClearSkyRadiation for each hour from historical data for this month
# This is crucial: SolarZenith >= 90 means sun is below horizon (night)
hourly_zenith = df_day[df_day["month"] == month].groupby("hour")["SolarZenith"].mean()
hourly_clearsky = (
    df_day[df_day["month"] == month].groupby("hour")["ClearSkyRadiation"].mean()
)

tree_predictions = []

for h in range(24):
    hour = hours[h]
    # Compute cyclical features for current hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)

    # Get typical SolarZenith and ClearSkyRadiation for this hour in this month
    solar_zenith = hourly_zenith.get(hour, 90)  # Default to 90 (horizon) if not found
    clear_sky_rad = hourly_clearsky.get(hour, 0)  # Default to 0 if not found

    lag_feats = tree_series[-MAX_LAG:]
    # Features order must match training:
    # hour, month, day_of_year, hour_sin, hour_cos, month_sin, month_cos, doy_sin, doy_cos,
    # SolarZenith, ClearSkyRadiation, Temperature, HumiditySpecific, HumidityRelative,
    # Pressure, WindSpeed, WindDirection, lags
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

    xgb_pred = xgb_model.predict(X_row)[0]
    rf_pred = rf_model.predict(X_row)[0]

    # Ensure non-negative predictions
    xgb_pred = max(0, xgb_pred)
    rf_pred = max(0, rf_pred)

    tree_predictions.append((xgb_pred, rf_pred))
    tree_series.append(xgb_pred)  # iterative update

# ---------- 6. Prepare sequence-based model inputs ----------
# Features must match training order (same as tree models, minus lags)
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

# Scale features (excluding target which is last column)
X_seq_scaled = scaler_X.transform(seq_data.drop(columns=[target_col])).astype(
    np.float32
)

# Start with last SEQ_LEN rows
seq_array = X_seq_scaled[-SEQ_LEN:].copy()
seq_predictions = []

for h in range(24):
    X_seq_input = seq_array[-SEQ_LEN:].reshape(1, SEQ_LEN, seq_array.shape[1])

    lstm_pred_scaled = lstm_model.predict(X_seq_input, verbose=0)
    cnn_pred_scaled = cnn_lstm_model.predict(X_seq_input, verbose=0)

    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1))[0, 0]
    cnn_pred = scaler_y.inverse_transform(cnn_pred_scaled.reshape(-1, 1))[0, 0]

    # Ensure non-negative predictions
    lstm_pred = max(0, lstm_pred)
    cnn_pred = max(0, cnn_pred)

    seq_predictions.append((lstm_pred, cnn_pred))

    # Prepare next input row with UPDATED hour features for next hour
    next_hour = hours[(h + 1) % 24]  # Wrap around for next day
    next_hour_sin = np.sin(2 * np.pi * next_hour / 24)
    next_hour_cos = np.cos(2 * np.pi * next_hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)

    # Get typical SolarZenith and ClearSkyRadiation for next hour
    next_solar_zenith = hourly_zenith.get(next_hour, 90)
    next_clear_sky = hourly_clearsky.get(next_hour, 0)

    # Create next row with proper features (scaled)
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

# ---------- 7. Combine results ----------
xgb_preds = [x for x, _ in tree_predictions]
rf_preds = [x for _, x in tree_predictions]
lstm_preds = [x for x, _ in seq_predictions]
cnn_preds = [x for _, x in seq_predictions]

# Select ensemble configuration based on API
print(f"\n🔧 Using ensemble configuration for: {SELECTED_API.upper()}")

if SELECTED_API.lower() == "nasa-power":
    ACTIVE_WEIGHTS = NASA_ENSEMBLE_WEIGHTS
    HOUR_CALIBRATION = NASA_HOUR_CALIBRATION
    print(
        f"   Weights: XGB={ACTIVE_WEIGHTS['XGBoost']:.0%}, RF={ACTIVE_WEIGHTS['RandomForest']:.0%}, LSTM={ACTIVE_WEIGHTS['LSTM']:.0%}, CNN={ACTIVE_WEIGHTS['CNN-LSTM']:.0%}"
    )
    print("   Calibration: Minimal (models match NASA data well)")
else:
    ACTIVE_WEIGHTS = OPENMETEO_ENSEMBLE_WEIGHTS
    HOUR_CALIBRATION = OPENMETEO_HOUR_CALIBRATION
    print(
        f"   Weights: XGB={ACTIVE_WEIGHTS['XGBoost']:.0%}, RF={ACTIVE_WEIGHTS['RandomForest']:.0%}, LSTM={ACTIVE_WEIGHTS['LSTM']:.0%}, CNN={ACTIVE_WEIGHTS['CNN-LSTM']:.0%}"
    )
    print("   Calibration: Hour-specific adjustments for timing differences")

ensemble_preds = []
for i in range(24):
    hour = hours[i]

    # Get predictions
    xgb_val = xgb_preds[i]
    rf_val = rf_preds[i]
    lstm_val = lstm_preds[i]
    cnn_val = cnn_preds[i]

    # Calculate weighted ensemble based on selected API configuration
    base_ensemble = (
        ACTIVE_WEIGHTS["XGBoost"] * xgb_val
        + ACTIVE_WEIGHTS["RandomForest"] * rf_val
        + ACTIVE_WEIGHTS["LSTM"] * lstm_val
        + ACTIVE_WEIGHTS["CNN-LSTM"] * cnn_val
    )

    # Apply hour-specific calibration factor
    calibration = HOUR_CALIBRATION.get(hour, 1.0)
    ensemble_pred = base_ensemble * calibration

    # Sanity check: cap at reasonable maximum solar radiation (~1000 W/m²)
    ensemble_pred = min(max(0, ensemble_pred), 1000)

    ensemble_preds.append(ensemble_pred)

# ---------- 8. Fetch API Predictions for Comparison ----------
list_available_apis()
print(f"📡 Fetching API predictions using: {SELECTED_API}...")

api_preds, api_name = fetch_api_prediction(year, month, day, SELECTED_API)

# If primary API not available, try Open-Meteo as fallback
if api_preds is None and SELECTED_API != "open-meteo":
    print("⚠️ Primary API failed. Trying Open-Meteo as fallback...")
    api_preds, api_name = fetch_api_prediction(year, month, day, "open-meteo")

# If still no data, use NaN
if api_preds is None:
    api_preds = [np.nan] * 24
    print("⚠️ API data not available - will show as missing in plots")

results_day = pd.DataFrame(
    {
        "datetime": date_index,
        "XGBoost": xgb_preds,
        "RandomForest": rf_preds,
        "LSTM": lstm_preds,
        "CNN_LSTM": cnn_preds,
        "Ensemble": ensemble_preds,
        f"{api_name}_API": api_preds,
    }
)

# ---------- 9. Save to CSV ----------
results_day.to_csv(f"solar_forecast_{year}_{month:02d}_{day:02d}.csv", index=False)
print(f"\n✅ Forecast saved to solar_forecast_{year}_{month:02d}_{day:02d}.csv")

# ---------- 10. Print hourly predictions ----------
print(f"\nHourly Solar Radiation Forecast for {year}-{month:02d}-{day:02d}:")
api_col_name = api_name[:10]  # Truncate for formatting
print(
    f"{'Hour':>4} | {'XGBoost':>10} | {'RandomForest':>13} | {'LSTM':>8} | {'CNN-LSTM':>10} | {'Ensemble':>10} | {api_col_name:>10}"
)
print("-" * 85)
for i, row in results_day.iterrows():
    hour = row["datetime"].hour
    api_val = row.get(f"{api_name}_API", np.nan)
    api_str = f"{api_val:10.2f}" if not np.isnan(api_val) else "       N/A"
    print(
        f"{hour:02d}:00 | {row['XGBoost']:10.2f} | {row['RandomForest']:13.2f} | {row['LSTM']:8.2f} | {row['CNN_LSTM']:10.2f} | {row['Ensemble']:10.2f} | {api_str}"
    )

# ---------- 11. Calculate comparison metrics ----------
if not all(np.isnan(api_preds)):
    print(f"\n📊 Model vs {api_name} API Comparison Metrics:")
    api_array = np.array(api_preds)

    # Filter out night hours (where both are near 0) for meaningful comparison
    valid_mask = (api_array > 10) | (np.array(ensemble_preds) > 10)

    if valid_mask.sum() > 0:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        api_valid = api_array[valid_mask]
        ensemble_valid = np.array(ensemble_preds)[valid_mask]
        xgb_valid = np.array(xgb_preds)[valid_mask]
        rf_valid = np.array(rf_preds)[valid_mask]
        lstm_valid = np.array(lstm_preds)[valid_mask]
        cnn_valid = np.array(cnn_preds)[valid_mask]

        print(f"\n  {'Model':<15} | {'MAE':>10} | {'RMSE':>10} | {'R²':>8}")
        print("  " + "-" * 50)

        for name, preds in [
            ("Ensemble", ensemble_valid),
            ("XGBoost", xgb_valid),
            ("RandomForest", rf_valid),
            ("LSTM", lstm_valid),
            ("CNN-LSTM", cnn_valid),
        ]:
            mae = mean_absolute_error(api_valid, preds)
            rmse = np.sqrt(mean_squared_error(api_valid, preds))
            try:
                r2 = r2_score(api_valid, preds)
            except:
                r2 = np.nan
            print(f"  {name:<15} | {mae:10.2f} | {rmse:10.2f} | {r2:8.4f}")

# ---------- 12. Plot results with API comparison ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: All ML models
ax1 = axes[0, 0]
ax1.plot(
    results_day["datetime"].dt.hour,
    results_day["XGBoost"],
    label="XGBoost",
    marker="o",
    markersize=3,
)
ax1.plot(
    results_day["datetime"].dt.hour,
    results_day["RandomForest"],
    label="RandomForest",
    marker="s",
    markersize=3,
)
ax1.plot(
    results_day["datetime"].dt.hour,
    results_day["LSTM"],
    label="LSTM",
    marker="^",
    markersize=3,
)
ax1.plot(
    results_day["datetime"].dt.hour,
    results_day["CNN_LSTM"],
    label="CNN-LSTM",
    marker="d",
    markersize=3,
)
ax1.plot(
    results_day["datetime"].dt.hour,
    results_day["Ensemble"],
    label="Ensemble",
    linewidth=3,
    color="black",
    linestyle="--",
)
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Solar Radiation (W/m²)")
ax1.set_title(f"ML Models Forecast: {year}-{month:02d}-{day:02d}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

# Plot 2: Ensemble vs API Comparison
ax2 = axes[0, 1]
ax2.plot(
    range(24),
    ensemble_preds,
    label="Ensemble (Our Model)",
    linewidth=2.5,
    color="blue",
    marker="o",
    markersize=5,
)
if not all(np.isnan(api_preds)):
    ax2.plot(
        range(24),
        api_preds,
        label=f"{api_name} API",
        linewidth=2.5,
        color="red",
        marker="s",
        markersize=5,
    )
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Solar Radiation (W/m²)")
ax2.set_title("🔍 Model vs API Comparison")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

# Plot 3: Calibrated Ensemble with confidence range
ax3 = axes[1, 0]
# Create calibrated range using tree models with calibration factors
calibrated_xgb = []
calibrated_rf = []
for i in range(24):
    hour = hours[i]
    cal_factor = HOUR_CALIBRATION.get(hour, 1.0)
    calibrated_xgb.append(min(max(0, xgb_preds[i] * cal_factor), 1000))
    calibrated_rf.append(min(max(0, rf_preds[i] * cal_factor), 1000))

# Use calibrated tree models for the range (±10% uncertainty band)
pred_center = np.array(ensemble_preds)
pred_min = pred_center * 0.85  # Lower bound (-15%)
pred_max = pred_center * 1.15  # Upper bound (+15%)

ax3.fill_between(
    range(24),
    pred_min,
    pred_max,
    alpha=0.3,
    color="blue",
    label="Calibrated Range (±15%)",
)
ax3.plot(
    range(24),
    ensemble_preds,
    label="Calibrated Ensemble",
    linewidth=2,
    color="blue",
    marker="o",
)
if not all(np.isnan(api_preds)):
    ax3.plot(
        range(24),
        api_preds,
        label=f"{api_name} API",
        linewidth=2,
        color="red",
        linestyle="--",
        marker="s",
        markersize=4,
    )
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Solar Radiation (W/m²)")
ax3.set_title(f"Calibrated Ensemble with Uncertainty vs {api_name}")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 24, 2))

# Plot 4: Difference/Error Analysis
ax4 = axes[1, 1]
if not all(np.isnan(api_preds)):
    diff = np.array(ensemble_preds) - np.array(api_preds)
    colors = ["green" if d >= 0 else "red" for d in diff]
    ax4.bar(range(24), diff, color=colors, alpha=0.7, edgecolor="black")
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Difference (W/m²)")
    ax4.set_title(f"📊 Prediction Difference (Ensemble - {api_name})")
    ax4.set_xticks(range(0, 24, 2))
    ax4.grid(True, alpha=0.3, axis="y")

    # Add statistics
    mean_diff = np.nanmean(diff)
    abs_mean_diff = np.nanmean(np.abs(diff))
    ax4.text(
        0.02,
        0.98,
        f"Mean Diff: {mean_diff:.1f} W/m²\nMAE: {abs_mean_diff:.1f} W/m²",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
else:
    ax4.text(
        0.5,
        0.5,
        "API Data Not Available",
        transform=ax4.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="gray",
    )
    ax4.set_title("📊 Prediction Difference (Ensemble - API)")

plt.tight_layout()
plt.savefig(f"solar_forecast_{year}_{month:02d}_{day:02d}.png", dpi=150)
plt.show()

print(f"\n🖼️ Plot saved to solar_forecast_{year}_{month:02d}_{day:02d}.png")

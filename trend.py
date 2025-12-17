# ==========================================
# 🌤 Daily Hourly Solar Radiation Forecast
# Using trained models from 2018–2025
# Input: year, month, day → Output: 24-hour forecast
# ==========================================

import os, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

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
xgb_model = joblib.load("saved_models/xgboost_model.pkl")
rf_model = joblib.load("saved_models/random_forest_model.pkl")
lstm_model = load_model("saved_models/lstm_model.h5")
cnn_lstm_model = load_model("saved_models/cnn_lstm_model.h5")

scaler_X = joblib.load("saved_models/scaler_X.pkl")
scaler_y = joblib.load("saved_models/scaler_y.pkl")

# Load ensemble weights
try:
    ensemble_weights = joblib.load("saved_models/ensemble_weights.pkl")
    print(f"Ensemble weights loaded: {ensemble_weights}")
except:
    ensemble_weights = {
        "XGBoost": 0.25,
        "RandomForest": 0.25,
        "LSTM": 0.25,
        "CNN-LSTM": 0.25,
    }
    print("Using equal ensemble weights")

# ---------- 3. Forecast settings ----------
MAX_LAG = 24
SEQ_LEN = 24

# ---------- 4. User input: day to forecast ----------
year = 2025
month = 6  # June
day = 15  # 15th

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

# Calculate ensemble prediction (weighted average)
ensemble_preds = []
for i in range(24):
    ensemble_pred = (
        ensemble_weights.get("XGBoost", 0.25) * xgb_preds[i]
        + ensemble_weights.get("RandomForest", 0.25) * rf_preds[i]
        + ensemble_weights.get("LSTM", 0.25) * lstm_preds[i]
        + ensemble_weights.get("CNN-LSTM", 0.25) * cnn_preds[i]
    )
    ensemble_preds.append(max(0, ensemble_pred))

results_day = pd.DataFrame(
    {
        "datetime": date_index,
        "XGBoost": xgb_preds,
        "RandomForest": rf_preds,
        "LSTM": lstm_preds,
        "CNN_LSTM": cnn_preds,
        "Ensemble": ensemble_preds,
    }
)

# ---------- 8. Save to CSV ----------
results_day.to_csv(f"solar_forecast_{year}_{month:02d}_{day:02d}.csv", index=False)
print(f"✅ Forecast saved to solar_forecast_{year}_{month:02d}_{day:02d}.csv")

# ---------- 9. Print hourly predictions ----------
print(f"\nHourly Solar Radiation Forecast for {year}-{month:02d}-{day:02d}:")
print(
    f"{'Hour':>4} | {'XGBoost':>10} | {'RandomForest':>13} | {'LSTM':>8} | {'CNN-LSTM':>10} | {'Ensemble':>10}"
)
print("-" * 70)
for i, row in results_day.iterrows():
    hour = row["datetime"].hour
    print(
        f"{hour:02d}:00 | {row['XGBoost']:10.2f} | {row['RandomForest']:13.2f} | {row['LSTM']:8.2f} | {row['CNN_LSTM']:10.2f} | {row['Ensemble']:10.2f}"
    )

# ---------- 10. Plot results ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: All models
ax1 = axes[0]
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
ax1.set_title(f"Hourly Solar Radiation Forecast: {year}-{month:02d}-{day:02d}")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

# Plot 2: Ensemble only with confidence range
ax2 = axes[1]
model_preds = np.array([xgb_preds, rf_preds, lstm_preds, cnn_preds])
pred_min = model_preds.min(axis=0)
pred_max = model_preds.max(axis=0)

ax2.fill_between(
    range(24), pred_min, pred_max, alpha=0.3, color="blue", label="Model Range"
)
ax2.plot(
    range(24), ensemble_preds, label="Ensemble", linewidth=2, color="blue", marker="o"
)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Solar Radiation (W/m²)")
ax2.set_title(f"Ensemble Forecast with Prediction Range")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(f"solar_forecast_{year}_{month:02d}_{day:02d}.png", dpi=150)
plt.show()

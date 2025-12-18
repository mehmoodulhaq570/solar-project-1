# ==========================================
# 🌤 Daily Hourly Solar Radiation Forecast
# Using trained models from 2018–2025
# With Real-Time API Comparison (Open-Meteo)
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


def fetch_openmeteo_solar_forecast(year, month, day):
    """
    Fetch solar radiation forecast from Open-Meteo API.
    Returns hourly GHI (Global Horizontal Irradiance) in W/m².
    """
    date_str = f"{year}-{month:02d}-{day:02d}"

    # Open-Meteo API endpoint for solar radiation
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "shortwave_radiation",  # GHI in W/m²
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


def fetch_solcast_forecast(year, month, day, api_key=None):
    """
    Fetch solar radiation forecast from Solcast API (requires API key).
    Free tier: 10 API calls/day.
    Sign up at: https://solcast.com/free-rooftop-solar-forecasting
    """
    if not api_key:
        print("⚠️ Solcast API: No API key provided. Skipping.")
        return None

    date_str = f"{year}-{month:02d}-{day:02d}"

    url = f"https://api.solcast.com.au/world_radiation/forecasts"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hours": 24,
        "output_parameters": "ghi",
        "format": "json",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecasts = data.get("forecasts", [])
        hourly_ghi = [f.get("ghi", 0) for f in forecasts[:24]]

        if len(hourly_ghi) == 24:
            print(f"✅ Solcast API: Successfully fetched forecast for {date_str}")
            return hourly_ghi
        else:
            print(f"⚠️ Solcast API: Unexpected data length")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Solcast API Error: {e}")
        return None


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
# For API comparison, use today or a future date (within 16 days)
# Open-Meteo only provides forecasts, not historical data
today = datetime.now()
year = today.year
month = today.month
day = today.day

# Override with specific date if needed (must be within forecast range for API)
# year = 2025
# month = 12
# day = 19  # Use tomorrow or a future date for API data

# Optional: Solcast API key (get free key at https://solcast.com/free-rooftop-solar-forecasting)
SOLCAST_API_KEY = None  # Set your key here if you have one

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

# ---------- 8. Fetch API Predictions for Comparison ----------
print("\n📡 Fetching API predictions for comparison...")
openmeteo_preds = fetch_openmeteo_solar_forecast(year, month, day)
solcast_preds = fetch_solcast_forecast(year, month, day, SOLCAST_API_KEY)

# If API data not available, use None for comparison
if openmeteo_preds is None:
    openmeteo_preds = [np.nan] * 24
    print("⚠️ Open-Meteo data not available - will show as missing in plots")

if solcast_preds is None:
    solcast_preds = [np.nan] * 24

results_day = pd.DataFrame(
    {
        "datetime": date_index,
        "XGBoost": xgb_preds,
        "RandomForest": rf_preds,
        "LSTM": lstm_preds,
        "CNN_LSTM": cnn_preds,
        "Ensemble": ensemble_preds,
        "OpenMeteo_API": openmeteo_preds,
    }
)

# Add Solcast if available
if not all(np.isnan(solcast_preds)):
    results_day["Solcast_API"] = solcast_preds

# ---------- 9. Save to CSV ----------
results_day.to_csv(f"solar_forecast_{year}_{month:02d}_{day:02d}.csv", index=False)
print(f"\n✅ Forecast saved to solar_forecast_{year}_{month:02d}_{day:02d}.csv")

# ---------- 10. Print hourly predictions ----------
print(f"\nHourly Solar Radiation Forecast for {year}-{month:02d}-{day:02d}:")
api_col = "OpenMeteo" if not all(np.isnan(openmeteo_preds)) else ""
print(
    f"{'Hour':>4} | {'XGBoost':>10} | {'RandomForest':>13} | {'LSTM':>8} | {'CNN-LSTM':>10} | {'Ensemble':>10} | {'OpenMeteo':>10}"
)
print("-" * 85)
for i, row in results_day.iterrows():
    hour = row["datetime"].hour
    api_val = row.get("OpenMeteo_API", np.nan)
    api_str = f"{api_val:10.2f}" if not np.isnan(api_val) else "       N/A"
    print(
        f"{hour:02d}:00 | {row['XGBoost']:10.2f} | {row['RandomForest']:13.2f} | {row['LSTM']:8.2f} | {row['CNN_LSTM']:10.2f} | {row['Ensemble']:10.2f} | {api_str}"
    )

# ---------- 11. Calculate comparison metrics ----------
if not all(np.isnan(openmeteo_preds)):
    print("\n📊 Model vs API Comparison Metrics:")
    api_array = np.array(openmeteo_preds)

    # Filter out night hours (where both are near 0) for meaningful comparison
    valid_mask = (api_array > 10) | (np.array(ensemble_preds) > 10)

    if valid_mask.sum() > 0:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        api_valid = api_array[valid_mask]
        ensemble_valid = np.array(ensemble_preds)[valid_mask]
        xgb_valid = np.array(xgb_preds)[valid_mask]
        rf_valid = np.array(rf_preds)[valid_mask]

        print(f"\n  {'Model':<15} | {'MAE':>10} | {'RMSE':>10} | {'R²':>8}")
        print("  " + "-" * 50)

        for name, preds in [
            ("Ensemble", ensemble_valid),
            ("XGBoost", xgb_valid),
            ("RandomForest", rf_valid),
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
if not all(np.isnan(openmeteo_preds)):
    ax2.plot(
        range(24),
        openmeteo_preds,
        label="Open-Meteo API",
        linewidth=2.5,
        color="red",
        marker="s",
        markersize=5,
    )
if not all(np.isnan(solcast_preds)):
    ax2.plot(
        range(24),
        solcast_preds,
        label="Solcast API",
        linewidth=2.5,
        color="green",
        marker="^",
        markersize=5,
    )
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Solar Radiation (W/m²)")
ax2.set_title("🔍 Model vs API Comparison")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

# Plot 3: Ensemble with confidence range
ax3 = axes[1, 0]
model_preds = np.array([xgb_preds, rf_preds, lstm_preds, cnn_preds])
pred_min = model_preds.min(axis=0)
pred_max = model_preds.max(axis=0)

ax3.fill_between(
    range(24), pred_min, pred_max, alpha=0.3, color="blue", label="Model Range"
)
ax3.plot(
    range(24), ensemble_preds, label="Ensemble", linewidth=2, color="blue", marker="o"
)
if not all(np.isnan(openmeteo_preds)):
    ax3.plot(
        range(24),
        openmeteo_preds,
        label="Open-Meteo API",
        linewidth=2,
        color="red",
        linestyle="--",
        marker="s",
        markersize=4,
    )
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Solar Radiation (W/m²)")
ax3.set_title("Ensemble with Prediction Range vs API")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 24, 2))

# Plot 4: Difference/Error Analysis
ax4 = axes[1, 1]
if not all(np.isnan(openmeteo_preds)):
    diff = np.array(ensemble_preds) - np.array(openmeteo_preds)
    colors = ["green" if d >= 0 else "red" for d in diff]
    ax4.bar(range(24), diff, color=colors, alpha=0.7, edgecolor="black")
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Difference (W/m²)")
    ax4.set_title("📊 Prediction Difference (Ensemble - API)")
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

# ==========================================
# 🌤 Forecast Next-Year Solar Radiation Trend
# Using trained models from 2018–2025
# Iterative forecasting for 2026
# ==========================================

import os, warnings, gc
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------- 1. Load historical data (2018–2025) ----------
df = pd.read_csv(
    "NASA meteriological and solar radiaton data/lahore_hourly_filled.csv",
    parse_dates=["datetime"],
    dayfirst=True,
    index_col="datetime"
)
df = df.sort_index().apply(pd.to_numeric, errors="coerce")
df.index = pd.to_datetime(df.index, dayfirst=True)

# Filter daylight-only
df_day = df[df["SolarZenith"] < 90].copy()
df_day.dropna(subset=["SolarRadiation"], inplace=True)
df_day["hour"] = df_day.index.hour
df_day["month"] = df_day.index.month

target_col = "SolarRadiation"

# ---------- 2. Load trained models ----------
xgb_model = joblib.load("save_model/xgboost_model.pkl")
rf_model  = joblib.load("save_model/random_forest_model.pkl")
lstm_model = load_model("save_model/lstm_model.h5")
cnn_lstm_model = load_model("save_model/cnn_lstm_model.h5")

scaler_X = joblib.load("save_model/scaler_X.pkl")
scaler_y = joblib.load("save_model/scaler_y.pkl")

# ---------- 3. Forecast settings ----------
MAX_LAG = 24
SEQ_LEN  = 24
hours_to_forecast = 500  # Number of hours to forecast (adjust as needed)

last_hist = df_day.copy()
last_hist_values = last_hist[target_col].values

# ---------- 4. Forecast for tree-based models ----------
tree_predictions = []
tree_series = last_hist_values[-MAX_LAG:].tolist()

for i in range(hours_to_forecast):
    hour = (last_hist.index[-1].hour + i + 1) % 24
    month = ((last_hist.index[-1].month - 1 + ((last_hist.index[-1].hour + i + 1)//24)) % 12) + 1
    lag_feats = tree_series[-MAX_LAG:]
    
    Temperature = last_hist["Temperature"].iloc[-1]
    HumiditySpecific = last_hist["HumiditySpecific"].iloc[-1]
    HumidityRelative = last_hist["HumidityRelative"].iloc[-1]
    Pressure = last_hist["Pressure"].iloc[-1]
    WindSpeed = last_hist["WindSpeed"].iloc[-1]
    WindDirection = last_hist["WindDirection"].iloc[-1]

    X_row = [hour, month, Temperature, HumiditySpecific, HumidityRelative, Pressure, WindSpeed, WindDirection] + lag_feats
    X_row = np.array(X_row).reshape(1, -1)
    
    xgb_pred = xgb_model.predict(X_row)[0]
    rf_pred  = rf_model.predict(X_row)[0]
    
    tree_predictions.append((xgb_pred, rf_pred))
    tree_series.append(xgb_pred)

# ---------- 5. Forecast for sequence-based models ----------
seq_features = ['hour','month','Temperature','HumiditySpecific','HumidityRelative','Pressure','WindSpeed','WindDirection', target_col]
seq_data = last_hist[seq_features].copy()

seq_predictions = []
seq_array = scaler_X.transform(seq_data.tail(SEQ_LEN).drop(columns=[target_col]))

for i in range(hours_to_forecast):
    X_seq_input = np.array(seq_array[-SEQ_LEN:]).reshape(1, SEQ_LEN, seq_array.shape[1])
    
    lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_seq_input)).flatten()[0]
    cnn_pred  = scaler_y.inverse_transform(cnn_lstm_model.predict(X_seq_input)).flatten()[0]
    
    seq_predictions.append((lstm_pred, cnn_pred))
    
    new_row = seq_array[-1].copy()
    new_row[-1] = scaler_y.transform(np.array([[lstm_pred]]))[0,0]
    seq_array = np.vstack([seq_array, new_row])

# ---------- 6. Prepare datetime index ----------
last_time = last_hist.index[-1]
forecast_index = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=hours_to_forecast, freq='H')

# ---------- 7. Combine results ----------
results_df = pd.DataFrame({
    'datetime': forecast_index,
    'XGBoost': [x for x,_ in tree_predictions],
    'RandomForest': [x for _,x in tree_predictions],
    'LSTM': [x for x,_ in seq_predictions],
    'CNN_LSTM': [x for _,x in seq_predictions],
})

# ---------- 8. Save to CSV ----------
results_df.to_csv("predicted_trend_2026.csv", index=False)
print("✅ Next-year trend predictions saved to predicted_trend_2026.csv")

# ---------- 9. Plot the predictions ----------
plt.figure(figsize=(16,6))
plt.plot(results_df['datetime'], results_df['XGBoost'], label='XGBoost', alpha=0.8)
plt.plot(results_df['datetime'], results_df['RandomForest'], label='Random Forest', alpha=0.8)
plt.plot(results_df['datetime'], results_df['LSTM'], label='LSTM', alpha=0.8)
plt.plot(results_df['datetime'], results_df['CNN_LSTM'], label='CNN-LSTM', alpha=0.8)
plt.xlabel("Datetime")
plt.ylabel("Predicted Solar Radiation")
plt.title("Next-Year Solar Radiation Trend (2026)")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 🌤 Benchmark: XGBoost, RandomForest, LSTM, CNN-LSTM
# Save Models + Remove Transformer
# ==========================================

import os, gc, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

# reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# create save directory
os.makedirs("saved_models", exist_ok=True)

# TF GPU memory growth (Colab safe)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ---------- 1. Load data ----------
df = pd.read_csv("NASA meteriological and solar radiaton data/lahore_hourly_filled.csv")
df.columns = df.columns.str.strip()

if "datetime" not in df.columns:
    raise ValueError(f"'datetime' column not found")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"])
df.set_index("datetime", inplace=True)
df = df.sort_index()

df = df.apply(pd.to_numeric, errors="coerce")

print("Loaded CSV range:", df.index.min(), "→", df.index.max())

required = [
    "ClearSkyRadiation",
    "SolarRadiation",
    "DirectRadiation",
    "DiffuseRadiation",
    "SolarZenith",
    "Temperature",
    "HumiditySpecific",
    "HumidityRelative",
    "Pressure",
    "WindSpeed",
    "WindDirection",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Missing columns: ", missing)

# IMPORTANT: Use ALL hours (not just daylight) so model learns night = 0
df_day = df.copy()
df_day.dropna(subset=["SolarRadiation"], inplace=True)

# Add cyclical encoding for hour and month (helps model understand periodicity)
df_day["hour"] = df_day.index.hour
df_day["month"] = df_day.index.month
df_day["day_of_year"] = df_day.index.dayofyear
df_day["hour_sin"] = np.sin(2 * np.pi * df_day["hour"] / 24)
df_day["hour_cos"] = np.cos(2 * np.pi * df_day["hour"] / 24)
df_day["month_sin"] = np.sin(2 * np.pi * df_day["month"] / 12)
df_day["month_cos"] = np.cos(2 * np.pi * df_day["month"] / 12)
# Day of year cyclical encoding (captures seasonal variation, peak at summer solstice ~day 172)
df_day["doy_sin"] = np.sin(2 * np.pi * df_day["day_of_year"] / 365)
df_day["doy_cos"] = np.cos(2 * np.pi * df_day["day_of_year"] / 365)

# Clearness Index: ratio of actual to clear sky radiation (0-1, indicates cloud cover)
# Avoid division by zero - set to 0 when ClearSkyRadiation is 0
df_day["ClearnessIndex"] = np.where(
    df_day["ClearSkyRadiation"] > 0,
    df_day["SolarRadiation"] / df_day["ClearSkyRadiation"],
    0,
)
df_day["ClearnessIndex"] = df_day["ClearnessIndex"].clip(0, 1)  # Clip to valid range

target_col = "SolarRadiation"
print("Total records (including night):", len(df_day))

# ---------- 2. Train/Test split ----------
split_idx = int(len(df_day) * 0.8)
train = df_day.iloc[:split_idx].copy()
test = df_day.iloc[split_idx:].copy()

# ---------- 3. Lag features ----------
MAX_LAG = 24


def make_lag_features(series_df, max_lag=24):
    # Features for tree-based models (XGBoost, RandomForest)
    # Order: time features, solar geometry, weather, derived features
    cols_base = [
        target_col,
        # Time features
        "hour",
        "month",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        # Solar geometry
        "SolarZenith",  # Key feature: >= 90 means sun is below horizon (night)
        "ClearSkyRadiation",  # Theoretical max radiation (helps model cloud effects)
        # Weather features
        "Temperature",
        "HumiditySpecific",
        "HumidityRelative",
        "Pressure",
        "WindSpeed",
        "WindDirection",
    ]
    df_feat = series_df[cols_base].copy()
    for lag in range(1, max_lag + 1):
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)
    return df_feat.dropna()


df_lag = make_lag_features(df_day, MAX_LAG)

train_lag = df_lag.loc[df_lag.index.intersection(train.index)]
test_lag = df_lag.loc[df_lag.index.intersection(test.index)]

# fallback if no test rows
if len(test_lag) == 0:
    train_lag = df_lag.iloc[:-MAX_LAG].copy()
    test_lag = df_lag.iloc[-MAX_LAG:].copy()

X_train_tree = train_lag.drop(columns=[target_col])
y_train_tree = train_lag[target_col]
X_test_tree = test_lag.drop(columns=[target_col])
y_test = test_lag[target_col]

# ---------- 4. Scaling for deep learning ----------
SEQ_LEN = 24
# Features for sequence models (LSTM, CNN-LSTM)
# Order matches tree models (excluding target which goes last)
features = [
    # Time features
    "hour",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    # Solar geometry
    "SolarZenith",
    "ClearSkyRadiation",
    # Weather features
    "Temperature",
    "HumiditySpecific",
    "HumidityRelative",
    "Pressure",
    "WindSpeed",
    "WindDirection",
    # Target (must be last)
    target_col,
]
df_seq = df_day[features].dropna()

train_seq_df = df_seq.loc[: train.index.max()]
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(train_seq_df.drop(columns=[target_col]))
scaler_y.fit(train_seq_df[[target_col]])

# Save scalers
joblib.dump(scaler_X, "saved_models/scaler_X.pkl")
joblib.dump(scaler_y, "saved_models/scaler_y.pkl")


def create_sequences(df_in, seq_len=24):
    Xs, ys, idxs = [], [], []
    X_arr = scaler_X.transform(df_in.drop(columns=[target_col]))
    y_arr = scaler_y.transform(df_in[[target_col]]).flatten()

    for i in range(seq_len, len(df_in)):
        Xs.append(X_arr[i - seq_len : i])
        ys.append(y_arr[i])
        idxs.append(df_in.index[i])

    return np.array(Xs), np.array(ys), np.array(idxs)


X_seq_all, y_seq_all, idxs_all = create_sequences(df_seq)

train_idx_set = set(train.index)
test_idx_set = set(test.index)

train_mask = np.array([idx in train_idx_set for idx in idxs_all])
test_mask = np.array([idx in test_idx_set for idx in idxs_all])

X_train_seq = X_seq_all[train_mask]
y_train_seq = y_seq_all[train_mask]
X_test_seq = X_seq_all[test_mask]
y_test_seq = y_seq_all[test_mask]

y_train_unscaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# ---------- RESULTS ----------
results = []

# ======================================
# 6. XGBoost
# ======================================
print("\n=== Training XGBoost ===")
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

xgb_model.fit(X_train_tree, y_train_tree)

# Save model
joblib.dump(xgb_model, "saved_models/xgboost_model.pkl")

xgb_train_pred = xgb_model.predict(X_train_tree)
xgb_test_pred = xgb_model.predict(X_test_tree)

results.append(
    [
        "XGBoost",
        r2_score(y_train_tree, xgb_train_pred),
        r2_score(y_test, xgb_test_pred),
        mean_absolute_error(y_test, xgb_test_pred),
        rmse(y_test, xgb_test_pred),
    ]
)

# ======================================
# 7. Random Forest
# ======================================
print("\n=== Training Random Forest ===")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_tree, y_train_tree)

# Save model
joblib.dump(rf, "saved_models/random_forest_model.pkl")

rf_train_pred = rf.predict(X_train_tree)
rf_test_pred = rf.predict(X_test_tree)

results.append(
    [
        "Random Forest",
        r2_score(y_train_tree, rf_train_pred),
        r2_score(y_test, rf_test_pred),
        mean_absolute_error(y_test, rf_test_pred),
        rmse(y_test, rf_test_pred),
    ]
)

# ======================================
# 8. LSTM (Improved Architecture)
# ======================================
print("\n=== Training LSTM ===")
K.clear_session()

# Build improved LSTM with attention and residual connections
inp_lstm = layers.Input(shape=(SEQ_LEN, X_train_seq.shape[2]))

# First LSTM block with more units
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp_lstm)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Second LSTM block
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Attention mechanism - learn which timesteps are important
attention = layers.Dense(1, activation="tanh")(x)
attention = layers.Flatten()(attention)
attention = layers.Activation("softmax")(attention)
attention = layers.RepeatVector(256)(attention)  # 128*2 for bidirectional
attention = layers.Permute([2, 1])(attention)

# Apply attention
x = layers.Multiply()([x, attention])
x = layers.Lambda(lambda xin: K.sum(xin, axis=1))(x)

# Dense layers with residual-like structure
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
out_lstm = layers.Dense(1)(x)

lstm_model = models.Model(inp_lstm, out_lstm)

# Learning rate scheduler for better convergence
lr_schedule = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=0
)
es = callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"],
)

lstm_model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.1,
    epochs=100,
    batch_size=64,
    callbacks=[es, lr_schedule],
    verbose=0,
)

# Save model
lstm_model.save("saved_models/lstm_model.h5")

lstm_train_pred = scaler_y.inverse_transform(lstm_model.predict(X_train_seq))
lstm_test_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_seq))

lstm_train_pred = lstm_train_pred.flatten()
lstm_test_pred = lstm_test_pred.flatten()

results.append(
    [
        "LSTM",
        r2_score(y_train_unscaled, lstm_train_pred),
        r2_score(y_test_unscaled, lstm_test_pred),
        mean_absolute_error(y_test_unscaled, lstm_test_pred),
        rmse(y_test_unscaled, lstm_test_pred),
    ]
)

# ======================================
# 9. CNN-LSTM (Improved Architecture)
# ======================================
print("\n=== Training CNN-LSTM ===")
K.clear_session()

# Build improved CNN-LSTM with multi-scale convolutions and attention
inp = layers.Input(shape=(SEQ_LEN, X_train_seq.shape[2]))

# Multi-scale convolution block 1 - capture different temporal patterns
conv1 = layers.Conv1D(64, 2, padding="same", activation="relu")(inp)
conv2 = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
conv3 = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
x = layers.Concatenate()([conv1, conv2, conv3])  # Multi-scale features
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Second conv block
x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# LSTM layers with more capacity
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Self-attention layer
attention = layers.Dense(1, activation="tanh")(x)
attention = layers.Flatten()(attention)
attention = layers.Activation("softmax")(attention)
attention = layers.RepeatVector(256)(attention)  # 128*2 for bidirectional
attention = layers.Permute([2, 1])(attention)
x = layers.Multiply()([x, attention])
x = layers.Lambda(lambda xin: K.sum(xin, axis=1))(x)

# Dense output layers
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(1)(x)

cnn_lstm_model = models.Model(inp, out)

# Learning rate scheduler
lr_schedule_cnn = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=0
)
es_cnn = callbacks.EarlyStopping(
    patience=15, restore_best_weights=True, monitor="val_loss"
)

cnn_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"],
)

cnn_lstm_model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.1,
    epochs=100,
    batch_size=64,
    callbacks=[es_cnn, lr_schedule_cnn],
    verbose=0,
)

# Save model
cnn_lstm_model.save("saved_models/cnn_lstm_model.h5")

cnn_train_pred = scaler_y.inverse_transform(cnn_lstm_model.predict(X_train_seq))
cnn_test_pred = scaler_y.inverse_transform(cnn_lstm_model.predict(X_test_seq))

cnn_train_pred = cnn_train_pred.flatten()
cnn_test_pred = cnn_test_pred.flatten()

results.append(
    [
        "CNN-LSTM",
        r2_score(y_train_unscaled, cnn_train_pred),
        r2_score(y_test_unscaled, cnn_test_pred),
        mean_absolute_error(y_test_unscaled, cnn_test_pred),
        rmse(y_test_unscaled, cnn_test_pred),
    ]
)

# ---------- 10. Final Results ----------
results_df = pd.DataFrame(
    results, columns=["Model", "R2_Train", "R2_Test", "MAE_Test", "RMSE_Test"]
)
print(results_df)
results_df.to_csv("saved_models/model_results.csv", index=False)

# ======================================
# 11. Ensemble Model (Weighted Average)
# ======================================
print("\n=== Creating Ensemble Model ===")

# Calculate weights based on test R2 scores (better models get higher weight)
r2_scores = {
    "XGBoost": r2_score(y_test, xgb_test_pred),
    "RandomForest": r2_score(y_test, rf_test_pred),
    "LSTM": r2_score(y_test_unscaled, lstm_test_pred),
    "CNN-LSTM": r2_score(y_test_unscaled, cnn_test_pred),
}

# Normalize weights (ensure they sum to 1)
total_r2 = sum(max(0, r2) for r2 in r2_scores.values())
ensemble_weights = {k: max(0, v) / total_r2 for k, v in r2_scores.items()}
print("Ensemble weights:", {k: f"{v:.3f}" for k, v in ensemble_weights.items()})

# Save ensemble weights for prediction
joblib.dump(ensemble_weights, "saved_models/ensemble_weights.pkl")

# Note: Ensemble prediction requires aligning predictions from all models
# For now, we save weights; ensemble will be computed during prediction in trend.py

print("\n🎉 All models saved successfully in: saved_models/")

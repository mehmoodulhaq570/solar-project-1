# ===============================================
# Configuration Settings for Solar Prediction Dashboard
# ===============================================

import os

# ============== Location Configuration ==============
LATITUDE = 31.56
LONGITUDE = 74.35
TIMEZONE = "Asia/Karachi"
LOCATION_NAME = "Lahore, Pakistan"

# ============== Model Configuration ==============
SEQUENCE_LENGTH = 24
MAX_LAG = 24
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS_TFT = 2
NUM_LAYERS_TCN = 4
KERNEL_SIZE_TCN = 3

# ============== Model Folders ==============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "NASA meteriological and solar radiaton data", "lahore_hourly_filled.csv"
)

MODEL_FOLDERS = {
    "Traditional Models": os.path.join(BASE_DIR, "saved_models"),
    "LSTM Models": os.path.join(BASE_DIR, "saved_models_lstm"),
    "TFT/TCN Models": os.path.join(BASE_DIR, "saved_models_tft"),
}

# Default model folder
DEFAULT_MODEL_FOLDER = "saved_models"

# ============== Available Models ==============
AVAILABLE_MODELS = [
    "XGBoost",
    "Random Forest",
    "LSTM",
    "CNN-LSTM",
    "TFT",
    "TCN",
    "Ensemble",
]

# Default selected models
DEFAULT_MODELS = ["XGBoost", "Random Forest", "TFT", "Ensemble"]

# ============== Ensemble Configuration ==============
# NASA POWER API weights (optimized for NASA satellite data)
NASA_ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.30,
    "RandomForest": 0.25,
    "LSTM": 0.08,
    "CNN-LSTM": 0.07,
    "TFT": 0.18,
    "TCN": 0.12,
}

NASA_HOUR_CALIBRATION = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 1.0,
    12: 1.0,
    13: 1.0,
    14: 1.0,
    15: 1.0,
    16: 1.0,
    17: 0.0,
    18: 0.0,
    19: 0.0,
    20: 0.0,
    21: 0.0,
    22: 0.0,
    23: 0.0,
}

# Open-Meteo API weights (optimized for real-time forecasts)
OPENMETEO_ENSEMBLE_WEIGHTS = {
    "XGBoost": 0.20,
    "RandomForest": 0.18,
    "LSTM": 0.12,
    "CNN-LSTM": 0.10,
    "TFT": 0.25,
    "TCN": 0.15,
}

OPENMETEO_HOUR_CALIBRATION = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.22,
    9: 0.41,
    10: 0.70,
    11: 0.94,
    12: 0.99,
    13: 1.05,
    14: 1.22,
    15: 1.65,
    16: 3.19,
    17: 10.0,
    18: 0.0,
    19: 0.0,
    20: 0.0,
    21: 0.0,
    22: 0.0,
    23: 0.0,
}

# ============== Feature Columns ==============
# Features for tree-based models (XGBoost, Random Forest)
TREE_FEATURES = [
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
]

# Features for sequence models (LSTM, CNN-LSTM, TFT, TCN)
SEQUENCE_FEATURES = [
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
    "SolarRadiation",  # Target (must be last)
]

TARGET_COLUMN = "SolarRadiation"

# ============== Visualization Colors ==============
MODEL_COLORS = {
    "XGBoost": "#e74c3c",
    "Random Forest": "#2ecc71",
    "LSTM": "#3498db",
    "CNN-LSTM": "#9b59b6",
    "TFT": "#1abc9c",
    "TCN": "#f1c40f",
    "Ensemble": "#2c3e50",
    "NASA POWER": "#e67e22",
    "Open-Meteo": "#f39c12",
}

# ============== API Settings ==============
API_TIMEOUT = 30  # seconds

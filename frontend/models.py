# ===============================================
# Model Loader Module
# Load all trained models (Traditional + Deep Learning)
# ===============================================

import os
import sys
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============== Check Available Frameworks ==============
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model as keras_load_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    pass


def load_traditional_models(model_folder: str) -> Dict[str, Any]:
    """
    Load traditional ML models (XGBoost, Random Forest).

    Args:
        model_folder: Path to folder containing model files

    Returns:
        Dictionary with loaded models
    """
    models = {}

    # Load XGBoost
    if XGBOOST_AVAILABLE:
        try:
            xgb_path = os.path.join(model_folder, "xgboost_model.pkl")
            if os.path.exists(xgb_path):
                models["XGBoost"] = joblib.load(xgb_path)
        except Exception as e:
            print(f"Warning: XGBoost not loaded: {e}")

    # Load Random Forest
    try:
        rf_path = os.path.join(model_folder, "random_forest_model.pkl")
        if os.path.exists(rf_path):
            models["Random Forest"] = joblib.load(rf_path)
    except Exception as e:
        print(f"Warning: Random Forest not loaded: {e}")

    return models


def load_keras_models(model_folder: str) -> Dict[str, Any]:
    """
    Load Keras/TensorFlow models (LSTM, CNN-LSTM).

    Args:
        model_folder: Path to folder containing model files

    Returns:
        Dictionary with loaded models
    """
    models = {}

    if not TENSORFLOW_AVAILABLE:
        return models

    # Load LSTM
    try:
        lstm_path = os.path.join(model_folder, "lstm_model.h5")
        if os.path.exists(lstm_path):
            models["LSTM"] = keras_load_model(lstm_path)
    except Exception as e:
        print(f"Warning: LSTM not loaded: {e}")

    # Load CNN-LSTM
    try:
        cnn_lstm_path = os.path.join(model_folder, "cnn_lstm_model.h5")
        if os.path.exists(cnn_lstm_path):
            models["CNN-LSTM"] = keras_load_model(cnn_lstm_path)
    except Exception as e:
        print(f"Warning: CNN-LSTM not loaded: {e}")

    return models


def load_pytorch_models(model_folder: str, input_size: int = 17) -> Dict[str, Any]:
    """
    Load PyTorch models (TFT, TCN).

    Args:
        model_folder: Path to folder containing model files
        input_size: Number of input features (default: 17)

    Returns:
        Dictionary with loaded models
    """
    models = {}

    if not PYTORCH_AVAILABLE:
        return models

    # Add tft folder to path for imports
    tft_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tft")
    if tft_folder not in sys.path:
        sys.path.insert(0, tft_folder)

    try:
        from tft import LitTFT
        from tcn import LitTCN
    except ImportError as e:
        print(f"Warning: Could not import TFT/TCN modules: {e}")
        return models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TFT
    try:
        tft_path = os.path.join(model_folder, "tft_model.pt")
        tft_ckpt_path = os.path.join(model_folder, "tft_model_best.ckpt")

        if os.path.exists(tft_ckpt_path):
            # Load from checkpoint (preferred)
            model = LitTFT.load_from_checkpoint(
                tft_ckpt_path,
                input_size=input_size,
                hidden_size=64,
                output_size=1,
                map_location=device,
            )
            model.eval()
            models["TFT"] = model
        elif os.path.exists(tft_path):
            # Load from state dict
            model = LitTFT(
                input_size=input_size,
                hidden_size=64,
                output_size=1,
            )
            model.load_state_dict(torch.load(tft_path, map_location=device))
            model.eval()
            models["TFT"] = model
    except Exception as e:
        print(f"Warning: TFT not loaded: {e}")

    # Load TCN
    try:
        tcn_path = os.path.join(model_folder, "tcn_model.pt")
        tcn_ckpt_path = os.path.join(model_folder, "tcn_model_best.ckpt")

        if os.path.exists(tcn_ckpt_path):
            # Load from checkpoint (preferred)
            model = LitTCN.load_from_checkpoint(
                tcn_ckpt_path,
                input_size=input_size,
                hidden_size=64,
                output_size=1,
                map_location=device,
            )
            model.eval()
            models["TCN"] = model
        elif os.path.exists(tcn_path):
            # Load from state dict
            model = LitTCN(
                input_size=input_size,
                hidden_size=64,
                output_size=1,
            )
            model.load_state_dict(torch.load(tcn_path, map_location=device))
            model.eval()
            models["TCN"] = model
    except Exception as e:
        print(f"Warning: TCN not loaded: {e}")

    return models


def load_scalers(model_folder: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load feature and target scalers.

    Args:
        model_folder: Path to folder containing scaler files

    Returns:
        Tuple of (scaler_X, scaler_y) or (None, None) if not found
    """
    scaler_X, scaler_y = None, None

    try:
        scaler_X_path = os.path.join(model_folder, "scaler_X.pkl")
        scaler_y_path = os.path.join(model_folder, "scaler_y.pkl")

        if os.path.exists(scaler_X_path):
            scaler_X = joblib.load(scaler_X_path)
        if os.path.exists(scaler_y_path):
            scaler_y = joblib.load(scaler_y_path)
    except Exception as e:
        print(f"Warning: Scalers not loaded: {e}")

    return scaler_X, scaler_y


def load_ensemble_weights(model_folder: str) -> Dict[str, float]:
    """
    Load ensemble weights from file or return defaults.

    Args:
        model_folder: Path to folder containing weights file

    Returns:
        Dictionary of model weights
    """
    default_weights = {
        "XGBoost": 0.20,
        "RandomForest": 0.18,
        "LSTM": 0.12,
        "CNN-LSTM": 0.10,
        "TFT": 0.25,
        "TCN": 0.15,
    }

    try:
        weights_path = os.path.join(model_folder, "ensemble_weights.pkl")
        if os.path.exists(weights_path):
            return joblib.load(weights_path)
    except Exception as e:
        print(f"Warning: Ensemble weights not loaded: {e}")

    return default_weights


def load_all_models(
    traditional_folder: str, lstm_folder: str, tft_folder: str, selected_models: list
) -> Dict[str, Any]:
    """
    Load all required models based on selection.

    Args:
        traditional_folder: Path to traditional models
        lstm_folder: Path to LSTM models
        tft_folder: Path to TFT/TCN models
        selected_models: List of selected model names

    Returns:
        Dictionary with all loaded models
    """
    all_models = {}

    # Load traditional models
    needs_traditional = any(
        m in selected_models for m in ["XGBoost", "Random Forest", "Ensemble"]
    )
    if needs_traditional:
        traditional_models = load_traditional_models(traditional_folder)
        all_models.update(traditional_models)

    # Load Keras models
    needs_keras = any(m in selected_models for m in ["LSTM", "CNN-LSTM", "Ensemble"])
    if needs_keras:
        keras_models = load_keras_models(lstm_folder)
        all_models.update(keras_models)

    # Load PyTorch models
    needs_pytorch = any(m in selected_models for m in ["TFT", "TCN", "Ensemble"])
    if needs_pytorch:
        pytorch_models = load_pytorch_models(tft_folder)
        all_models.update(pytorch_models)

    return all_models


# Export framework availability flags
__all__ = [
    "TENSORFLOW_AVAILABLE",
    "PYTORCH_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "load_traditional_models",
    "load_keras_models",
    "load_pytorch_models",
    "load_scalers",
    "load_ensemble_weights",
    "load_all_models",
]

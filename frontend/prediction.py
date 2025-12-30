# ===============================================
# Prediction Engine Module
# Generate predictions from all model types
# ===============================================

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Check for PyTorch
try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .config import (
    SEQUENCE_LENGTH,
    MAX_LAG,
    TARGET_COLUMN,
    NASA_ENSEMBLE_WEIGHTS,
    OPENMETEO_ENSEMBLE_WEIGHTS,
    NASA_HOUR_CALIBRATION,
    OPENMETEO_HOUR_CALIBRATION,
)


class PredictionEngine:
    """
    Unified prediction engine for all model types.
    """

    def __init__(
        self, models: Dict[str, Any], scaler_X: Any, scaler_y: Any, df_day: pd.DataFrame
    ):
        """
        Initialize prediction engine.

        Args:
            models: Dictionary of loaded models
            scaler_X: Feature scaler
            scaler_y: Target scaler
            df_day: Prepared dataframe with historical data
        """
        self.models = models
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.df_day = df_day

        # Get device for PyTorch models
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def get_weather_values(self) -> Dict[str, float]:
        """Get the latest weather values from historical data."""
        return {
            "Temperature": self.df_day["Temperature"].iloc[-1],
            "HumiditySpecific": self.df_day["HumiditySpecific"].iloc[-1],
            "HumidityRelative": self.df_day["HumidityRelative"].iloc[-1],
            "Pressure": self.df_day["Pressure"].iloc[-1],
            "WindSpeed": self.df_day["WindSpeed"].iloc[-1],
            "WindDirection": self.df_day["WindDirection"].iloc[-1],
        }

    def get_hourly_solar_stats(self, month: int) -> Tuple[pd.Series, pd.Series]:
        """Get typical solar zenith and clear sky radiation for each hour."""
        monthly_data = self.df_day[self.df_day["month"] == month]
        hourly_zenith = monthly_data.groupby("hour")["SolarZenith"].mean()
        hourly_clearsky = monthly_data.groupby("hour")["ClearSkyRadiation"].mean()
        return hourly_zenith, hourly_clearsky

    def predict_tree_models(
        self, year: int, month: int, day: int, hours: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Generate predictions using tree-based models (XGBoost, Random Forest).

        Args:
            year, month, day: Forecast date
            hours: Array of hours (0-23)

        Returns:
            Dictionary with predictions for each tree model
        """
        predictions = {"XGBoost": [], "Random Forest": []}

        has_xgb = "XGBoost" in self.models
        has_rf = "Random Forest" in self.models

        if not has_xgb and not has_rf:
            return predictions

        # Get weather values and solar stats
        weather = self.get_weather_values()
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        hourly_zenith, hourly_clearsky = self.get_hourly_solar_stats(month)

        # Initialize lag series with historical values
        last_hist_values = self.df_day[TARGET_COLUMN].values
        tree_series = last_hist_values[-MAX_LAG:].tolist()

        for h in range(24):
            hour = hours[h]

            # Time features with cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            doy_sin = np.sin(2 * np.pi * day_of_year / 365)
            doy_cos = np.cos(2 * np.pi * day_of_year / 365)

            # Solar features
            solar_zenith = hourly_zenith.get(hour, 90)
            clear_sky_rad = hourly_clearsky.get(hour, 0)

            # Lag features
            lag_feats = tree_series[-MAX_LAG:]

            # Build feature row (matching training.py order)
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
                weather["Temperature"],
                weather["HumiditySpecific"],
                weather["HumidityRelative"],
                weather["Pressure"],
                weather["WindSpeed"],
                weather["WindDirection"],
            ] + lag_feats

            X_row = np.array(X_row).reshape(1, -1)

            # XGBoost prediction
            if has_xgb:
                xgb_pred = self.models["XGBoost"].predict(X_row)[0]
                predictions["XGBoost"].append(max(0, xgb_pred))
                tree_series.append(xgb_pred)

            # Random Forest prediction
            if has_rf:
                rf_pred = self.models["Random Forest"].predict(X_row)[0]
                predictions["Random Forest"].append(max(0, rf_pred))
                if not has_xgb:
                    tree_series.append(rf_pred)

        return predictions

    def predict_keras_models(
        self, year: int, month: int, day: int, hours: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Generate predictions using Keras models (LSTM, CNN-LSTM).

        Args:
            year, month, day: Forecast date
            hours: Array of hours (0-23)

        Returns:
            Dictionary with predictions for each Keras model
        """
        predictions = {"LSTM": [], "CNN-LSTM": []}

        has_lstm = "LSTM" in self.models
        has_cnn = "CNN-LSTM" in self.models

        if (
            (not has_lstm and not has_cnn)
            or self.scaler_X is None
            or self.scaler_y is None
        ):
            return predictions

        # Get weather values and solar stats
        weather = self.get_weather_values()
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        hourly_zenith, hourly_clearsky = self.get_hourly_solar_stats(month)

        # Prepare sequence features
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
            TARGET_COLUMN,
        ]

        seq_data = self.df_day[seq_features].copy()
        X_seq_scaled = self.scaler_X.transform(
            seq_data.drop(columns=[TARGET_COLUMN])
        ).astype(np.float32)

        seq_array = X_seq_scaled[-SEQUENCE_LENGTH:].copy()

        for h in range(24):
            X_seq_input = seq_array[-SEQUENCE_LENGTH:].reshape(
                1, SEQUENCE_LENGTH, seq_array.shape[1]
            )

            # LSTM prediction
            if has_lstm:
                lstm_pred_scaled = self.models["LSTM"].predict(X_seq_input, verbose=0)
                lstm_pred = self.scaler_y.inverse_transform(
                    lstm_pred_scaled.reshape(-1, 1)
                )[0, 0]
                predictions["LSTM"].append(max(0, lstm_pred))

            # CNN-LSTM prediction
            if has_cnn:
                cnn_pred_scaled = self.models["CNN-LSTM"].predict(
                    X_seq_input, verbose=0
                )
                cnn_pred = self.scaler_y.inverse_transform(
                    cnn_pred_scaled.reshape(-1, 1)
                )[0, 0]
                predictions["CNN-LSTM"].append(max(0, cnn_pred))

            # Prepare next row for sequence
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
                    weather["Temperature"],
                    weather["HumiditySpecific"],
                    weather["HumidityRelative"],
                    weather["Pressure"],
                    weather["WindSpeed"],
                    weather["WindDirection"],
                ]
            ).reshape(1, -1)

            next_row = self.scaler_X.transform(next_row_raw).flatten()
            seq_array = np.vstack([seq_array, next_row])

        return predictions

    def predict_pytorch_models(
        self, year: int, month: int, day: int, hours: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Generate predictions using PyTorch models (TFT, TCN).

        Args:
            year, month, day: Forecast date
            hours: Array of hours (0-23)

        Returns:
            Dictionary with predictions for each PyTorch model
        """
        predictions = {"TFT": [], "TCN": []}

        if not PYTORCH_AVAILABLE:
            return predictions

        has_tft = "TFT" in self.models
        has_tcn = "TCN" in self.models

        if (
            (not has_tft and not has_tcn)
            or self.scaler_X is None
            or self.scaler_y is None
        ):
            return predictions

        # Get weather values and solar stats
        weather = self.get_weather_values()
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        hourly_zenith, hourly_clearsky = self.get_hourly_solar_stats(month)

        # Prepare sequence features (same as Keras models)
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
            TARGET_COLUMN,
        ]

        seq_data = self.df_day[seq_features].copy()
        X_seq_scaled = self.scaler_X.transform(
            seq_data.drop(columns=[TARGET_COLUMN])
        ).astype(np.float32)

        seq_array = X_seq_scaled[-SEQUENCE_LENGTH:].copy()

        for h in range(24):
            X_seq_input = seq_array[-SEQUENCE_LENGTH:].reshape(
                1, SEQUENCE_LENGTH, seq_array.shape[1]
            )

            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_seq_input, dtype=torch.float32).to(self.device)

            # TFT prediction
            if has_tft:
                with torch.no_grad():
                    tft_pred_scaled = self.models["TFT"](X_tensor).cpu().numpy()
                tft_pred = self.scaler_y.inverse_transform(
                    tft_pred_scaled.reshape(-1, 1)
                )[0, 0]
                predictions["TFT"].append(max(0, tft_pred))

            # TCN prediction
            if has_tcn:
                with torch.no_grad():
                    tcn_pred_scaled = self.models["TCN"](X_tensor).cpu().numpy()
                tcn_pred = self.scaler_y.inverse_transform(
                    tcn_pred_scaled.reshape(-1, 1)
                )[0, 0]
                predictions["TCN"].append(max(0, tcn_pred))

            # Prepare next row for sequence
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
                    weather["Temperature"],
                    weather["HumiditySpecific"],
                    weather["HumidityRelative"],
                    weather["Pressure"],
                    weather["WindSpeed"],
                    weather["WindDirection"],
                ]
            ).reshape(1, -1)

            next_row = self.scaler_X.transform(next_row_raw).flatten()
            seq_array = np.vstack([seq_array, next_row])

        return predictions

    def calculate_ensemble(
        self,
        all_predictions: Dict[str, List[float]],
        hours: np.ndarray,
        use_openmeteo_weights: bool = False,
    ) -> List[float]:
        """
        Calculate calibrated ensemble prediction.

        Args:
            all_predictions: Dictionary with all model predictions
            hours: Array of hours
            use_openmeteo_weights: Whether to use Open-Meteo optimized weights

        Returns:
            List of ensemble predictions
        """
        # Select weights based on API
        if use_openmeteo_weights:
            weights = OPENMETEO_ENSEMBLE_WEIGHTS
            calibration = OPENMETEO_HOUR_CALIBRATION
        else:
            weights = NASA_ENSEMBLE_WEIGHTS
            calibration = NASA_HOUR_CALIBRATION

        # Get predictions with fallback to zeros
        xgb_preds = all_predictions.get("XGBoost", [0] * 24)
        rf_preds = all_predictions.get("Random Forest", [0] * 24)
        lstm_preds = all_predictions.get("LSTM", [0] * 24)
        cnn_preds = all_predictions.get("CNN-LSTM", [0] * 24)
        tft_preds = all_predictions.get("TFT", [0] * 24)
        tcn_preds = all_predictions.get("TCN", [0] * 24)

        # Fill empty predictions
        if not xgb_preds:
            xgb_preds = [0] * 24
        if not rf_preds:
            rf_preds = [0] * 24
        if not lstm_preds:
            lstm_preds = [0] * 24
        if not cnn_preds:
            cnn_preds = [0] * 24
        if not tft_preds:
            tft_preds = [0] * 24
        if not tcn_preds:
            tcn_preds = [0] * 24

        ensemble_preds = []
        for i in range(24):
            hour = hours[i]

            # Calculate weighted ensemble
            base_ensemble = (
                weights.get("XGBoost", 0) * xgb_preds[i]
                + weights.get("RandomForest", 0) * rf_preds[i]
                + weights.get("LSTM", 0) * lstm_preds[i]
                + weights.get("CNN-LSTM", 0) * cnn_preds[i]
                + weights.get("TFT", 0) * tft_preds[i]
                + weights.get("TCN", 0) * tcn_preds[i]
            )

            # Apply hour-specific calibration
            hour_calibration = calibration.get(hour, 1.0)
            ensemble_pred = base_ensemble * hour_calibration

            # Cap at reasonable maximum
            ensemble_pred = min(max(0, ensemble_pred), 1000)
            ensemble_preds.append(ensemble_pred)

        return ensemble_preds

    def predict_all(
        self, year: int, month: int, day: int, use_openmeteo_weights: bool = False
    ) -> Dict[str, List[float]]:
        """
        Generate predictions from all loaded models.

        Args:
            year, month, day: Forecast date
            use_openmeteo_weights: Whether to use Open-Meteo optimized weights

        Returns:
            Dictionary with all predictions including ensemble
        """
        # Create date index and hours
        date_index = pd.date_range(
            start=f"{year}-{month:02d}-{day:02d} 00:00",
            end=f"{year}-{month:02d}-{day:02d} 23:00",
            freq="H",
        )
        hours = date_index.hour.values

        # Get predictions from each model type
        all_predictions = {}

        # Tree-based models
        tree_preds = self.predict_tree_models(year, month, day, hours)
        all_predictions.update(tree_preds)

        # Keras models
        keras_preds = self.predict_keras_models(year, month, day, hours)
        all_predictions.update(keras_preds)

        # PyTorch models
        pytorch_preds = self.predict_pytorch_models(year, month, day, hours)
        all_predictions.update(pytorch_preds)

        # Calculate ensemble
        ensemble_preds = self.calculate_ensemble(
            all_predictions, hours, use_openmeteo_weights
        )
        all_predictions["Ensemble"] = ensemble_preds

        return all_predictions

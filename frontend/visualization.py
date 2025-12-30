# ===============================================
# Visualization Module
# Charts and plots for solar radiation forecasts
# ===============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODEL_COLORS


def create_forecast_plots(
    results_df: pd.DataFrame,
    selected_models: List[str],
    api_preds: Optional[List[float]],
    api_name: str,
    ensemble_preds: List[float],
    forecast_date: str,
) -> plt.Figure:
    """
    Create comprehensive forecast visualization.

    Args:
        results_df: DataFrame with all predictions
        selected_models: List of selected model names
        api_preds: API predictions (optional)
        api_name: Name of the API used
        ensemble_preds: Ensemble predictions
        forecast_date: Date string for title

    Returns:
        Matplotlib figure
    """
    # Create 2x2 plot grid if API data is available, else 1x2
    if api_preds:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes[0], axes[1]
        ax3, ax4 = None, None

    colors = MODEL_COLORS.copy()
    colors[api_name] = "#f39c12"

    # ========== Plot 1: All Models ==========
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

    # Add API to first plot
    if api_preds:
        ax1.plot(
            range(24),
            api_preds,
            label=api_name,
            color="#f39c12",
            linewidth=2.5,
            linestyle=":",
            marker="s",
            markersize=4,
        )

    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
    ax1.set_title(f"Solar Forecast: {forecast_date}", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.5, 23.5)

    # ========== Plot 2: Ensemble with Uncertainty ==========
    pred_center = np.array(ensemble_preds)
    pred_min = pred_center * 0.85  # Lower bound (-15%)
    pred_max = pred_center * 1.15  # Upper bound (+15%)

    ax2.fill_between(
        range(24),
        pred_min,
        pred_max,
        alpha=0.3,
        color="#3498db",
        label="Calibrated Range (±15%)",
    )
    ax2.plot(
        range(24),
        ensemble_preds,
        label="Calibrated Ensemble",
        linewidth=2.5,
        color="#2c3e50",
        marker="o",
        markersize=5,
    )
    if api_preds:
        ax2.plot(
            range(24),
            api_preds,
            label=api_name,
            linewidth=2.5,
            color="#f39c12",
            linestyle="--",
            marker="s",
            markersize=4,
        )
    ax2.set_xlabel("Hour of Day", fontsize=12)
    ax2.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
    ax2.set_title("Calibrated Ensemble vs API", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.5, 23.5)

    # ========== Plot 3 & 4: Additional Analysis ==========
    if api_preds and ax3 is not None and ax4 is not None:
        # Plot 3: TFT/TCN vs Traditional Models
        tft_preds = results_df.get("TFT", None)
        tcn_preds = results_df.get("TCN", None)
        xgb_preds = results_df.get("XGBoost", None)
        rf_preds = results_df.get("Random Forest", None)

        if xgb_preds is not None:
            ax3.plot(
                range(24),
                results_df["XGBoost"],
                label="XGBoost",
                color=colors["XGBoost"],
                linewidth=2,
                marker="o",
                markersize=4,
            )
        if rf_preds is not None:
            ax3.plot(
                range(24),
                results_df["Random Forest"],
                label="Random Forest",
                color=colors["Random Forest"],
                linewidth=2,
                marker="s",
                markersize=4,
            )
        if tft_preds is not None:
            ax3.plot(
                range(24),
                results_df["TFT"],
                label="TFT",
                color=colors["TFT"],
                linewidth=2,
                marker="^",
                markersize=4,
            )
        if tcn_preds is not None:
            ax3.plot(
                range(24),
                results_df["TCN"],
                label="TCN",
                color=colors["TCN"],
                linewidth=2,
                marker="d",
                markersize=4,
            )
        ax3.plot(
            range(24),
            api_preds,
            label=api_name,
            color="#f39c12",
            linewidth=2.5,
            linestyle="--",
            marker="p",
            markersize=4,
        )

        ax3.set_xlabel("Hour of Day", fontsize=12)
        ax3.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
        ax3.set_title("Model Comparison vs API", fontsize=14)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))

        # Plot 4: Difference Analysis
        diff = np.array(ensemble_preds) - np.array(api_preds)
        colors_bar = ["#2ecc71" if d >= 0 else "#e74c3c" for d in diff]
        ax4.bar(range(24), diff, color=colors_bar, alpha=0.7, edgecolor="black")
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax4.set_xlabel("Hour of Day", fontsize=12)
        ax4.set_ylabel("Difference (W/m²)", fontsize=12)
        ax4.set_title("📊 Prediction Difference (Ensemble - API)", fontsize=14)
        ax4.set_xticks(range(0, 24, 2))
        ax4.grid(True, alpha=0.3, axis="y")

        # Add MAE annotation
        mae = np.mean(np.abs(diff))
        ax4.text(
            0.02,
            0.98,
            f"MAE: {mae:.1f} W/m²",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    return fig


def create_deep_learning_comparison_plot(
    results_df: pd.DataFrame, api_preds: Optional[List[float]], api_name: str
) -> plt.Figure:
    """
    Create comparison plot specifically for deep learning models.

    Args:
        results_df: DataFrame with predictions
        api_preds: API predictions
        api_name: Name of API used

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    # Plot 1: LSTM vs TFT
    has_lstm = "LSTM" in results_df.columns
    has_tft = "TFT" in results_df.columns
    has_cnn = "CNN-LSTM" in results_df.columns
    has_tcn = "TCN" in results_df.columns

    if has_lstm:
        ax1.plot(
            range(24),
            results_df["LSTM"],
            label="LSTM",
            color=MODEL_COLORS["LSTM"],
            linewidth=2,
            marker="o",
            markersize=4,
        )
    if has_cnn:
        ax1.plot(
            range(24),
            results_df["CNN-LSTM"],
            label="CNN-LSTM",
            color=MODEL_COLORS["CNN-LSTM"],
            linewidth=2,
            marker="s",
            markersize=4,
        )
    if has_tft:
        ax1.plot(
            range(24),
            results_df["TFT"],
            label="TFT",
            color=MODEL_COLORS["TFT"],
            linewidth=2,
            marker="^",
            markersize=4,
        )
    if has_tcn:
        ax1.plot(
            range(24),
            results_df["TCN"],
            label="TCN",
            color=MODEL_COLORS["TCN"],
            linewidth=2,
            marker="d",
            markersize=4,
        )

    if api_preds:
        ax1.plot(
            range(24),
            api_preds,
            label=api_name,
            color="#f39c12",
            linewidth=2.5,
            linestyle="--",
            marker="p",
            markersize=4,
        )

    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
    ax1.set_title("Deep Learning Models Comparison", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # Plot 2: TFT vs TCN (Transformer vs Convolutional)
    if has_tft and has_tcn:
        ax2.fill_between(
            range(24),
            np.minimum(results_df["TFT"], results_df["TCN"]),
            np.maximum(results_df["TFT"], results_df["TCN"]),
            alpha=0.3,
            color="#1abc9c",
            label="TFT-TCN Range",
        )
        ax2.plot(
            range(24),
            results_df["TFT"],
            label="TFT (Transformer)",
            color=MODEL_COLORS["TFT"],
            linewidth=2.5,
            marker="^",
            markersize=5,
        )
        ax2.plot(
            range(24),
            results_df["TCN"],
            label="TCN (Convolutional)",
            color=MODEL_COLORS["TCN"],
            linewidth=2.5,
            marker="d",
            markersize=5,
        )

        if api_preds:
            ax2.plot(
                range(24),
                api_preds,
                label=api_name,
                color="#f39c12",
                linewidth=2,
                linestyle="--",
                marker="s",
                markersize=4,
            )

        ax2.set_xlabel("Hour of Day", fontsize=12)
        ax2.set_ylabel("Solar Radiation (W/m²)", fontsize=12)
        ax2.set_title("TFT vs TCN: Transformer vs Convolutional", fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
    else:
        ax2.text(
            0.5,
            0.5,
            "TFT and TCN models required\nfor this comparison",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.set_title("TFT vs TCN Comparison", fontsize=14)

    plt.tight_layout()
    return fig


def calculate_metrics(
    predictions: Dict[str, List[float]], api_preds: List[float]
) -> pd.DataFrame:
    """
    Calculate comparison metrics between models and API.

    Args:
        predictions: Dictionary of model predictions
        api_preds: API predictions (ground truth)

    Returns:
        DataFrame with metrics for each model
    """
    api_array = np.array(api_preds)

    # Focus on daylight hours for meaningful comparison
    valid_mask = api_array > 10

    if valid_mask.sum() == 0:
        return pd.DataFrame()

    metrics_data = []

    for model_name, preds in predictions.items():
        if not preds or len(preds) != 24:
            continue

        preds_arr = np.array(preds)

        try:
            mae = mean_absolute_error(api_array[valid_mask], preds_arr[valid_mask])
            rmse = np.sqrt(
                mean_squared_error(api_array[valid_mask], preds_arr[valid_mask])
            )
            r2 = r2_score(api_array[valid_mask], preds_arr[valid_mask])

            # Calculate bias
            bias = np.mean(preds_arr[valid_mask] - api_array[valid_mask])

            metrics_data.append(
                {
                    "Model": model_name,
                    "MAE (W/m²)": mae,
                    "RMSE (W/m²)": rmse,
                    "R²": r2,
                    "Bias (W/m²)": bias,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(metrics_data)


def create_model_architecture_info() -> Dict[str, Dict[str, str]]:
    """
    Return information about each model architecture.

    Returns:
        Dictionary with model information
    """
    return {
        "XGBoost": {
            "type": "Tree-Based Ensemble",
            "description": "Gradient boosting with decision trees",
            "features": "Uses 24-hour lag features",
            "strength": "Fast, interpretable, handles missing data",
        },
        "Random Forest": {
            "type": "Tree-Based Ensemble",
            "description": "Bootstrap aggregated decision trees",
            "features": "Uses 24-hour lag features",
            "strength": "Robust, less overfitting than single trees",
        },
        "LSTM": {
            "type": "Recurrent Neural Network",
            "description": "Bidirectional Long Short-Term Memory",
            "features": "24-step sequence input",
            "strength": "Captures long-range temporal dependencies",
        },
        "CNN-LSTM": {
            "type": "Hybrid Neural Network",
            "description": "1D Convolution + LSTM layers",
            "features": "24-step sequence input",
            "strength": "Local pattern extraction + temporal modeling",
        },
        "TFT": {
            "type": "Transformer",
            "description": "Temporal Fusion Transformer with attention",
            "features": "24-step sequence, multi-head attention",
            "strength": "State-of-the-art temporal modeling, interpretable attention",
        },
        "TCN": {
            "type": "Convolutional Neural Network",
            "description": "Temporal Convolutional Network with dilated convolutions",
            "features": "24-step sequence, causal convolutions",
            "strength": "Parallelizable, large receptive field, no recurrence",
        },
        "Ensemble": {
            "type": "Weighted Average",
            "description": "Calibrated weighted combination of all models",
            "features": "Hour-specific calibration",
            "strength": "Reduces individual model errors, more robust",
        },
    }

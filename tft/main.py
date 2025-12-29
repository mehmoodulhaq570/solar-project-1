# ==========================================
# 🌤 TFT (Temporal Fusion Transformer) Training
# Aligned with training.py for Solar Radiation Prediction
# ==========================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add parent directory to path to access shared files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TFT model from tft.py
from tft import LitTFT

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Create save directory (same as training.py)
os.makedirs("../saved_models", exist_ok=True)


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ---------------------------------------------------
# Custom Dataset for TFT (Sequence-based)
# ---------------------------------------------------
class SolarSequenceDataset(Dataset):
    """
    Creates sequences from scaled data for TFT training.
    Matches the sequence creation logic in training.py.
    """

    def __init__(self, X_scaled, y_scaled, seq_len=24):
        self.X = X_scaled
        self.y = y_scaled
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Sequence of features
        x_seq = torch.tensor(self.X[idx : idx + self.seq_len], dtype=torch.float32)
        # Target is the value at the end of the sequence
        y_val = torch.tensor([self.y[idx + self.seq_len]], dtype=torch.float32)
        return x_seq, y_val


# ---------------------------------------------------
# Main Training Function
# ---------------------------------------------------
def train_tft():
    print("\n" + "=" * 60)
    print("🌤 TFT (Temporal Fusion Transformer) Training")
    print("=" * 60)

    # ---------- 1. Load data (same as training.py) ----------
    data_path = (
        "../NASA meteriological and solar radiaton data/lahore_hourly_filled.csv"
    )
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    if "datetime" not in df.columns:
        raise ValueError("'datetime' column not found")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"Loaded CSV range: {df.index.min()} → {df.index.max()}")

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
        raise ValueError(f"Missing columns: {missing}")

    # Copy data (use all hours like training.py)
    df_day = df.copy()
    df_day.dropna(subset=["SolarRadiation"], inplace=True)

    # ---------- 2. Feature Engineering (exactly as training.py) ----------
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

    # Clearness Index
    df_day["ClearnessIndex"] = np.where(
        df_day["ClearSkyRadiation"] > 0,
        df_day["SolarRadiation"] / df_day["ClearSkyRadiation"],
        0,
    )
    df_day["ClearnessIndex"] = df_day["ClearnessIndex"].clip(0, 1)

    target_col = "SolarRadiation"
    print(f"Total records (including night): {len(df_day)}")

    # ---------- 3. Train/Test split (same as training.py) ----------
    split_idx = int(len(df_day) * 0.8)
    train = df_day.iloc[:split_idx].copy()
    test = df_day.iloc[split_idx:].copy()

    # ---------- 4. Prepare features for TFT ----------
    SEQ_LEN = 24

    # Features for sequence models (same order as training.py)
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
        # Target (must be last for consistency)
        target_col,
    ]

    df_seq = df_day[features].dropna()

    # ---------- 5. Scaling (same as training.py) ----------
    train_seq_df = df_seq.loc[: train.index.max()]

    # Try to load existing scalers, otherwise create new ones
    try:
        scaler_X = joblib.load("../saved_models/scaler_X.pkl")
        scaler_y = joblib.load("../saved_models/scaler_y.pkl")
        print("Loaded existing scalers from saved_models/")
    except FileNotFoundError:
        print("Creating new scalers...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        scaler_X.fit(train_seq_df.drop(columns=[target_col]))
        scaler_y.fit(train_seq_df[[target_col]])
        # Save scalers
        joblib.dump(scaler_X, "../saved_models/scaler_X.pkl")
        joblib.dump(scaler_y, "../saved_models/scaler_y.pkl")

    # Create sequences (same logic as training.py)
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

    # Unscaled targets for evaluation
    y_train_unscaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    print(f"\nTrain sequences: {len(X_train_seq)}")
    print(f"Test sequences: {len(X_test_seq)}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Number of features: {X_train_seq.shape[2]}")

    # ---------- 6. Create DataLoaders ----------
    BATCH_SIZE = 64

    # Create datasets using the pre-created sequences
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32),
    )

    # Split training into train/val (90/10 like training.py)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_seq.reshape(-1, 1), dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ---------- 7. Initialize TFT Model ----------
    input_size = X_train_seq.shape[2]  # Number of features
    hidden_size = 64
    output_size = 1
    learning_rate = 1e-3

    print(f"\n=== Training TFT ===")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")

    model = LitTFT(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    )

    # ---------- 8. Callbacks (similar to training.py) ----------
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    checkpoint = ModelCheckpoint(
        dirpath="../saved_models",
        filename="tft_model_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---------- 9. Train TFT ----------
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[early_stop, checkpoint, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_model_path = checkpoint.best_model_path
    if best_model_path:
        print(f"\nLoading best model from: {best_model_path}")
        model = LitTFT.load_from_checkpoint(best_model_path)

    # ---------- 10. Evaluate on Test Set ----------
    print("\n=== Evaluating TFT ===")

    model.eval()

    # Get predictions
    train_preds = []
    test_preds = []

    with torch.no_grad():
        for batch in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False):
            x, _ = batch
            pred = model(x)
            train_preds.append(pred.cpu().numpy())

        for batch in test_loader:
            x, _ = batch
            pred = model(x)
            test_preds.append(pred.cpu().numpy())

    train_preds = np.concatenate(train_preds).flatten()
    test_preds = np.concatenate(test_preds).flatten()

    # Inverse transform predictions to original scale
    train_preds_unscaled = scaler_y.inverse_transform(
        train_preds.reshape(-1, 1)
    ).flatten()
    test_preds_unscaled = scaler_y.inverse_transform(
        test_preds.reshape(-1, 1)
    ).flatten()

    # Calculate metrics (same as training.py)
    train_r2 = r2_score(y_train_unscaled, train_preds_unscaled)
    test_r2 = r2_score(y_test_unscaled, test_preds_unscaled)
    test_mae = mean_absolute_error(y_test_unscaled, test_preds_unscaled)
    test_rmse = rmse(y_test_unscaled, test_preds_unscaled)

    print(f"\n{'='*50}")
    print(f"TFT Results:")
    print(f"{'='*50}")
    print(f"Train R²:  {train_r2:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"{'='*50}")

    # ---------- 11. Save Results (same format as training.py) ----------
    # Save model state dict
    torch.save(model.state_dict(), "../saved_models/tft_model.pt")
    print(f"\nModel saved to: saved_models/tft_model.pt")

    # Load existing results if available
    results_path = "../saved_models/model_results.csv"
    try:
        results_df = pd.read_csv(results_path)
        # Remove existing TFT row if present
        results_df = results_df[results_df["Model"] != "TFT"]
    except FileNotFoundError:
        results_df = pd.DataFrame(
            columns=["Model", "R2_Train", "R2_Test", "MAE_Test", "RMSE_Test"]
        )

    # Add TFT results
    tft_results = pd.DataFrame(
        [
            {
                "Model": "TFT",
                "R2_Train": train_r2,
                "R2_Test": test_r2,
                "MAE_Test": test_mae,
                "RMSE_Test": test_rmse,
            }
        ]
    )

    results_df = pd.concat([results_df, tft_results], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results appended to: {results_path}")

    # Print comparison with other models
    print(f"\n{'='*60}")
    print("Model Comparison:")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"{'='*60}")

    print("\n🎉 TFT training complete!")

    return model, {
        "R2_Train": train_r2,
        "R2_Test": test_r2,
        "MAE_Test": test_mae,
        "RMSE_Test": test_rmse,
    }


if __name__ == "__main__":
    model, metrics = train_tft()

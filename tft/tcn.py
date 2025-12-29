# ==========================================
# 🌤 TCN (Temporal Convolutional Network) Model
# Aligned with training.py for Solar Radiation Prediction
# ==========================================

import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


class CausalConv1d(nn.Module):
    """
    Causal convolution layer that ensures no future information leakage.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        # Remove the extra padding from the right (future)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    A residual block with causal dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        # First causal convolution
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (downsample if channels don't match)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TemporalConvolutionalNetwork(nn.Module):
    """
    TCN architecture for time series forecasting.
    Uses dilated causal convolutions to capture long-range dependencies.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Build TCN layers with exponentially increasing dilation
        layers = []
        num_channels = [hidden_size] * num_layers

        for i in range(num_layers):
            dilation = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)

        # Output layers
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_activation = nn.GELU()
        self.output_dropout = nn.Dropout(dropout)
        self.output_fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Apply TCN
        out = self.tcn(x)

        # Take the last timestep
        out = out[:, :, -1]  # (batch, hidden_size)

        # Output layers
        out = self.output_norm(out)
        out = self.output_fc1(out)
        out = self.output_activation(out)
        out = self.output_dropout(out)
        prediction = self.output_fc2(out)

        return prediction


class LitTCN(L.LightningModule):
    """
    PyTorch Lightning wrapper for TCN model.
    Aligned with training.py metrics and evaluation structure.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=1e-3,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        self.model = TemporalConvolutionalNetwork(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Buffers to store predictions and targets for metrics
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    # -------------------
    # Training
    # -------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # Store for epoch-end metrics
        self.train_preds.append(y_hat.detach())
        self.train_targets.append(y.detach())

        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) > 0:
            preds = torch.cat(self.train_preds, dim=0).cpu().numpy().flatten()
            targets = torch.cat(self.train_targets, dim=0).cpu().numpy().flatten()

            r2 = r2_score(targets, preds)
            self.log("train_r2", r2, prog_bar=True)

            self.train_preds.clear()
            self.train_targets.clear()

    # -------------------
    # Validation
    # -------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds, dim=0).cpu().numpy().flatten()
            targets = torch.cat(self.val_targets, dim=0).cpu().numpy().flatten()

            r2 = r2_score(targets, preds)
            mae = mean_absolute_error(targets, preds)

            self.log("val_r2", r2, prog_bar=True)
            self.log("val_mae", mae)

            self.val_preds.clear()
            self.val_targets.clear()

    # -------------------
    # Testing
    # -------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_preds.append(y_hat.detach())
        self.test_targets.append(y.detach())

        return nn.functional.mse_loss(y_hat, y)

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return

        preds = torch.cat(self.test_preds, dim=0).cpu().numpy().flatten()
        targets = torch.cat(self.test_targets, dim=0).cpu().numpy().flatten()

        # Compute metrics (same as training.py)
        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)
        rmse = math.sqrt(mean_squared_error(targets, preds))

        self.log("test_r2", r2)
        self.log("test_mae", mae)
        self.log("test_rmse", rmse)

        self.test_preds.clear()
        self.test_targets.clear()

    # -------------------
    # Prediction helper
    # -------------------
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    # -------------------
    # Optimizer with scheduler (like training.py)
    # -------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

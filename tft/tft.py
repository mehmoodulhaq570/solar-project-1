import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT architecture with multi-head attention.
    Aligned with the training.py workflow for solar radiation prediction.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, hidden_size) * 0.1)

        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Output layers with residual connection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_activation = nn.GELU()
        self.output_dropout = nn.Dropout(dropout)
        self.output_fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        projected_x = self.input_projection(x)

        # Add positional encoding
        projected_x = projected_x + self.pos_encoder[:, :seq_len, :]

        # Transformer encoder
        encoded_output = self.transformer_encoder(projected_x)

        # Use last timestep output
        last_step_output = encoded_output[:, -1, :]

        # Output layers
        out = self.output_norm(last_step_output)
        out = self.output_fc1(out)
        out = self.output_activation(out)
        out = self.output_dropout(out)
        prediction = self.output_fc2(out)

        return prediction


class LitTFT(L.LightningModule):
    """
    PyTorch Lightning wrapper for TFT model.
    Aligned with training.py metrics and evaluation structure.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=1e-3,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.model = TemporalFusionTransformer(
            input_size,
            hidden_size,
            output_size,
            num_heads=num_heads,
            num_layers=num_layers,
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

"""
Convert Keras 2.x h5 models to Keras 3.x format
This script rebuilds the model architecture and loads weights
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

print("Converting LSTM model...")
# Recreate LSTM architecture
lstm_model = models.Sequential(
    [
        layers.Input(shape=(24, 17)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)

# Load weights from old h5 file
lstm_model.load_weights("saved_models/lstm_model.h5")
lstm_model.save("saved_models/lstm_model_v3.keras")
print("LSTM model saved as lstm_model_v3.keras")

print("\nConverting CNN-LSTM model...")
# Recreate CNN-LSTM architecture (matching the saved model with BatchNormalization)
inp = layers.Input(shape=(24, 17), name="input_1")
x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d")(inp)
x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(x)
x = layers.BatchNormalization(name="batch_normalization")(x)
x = layers.MaxPool1D(2, name="max_pooling1d")(x)
x = layers.Dropout(0.2, name="dropout")(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bidirectional")(
    x
)
x = layers.Dropout(0.2, name="dropout_1")(x)
x = layers.Bidirectional(layers.LSTM(32), name="bidirectional_1")(x)
x = layers.Dropout(0.2, name="dropout_2")(x)
# Match saved model: dense expects input 64 -> output 32
x = layers.Dense(32, activation="relu", name="dense")(x)
x = layers.Dense(16, activation="relu", name="dense_1")(x)
out = layers.Dense(1, name="dense_2")(x)
cnn_lstm_model = models.Model(inp, out)

# Load weights by name
cnn_lstm_model.load_weights("saved_models/cnn_lstm_model.h5", by_name=True)
cnn_lstm_model.save("saved_models/cnn_lstm_model_v3.keras")
print("CNN-LSTM model saved as cnn_lstm_model_v3.keras")

print("\nDone! Models converted to Keras 3.x format.")

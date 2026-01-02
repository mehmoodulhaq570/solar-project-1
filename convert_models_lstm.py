"""
Convert Keras 2.x h5 models to Keras 3.x format for saved_models_lstm folder
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_FOLDER = "saved_models_lstm"

print("Converting LSTM model from saved_models_lstm...")
# Recreate LSTM architecture (matches the saved model structure)
lstm_model = models.Sequential(
    [
        layers.Input(shape=(24, 15)),  # 15 features based on saved_models_lstm
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

# Load weights from old h5 file
try:
    lstm_model.load_weights(f"{MODEL_FOLDER}/lstm_model.h5")
    lstm_model.save(f"{MODEL_FOLDER}/lstm_model_v3.keras")
    print("LSTM model saved as lstm_model_v3.keras")
except Exception as e:
    print(f"LSTM load failed with 15 features, trying 17: {e}")
    # Try with 17 features
    lstm_model = models.Sequential(
        [
            layers.Input(shape=(24, 17)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    lstm_model.load_weights(f"{MODEL_FOLDER}/lstm_model.h5")
    lstm_model.save(f"{MODEL_FOLDER}/lstm_model_v3.keras")
    print("LSTM model saved as lstm_model_v3.keras (17 features)")

print("\nConverting CNN-LSTM model from saved_models_lstm...")
# Recreate CNN-LSTM architecture (no BatchNormalization based on layer list)
inp = layers.Input(shape=(24, 15), name="input_layer")
x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d")(inp)
x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(x)
x = layers.MaxPool1D(2, name="max_pooling1d")(x)
x = layers.Dropout(0.2, name="dropout")(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bidirectional")(
    x
)
x = layers.Dropout(0.2, name="dropout_1")(x)
x = layers.Bidirectional(layers.LSTM(32), name="bidirectional_1")(x)
x = layers.Dropout(0.2, name="dropout_2")(x)
x = layers.Dense(64, activation="relu", name="dense")(x)
x = layers.Dense(32, activation="relu", name="dense_1")(x)
out = layers.Dense(1, name="dense_2")(x)
cnn_lstm_model = models.Model(inp, out)

try:
    cnn_lstm_model.load_weights(f"{MODEL_FOLDER}/cnn_lstm_model.h5", by_name=True)
    cnn_lstm_model.save(f"{MODEL_FOLDER}/cnn_lstm_model_v3.keras")
    print("CNN-LSTM model saved as cnn_lstm_model_v3.keras")
except Exception as e:
    print(f"CNN-LSTM load failed with 15 features, trying 17: {e}")
    # Try with 17 features
    inp = layers.Input(shape=(24, 17), name="input_layer")
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(x)
    x = layers.MaxPool1D(2, name="max_pooling1d")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bidirectional"
    )(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.Bidirectional(layers.LSTM(32), name="bidirectional_1")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, name="dense_2")(x)
    cnn_lstm_model = models.Model(inp, out)
    cnn_lstm_model.load_weights(f"{MODEL_FOLDER}/cnn_lstm_model.h5", by_name=True)
    cnn_lstm_model.save(f"{MODEL_FOLDER}/cnn_lstm_model_v3.keras")
    print("CNN-LSTM model saved as cnn_lstm_model_v3.keras (17 features)")

print("\nDone! Models converted to Keras 3.x format in saved_models_lstm/")

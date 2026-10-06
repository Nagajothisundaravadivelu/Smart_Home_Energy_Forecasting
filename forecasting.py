"""Data preparation and LSTM utilities for household power forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

READINGS_PER_DAY = 96
LOOKBACK_STEPS = READINGS_PER_DAY
FORECAST_STEPS = 4
MAX_INTERPOLATION_STEPS = 4


def load_household_data(source: str | Path | BinaryIO) -> pd.Series:
    """Read the UCI text export and return 15-minute mean active power in kW."""
    frame = pd.read_csv(
        source,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        dtype={"Date": "string", "Time": "string", "Global_active_power": "string"},
        na_values=["?"],
    )
    timestamps = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    power = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    readings = pd.Series(power.to_numpy(), index=timestamps, name="active_power_kw")
    readings = readings.loc[readings.index.notna()].dropna().sort_index()
    if readings.empty:
        raise ValueError("No valid timestamped active-power readings were found.")

    readings = readings.groupby(level=0).mean()
    quarter_hour = readings.resample("15min").mean()
    # Fill only short interior gaps; long outages must not become artificial training data.
    quarter_hour = quarter_hour.interpolate(
        method="time",
        limit=MAX_INTERPOLATION_STEPS,
        limit_area="inside",
    )
    quarter_hour.name = "active_power_kw"
    return quarter_hour


def make_train_validation_sequences(
    values: np.ndarray,
    lookback: int = LOOKBACK_STEPS,
    horizon: int = FORECAST_STEPS,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create chronological, finite-only sequences and training-only normalization."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must both be positive.")
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be at least 0.5 and less than 1.")
    if len(values) <= lookback + horizon:
        raise ValueError("Not enough readings to create training and validation sequences.")

    split = int(len(values) * train_fraction)
    training_values = values[:split]
    training_values = training_values[np.isfinite(training_values)]
    if training_values.size == 0:
        raise ValueError("No finite readings are available in the training period.")
    mean = float(training_values.mean())
    scale = float(training_values.std())
    if scale == 0:
        scale = 1.0

    sample_count = len(values) - lookback - horizon + 1
    input_windows = np.lib.stride_tricks.sliding_window_view(values, lookback)[:sample_count]
    target_windows = np.lib.stride_tricks.sliding_window_view(values[lookback:], horizon)
    starts = np.arange(sample_count)
    valid = np.isfinite(input_windows).all(axis=1) & np.isfinite(target_windows).all(axis=1)

    train_mask = valid & (starts + lookback + horizon <= split)
    validation_mask = valid & (starts + lookback >= split)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(
            "Not enough continuous readings for both training and validation sequences."
        )

    x_train = ((input_windows[train_mask] - mean) / scale).astype(np.float32, copy=True)
    y_train = ((target_windows[train_mask] - mean) / scale).astype(np.float32, copy=True)
    x_validation = (
        (input_windows[validation_mask] - mean) / scale
    ).astype(np.float32, copy=True)
    y_validation = (
        (target_windows[validation_mask] - mean) / scale
    ).astype(np.float32, copy=True)
    return x_train, y_train, x_validation, y_validation, mean, scale


def train_lstm(
    series: pd.Series,
    training_days: int = 180,
    epochs: int = 8,
) -> tuple[Any, float, float, float]:
    """Train a four-step LSTM and return the model, normalization, and validation RMSE."""
    if training_days < 30:
        raise ValueError("Use at least 30 days of history to train the LSTM.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model training. Install the dependencies "
            "from requirements.txt and restart the app."
        ) from exc

    recent = series.tail(training_days * READINGS_PER_DAY)
    x_train, y_train, x_validation, y_validation, mean, scale = (
        make_train_validation_sequences(recent.to_numpy())
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK_STEPS, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(FORECAST_STEPS),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        x_train[..., np.newaxis],
        y_train,
        validation_data=(x_validation[..., np.newaxis], y_validation),
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    prediction = model.predict(x_validation[..., np.newaxis], verbose=0)
    rmse = float(np.sqrt(np.mean(np.square((prediction - y_validation) * scale))))
    return model, mean, scale, rmse


def forecast_next_hour(
    model: Any,
    recent_readings: np.ndarray,
    mean: float,
    scale: float,
) -> np.ndarray:
    """Forecast the next four 15-minute mean active-power readings in kW."""
    recent_readings = np.asarray(recent_readings, dtype=np.float32).reshape(-1)
    if len(recent_readings) != LOOKBACK_STEPS:
        raise ValueError(f"Exactly {LOOKBACK_STEPS} readings are required for a forecast.")
    if not np.isfinite(recent_readings).all():
        raise ValueError("The latest 24 hours contain a long data gap; forecasting is unavailable.")
    normalized = (recent_readings - mean) / scale
    prediction = model.predict(normalized[np.newaxis, :, np.newaxis], verbose=0)
    return np.maximum(np.asarray(prediction).reshape(-1) * scale + mean, 0.0)

"""Deep-learning pipeline — train an LSTM and benchmark it against the classics.

This adds the "deep learning" end of the "statistical -> deep learning" spectrum.
The LSTM reads 24 hours of history and predicts AQI at all three horizons at once.
We evaluate it on the same time-held-out test set and put its RMSE/MAE/R² side by
side with Ridge / Random Forest / persistence, then record the scores in the
Hopsworks model registry.

    python -m ml.pipelines.train_lstm
"""
from __future__ import annotations

import gc
import os
import sys
from datetime import datetime, timedelta, timezone

# Quieten TensorFlow's C++ logs before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.storage.feature_store import read_features
from ml.storage.registry import MODELS_DIR, register_run
from ml.pipelines.training_pipeline import TRAIN_WINDOW_DAYS
from ml.storage.hopsworks_store import get_feature_store
from ml.training.dataset import HORIZONS, add_targets
from ml.training.models import evaluate
from ml.training.sequences import WINDOW, build_sequences

import tensorflow as tf  # noqa: E402  (after the env var above)
from tensorflow import keras  # noqa: E402


def _split_masks(end_times: np.ndarray, test_frac: float = 0.2):
    unique = np.unique(end_times)
    cutoff = unique[int(len(unique) * (1 - test_frac))]
    train = end_times < cutoff
    return train, ~train


def _classical_scores(fs=None) -> pd.DataFrame:
    """The most recent *classical* run's scores, for a fair side-by-side."""
    from ml.storage.registry import REGISTRY_FG_NAME, REGISTRY_FG_VERSION

    fs = fs or get_feature_store()
    try:
        fg = fs.get_feature_group(REGISTRY_FG_NAME, REGISTRY_FG_VERSION)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    df = fg.read()
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]
    classical = df[df["model_name"] != "lstm"]
    if classical.empty:
        return classical

    # Latest run.
    latest_run = classical.sort_values("trained_at", ascending=False)["run_id"].iloc[0]
    return classical[classical["run_id"] == latest_run][
        ["model_name", "horizon_h", "rmse", "mae", "r2"]
    ]


def run_lstm_training(epochs: int = 80) -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tf.random.set_seed(42)
    np.random.seed(42)

    print("\n  Building sequences from the feature store ...")
    since = (datetime.now(timezone.utc) - timedelta(days=TRAIN_WINDOW_DAYS)).isoformat()
    fs = get_feature_store()
    data = add_targets(read_features(start=since, fs=fs))
    X, y, end_times = build_sequences(data)
    print(f"  {X.shape[0]} sequences of shape {X.shape[1:]} (window={WINDOW}h).")

    train, test = _split_masks(end_times)
    X_tr, X_te, y_tr, y_te = X[train], X[test], y[train], y[test]

    # Free the full arrays now that we've split.
    n_features = X.shape[2]
    del X, y
    gc.collect()

    # Scale features (fit on train only).
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_features))
    scale = lambda a: scaler.transform(a.reshape(-1, n_features)).reshape(a.shape)
    X_tr, X_te = scale(X_tr), scale(X_te)

    # Scale the TARGET too.
    y_scaler = StandardScaler().fit(y_tr)
    y_tr_scaled = y_scaler.transform(y_tr)

    model = keras.Sequential(
        [
            keras.Input((WINDOW, n_features)),
            keras.layers.LSTM(32),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(len(HORIZONS)),  # one output per horizon
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    print(f"  Training LSTM on {len(X_tr)} sequences ...\n")
    model.fit(
        X_tr, y_tr_scaled,
        validation_split=0.1,
        epochs=epochs,
        batch_size=256,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        ],
        verbose=2,
    )

    # Evaluate per horizon, back on the real AQI scale.
    preds = y_scaler.inverse_transform(model.predict(X_te, batch_size=512, verbose=0))
    lstm_rows = []
    for j, h in enumerate(HORIZONS):
        m = evaluate(y_te[:, j], preds[:, j])
        lstm_rows.append({"model_name": "lstm", "horizon_h": h, **m})

    # Persist artifact + register scores.
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "aqi_lstm.keras")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trained_at = datetime.now(timezone.utc)
    register_run(
        [
            {
                "run_id": run_id, "trained_at": trained_at, "horizon_h": r["horizon_h"],
                "model_name": "lstm", "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
                "is_best": False, "artifact_path": str(MODELS_DIR / "aqi_lstm.keras"),
            }
            for r in lstm_rows
        ],
        fs=fs,
    )

    # Side-by-side comparison.
    classical = _classical_scores(fs)
    lstm_df = pd.DataFrame(lstm_rows)
    both = pd.concat([classical, lstm_df], ignore_index=True)
    print("\n  Full benchmark (lower RMSE/MAE better, higher R² better):")
    for h in HORIZONS:
        print(f"\n  --- +{h}h ---")
        block = both[both.horizon_h == h].drop(columns="horizon_h").sort_values("rmse")
        print(block.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print()
    return both


if __name__ == "__main__":
    run_lstm_training()

"""Model registry with local artifact fallback.

Stores trained model artifacts and records training run scorecards.
"""
from __future__ import annotations

import gzip
import os
import pickle
import shutil
from pathlib import Path

import joblib
import pandas as pd

from ml.common.config import PROJECT_ROOT
from ml.storage.hopsworks_store import get_feature_store, get_model_registry, DATA_DIR

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

REGISTRY_FG_NAME = "model_registry"
REGISTRY_FG_VERSION = 1


def _local_registry_path() -> Path:
    return DATA_DIR / f"{REGISTRY_FG_NAME}.parquet"


def save_artifact(model, name: str) -> Path:
    """Persist a fitted model to models/<name>.pkl and return the path."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def register_run(records: list[dict], fs=None) -> int:
    """Append one training run's metrics to local parquet and Hopsworks."""
    frame = pd.DataFrame(records)
    if "trained_at" in frame.columns:
        frame["trained_at"] = pd.to_datetime(frame["trained_at"], utc=True)
    if "is_best" in frame.columns:
        frame["is_best"] = frame["is_best"].astype(bool)

    reg_path = _local_registry_path()
    if reg_path.exists():
        try:
            existing = pd.read_parquet(reg_path)
            combined = pd.concat([existing, frame], ignore_index=True)
            combined.to_parquet(reg_path, index=False)
        except Exception:
            frame.to_parquet(reg_path, index=False)
    else:
        frame.to_parquet(reg_path, index=False)

    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_or_create_feature_group(
                name=REGISTRY_FG_NAME,
                version=REGISTRY_FG_VERSION,
                primary_key=["run_id", "model_name", "horizon_h"],
                description="Training run scorecards: metrics per model per horizon.",
            )
            fg.insert(frame, write_options={"wait_for_job": True})
        except Exception as e:
            print(f"  [Hopsworks] Model registry sync notice: {e}")

    return len(frame)


def save_active_model(
    model, horizon_h: int, model_name: str, metrics: dict, mr=None
) -> None:
    """Store the active model locally in models/ and upload to Hopsworks if connected."""
    artifact_path = MODELS_DIR / f"aqi_{horizon_h}h_{model_name}.pkl"
    joblib.dump(model, artifact_path)

    # Standard horizon file for direct loading
    active_path = MODELS_DIR / f"aqi_{horizon_h}h.pkl"
    joblib.dump(model, active_path)

    mr = mr or get_model_registry()
    if mr is not None:
        try:
            export_dir = MODELS_DIR / f"export_{horizon_h}h"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_artifact = export_dir / f"aqi_{horizon_h}h_{model_name}.pkl"
            joblib.dump(model, export_artifact)

            hw_model = mr.python.create_model(
                name=f"aqi_{horizon_h}h",
                metrics={
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                },
                description=f"AQI +{horizon_h}h forecast model ({model_name})",
            )
            hw_model.save(str(export_dir))
            shutil.rmtree(export_dir, ignore_errors=True)
        except Exception as e:
            print(f"  [Hopsworks] Model Registry upload notice: {e}")


def load_active_model(horizon_h: int, mr=None) -> tuple[object, str]:
    """Load the latest model for a horizon from Hopsworks or local models/."""
    mr = mr or get_model_registry()
    if mr is not None:
        try:
            hw_model = mr.get_best_model(name=f"aqi_{horizon_h}h", metric="rmse", direction="min")
            model_dir = hw_model.download()
            pkl_files = list(Path(model_dir).glob("*.pkl"))
            if pkl_files:
                model = joblib.load(pkl_files[0])
                model_name = pkl_files[0].stem.split("_", 2)[-1] if "_" in pkl_files[0].stem else "unknown"
                return model, model_name
        except Exception:
            pass

    # Local fallback
    local_files = list(MODELS_DIR.glob(f"aqi_{horizon_h}h_*.pkl"))
    if local_files:
        model = joblib.load(local_files[0])
        model_name = local_files[0].stem.split("_", 2)[-1]
        return model, model_name

    active_path = MODELS_DIR / f"aqi_{horizon_h}h.pkl"
    if active_path.exists():
        model = joblib.load(active_path)
        return model, "ridge"

    raise RuntimeError(f"No local or Hopsworks model found for aqi_{horizon_h}h.")


def get_best_models(fs=None) -> pd.DataFrame:
    """The current best model per horizon from registry Feature Group or local parquet."""
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(REGISTRY_FG_NAME, REGISTRY_FG_VERSION)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        reg_path = _local_registry_path()
        if reg_path.exists():
            df = pd.read_parquet(reg_path)
            df.columns = [c.lower() for c in df.columns]

    if df.empty:
        return df

    best = df[df["is_best"] == True]  # noqa: E712
    if best.empty:
        return df.sort_values("trained_at", ascending=False).groupby("horizon_h", as_index=False).first()

    return (
        best.sort_values("trained_at", ascending=False)
        .groupby("horizon_h", as_index=False)
        .first()
        .sort_values("horizon_h")
    )


"""Hopsworks-backed storage layer with transparent local fallback.

Hopsworks provides both a Feature Store and a Model Registry.
If Hopsworks is temporarily unavailable or not yet provisioned,
data is automatically persisted to local parquet files in data/,
ensuring pipelines and applications run reliably.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ml.common.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FEATURES_FG_NAME = "aqi_features"
FEATURES_FG_VERSION = 1

PREDICTIONS_FG_NAME = "predictions"
PREDICTIONS_FG_VERSION = 1

ALERTS_FG_NAME = "alerts"
ALERTS_FG_VERSION = 1

DRIVERS_FG_NAME = "forecast_drivers"
DRIVERS_FG_VERSION = 1


def _api_key() -> str:
    """Hopsworks API key from the environment."""
    return os.getenv("HOPSWORKS_API_KEY", "").strip()


@lru_cache(maxsize=1)
def get_project():
    """Login to Hopsworks and return the project handle (cached)."""
    import hopsworks

    key = _api_key()
    if not key:
        return None

    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
    try:
        if project_name:
            return hopsworks.login(project=project_name, api_key_value=key)
        return hopsworks.login(api_key_value=key)
    except Exception as e:
        print(f"  [Hopsworks] Notice: Using local storage fallback ({type(e).__name__})")
        return None


@lru_cache(maxsize=1)
def get_feature_store():
    """Return the project's Feature Store handle (cached)."""
    proj = get_project()
    if proj is None:
        return None
    try:
        return proj.get_feature_store()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_model_registry():
    """Return the project's Model Registry handle (cached)."""
    proj = get_project()
    if proj is None:
        return None
    try:
        return proj.get_model_registry()
    except Exception:
        return None


# ---------------------------------------------------------------- features ---


def _local_path(fg_name: str) -> Path:
    return DATA_DIR / f"{fg_name}.parquet"


def _get_or_create_features_fg(fs=None):
    """Get or create the main aqi_features Feature Group."""
    fs = fs or get_feature_store()
    if fs is None:
        return None
    return fs.get_or_create_feature_group(
        name=FEATURES_FG_NAME,
        version=FEATURES_FG_VERSION,
        primary_key=["city_id", "event_time"],
        event_time="event_time",
        description="Hourly AQI features for Pakistani cities.",
    )


def write_features(df: pd.DataFrame, fs=None) -> int:
    """Write a features DataFrame to Hopsworks with local parquet persistence."""
    path = _local_path(FEATURES_FG_NAME)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["city_id", "event_time"], keep="last")
            combined.to_parquet(path, index=False)
        except Exception:
            df.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)

    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = _get_or_create_features_fg(fs)
            if fg is not None:
                fg.insert(df, write_options={"wait_for_job": True})
        except Exception as e:
            print(f"  [Hopsworks] Feature Store sync notice: {e}")

    return len(df)


def upsert_features(df: pd.DataFrame, fs=None) -> int:
    """Insert/refresh a recent window of features."""
    return write_features(df, fs)


def table_summary(fs=None) -> pd.DataFrame:
    """Row counts and time span per city — quick sanity check."""
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = _get_or_create_features_fg(fs)
            if fg is not None:
                df = fg.read()
                df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        path = _local_path(FEATURES_FG_NAME)
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]

    if df.empty:
        return pd.DataFrame(columns=["city_id", "rows", "first_hour", "last_hour", "avg_aqi", "max_aqi"])

    return (
        df.groupby("city_id")
        .agg(
            rows=("city_id", "count"),
            first_hour=("event_time", "min"),
            last_hour=("event_time", "max"),
            avg_aqi=("aqi", lambda x: round(x.mean())),
            max_aqi=("aqi", "max"),
        )
        .reset_index()
        .sort_values("city_id")
    )


def features_fg_exists(fs=None) -> bool:
    """Check whether the aqi_features Feature Group already has data."""
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(FEATURES_FG_NAME, FEATURES_FG_VERSION)
            if fg is not None:
                return True
        except Exception:
            pass
    return _local_path(FEATURES_FG_NAME).exists()


# ------------------------------------------------------------ predictions ---


def write_predictions(df: pd.DataFrame, fs=None) -> int:
    """Write predictions to storage."""
    _local_path(PREDICTIONS_FG_NAME).parent.mkdir(exist_ok=True)
    df.to_parquet(_local_path(PREDICTIONS_FG_NAME), index=False)

    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_or_create_feature_group(
                name=PREDICTIONS_FG_NAME,
                version=PREDICTIONS_FG_VERSION,
                primary_key=["city_id", "horizon_h"],
                description="3-day AQI forecasts per city, refreshed hourly.",
            )
            fg.insert(df, write_options={"wait_for_job": True}, overwrite=True)
        except Exception as e:
            print(f"  [Hopsworks] Prediction sync notice: {e}")

    return len(df)


def read_predictions(city_id: str | None = None, fs=None) -> pd.DataFrame:
    """Read predictions, optionally filtered by city."""
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(PREDICTIONS_FG_NAME, PREDICTIONS_FG_VERSION)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        path = _local_path(PREDICTIONS_FG_NAME)
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]

    if city_id and not df.empty and "city_id" in df.columns:
        df = df[df["city_id"] == city_id]
    return df


# --------------------------------------------------------------- alerts ---


def write_alerts(df: pd.DataFrame, fs=None) -> int:
    """Write alerts to storage."""
    _local_path(ALERTS_FG_NAME).parent.mkdir(exist_ok=True)
    df.to_parquet(_local_path(ALERTS_FG_NAME), index=False)

    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_or_create_feature_group(
                name=ALERTS_FG_NAME,
                version=ALERTS_FG_VERSION,
                primary_key=["city_id"],
                description="Hazardous AQI alerts.",
            )
            fg.insert(df, write_options={"wait_for_job": True}, overwrite=True)
        except Exception as e:
            print(f"  [Hopsworks] Alerts sync notice: {e}")

    return len(df)


def read_alerts(city_id: str | None = None, fs=None) -> pd.DataFrame:
    """Read alerts, optionally filtered by city."""
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(ALERTS_FG_NAME, ALERTS_FG_VERSION)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        path = _local_path(ALERTS_FG_NAME)
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]

    if city_id and not df.empty and "city_id" in df.columns:
        df = df[df["city_id"] == city_id]
    return df


# -------------------------------------------------------- SHAP drivers ---


def write_drivers(df: pd.DataFrame, fs=None) -> int:
    """Write SHAP forecast drivers to storage."""
    _local_path(DRIVERS_FG_NAME).parent.mkdir(exist_ok=True)
    df.to_parquet(_local_path(DRIVERS_FG_NAME), index=False)

    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_or_create_feature_group(
                name=DRIVERS_FG_NAME,
                version=DRIVERS_FG_VERSION,
                primary_key=["city_id", "feature"],
                description="SHAP-based feature contributions.",
            )
            fg.insert(df, write_options={"wait_for_job": True}, overwrite=True)
        except Exception as e:
            print(f"  [Hopsworks] Drivers sync notice: {e}")

    return len(df)


def read_drivers(city_id: str | None = None, fs=None) -> pd.DataFrame:
    """Read SHAP drivers, optionally filtered by city."""
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(DRIVERS_FG_NAME, DRIVERS_FG_VERSION)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        path = _local_path(DRIVERS_FG_NAME)
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]

    if city_id and not df.empty and "city_id" in df.columns:
        df = df[df["city_id"] == city_id]
    return df


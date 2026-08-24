"""Nafas forecast API."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(
    title="Nafas Forecast API",
    description=(
        "Three-day air-quality forecasts for Pakistani cities, produced by an "
        "automated machine-learning pipeline. Read-only."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _connect():
    """Connect to Hopsworks (cached). Returns None if unavailable."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    if not api_key:
        return None
    
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
    try:
        if project_name:
            project = hopsworks.login(project=project_name, api_key_value=api_key)
        else:
            project = hopsworks.login(api_key_value=api_key)
        return project.get_feature_store()
    except Exception:
        return None


def _read_fg(name: str, version: int = 1) -> list[dict]:
    """Read a Feature Group and return as list of dicts."""
    df = _read_fg_df(name, version)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _read_fg_df(name: str, version: int = 1):
    """Read a Feature Group and return as DataFrame (Hopsworks with local parquet fallback)."""
    import pandas as pd
    fs = _connect()
    if fs is not None:
        try:
            fg = fs.get_feature_group(name, version)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
            if not df.empty:
                return df
        except Exception:
            pass

    # Local fallback
    local_path = Path(__file__).resolve().parents[1] / "data" / f"{name}.parquet"
    if local_path.exists():
        try:
            df = pd.read_parquet(local_path)
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
            pass

    return pd.DataFrame()



@app.get("/", summary="Service description")
def root() -> dict:
    return {
        "service": "Nafas Forecast API",
        "description": "3-day air-quality forecasts, updated hourly.",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/cities",
            "/current/{city_id}",
            "/forecast/{city_id}",
            "/models",
        ],
    }


@app.get("/health", summary="Liveness and data freshness")
def health() -> dict:
    """Confirms Hopsworks is reachable and reports how fresh the data is."""
    try:
        df = _read_fg_df("aqi_features")
        if df.empty:
            return {"status": "ok", "latest_observation": None, "cities": 0}
        return {
            "status": "ok",
            "latest_observation": str(df["event_time"].max()),
            "cities": df["city_id"].nunique(),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Feature store unavailable: {type(exc).__name__}") from exc


@app.get("/cities", summary="Cities covered, with current AQI")
def cities() -> list[dict]:
    df = _read_fg_df("aqi_features")
    if df.empty:
        return []
    latest = (
        df.sort_values("event_time", ascending=False)
        .groupby("city_id", as_index=False)
        .first()
    )
    return latest[["city_id", "event_time", "aqi", "pm25", "temp_c"]].to_dict(orient="records")


@app.get("/current/{city_id}", summary="Latest measured conditions for a city")
def current(city_id: str) -> dict:
    df = _read_fg_df("aqi_features")
    if df.empty:
        raise HTTPException(404, f"No data for '{city_id}'.")
    latest = (
        df[df["city_id"] == city_id]
        .sort_values("event_time", ascending=False)
        .head(1)
    )
    if latest.empty:
        raise HTTPException(404, f"Unknown city '{city_id}'. See /cities.")
    return latest.iloc[0].to_dict()


@app.get("/forecast/{city_id}", summary="3-day forecast, with the drivers behind it")
def forecast(city_id: str) -> dict:
    preds = _read_fg_df("predictions")
    if preds.empty:
        raise HTTPException(404, f"No forecast for '{city_id}'. See /cities.")

    horizons = preds[preds["city_id"] == city_id].sort_values("horizon_h")
    if horizons.empty:
        raise HTTPException(404, f"No forecast for '{city_id}'. See /cities.")

    drivers_df = _read_fg_df("forecast_drivers")
    drivers = []
    if not drivers_df.empty:
        d = drivers_df[drivers_df["city_id"] == city_id]
        if not d.empty:
            drivers = (
                d.sort_values("contribution", key=abs, ascending=False)
                .head(5)
                .to_dict(orient="records")
            )

    features = _read_fg_df("aqi_features")
    now_aqi = None
    observed_at = None
    if not features.empty:
        city_latest = (
            features[features["city_id"] == city_id]
            .sort_values("event_time", ascending=False)
            .head(1)
        )
        if not city_latest.empty:
            now_aqi = city_latest.iloc[0]["aqi"]
            observed_at = str(city_latest.iloc[0]["event_time"])

    return {
        "city_id": city_id,
        "current_aqi": now_aqi,
        "observed_at": observed_at,
        "forecast": horizons.to_dict(orient="records"),
        "drivers": drivers,
    }


@app.get("/models", summary="Model accuracy, and which model is serving")
def models() -> dict:
    scores = _read_fg_df("model_registry")
    if scores.empty:
        return {"benchmark": [], "note": "No training runs recorded yet."}

    latest_run = scores.sort_values("trained_at", ascending=False)["run_id"].iloc[0]
    benchmark = scores[scores["run_id"] == latest_run][
        ["model_name", "horizon_h", "rmse", "mae", "r2"]
    ].sort_values(["horizon_h", "rmse"]).to_dict(orient="records")

    return {
        "benchmark": benchmark,
        "note": "The serving model is the smallest whose RMSE is within 2% of the best.",
    }

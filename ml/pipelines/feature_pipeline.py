"""Feature pipeline — the hourly refresh."""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from ml.clients.openmeteo import OpenMeteoClient
from ml.common.cities import CITIES
from ml.features.engineering import build_features
from ml.storage.hopsworks_store import get_feature_store, upsert_features

DEFAULT_WINDOW_DAYS = 10


def run_feature_pipeline(window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=window_days)
    client = OpenMeteoClient()


    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for city in CITIES:
        try:
            raw = client.get_history(city, start.isoformat(), end.isoformat())
            frames.append(build_features(raw).rename(columns={"time": "event_time"}))
        except Exception as exc:  # noqa: BLE001 — report and carry on
            print(f"    {city.name}: FAILED — {type(exc).__name__}: {exc}")
            failures.append(city.id)

    if not frames:
        raise RuntimeError(
            f"Feature pipeline failed for all {len(CITIES)} cities: {failures}"
        )

    window = pd.concat(frames, ignore_index=True)

    fs = get_feature_store()
    n = upsert_features(window, fs=fs)

    if failures:
        print(f"\n  WARNING: skipped {len(failures)} city/cities: {', '.join(failures)}")

    newest = window.groupby("city_id")["event_time"].max()
    print(f"\n  Feature pipeline: upserted {n} rows ({start} -> {end}).")
    print("  Newest hour per city now in the store:")
    for city_id, ts in newest.items():
        print(f"    {city_id:<11} {ts}")
    print()
    return window


if __name__ == "__main__":
    run_feature_pipeline()

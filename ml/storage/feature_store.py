"""Feature store access layer — backed by Hopsworks with local parquet fallback."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ml.common.config import PROJECT_ROOT
from ml.storage.hopsworks_store import (
    get_feature_store,
    _get_or_create_features_fg,
    FEATURES_FG_NAME,
    FEATURES_FG_VERSION,
    DATA_DIR,
)


def _read_raw_features_df(fs=None) -> pd.DataFrame:
    df = pd.DataFrame()
    fs = fs or get_feature_store()
    if fs is not None:
        try:
            fg = fs.get_feature_group(FEATURES_FG_NAME, FEATURES_FG_VERSION)
            df = fg.read()
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        local_path = DATA_DIR / f"{FEATURES_FG_NAME}.parquet"
        if local_path.exists():
            df = pd.read_parquet(local_path)
            df.columns = [c.lower() for c in df.columns]

    return df


def read_features(
    city_ids: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    stride: int = 1,
    fs=None,
) -> pd.DataFrame:
    """OFFLINE retrieval: historical feature rows, optionally filtered."""
    df = _read_raw_features_df(fs)
    if df.empty:
        return df

    # Normalise column names to lowercase.
    df.columns = [c.lower() for c in df.columns]

    if city_ids:
        df = df[df["city_id"].isin(city_ids)]
    if start:
        df = df[df["event_time"] >= pd.Timestamp(start, tz="UTC" if df["event_time"].dt.tz else None)]
    if end:
        df = df[df["event_time"] <= pd.Timestamp(end, tz="UTC" if df["event_time"].dt.tz else None)]

    df = df.sort_values(["city_id", "event_time"]).reset_index(drop=True)

    if stride > 1:
        df = (
            df.groupby("city_id", group_keys=False)
            .apply(lambda g: g.iloc[::stride])
            .reset_index(drop=True)
        )

    return df


def get_latest_features(
    city_ids: Sequence[str] | None = None,
    fs=None,
) -> pd.DataFrame:
    """ONLINE retrieval: the latest feature row per city (for live inference)."""
    df = _read_raw_features_df(fs)
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]

    if city_ids:
        df = df[df["city_id"].isin(city_ids)]

    latest = (
        df.sort_values("event_time", ascending=False)
        .groupby("city_id", as_index=False)
        .first()
        .sort_values("city_id")
        .reset_index(drop=True)
    )
    return latest


def refresh_latest_view(fs=None) -> None:
    """No-op: Hopsworks / local handles online serving natively."""
    pass


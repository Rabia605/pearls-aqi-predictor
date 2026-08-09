"""Central configuration.

Every secret (API key, connection string) is read here from the `.env` file and
nowhere else. That keeps secrets out of the code and gives us one place to look
when something is misconfigured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# `.env` lives at the project root, which is two levels up from this file
# (ml/common/config.py -> ml/common -> ml -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Typed view of our environment. Frozen so it can't be mutated by accident."""

    # Data sources — required
    aqicn_token: str
    openweather_api_key: str

    # Feature Store & Model Registry — Hopsworks
    hopsworks_api_key: str


def _require(name: str) -> str:
    """Fetch an env var, failing loudly with a helpful message if it's missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def get_settings() -> Settings:
    return Settings(
        aqicn_token=_require("AQICN_TOKEN"),
        openweather_api_key=_require("OPENWEATHER_API_KEY"),
        hopsworks_api_key=_require("HOPSWORKS_API_KEY"),
    )

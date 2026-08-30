"""Hazardous-AQI alerts (dashboard-only)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from ml.common.aqi import aqi_category
from ml.common.cities import get_city
from ml.storage.hopsworks_store import (
    get_feature_store,
    read_predictions,
    write_alerts,
)


LEVELS: list[tuple[int, str, str, str]] = [
    (300, "emergency", "Everyone",
     "Avoid all outdoor activity. Keep windows shut and run an air purifier."),
    (200, "severe", "Everyone",
     "Avoid outdoor exertion. Sensitive groups should stay indoors."),
    (150, "high", "Everyone, especially sensitive groups",
     "Limit prolonged outdoor exertion. Wear an N95 mask outside."),
    (100, "moderate", "Sensitive groups (asthma, children, elderly, heart/lung conditions)",
     "Reduce prolonged outdoor exertion. Keep reliever medication handy."),
]


def _level_for(aqi: float) -> tuple[str, str, str] | None:
    for threshold, severity, who, advice in LEVELS:
        if aqi > threshold:
            return severity, who, advice
    return None


def run_alerts() -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fs = get_feature_store()
    preds = read_predictions(fs=fs)

    if preds.empty:
        print("\n  No predictions found — run the inference pipeline first.\n")
        return preds


    preds.columns = [c.lower() for c in preds.columns]

    records: list[dict] = []
    for city_id, group in preds.groupby("city_id"):
        crossing = group[group["predicted_aqi"] > LEVELS[-1][0]]
        if crossing.empty:
            continue


        worst = crossing.loc[crossing["predicted_aqi"].idxmax()]
        first = crossing.loc[crossing["horizon_h"].idxmin()]
        level = _level_for(float(worst["predicted_aqi"]))
        if level is None:
            continue
        severity, who, advice = level

        records.append(
            {
                "city_id": city_id,
                "severity": severity,
                "peak_aqi": round(float(worst["predicted_aqi"]), 1),
                "category": aqi_category(float(worst["predicted_aqi"])),
                "peak_time": str(worst["forecast_time"]),
                "starts_in_h": int(first["horizon_h"]),
                "affects": who,
                "advice": advice,
                "created_at": str(datetime.now(timezone.utc)),
            }
        )

    alerts = pd.DataFrame(records)

    if not alerts.empty:
        write_alerts(alerts, fs=fs)

    print("\n  Hazardous-AQI alert check")
    print("  " + "=" * 62)
    if alerts.empty:
        print("  No city is forecast to exceed AQI 100 in the next 3 days.")
    else:
        for r in alerts.itertuples():
            city = get_city(str(r.city_id))
            print(f"\n  [{r.severity.upper()}] {city.name} — peak AQI {r.peak_aqi} ({r.category})")
            print(f"    starts in ~{r.starts_in_h}h")
            print(f"    affects: {r.affects}")
            print(f"    advice : {r.advice}")
    print("\n  " + "=" * 62)
    print(f"  Wrote {len(alerts)} alert(s) to Hopsworks '{alerts}'.\n")
    return alerts


if __name__ == "__main__":
    run_alerts()

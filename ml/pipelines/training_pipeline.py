"""Training pipeline — reads features, trains & benchmarks models, registers the best."""
from __future__ import annotations

import gzip
import pickle
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd


SELECTION_TOLERANCE = 0.02


TRAIN_WINDOW_DAYS = 550


def _artifact_size(model) -> int:
    """Stored size in bytes — what the hourly inference job has to download."""
    return len(gzip.compress(pickle.dumps(model), compresslevel=6))


from ml.storage.feature_store import read_features
from ml.storage.registry import register_run, save_active_model, save_artifact
from ml.storage.hopsworks_store import get_feature_store
from ml.training.dataset import (
    HORIZONS,
    add_targets,
    build_design_matrix,
    feature_columns,
    time_split_mask,
)
from ml.training.models import evaluate, make_models


def _print_importances(model, feature_names: list[str], top: int = 8) -> None:
    """Show what a Random Forest leaned on most (native feature importances)."""
    rf = model.named_steps.get("model")
    if not hasattr(rf, "feature_importances_"):
        return
    ranked = sorted(
        zip(feature_names, rf.feature_importances_), key=lambda t: t[1], reverse=True
    )
    print("\n  Top drivers (Random Forest, +24h):")
    for name, imp in ranked[:top]:
        bar = "#" * int(imp * 60)
        print(f"    {name:<22} {imp:6.3f}  {bar}")


def run_training(test_frac: float = 0.2) -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n  Loading features from the feature store ...")
    since = (datetime.now(timezone.utc) - timedelta(days=TRAIN_WINDOW_DAYS)).isoformat()
    features = read_features(start=since)
    data = add_targets(features)
    feat_cols = feature_columns(data)
    X_all = build_design_matrix(data, feat_cols)
    train_mask, test_mask = time_split_mask(data, test_frac)

    split_time = data.loc[test_mask, "event_time"].min()
    print(
        f"  {len(data)} rows, {X_all.shape[1]} model inputs. "
        f"Test set = everything from {split_time.date()} onward.\n"
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trained_at = datetime.now(timezone.utc)

    rows: list[dict] = []
    registry_records: list[dict] = []

    fs = get_feature_store()

    for h in HORIZONS:
        target = data[f"target_aqi_{h}h"]
        valid = target.notna()
        tr, te = train_mask & valid, test_mask & valid
        X_tr, y_tr = X_all[tr], target[tr]
        X_te, y_te = X_all[te], target[te]


        scored = {"persistence": evaluate(y_te, data.loc[te, "aqi"])}


        fitted = {}
        for name, model in make_models().items():
            model.fit(X_tr, y_tr)
            scored[name] = evaluate(y_te, model.predict(X_te))
            fitted[name] = model


        best_rmse = min(scored[n]["rmse"] for n in fitted)
        contenders = {
            n: m
            for n, m in fitted.items()
            if scored[n]["rmse"] <= best_rmse * (1 + SELECTION_TOLERANCE)
        }
        best_name = min(contenders, key=lambda n: _artifact_size(contenders[n]))
        if scored[best_name]["rmse"] > best_rmse:
            print(
                f"  +{h}h: serving {best_name} "
                f"(RMSE {scored[best_name]['rmse']:.2f} vs best {best_rmse:.2f}) "
                f"— within {SELECTION_TOLERANCE:.0%}, and far cheaper to serve."
            )
        artifact = save_artifact(fitted[best_name], f"aqi_{h}h_{best_name}")

        save_active_model(fitted[best_name], h, best_name, scored[best_name])

        for name, m in scored.items():
            rows.append({"horizon_h": h, "model": name, **m})
            registry_records.append(
                {
                    "run_id": run_id,
                    "trained_at": trained_at,
                    "horizon_h": h,
                    "model_name": name,
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "is_best": name == best_name,
                    "artifact_path": str(artifact) if name == best_name else None,
                }
            )

        if h == 24 and best_name == "random_forest":
            _print_importances(fitted[best_name], list(X_all.columns))


    results = pd.DataFrame(rows)
    print("\n  Model comparison (lower RMSE/MAE better, higher R² better):")
    for h in HORIZONS:
        print(f"\n  --- +{h}h forecast ---")
        block = results[results.horizon_h == h].drop(columns="horizon_h")
        print(block.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    n = register_run(registry_records, fs=fs)
    print(f"\n  Registered {n} scorecards to Hopsworks. Best models saved to models/.\n")
    return results


if __name__ == "__main__":
    run_training()

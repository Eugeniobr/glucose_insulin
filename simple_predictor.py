import json
from pathlib import Path

import numpy as np

from predictor import predict_ridge_model, train_ridge_model


SIMPLE_FEATURE_NAMES = (
    "intercept",
    "current_glucose",
    "delta_15",
    "delta_30",
    "delta_60",
    "delta_120",
    "slope_30",
    "slope_60",
    "mean_30",
    "mean_60",
    "std_30",
    "std_60",
    "range_60",
    "carbs_last_30",
    "carbs_last_60",
    "carbs_last_120",
    "rapid_last_30",
    "rapid_last_60",
    "rapid_last_120",
    "future_carbs_total",
    "future_rapid_total",
    "minutes_since_meal",
    "minutes_since_rapid",
    "hour_sin",
    "hour_cos",
)


def _feature_vector(sample: dict) -> np.ndarray:
    return np.array([float(sample.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES], dtype=float)


def build_simple_feature_payload(
    observed_timestamps,
    observed_glucose,
    future_meals,
    future_insulin,
    horizon_minutes: int,
):
    observed_glucose = np.asarray(observed_glucose, dtype=float)
    current_glucose = float(observed_glucose[-1])

    def _history_value(minutes_back: float) -> float:
        if len(observed_glucose) < 2:
            return current_glucose
        target_time = observed_timestamps[-1].timestamp() - minutes_back * 60.0
        series_times = np.array([timestamp.timestamp() for timestamp in observed_timestamps], dtype=float)
        return float(np.interp(target_time, series_times, observed_glucose))

    prev_15 = _history_value(15.0)
    prev_30 = _history_value(30.0)
    prev_60 = _history_value(60.0)
    prev_120 = _history_value(120.0)
    now_time = observed_timestamps[-1].timestamp()
    series_times = np.array([timestamp.timestamp() for timestamp in observed_timestamps], dtype=float)

    def _window_stats(window_minutes: float):
        mask = series_times >= now_time - window_minutes * 60.0
        values = observed_glucose[mask]
        if len(values) == 0:
            values = np.array([current_glucose], dtype=float)
        return float(np.mean(values)), float(np.std(values)), float(np.max(values) - np.min(values))

    mean_30, std_30, _ = _window_stats(30.0)
    mean_60, std_60, range_60 = _window_stats(60.0)

    def _sum_recent(items, key, cutoff):
        return float(sum(item.get(key, 0.0) for item in items if 0.0 <= float(item.get("minutes", -1.0)) <= cutoff))

    future_carbs_total = float(sum(item.get("carbs_g", 0.0) for item in future_meals if 0.0 <= float(item.get("minutes", -1.0)) <= horizon_minutes))
    future_rapid_total = float(
        sum(item.get("units", 0.0) for item in future_insulin if item.get("insulin_type") == "rapida" and 0.0 <= float(item.get("minutes", -1.0)) <= horizon_minutes)
    )
    minutes_since_meal = -1.0
    minutes_since_rapid = -1.0
    hour = observed_timestamps[-1].hour + observed_timestamps[-1].minute / 60.0
    angle = 2.0 * np.pi * hour / 24.0
    return {
        "intercept": 1.0,
        "current_glucose": current_glucose,
        "delta_15": current_glucose - prev_15,
        "delta_30": current_glucose - prev_30,
        "delta_60": current_glucose - prev_60,
        "delta_120": current_glucose - prev_120,
        "slope_30": float((current_glucose - prev_30) / 30.0),
        "slope_60": float((current_glucose - prev_60) / 60.0),
        "mean_30": mean_30,
        "mean_60": mean_60,
        "std_30": std_30,
        "std_60": std_60,
        "range_60": range_60,
        "carbs_last_30": 0.0,
        "carbs_last_60": 0.0,
        "carbs_last_120": 0.0,
        "rapid_last_30": 0.0,
        "rapid_last_60": 0.0,
        "rapid_last_120": 0.0,
        "future_carbs_total": future_carbs_total,
        "future_rapid_total": future_rapid_total,
        "minutes_since_meal": minutes_since_meal,
        "minutes_since_rapid": minutes_since_rapid,
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
    }


def train_simple_forecast_model(samples: list[dict], model_path: Path, min_points: int = 96, l2: float = 6.0) -> dict:
    payload = {
        "feature_names": list(SIMPLE_FEATURE_NAMES),
        "horizons": {},
        "min_points": int(min_points),
        "l2": float(l2),
    }
    if not samples:
        model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    samples = sorted(samples, key=lambda item: item["timestamp"])
    for horizon in (15, 30, 60, 120):
        rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
        if len(rows) < min_points:
            payload["horizons"][str(horizon)] = {"status": "insufficient_data", "points_used": len(rows)}
            continue

        split_index = max(int(len(rows) * 0.8), min_points)
        split_index = min(split_index, len(rows) - 1)
        train_rows = rows[:split_index]
        valid_rows = rows[split_index:]
        train_features = np.asarray([_feature_vector(row) for row in train_rows], dtype=float)
        train_targets = np.asarray([float(row["target_delta"]) for row in train_rows], dtype=float)
        model = train_ridge_model(train_features, train_targets, l2)

        valid_features = np.asarray([_feature_vector(row) for row in valid_rows], dtype=float)
        valid_targets = np.asarray([float(row["target_delta"]) for row in valid_rows], dtype=float)
        valid_predictions = np.asarray([predict_ridge_model(model, vector) for vector in valid_features], dtype=float)
        valid_residuals = valid_predictions - valid_targets
        payload["horizons"][str(horizon)] = {
            "status": "trained",
            "points_used": int(len(rows)),
            "train_points": int(len(train_rows)),
            "valid_points": int(len(valid_rows)),
            "feature_means": model["feature_means"].tolist(),
            "feature_scales": model["feature_scales"].tolist(),
            "weights": model["weights"].tolist(),
            "train_target_rmse_mg_dl": float(model["rmse"]),
            "train_target_mae_mg_dl": float(model["mae"]),
            "valid_target_rmse_mg_dl": float(np.sqrt(np.mean(valid_residuals**2))),
            "valid_target_mae_mg_dl": float(np.mean(np.abs(valid_residuals))),
        }

    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_simple_forecast_model(config) -> dict:
    if not getattr(config, "simple_forecast_model_enabled", False):
        return {}
    path = getattr(config, "simple_forecast_model_path", None)
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def predict_with_simple_model(model_payload: dict, feature_payload: dict, horizon_minutes: int):
    horizon_payload = model_payload.get("horizons", {}).get(str(int(horizon_minutes)), {})
    if horizon_payload.get("status") != "trained":
        return {"status": horizon_payload.get("status", "unavailable"), "applied": False}
    vector = _feature_vector(feature_payload)
    predicted_delta = predict_ridge_model(horizon_payload, vector)
    current_glucose = float(feature_payload.get("current_glucose", 0.0))
    prediction = current_glucose + predicted_delta
    return {
        "status": "trained",
        "prediction_mg_dl": float(prediction),
        "predicted_delta_mg_dl": float(predicted_delta),
        "applied": True,
    }

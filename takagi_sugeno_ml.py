import json
from datetime import datetime, timedelta

import numpy as np

from predictor import predict_ridge_model
from takagi_sugeno_dynamic import build_dynamic_feature_payload, predict_dynamic_ts_delta


TS_ML_HORIZONS = (15, 30, 60)

RESIDUAL_FEATURE_NAMES = (
    "intercept",
    "current_glucose",
    "delta_15",
    "delta_30",
    "delta_60",
    "net_input_60",
    "mean_30",
    "std_30",
    "carbs_last_60",
    "rapid_last_60",
    "basal_last_60",
    "rapid_iob_60",
    "rapid_action_60",
    "basal_iob_12h",
    "basal_action_12h",
    "future_carbs_15",
    "future_rapid_15",
    "future_basal_15",
    "future_rapid_action_15",
    "future_basal_action_15",
    "minutes_since_meal",
    "minutes_since_rapid",
    "minutes_since_basal",
    "ts_delta",
    "regime_alta_abrupta",
    "regime_queda_abrupta",
    "regime_estabilidade",
)


def load_takagi_sugeno_ml_model(config) -> dict:
    benchmark_path = getattr(config, "takagi_sugeno_ml_benchmark_path", None)
    ts_model_path = getattr(config, "takagi_sugeno_dynamic_model_path", None)
    if not getattr(config, "takagi_sugeno_ml_enabled", False):
        return {}
    if not benchmark_path or not benchmark_path.exists() or not ts_model_path or not ts_model_path.exists():
        return {}
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        benchmark_payload = json.load(handle)
    with open(ts_model_path, "r", encoding="utf-8") as handle:
        ts_model_payload = json.load(handle)
    if not isinstance(benchmark_payload, dict) or not isinstance(ts_model_payload, dict):
        return {}
    if benchmark_payload.get("ts_model_status") != "trained" or ts_model_payload.get("status") != "trained":
        return {}
    residual_model = benchmark_payload.get("residual_model", {})
    if not residual_model.get("weights"):
        return {}
    return {
        "ts_model": ts_model_payload,
        "residual_model": residual_model,
        "benchmark": benchmark_payload,
    }


def _residual_feature_payload(feature_payload: dict, ts_delta: float) -> dict:
    regime = str(feature_payload.get("regime", "estabilidade"))
    return {
        "intercept": 1.0,
        "current_glucose": float(feature_payload.get("current_glucose", 0.0)),
        "delta_15": float(feature_payload.get("delta_15", 0.0)),
        "delta_30": float(feature_payload.get("delta_30", 0.0)),
        "delta_60": float(feature_payload.get("delta_60", 0.0)),
        "net_input_60": float(feature_payload.get("net_input_60", 0.0)),
        "mean_30": float(feature_payload.get("mean_30", 0.0)),
        "std_30": float(feature_payload.get("std_30", 0.0)),
        "carbs_last_60": float(feature_payload.get("carbs_last_60", 0.0)),
        "rapid_last_60": float(feature_payload.get("rapid_last_60", 0.0)),
        "basal_last_60": float(feature_payload.get("basal_last_60", 0.0)),
        "rapid_iob_60": float(feature_payload.get("rapid_iob_60", 0.0)),
        "rapid_action_60": float(feature_payload.get("rapid_action_60", 0.0)),
        "basal_iob_12h": float(feature_payload.get("basal_iob_12h", 0.0)),
        "basal_action_12h": float(feature_payload.get("basal_action_12h", 0.0)),
        "future_carbs_15": float(feature_payload.get("future_carbs_15", 0.0)),
        "future_rapid_15": float(feature_payload.get("future_rapid_15", 0.0)),
        "future_basal_15": float(feature_payload.get("future_basal_15", 0.0)),
        "future_rapid_action_15": float(feature_payload.get("future_rapid_action_15", 0.0)),
        "future_basal_action_15": float(feature_payload.get("future_basal_action_15", 0.0)),
        "minutes_since_meal": float(feature_payload.get("minutes_since_meal", -1.0)),
        "minutes_since_rapid": float(feature_payload.get("minutes_since_rapid", -1.0)),
        "minutes_since_basal": float(feature_payload.get("minutes_since_basal", -1.0)),
        "ts_delta": float(ts_delta),
        "regime_alta_abrupta": 1.0 if regime == "alta_abrupta" else 0.0,
        "regime_queda_abrupta": 1.0 if regime == "queda_abrupta" else 0.0,
        "regime_estabilidade": 1.0 if regime == "estabilidade" else 0.0,
    }


def _feature_vector(payload: dict) -> np.ndarray:
    return np.array([float(payload.get(name, 0.0)) for name in RESIDUAL_FEATURE_NAMES], dtype=float)


def _build_window_from_observations(timestamps, observed_glucose, meal_events, insulin_events, forecast_minutes: int = 120):
    if len(timestamps) < 2:
        return None
    end_time = timestamps[-1]
    grid_offsets = np.arange(-60, forecast_minutes + 1, 15, dtype=float)
    series_times = np.array([timestamp.timestamp() for timestamp in timestamps], dtype=float)
    observed = np.interp(
        np.array([(end_time + timedelta(minutes=float(offset))).timestamp() for offset in grid_offsets], dtype=float),
        series_times,
        np.asarray(observed_glucose, dtype=float),
    ).astype(float)
    start_time = end_time - timedelta(minutes=60)
    events = []
    for meal in meal_events:
        delta_minutes = (meal.timestamp - start_time).total_seconds() / 60.0
        if -1e-6 <= delta_minutes <= forecast_minutes + 60 + 1e-6:
            events.append({"minutes": float(delta_minutes), "carbs_g": float(meal.carbs_g), "rapid_units": 0.0})
    for insulin in insulin_events:
        delta_minutes = (insulin.timestamp - start_time).total_seconds() / 60.0
        if -1e-6 <= delta_minutes <= forecast_minutes + 60 + 1e-6:
            rapid_units = float(insulin.units) if insulin.insulin_type == "rapida" else 0.0
            basal_units = float(insulin.units) if insulin.insulin_type == "basal" else 0.0
            events.append({"minutes": float(delta_minutes), "carbs_g": 0.0, "rapid_units": rapid_units, "basal_units": basal_units})
    return {
        "start_timestamp": start_time.isoformat(),
        "grid_minutes": (grid_offsets + 60.0).tolist(),
        "observed_glucose": observed.tolist(),
        "events": events,
    }


def rollout_takagi_sugeno_ml_window(model_payload: dict, window: dict, step_minutes: int = 15) -> dict:
    observed = np.asarray(window["observed_glucose"], dtype=float)
    grid_minutes = np.asarray(window["grid_minutes"], dtype=float)
    event_items = list(window.get("events", []))
    ts_model = {"global": model_payload["ts_model"].get("global"), "by_regime": model_payload["ts_model"].get("by_regime", {})}
    residual_model = model_payload["residual_model"]
    predicted_ts = observed[:5].tolist()
    predicted_ml = observed[:5].tolist()
    if len(observed) < 5:
        return {
            "takagi_sugeno_glucose": observed.tolist(),
            "takagi_sugeno_ml_glucose": observed.tolist(),
        }
    for index in range(4, len(observed) - 1):
        current_minute = float(grid_minutes[index])
        future_events = [item for item in event_items if 0.0 < float(item["minutes"]) - current_minute <= step_minutes]
        previous_events_30 = [item for item in event_items if 0.0 <= current_minute - float(item["minutes"]) <= 30.0]
        previous_events_60 = [item for item in event_items if 0.0 <= current_minute - float(item["minutes"]) <= 60.0]

        for predicted_series, collector in ((predicted_ts, "ts"), (predicted_ml, "ml")):
            current = float(predicted_series[-1])
            prev_15 = float(predicted_series[-2])
            prev_30 = float(predicted_series[-3])
            prev_60 = float(predicted_series[-5])
            recent = np.asarray(predicted_series[-3:], dtype=float)
            feature_payload = build_dynamic_feature_payload(
                current=current,
                prev_15=prev_15,
                prev_30=prev_30,
                prev_60=prev_60,
                recent_values=recent,
                previous_events_30=previous_events_30,
                previous_events_60=previous_events_60,
                future_events=future_events,
                current_minute=current_minute,
            )
            ts_delta = predict_dynamic_ts_delta(ts_model, feature_payload)
            if collector == "ts":
                predicted_series.append(current + ts_delta)
            else:
                residual = predict_ridge_model(
                    residual_model,
                    _feature_vector(_residual_feature_payload(feature_payload, ts_delta)),
                )
                predicted_series.append(current + ts_delta + residual)
    return {
        "takagi_sugeno_glucose": predicted_ts,
        "takagi_sugeno_ml_glucose": predicted_ml,
        "grid_minutes": window["grid_minutes"],
    }


def predict_with_takagi_sugeno_ml(model_payload: dict, timestamps, observed_glucose, meal_events, insulin_events):
    if not model_payload:
        return {}
    window = _build_window_from_observations(timestamps, observed_glucose, meal_events, insulin_events, forecast_minutes=120)
    if not window:
        return {}
    rollout = rollout_takagi_sugeno_ml_window(model_payload, window)
    results = {}
    grid = np.asarray(rollout["grid_minutes"], dtype=float)
    for horizon in TS_ML_HORIZONS:
        target_minute = 60.0 + float(horizon)
        results[int(horizon)] = {
            "takagi_sugeno_predicted_glucose_mg_dl": float(np.interp(target_minute, grid, np.asarray(rollout["takagi_sugeno_glucose"], dtype=float))),
            "takagi_sugeno_ml_predicted_glucose_mg_dl": float(np.interp(target_minute, grid, np.asarray(rollout["takagi_sugeno_ml_glucose"], dtype=float))),
        }
    results["rollout"] = {
        "start_timestamp": window.get("start_timestamp"),
        "grid_minutes": rollout.get("grid_minutes", []),
        "observed_glucose": window.get("observed_glucose", []),
        "takagi_sugeno_glucose": rollout.get("takagi_sugeno_glucose", []),
        "takagi_sugeno_ml_glucose": rollout.get("takagi_sugeno_ml_glucose", []),
    }
    return results

import json
from datetime import datetime, timedelta

import numpy as np


FEATURE_NAMES = (
    "intercept",
    "phys_pred",
    "current_glucose",
    "delta_phys_current",
    "horizon_minutes",
    "subcutaneous_insulin",
    "plasma_insulin",
    "remote_insulin",
    "h3",
    "future_carbs_total",
    "future_insulin_total",
    "future_rapid_units",
    "future_basal_units",
    "future_regular_units",
    "future_unknown_units",
    "meal_count",
    "insulin_count",
    "minutes_to_first_meal",
    "minutes_to_first_insulin",
    "recent_slope_mgdl_per_min",
    "recent_variability_mgdl",
    "minutes_since_meal",
    "minutes_since_insulin",
    "regime_estavel",
    "regime_pos_prandial",
    "regime_correcao",
    "regime_jejum",
    "regime_noturno",
)


def _safe_minutes_to_first(items, key):
    if not items:
        return -1.0
    return float(min(item.get(key, -1.0) for item in items))


def _regime_flags(regime_payload: dict | None) -> dict:
    default = {
        "regime_estavel": 1.0,
        "regime_pos_prandial": 0.0,
        "regime_correcao": 0.0,
        "regime_jejum": 0.0,
        "regime_noturno": 0.0,
    }
    if not regime_payload:
        return default
    name = str(regime_payload.get("name", "estavel"))
    flags = {key: 0.0 for key in default}
    key = f"regime_{name}"
    if key not in flags:
        key = "regime_estavel"
    flags[key] = 1.0
    return flags


def _optional_float(value, default=-1.0):
    return float(default if value is None else value)


def _feature_payload(anchor_state, future_meals, future_insulin, physiological_prediction, horizon_minutes, regime_payload=None):
    anchor = np.asarray(anchor_state, dtype=float)
    current_glucose = float(anchor[3] / 100.0)
    future_carbs_total = float(sum(item.get("carbs_g", 0.0) for item in future_meals))
    future_insulin_total = float(sum(item.get("units", 0.0) for item in future_insulin))
    type_totals = {"rapida": 0.0, "basal": 0.0, "regular": 0.0, "desconhecido": 0.0}
    for item in future_insulin:
        insulin_type = item.get("insulin_type", "desconhecido")
        if insulin_type not in type_totals:
            insulin_type = "desconhecido"
        type_totals[insulin_type] += float(item.get("units", 0.0))

    regime_flags = _regime_flags(regime_payload)
    payload = {
        "intercept": 1.0,
        "phys_pred": float(physiological_prediction),
        "current_glucose": current_glucose,
        "delta_phys_current": float(physiological_prediction - current_glucose),
        "horizon_minutes": float(horizon_minutes),
        "subcutaneous_insulin": float(anchor[0] / 1000.0),
        "plasma_insulin": float(anchor[1] / 100.0),
        "remote_insulin": float(anchor[2] / 100.0),
        "h3": float(anchor[6] / 100.0),
        "future_carbs_total": future_carbs_total,
        "future_insulin_total": future_insulin_total,
        "future_rapid_units": type_totals["rapida"],
        "future_basal_units": type_totals["basal"],
        "future_regular_units": type_totals["regular"],
        "future_unknown_units": type_totals["desconhecido"],
        "meal_count": float(len(future_meals)),
        "insulin_count": float(len(future_insulin)),
        "minutes_to_first_meal": _safe_minutes_to_first(future_meals, "minutes"),
        "minutes_to_first_insulin": _safe_minutes_to_first(future_insulin, "minutes"),
        "recent_slope_mgdl_per_min": float((regime_payload or {}).get("recent_slope_mgdl_per_min", 0.0)),
        "recent_variability_mgdl": float((regime_payload or {}).get("recent_variability_mgdl", 0.0)),
        "minutes_since_meal": _optional_float((regime_payload or {}).get("minutes_since_meal"), -1.0),
        "minutes_since_insulin": _optional_float((regime_payload or {}).get("minutes_since_insulin"), -1.0),
    }
    payload.update(regime_flags)
    return payload


def _vectorize_features(payload):
    return np.array([float(payload[name]) for name in FEATURE_NAMES], dtype=float)


def _history_cutoff(generated_at: str, lookback_days: int) -> datetime:
    return datetime.fromisoformat(generated_at) - timedelta(days=lookback_days)


def train_ridge_model(features: np.ndarray, targets: np.ndarray, l2: float):
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = (features - means) / scales
    identity = np.eye(normalized.shape[1], dtype=float)
    identity[0, 0] = 0.0
    system = normalized.T @ normalized + float(l2) * identity
    rhs = normalized.T @ targets
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ rhs
    predictions = normalized @ weights
    residuals = predictions - targets
    return {
        "feature_means": means,
        "feature_scales": scales,
        "weights": weights,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def predict_ridge_model(model_payload: dict, feature_vector: np.ndarray) -> float:
    means = np.asarray(model_payload["feature_means"], dtype=float)
    scales = np.asarray(model_payload["feature_scales"], dtype=float)
    weights = np.asarray(model_payload["weights"], dtype=float)
    normalized = (np.asarray(feature_vector, dtype=float) - means) / scales
    return float(normalized @ weights)


def train_forecast_model(config, history: list[dict], generated_at: str) -> dict:
    cutoff = _history_cutoff(generated_at, config.forecast_model_lookback_days)
    payload = {
        "updated_at": generated_at,
        "enabled": bool(config.forecast_model_enabled),
        "min_points": int(config.forecast_model_min_points),
        "lookback_days": int(config.forecast_model_lookback_days),
        "l2": float(config.forecast_model_l2),
        "horizons": {},
    }
    if not config.forecast_model_enabled:
        with open(config.forecast_model_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return payload

    usable = []
    for item in history:
        if not item.get("is_resolved"):
            continue
        if "anchor_state" not in item or "future_meals" not in item or "future_insulin" not in item:
            continue
        if datetime.fromisoformat(item["generated_at"]) < cutoff:
            continue
        usable.append(item)

    for horizon in (15, 30, 60, 120):
        rows = [item for item in usable if int(item.get("horizon_minutes", -1)) == horizon]
        if len(rows) < config.forecast_model_min_points:
            payload["horizons"][str(horizon)] = {
                "status": "insufficient_data",
                "points_used": len(rows),
            }
            continue

        feature_rows = []
        targets = []
        for item in rows:
            physiological_prediction = float(item.get("physiological_predicted_glucose_mg_dl", item["predicted_glucose_mg_dl"]))
            feature_payload = _feature_payload(
                anchor_state=item["anchor_state"],
                future_meals=item.get("future_meals", []),
                future_insulin=item.get("future_insulin", []),
                physiological_prediction=physiological_prediction,
                horizon_minutes=horizon,
                regime_payload=item.get("regime"),
            )
            feature_rows.append(_vectorize_features(feature_payload))
            targets.append(float(item["observed_glucose_mg_dl"] - physiological_prediction))

        features = np.asarray(feature_rows, dtype=float)
        target_vector = np.asarray(targets, dtype=float)
        model = train_ridge_model(features, target_vector, config.forecast_model_l2)
        payload["horizons"][str(horizon)] = {
            "status": "trained",
            "points_used": int(len(rows)),
            "feature_names": list(FEATURE_NAMES),
            "feature_means": model["feature_means"].tolist(),
            "feature_scales": model["feature_scales"].tolist(),
            "weights": model["weights"].tolist(),
            "target_rmse_mg_dl": model["rmse"],
            "target_mae_mg_dl": model["mae"],
        }

    with open(config.forecast_model_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def predict_with_forecast_model(model_payload: dict, anchor_state, future_meals, future_insulin, physiological_prediction, horizon_minutes: int, regime_payload=None):
    horizon_payload = model_payload.get("horizons", {}).get(str(int(horizon_minutes)), {})
    if horizon_payload.get("status") != "trained":
        return {
            "status": horizon_payload.get("status", "unavailable"),
            "prediction_mg_dl": float(physiological_prediction),
            "physiological_prediction_mg_dl": float(physiological_prediction),
            "predicted_residual_mg_dl": 0.0,
            "applied": False,
        }

    feature_payload = _feature_payload(
        anchor_state=anchor_state,
        future_meals=future_meals,
        future_insulin=future_insulin,
        physiological_prediction=physiological_prediction,
        horizon_minutes=horizon_minutes,
        regime_payload=regime_payload,
    )
    vector = _vectorize_features(feature_payload)
    residual = predict_ridge_model(horizon_payload, vector)
    prediction = float(physiological_prediction + residual)
    return {
        "status": "trained",
        "prediction_mg_dl": prediction,
        "physiological_prediction_mg_dl": float(physiological_prediction),
        "predicted_residual_mg_dl": residual,
        "applied": True,
    }


def load_external_forecast_model(config) -> dict:
    if not getattr(config, "external_forecast_model_enabled", False):
        return {}
    path = getattr(config, "external_forecast_model_path", None)
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _external_feature_vector(anchor_state, future_meals, future_insulin, horizon_minutes: int) -> np.ndarray:
    anchor = np.asarray(anchor_state, dtype=float)
    current_glucose = float(anchor[3] / 100.0)
    future_carbs_total = float(sum(item.get("carbs_g", 0.0) for item in future_meals))
    future_rapid_total = float(sum(item.get("units", 0.0) for item in future_insulin if item.get("insulin_type") == "rapida"))
    future_rapid_any = float(sum(item.get("units", 0.0) for item in future_insulin))
    minutes_since_event = -1.0
    if future_meals or future_insulin:
        minutes_since_event = 0.0
    hour_angle = 0.0
    return np.array(
        [
            1.0,
            current_glucose,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            future_carbs_total,
            future_rapid_total,
            minutes_since_event,
            np.sin(hour_angle),
            np.cos(hour_angle),
        ],
        dtype=float,
    )


def predict_with_external_forecast_model(model_payload: dict, anchor_state, future_meals, future_insulin, horizon_minutes: int):
    horizon_payload = model_payload.get("horizons", {}).get(str(int(horizon_minutes)), {})
    if horizon_payload.get("status") != "trained":
        return {"status": horizon_payload.get("status", "unavailable"), "applied": False}
    vector = _external_feature_vector(anchor_state, future_meals, future_insulin, horizon_minutes)
    prediction = predict_ridge_model(horizon_payload, vector)
    return {
        "status": "trained",
        "prediction_mg_dl": float(prediction),
        "applied": True,
    }


def blend_forecast_predictions(config, physiological_prediction: float, internal_prediction: dict, external_prediction: dict):
    blended = float(internal_prediction.get("prediction_mg_dl", physiological_prediction))
    applied_sources = []
    if internal_prediction.get("applied"):
        applied_sources.append("internal")
    if external_prediction.get("applied"):
        weight = float(getattr(config, "external_forecast_weight", 0.35))
        blended = (1.0 - weight) * blended + weight * float(external_prediction["prediction_mg_dl"])
        applied_sources.append("external")
    return {
        "prediction_mg_dl": float(blended),
        "sources": applied_sources,
        "external_prediction_mg_dl": float(external_prediction.get("prediction_mg_dl", physiological_prediction)),
        "internal_prediction_mg_dl": float(internal_prediction.get("prediction_mg_dl", physiological_prediction)),
        "physiological_prediction_mg_dl": float(physiological_prediction),
    }

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import least_squares

from forward import run_forward
from inputs import InsulinEvent, MealEvent
from model import UltradianParams


CALIBRATED_PARAM_NAMES = (
    "ka",
    "insulin_bioavailability",
    "meal_tau",
    "insulin_sensitivity_scale",
)

CALIBRATED_PARAM_BOUNDS = {
    "ka": (0.01, 0.08),
    "insulin_bioavailability": (0.5, 1.5),
    "meal_tau": (20.0, 90.0),
    "insulin_sensitivity_scale": (0.5, 1.8),
}


def load_calibrated_params(config) -> dict:
    if not config.calibrated_params_path.exists():
        return {}
    with open(config.calibrated_params_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("params", {}) if isinstance(payload, dict) else {}


def apply_calibrated_params(base_params: UltradianParams, overrides: dict) -> UltradianParams:
    valid = {name: overrides[name] for name in CALIBRATED_PARAM_NAMES if name in overrides}
    return replace(base_params, **valid)


def _vector_from_params(params: UltradianParams) -> np.ndarray:
    return np.array([float(getattr(params, name)) for name in CALIBRATED_PARAM_NAMES], dtype=float)


def _params_from_vector(base_params: UltradianParams, vector: np.ndarray) -> UltradianParams:
    updates = {name: float(vector[index]) for index, name in enumerate(CALIBRATED_PARAM_NAMES)}
    return replace(base_params, **updates)


def _history_for_calibration(history: list[dict], generated_at: str, lookback_days: int) -> list[dict]:
    cutoff = datetime.fromisoformat(generated_at) - timedelta(days=lookback_days)
    filtered = []
    for item in history:
        if not item.get("is_resolved"):
            continue
        if "anchor_state" not in item or "future_meals" not in item or "future_insulin" not in item:
            continue
        if datetime.fromisoformat(item["generated_at"]) < cutoff:
            continue
        filtered.append(item)
    return filtered


def _deserialize_meals(items: list[dict]) -> list[MealEvent]:
    return [
        MealEvent(timestamp=datetime.fromisoformat(item["timestamp"]), minutes=float(item["minutes"]), carbs_g=float(item["carbs_g"]))
        for item in items
    ]


def _deserialize_insulin(items: list[dict]) -> list[InsulinEvent]:
    return [
        InsulinEvent(
            timestamp=datetime.fromisoformat(item["timestamp"]),
            minutes=float(item["minutes"]),
            units=float(item["units"]),
            insulin_type=item.get("insulin_type", "desconhecido"),
        )
        for item in items
    ]


def _residuals_for_history(vector, base_params, history, config, reference_vector):
    params = _params_from_vector(base_params, vector)
    residuals = []
    for item in history:
        anchor_state = np.asarray(item["anchor_state"], dtype=float)
        horizon_minutes = float(item["horizon_minutes"])
        meals = _deserialize_meals(item.get("future_meals", []))
        insulin = _deserialize_insulin(item.get("future_insulin", []))
        result = run_forward(
            params=params,
            initial_state=anchor_state,
            start_time_minutes=0.0,
            end_time_minutes=horizon_minutes,
            meal_events=meals,
            insulin_events=insulin,
            exercise_events=[],
            disturbance=None,
            dt_minutes=config.forward_dt_minutes,
            extra_times=[horizon_minutes],
        )
        predicted = float(np.interp(horizon_minutes, result["t"], result["glucose_mg_dl"]))
        observed = float(item["observed_glucose_mg_dl"])
        residuals.append((predicted - observed) / max(observed, 80.0))

    regularization = 0.15 * (vector - reference_vector)
    return np.concatenate([np.asarray(residuals, dtype=float), regularization])


def calibrate_online_params(config, base_params: UltradianParams, history: list[dict], generated_at: str):
    usable_history = _history_for_calibration(history, generated_at, config.calibration_lookback_days)
    if len(usable_history) < config.calibration_min_points:
        return base_params, {
            "status": "insufficient_data",
            "points_used": len(usable_history),
            "params": {name: float(getattr(base_params, name)) for name in CALIBRATED_PARAM_NAMES},
        }

    x0 = _vector_from_params(base_params)
    lower = np.array([CALIBRATED_PARAM_BOUNDS[name][0] for name in CALIBRATED_PARAM_NAMES], dtype=float)
    upper = np.array([CALIBRATED_PARAM_BOUNDS[name][1] for name in CALIBRATED_PARAM_NAMES], dtype=float)
    result = least_squares(
        fun=_residuals_for_history,
        x0=x0,
        bounds=(lower, upper),
        args=(base_params, usable_history, config, x0),
        max_nfev=60,
    )
    optimized = result.x
    blended = x0 + float(config.calibration_blend) * (optimized - x0)
    blended = np.clip(blended, lower, upper)
    calibrated = _params_from_vector(base_params, blended)
    payload = {
        "updated_at": generated_at,
        "status": "calibrated",
        "points_used": len(usable_history),
        "cost": float(result.cost),
        "params": {name: float(getattr(calibrated, name)) for name in CALIBRATED_PARAM_NAMES},
    }
    with open(config.calibrated_params_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return calibrated, payload


def save_initial_calibration_snapshot(config, params: UltradianParams, generated_at: str):
    if config.calibrated_params_path.exists():
        return
    payload = {
        "updated_at": generated_at,
        "status": "seed",
        "points_used": 0,
        "params": {name: float(getattr(params, name)) for name in CALIBRATED_PARAM_NAMES},
    }
    with open(config.calibrated_params_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

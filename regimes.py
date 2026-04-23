from dataclasses import asdict
from datetime import datetime

import numpy as np


REGIME_NAMES = ("estavel", "pos_prandial", "correcao", "jejum", "noturno")


def _event_stats(events, now_minutes: float, quiet_window_minutes: float, amount_key: str):
    recent = [
        event
        for event in events
        if 0.0 <= float(now_minutes - event.minutes) <= float(quiet_window_minutes)
    ]
    last_minutes = min((float(now_minutes - event.minutes) for event in recent), default=None)
    total = float(sum(getattr(event, amount_key) for event in recent))
    return recent, last_minutes, total


def detect_regime(config, observed_minutes, observed_glucose, timestamps, meal_events, insulin_events) -> dict:
    current_time_minutes = float(observed_minutes[-1])
    current_timestamp = timestamps[-1]
    recent_window = max(float(config.forecast_anchor_slope_window_minutes), 20.0)
    start_time = max(float(observed_minutes[0]), current_time_minutes - recent_window)
    mask = observed_minutes >= start_time
    trend_minutes = np.asarray(observed_minutes[mask], dtype=float)
    trend_glucose = np.asarray(observed_glucose[mask], dtype=float)
    if len(trend_minutes) >= 2:
        centered = trend_minutes - trend_minutes.mean()
        slope_mgdl_per_min = float(np.polyfit(centered, trend_glucose, 1)[0])
    else:
        slope_mgdl_per_min = 0.0
    variability = float(np.std(trend_glucose)) if len(trend_glucose) >= 2 else 0.0

    recent_meals, minutes_since_meal, recent_carbs = _event_stats(
        meal_events,
        current_time_minutes,
        float(config.regime_quiet_minutes),
        "carbs_g",
    )
    recent_insulin, minutes_since_insulin, recent_units = _event_stats(
        insulin_events,
        current_time_minutes,
        float(config.regime_quiet_minutes),
        "units",
    )
    rapid_recent_units = float(
        sum(event.units for event in recent_insulin if getattr(event, "insulin_type", "desconhecido") == "rapida")
    )

    hour = current_timestamp.hour + current_timestamp.minute / 60.0
    regime = "estavel"
    reason = "default"
    if recent_meals and minutes_since_meal is not None and minutes_since_meal <= float(config.regime_recent_meal_minutes):
        regime = "pos_prandial"
        reason = "meal_recent"
    elif (
        rapid_recent_units > 0.0
        and (minutes_since_insulin is not None and minutes_since_insulin <= float(config.regime_recent_insulin_minutes))
        and (minutes_since_meal is None or minutes_since_meal > 120.0)
    ):
        regime = "correcao"
        reason = "rapid_without_meal"
    elif 0.0 <= hour < 6.0 and (minutes_since_meal is None or minutes_since_meal > 240.0):
        regime = "noturno"
        reason = "overnight"
    elif (minutes_since_meal is None or minutes_since_meal > float(config.regime_quiet_minutes)) and abs(slope_mgdl_per_min) < 0.35:
        regime = "jejum"
        reason = "quiet_window"

    payload = {
        "name": regime,
        "reason": reason,
        "current_hour": float(hour),
        "current_glucose_mg_dl": float(observed_glucose[-1]),
        "recent_slope_mgdl_per_min": slope_mgdl_per_min,
        "recent_variability_mgdl": variability,
        "minutes_since_meal": float(minutes_since_meal) if minutes_since_meal is not None else None,
        "minutes_since_insulin": float(minutes_since_insulin) if minutes_since_insulin is not None else None,
        "recent_carbs_g": recent_carbs,
        "recent_insulin_units": recent_units,
        "recent_rapid_units": rapid_recent_units,
        "flags": {name: name == regime for name in REGIME_NAMES},
    }
    return payload


def regime_parameter_overrides(params, regime_payload: dict) -> dict:
    regime = regime_payload.get("name", "estavel")
    if regime == "pos_prandial":
        return {
            "meal_tau": float(params.meal_tau * 1.06),
            "meal_carb_scale": float(params.meal_carb_scale * 1.04),
            "insulin_sensitivity_scale": float(params.insulin_sensitivity_scale * 0.97),
        }
    if regime == "correcao":
        return {
            "insulin_sensitivity_scale": float(params.insulin_sensitivity_scale * 1.05),
            "insulin_rapid_bio_scale": float(params.insulin_rapid_bio_scale * 1.03),
            "insulin_rapid_tau": float(params.insulin_rapid_tau * 0.96),
        }
    if regime == "jejum":
        return {
            "meal_carb_scale": float(params.meal_carb_scale * 0.98),
            "insulin_sensitivity_scale": float(params.insulin_sensitivity_scale * 1.02),
        }
    if regime == "noturno":
        return {
            "insulin_sensitivity_scale": float(params.insulin_sensitivity_scale * 1.04),
            "meal_carb_scale": float(params.meal_carb_scale * 0.97),
        }
    return {}

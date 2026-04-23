import json
from datetime import timedelta
from pathlib import Path

import numpy as np


DYNAMIC_FEATURE_NAMES = (
    "intercept",
    "current_glucose",
    "delta_15",
    "delta_30",
    "delta_60",
    "mean_30",
    "std_30",
    "carbs_last_30",
    "carbs_last_60",
    "rapid_last_30",
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
    "hour_sin",
    "hour_cos",
)

PREMISE_FEATURES = ("current_glucose", "delta_15", "net_input_60")
REGIME_NAMES = ("alta_abrupta", "queda_abrupta", "estabilidade")


def _decay(minutes: float, tau: float) -> float:
    if minutes < 0.0:
        return 0.0
    return float(np.exp(-minutes / max(tau, 1e-6)))


def _gamma_action(minutes: float, tau: float) -> float:
    if minutes < 0.0:
        return 0.0
    scaled = minutes / max(tau, 1e-6)
    return float(scaled * np.exp(1.0 - scaled))


def _insulin_kinetics(events: list[dict], current_minute: float) -> dict:
    rapid_iob_60 = 0.0
    rapid_action_60 = 0.0
    basal_iob_12h = 0.0
    basal_action_12h = 0.0
    future_rapid_action_15 = 0.0
    future_basal_action_15 = 0.0
    for item in events:
        event_minute = float(item.get("minutes", 0.0))
        rapid_units = float(item.get("rapid_units", 0.0))
        basal_units = float(item.get("basal_units", 0.0))
        delta = current_minute - event_minute
        if delta >= 0.0:
            rapid_iob_60 += rapid_units * _decay(delta, 180.0)
            rapid_action_60 += rapid_units * _gamma_action(delta, 75.0)
            basal_iob_12h += basal_units * _decay(delta, 960.0)
            basal_action_12h += basal_units * _gamma_action(delta, 480.0)
        else:
            lead = -delta
            future_rapid_action_15 += rapid_units * _gamma_action(lead, 75.0)
            future_basal_action_15 += basal_units * _gamma_action(lead, 480.0)
    return {
        "rapid_iob_60": float(rapid_iob_60),
        "rapid_action_60": float(rapid_action_60),
        "basal_iob_12h": float(basal_iob_12h),
        "basal_action_12h": float(basal_action_12h),
        "future_rapid_action_15": float(future_rapid_action_15),
        "future_basal_action_15": float(future_basal_action_15),
    }


def _feature_vector(row: dict) -> np.ndarray:
    return np.array([float(row.get(name, 0.0)) for name in DYNAMIC_FEATURE_NAMES], dtype=float)


def _premise_vector(row: dict) -> np.ndarray:
    return np.array([float(row.get(name, 0.0)) for name in PREMISE_FEATURES], dtype=float)


def _compute_centers(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.quantile(values, [0.2, 0.5, 0.8]), dtype=float)


def _compute_widths(centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    spread = max(float(np.std(values)), 1e-3)
    widths = np.zeros_like(centers)
    fallback = spread * 0.75
    for index, center in enumerate(centers):
        neighbors = []
        if index > 0:
            neighbors.append(abs(center - centers[index - 1]))
        if index < len(centers) - 1:
            neighbors.append(abs(centers[index + 1] - center))
        widths[index] = max(float(np.mean(neighbors)) if neighbors else fallback, fallback)
    return widths


def _gaussian(value: float, center: float, width: float) -> float:
    return float(np.exp(-0.5 * ((value - center) / max(width, 1e-6)) ** 2))


def _classify_dynamic_regime(delta_15: float, delta_30: float) -> str:
    if delta_15 >= 12.0 or delta_30 >= 25.0:
        return "alta_abrupta"
    if delta_15 <= -12.0 or delta_30 <= -25.0:
        return "queda_abrupta"
    return "estabilidade"


def _rule_weights(premise: np.ndarray, model_payload: dict) -> np.ndarray:
    weights = []
    glucose = float(premise[0])
    delta = float(premise[1])
    net_input = float(premise[2])
    g_centers = np.asarray(model_payload["glucose_centers"], dtype=float)
    g_widths = np.asarray(model_payload["glucose_widths"], dtype=float)
    d_centers = np.asarray(model_payload["delta_centers"], dtype=float)
    d_widths = np.asarray(model_payload["delta_widths"], dtype=float)
    n_centers = np.asarray(model_payload["net_input_centers"], dtype=float)
    n_widths = np.asarray(model_payload["net_input_widths"], dtype=float)
    for g_idx, g_center in enumerate(g_centers):
        mu_g = _gaussian(glucose, g_center, g_widths[g_idx])
        for d_idx, d_center in enumerate(d_centers):
            mu_d = _gaussian(delta, d_center, d_widths[d_idx])
            for n_idx, n_center in enumerate(n_centers):
                mu_n = _gaussian(net_input, n_center, n_widths[n_idx])
                weights.append(mu_g * mu_d * mu_n)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def _weighted_ridge(features: np.ndarray, targets: np.ndarray, sample_weights: np.ndarray, l2: float) -> dict:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = (features - means) / scales
    sqrt_w = np.sqrt(np.clip(sample_weights, 1e-9, None))[:, None]
    weighted_x = normalized * sqrt_w
    weighted_y = targets * sqrt_w[:, 0]
    identity = np.eye(normalized.shape[1], dtype=float)
    identity[0, 0] = 0.0
    system = weighted_x.T @ weighted_x + float(l2) * identity
    rhs = weighted_x.T @ weighted_y
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ rhs
    return {
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "weights": weights.tolist(),
    }


def _predict_from_single_model(model_payload: dict, feature_payload: dict) -> float:
    premise = _premise_vector(feature_payload)
    weights = _rule_weights(premise, model_payload)
    feature_vector = _feature_vector(feature_payload)
    outputs = []
    for rule in model_payload["rule_models"]:
        means = np.asarray(rule["feature_means"], dtype=float)
        scales = np.asarray(rule["feature_scales"], dtype=float)
        coeffs = np.asarray(rule["weights"], dtype=float)
        outputs.append(float(((feature_vector - means) / scales) @ coeffs))
    return float(np.dot(weights, np.asarray(outputs, dtype=float)))


def predict_dynamic_ts_delta(model_payload: dict, feature_payload: dict) -> float:
    regime = str(feature_payload.get("regime", ""))
    by_regime = model_payload.get("by_regime", {})
    if regime in by_regime and by_regime[regime].get("status") == "trained":
        return _predict_from_single_model(by_regime[regime], feature_payload)
    return _predict_from_single_model(model_payload["global"], feature_payload)


def _train_single_ts_model(rows: list[dict], l2: float) -> dict:
    features = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    targets = np.asarray([float(row["target_delta_15"]) for row in rows], dtype=float)
    premise = np.asarray([_premise_vector(row) for row in rows], dtype=float)
    g_centers = _compute_centers(premise[:, 0])
    d_centers = _compute_centers(premise[:, 1])
    n_centers = _compute_centers(premise[:, 2])
    g_widths = _compute_widths(g_centers, premise[:, 0])
    d_widths = _compute_widths(d_centers, premise[:, 1])
    n_widths = _compute_widths(n_centers, premise[:, 2])
    base_payload = {
        "status": "trained",
        "glucose_centers": g_centers.tolist(),
        "glucose_widths": g_widths.tolist(),
        "delta_centers": d_centers.tolist(),
        "delta_widths": d_widths.tolist(),
        "net_input_centers": n_centers.tolist(),
        "net_input_widths": n_widths.tolist(),
    }
    memberships = np.asarray([_rule_weights(prem, base_payload) for prem in premise], dtype=float)
    rule_models = []
    for rule_index in range(memberships.shape[1]):
        rule_models.append(_weighted_ridge(features, targets, memberships[:, rule_index], l2))
    base_payload["rule_models"] = rule_models
    return base_payload


def train_dynamic_ts_model(rows: list[dict], model_path: Path, l2: float = 8.0) -> dict:
    payload = {
        "feature_names": list(DYNAMIC_FEATURE_NAMES),
        "premise_feature_names": list(PREMISE_FEATURES),
        "rules_per_axis": 3,
        "status": "insufficient_data",
        "regime_names": list(REGIME_NAMES),
    }
    if len(rows) < 32:
        model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    global_model = _train_single_ts_model(rows, l2)
    by_regime = {}
    for regime in REGIME_NAMES:
        subset = [row for row in rows if str(row.get("regime")) == regime]
        if len(subset) < 48:
            continue
        by_regime[regime] = _train_single_ts_model(subset, l2)
    payload.update(
        {
            "status": "trained",
            "l2": float(l2),
            "global": global_model,
            "by_regime": by_regime,
            "points_used": len(rows),
        }
    )
    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_dynamic_rows(glucose_points, events, step_minutes: int = 15) -> list[dict]:
    if len(glucose_points) < 64:
        return []
    rows = []
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    glucose_values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    step_seconds = int(step_minutes) * 60
    event_records = [
        {
            "timestamp": event.timestamp.timestamp(),
            "carbs_g": float(event.carbs_g),
            "rapid_units": float(event.rapid_units),
            "basal_units": float(getattr(event, "basal_units", 0.0)),
        }
        for event in events
    ]
    event_days = {
        event.timestamp.date()
        for event in events
        if event.carbs_g > 0.0 or event.rapid_units > 0.0 or getattr(event, "basal_units", 0.0) > 0.0
    }

    def glucose_at(ts_seconds: float) -> float:
        return float(np.interp(ts_seconds, timestamps, glucose_values))

    cursor = int(timestamps[0] + 120 * 60)
    end_limit = int(timestamps[-1] - step_seconds)
    while cursor <= end_limit:
        current_dt = timedelta(seconds=cursor - timestamps[0])
        timestamp_date = glucose_points[0].timestamp + current_dt
        if timestamp_date.date() not in event_days:
            cursor += step_seconds
            continue
        current = glucose_at(cursor)
        prev_15 = glucose_at(cursor - 15 * 60)
        prev_30 = glucose_at(cursor - 30 * 60)
        prev_60 = glucose_at(cursor - 60 * 60)
        recent_30 = glucose_values[(timestamps >= cursor - 30 * 60) & (timestamps <= cursor)]
        recent_60_events = [event for event in event_records if 0.0 <= cursor - event["timestamp"] <= 60 * 60]
        recent_30_events = [event for event in event_records if 0.0 <= cursor - event["timestamp"] <= 30 * 60]
        future_15 = [event for event in event_records if 0.0 < event["timestamp"] - cursor <= 15 * 60]
        kinetics = _insulin_kinetics(
            [
                {
                    "minutes": float((event["timestamp"] - (cursor - 60 * 60)) / 60.0),
                    "rapid_units": float(event["rapid_units"]),
                    "basal_units": float(event["basal_units"]),
                }
                for event in event_records
                if -15 * 60 <= event["timestamp"] - cursor <= 12 * 60 * 60
            ],
            current_minute=60.0,
        )
        minutes_since_meal = min(
            [float((cursor - event["timestamp"]) / 60.0) for event in event_records if event["carbs_g"] > 0.0 and event["timestamp"] <= cursor],
            default=-1.0,
        )
        minutes_since_rapid = min(
            [float((cursor - event["timestamp"]) / 60.0) for event in event_records if event["rapid_units"] > 0.0 and event["timestamp"] <= cursor],
            default=-1.0,
        )
        minutes_since_basal = min(
            [float((cursor - event["timestamp"]) / 60.0) for event in event_records if event["basal_units"] > 0.0 and event["timestamp"] <= cursor],
            default=-1.0,
        )
        dt = timestamp_date
        angle = 2.0 * np.pi * (dt.hour + dt.minute / 60.0) / 24.0
        next_glucose = glucose_at(cursor + step_seconds)
        rows.append(
            {
                "timestamp": dt.isoformat(),
                "current_glucose": current,
                "delta_15": current - prev_15,
                "delta_30": current - prev_30,
                "delta_60": current - prev_60,
                "net_input_60": float(sum(event["carbs_g"] for event in recent_60_events) - 8.0 * sum(event["rapid_units"] for event in recent_60_events)),
                "mean_30": float(np.mean(recent_30)) if len(recent_30) else current,
                "std_30": float(np.std(recent_30)) if len(recent_30) else 0.0,
                "carbs_last_30": float(sum(event["carbs_g"] for event in recent_30_events)),
                "carbs_last_60": float(sum(event["carbs_g"] for event in recent_60_events)),
                "rapid_last_30": float(sum(event["rapid_units"] for event in recent_30_events)),
                "rapid_last_60": float(sum(event["rapid_units"] for event in recent_60_events)),
                "basal_last_60": float(sum(event["basal_units"] for event in recent_60_events)),
                "rapid_iob_60": kinetics["rapid_iob_60"],
                "rapid_action_60": kinetics["rapid_action_60"],
                "basal_iob_12h": kinetics["basal_iob_12h"],
                "basal_action_12h": kinetics["basal_action_12h"],
                "future_carbs_15": float(sum(event["carbs_g"] for event in future_15)),
                "future_rapid_15": float(sum(event["rapid_units"] for event in future_15)),
                "future_basal_15": float(sum(event["basal_units"] for event in future_15)),
                "future_rapid_action_15": kinetics["future_rapid_action_15"],
                "future_basal_action_15": kinetics["future_basal_action_15"],
                "minutes_since_meal": minutes_since_meal,
                "minutes_since_rapid": minutes_since_rapid,
                "minutes_since_basal": minutes_since_basal,
                "hour_sin": float(np.sin(angle)),
                "hour_cos": float(np.cos(angle)),
                "target_delta_15": float(next_glucose - current),
                "regime": _classify_dynamic_regime(current - prev_15, current - prev_30),
            }
        )
        cursor += step_seconds
    return rows


def build_dynamic_windows(glucose_points, events, horizon_minutes: int = 120, step_minutes: int = 15) -> list[dict]:
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    glucose_values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    windows = []
    event_records = [
        {
            "timestamp": event.timestamp,
            "carbs_g": float(event.carbs_g),
            "rapid_units": float(event.rapid_units),
            "basal_units": float(getattr(event, "basal_units", 0.0)),
        }
        for event in events
        if event.carbs_g > 0.0 or event.rapid_units > 0.0 or getattr(event, "basal_units", 0.0) > 0.0
    ]
    for event in event_records:
        start = event["timestamp"] - timedelta(minutes=60)
        end = event["timestamp"] + timedelta(minutes=horizon_minutes)
        start_s = start.timestamp()
        end_s = end.timestamp()
        if start_s < timestamps[0] or end_s > timestamps[-1]:
            continue
        grid = np.arange(start_s, end_s + step_minutes * 60, step_minutes * 60, dtype=float)
        observed = np.interp(grid, timestamps, glucose_values).astype(float)
        future_events = []
        for other in event_records:
            delta_minutes = (other["timestamp"] - start).total_seconds() / 60.0
            if 0.0 <= delta_minutes <= horizon_minutes + 60:
                future_events.append(
                    {
                        "minutes": float(delta_minutes),
                        "carbs_g": float(other["carbs_g"]),
                        "rapid_units": float(other["rapid_units"]),
                        "basal_units": float(other["basal_units"]),
                    }
                )
        windows.append(
            {
                "start_timestamp": start.isoformat(),
                "grid_minutes": ((grid - start_s) / 60.0).tolist(),
                "observed_glucose": observed.tolist(),
                "events": future_events,
            }
        )
    return windows


def build_dynamic_feature_payload(
    current: float,
    prev_15: float,
    prev_30: float,
    prev_60: float,
    recent_values: np.ndarray,
    previous_events_30: list[dict],
    previous_events_60: list[dict],
    future_events: list[dict],
    current_minute: float,
) -> dict:
    past_meals = [current_minute - float(item["minutes"]) for item in previous_events_60 if item.get("carbs_g", 0.0) > 0.0]
    past_rapid = [current_minute - float(item["minutes"]) for item in previous_events_60 if item.get("rapid_units", 0.0) > 0.0]
    past_basal = [current_minute - float(item["minutes"]) for item in previous_events_60 if item.get("basal_units", 0.0) > 0.0]
    angle = 2.0 * np.pi * ((current_minute / 60.0) % 24.0) / 24.0
    delta_15 = current - prev_15
    delta_30 = current - prev_30
    kinetics = _insulin_kinetics(previous_events_60 + future_events, current_minute=current_minute)
    return {
        "intercept": 1.0,
        "current_glucose": current,
        "delta_15": delta_15,
        "delta_30": delta_30,
        "delta_60": current - prev_60,
        "net_input_60": float(sum(item["carbs_g"] for item in previous_events_60) - 8.0 * sum(item["rapid_units"] for item in previous_events_60)),
        "mean_30": float(np.mean(recent_values)),
        "std_30": float(np.std(recent_values)),
        "carbs_last_30": float(sum(item["carbs_g"] for item in previous_events_30)),
        "carbs_last_60": float(sum(item["carbs_g"] for item in previous_events_60)),
        "rapid_last_30": float(sum(item["rapid_units"] for item in previous_events_30)),
        "rapid_last_60": float(sum(item["rapid_units"] for item in previous_events_60)),
        "basal_last_60": float(sum(item.get("basal_units", 0.0) for item in previous_events_60)),
        "rapid_iob_60": kinetics["rapid_iob_60"],
        "rapid_action_60": kinetics["rapid_action_60"],
        "basal_iob_12h": kinetics["basal_iob_12h"],
        "basal_action_12h": kinetics["basal_action_12h"],
        "future_carbs_15": float(sum(item["carbs_g"] for item in future_events)),
        "future_rapid_15": float(sum(item["rapid_units"] for item in future_events)),
        "future_basal_15": float(sum(item.get("basal_units", 0.0) for item in future_events)),
        "future_rapid_action_15": kinetics["future_rapid_action_15"],
        "future_basal_action_15": kinetics["future_basal_action_15"],
        "minutes_since_meal": float(min(past_meals)) if past_meals else -1.0,
        "minutes_since_rapid": float(min(past_rapid)) if past_rapid else -1.0,
        "minutes_since_basal": float(min(past_basal)) if past_basal else -1.0,
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
        "regime": _classify_dynamic_regime(delta_15, delta_30),
    }


def rollout_dynamic_ts_window(model_payload: dict, window: dict, step_minutes: int = 15) -> dict:
    observed = np.asarray(window["observed_glucose"], dtype=float)
    grid_minutes = np.asarray(window["grid_minutes"], dtype=float)
    event_items = list(window.get("events", []))
    if len(observed) < 5:
        return {
            "predicted_glucose": observed.tolist(),
            "observed_glucose": observed.tolist(),
            "rmse_mg_dl": None,
            "mae_mg_dl": None,
        }
    predicted = observed[:5].tolist()
    for index in range(4, len(observed) - 1):
        current = float(predicted[-1])
        prev_15 = float(predicted[-2])
        prev_30 = float(predicted[-3])
        prev_60 = float(predicted[-5])
        recent = np.asarray(predicted[-3:], dtype=float)
        current_minute = float(grid_minutes[index])
        future_events = [item for item in event_items if 0.0 < float(item["minutes"]) - current_minute <= step_minutes]
        previous_events_30 = [item for item in event_items if 0.0 <= current_minute - float(item["minutes"]) <= 30.0]
        previous_events_60 = [item for item in event_items if 0.0 <= current_minute - float(item["minutes"]) <= 60.0]
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
        predicted.append(current + predict_dynamic_ts_delta(model_payload, feature_payload))
    predicted = np.asarray(predicted, dtype=float)
    target = observed[: len(predicted)]
    residuals = predicted - target
    return {
        "predicted_glucose": predicted.tolist(),
        "observed_glucose": target.tolist(),
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
    }

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
from scipy.optimize import least_squares


@dataclass
class PersonalModelParams:
    stomach_emptying_rate: float = 0.030
    gut_absorption_rate: float = 0.018
    meal_gain_mgdl_per_g: float = 3.2
    rapid_absorption_rate: float = 0.018
    rapid_action_rate: float = 0.012
    basal_absorption_rate: float = 0.003
    basal_action_rate: float = 0.004
    insulin_sensitivity_rapid: float = 10.0
    insulin_sensitivity_basal: float = 4.0
    insulin_effect_saturation: float = 80.0
    glucose_effectiveness: float = 0.020
    hepatic_production_gain: float = 0.85
    hepatic_suppression_gain: float = 0.10
    basal_recovery_rate: float = 0.003
    exercise_activation_rate: float = 0.080
    exercise_decay_rate: float = 0.015
    exercise_clearance_gain: float = 0.060
    exercise_sensitivity_gain: float = 0.040
    cgm_delay_rate: float = 0.060
    endogenous_production_offset: float = 0.0
    hypoglycemia_threshold_mg_dl: float = 85.0
    hypoglycemia_transition_width_mg_dl: float = 10.0
    hypoglycemia_hepatic_support: float = 1.2
    hypoglycemia_insulin_attenuation: float = 0.75


PARAMETER_NAMES = (
    "stomach_emptying_rate",
    "gut_absorption_rate",
    "meal_gain_mgdl_per_g",
    "rapid_absorption_rate",
    "rapid_action_rate",
    "basal_absorption_rate",
    "basal_action_rate",
    "insulin_sensitivity_rapid",
    "insulin_sensitivity_basal",
    "insulin_effect_saturation",
    "glucose_effectiveness",
    "hepatic_production_gain",
    "hepatic_suppression_gain",
    "basal_recovery_rate",
    "exercise_activation_rate",
    "exercise_decay_rate",
    "exercise_clearance_gain",
    "exercise_sensitivity_gain",
    "cgm_delay_rate",
    "endogenous_production_offset",
    "hypoglycemia_threshold_mg_dl",
    "hypoglycemia_transition_width_mg_dl",
    "hypoglycemia_hepatic_support",
    "hypoglycemia_insulin_attenuation",
)


PARAMETER_BOUNDS = {
    "stomach_emptying_rate": (0.003, 0.080),
    "gut_absorption_rate": (0.003, 0.060),
    "meal_gain_mgdl_per_g": (0.5, 12.0),
    "rapid_absorption_rate": (0.002, 0.060),
    "rapid_action_rate": (0.002, 0.060),
    "basal_absorption_rate": (0.0003, 0.020),
    "basal_action_rate": (0.0003, 0.020),
    "insulin_sensitivity_rapid": (0.1, 40.0),
    "insulin_sensitivity_basal": (0.05, 20.0),
    "insulin_effect_saturation": (5.0, 200.0),
    "glucose_effectiveness": (0.001, 0.080),
    "hepatic_production_gain": (0.1, 2.0),
    "hepatic_suppression_gain": (0.0, 0.5),
    "basal_recovery_rate": (0.0005, 0.020),
    "exercise_activation_rate": (0.010, 0.200),
    "exercise_decay_rate": (0.001, 0.050),
    "exercise_clearance_gain": (0.0, 0.150),
    "exercise_sensitivity_gain": (0.0, 0.150),
    "cgm_delay_rate": (0.005, 0.250),
    "endogenous_production_offset": (-5.0, 5.0),
    "hypoglycemia_threshold_mg_dl": (65.0, 110.0),
    "hypoglycemia_transition_width_mg_dl": (3.0, 20.0),
    "hypoglycemia_hepatic_support": (0.0, 4.0),
    "hypoglycemia_insulin_attenuation": (0.0, 0.95),
}


STATE_NAMES = (
    "q_stomach",
    "q_gut",
    "i_rapid_sc",
    "i_basal_sc",
    "x_rapid",
    "x_basal",
    "exercise_state",
    "g_blood",
    "g_cgm",
)


def classify_window_regime(window: dict) -> dict:
    sample_times = np.asarray(window.get("sample_minutes", []), dtype=float)
    sample_glucose = np.asarray(window.get("sample_glucose_mg_dl", []), dtype=float)
    if len(sample_times) >= 2 and len(sample_glucose) >= 2:
        current_glucose = float(sample_glucose[0])
        glucose_30 = float(np.interp(min(30.0, float(sample_times[-1])), sample_times, sample_glucose))
        glucose_60 = float(np.interp(min(60.0, float(sample_times[-1])), sample_times, sample_glucose))
        delta_30 = glucose_30 - current_glucose
        delta_60 = glucose_60 - current_glucose
        slope_30 = delta_30 / max(min(30.0, float(sample_times[-1])), 1.0)
        slope_60 = delta_60 / max(min(60.0, float(sample_times[-1])), 1.0)
    else:
        current_glucose = float(window.get("baseline_glucose_mg_dl", 0.0))
        delta_30 = 0.0
        delta_60 = 0.0
        slope_30 = 0.0
        slope_60 = 0.0

    if delta_30 >= 25.0 or slope_30 >= 0.8 or delta_60 >= 40.0:
        name = "alta_abrupta"
        reason = "subida rápida da glicose"
    elif delta_30 <= -25.0 or slope_30 <= -0.8 or delta_60 <= -40.0:
        name = "queda_abrupta"
        reason = "queda rápida da glicose"
    else:
        name = "estabilidade"
        reason = "variação curta limitada"
    return {
        "name": name,
        "reason": reason,
        "hour_of_day": float(window.get("hour_of_day", 12.0)),
        "current_glucose_mg_dl": current_glucose,
        "delta_30_mg_dl": float(delta_30),
        "delta_60_mg_dl": float(delta_60),
        "slope_30_mgdl_per_min": float(slope_30),
        "slope_60_mgdl_per_min": float(slope_60),
    }


def build_walk_forward_splits(windows: list[dict], min_train_windows: int = 4, validation_windows: int = 2, step_windows: int = 2) -> list[dict]:
    if len(windows) <= min_train_windows:
        return []
    splits = []
    train_end = int(min_train_windows)
    while train_end < len(windows):
        valid_end = min(train_end + int(validation_windows), len(windows))
        train_slice = windows[:train_end]
        valid_slice = windows[train_end:valid_end]
        if not valid_slice:
            break
        splits.append(
            {
                "train_windows": train_slice,
                "valid_windows": valid_slice,
                "train_start": train_slice[0]["timestamp"],
                "train_end": train_slice[-1]["timestamp"],
                "valid_start": valid_slice[0]["timestamp"],
                "valid_end": valid_slice[-1]["timestamp"],
            }
        )
        if valid_end == len(windows):
            break
        train_end += int(step_windows)
    return splits


def _vector_from_params(params: PersonalModelParams, active_param_names: tuple[str, ...]) -> np.ndarray:
    return np.array([float(getattr(params, name)) for name in active_param_names], dtype=float)


def _params_from_vector(vector: np.ndarray, active_param_names: tuple[str, ...], base_params: PersonalModelParams) -> PersonalModelParams:
    payload = asdict(base_params)
    payload.update({name: float(vector[index]) for index, name in enumerate(active_param_names)})
    return PersonalModelParams(**payload)


def _saturated_insulin_effect(action_state: float, sensitivity: float, saturation: float) -> float:
    return sensitivity * np.tanh(max(action_state, 0.0) / max(saturation, 1e-6))


def _rk4_step(
    state: np.ndarray,
    dt: float,
    meal_input: float,
    rapid_input: float,
    basal_input: float,
    exercise_input: float,
    baseline_glucose: float,
    params: PersonalModelParams,
) -> np.ndarray:
    def rhs(local_state):
        q_stomach, q_gut, i_rapid_sc, i_basal_sc, x_rapid, x_basal, ex_state, g_blood, g_cgm = local_state
        dq_stomach = -params.stomach_emptying_rate * q_stomach + meal_input
        dq_gut = params.stomach_emptying_rate * q_stomach - params.gut_absorption_rate * q_gut
        di_rapid = -params.rapid_absorption_rate * i_rapid_sc + rapid_input
        di_basal = -params.basal_absorption_rate * i_basal_sc + basal_input
        dx_rapid = -params.rapid_action_rate * x_rapid + params.rapid_absorption_rate * i_rapid_sc
        dx_basal = -params.basal_action_rate * x_basal + params.basal_absorption_rate * i_basal_sc
        dex = -params.exercise_decay_rate * ex_state + params.exercise_activation_rate * exercise_input

        insulin_effect = _saturated_insulin_effect(x_rapid, params.insulin_sensitivity_rapid, params.insulin_effect_saturation)
        insulin_effect += _saturated_insulin_effect(x_basal, params.insulin_sensitivity_basal, params.insulin_effect_saturation)
        exercise_multiplier = 1.0 + ex_state * (1.0 + params.exercise_sensitivity_gain)
        hypo_guard = 1.0 / (
            1.0
            + np.exp(
                (g_blood - params.hypoglycemia_threshold_mg_dl)
                / max(params.hypoglycemia_transition_width_mg_dl, 1e-6)
            )
        )
        glucose_excess = max(g_blood - (baseline_glucose - 10.0), 0.0)
        glucose_scale = max(baseline_glucose, 80.0)
        insulin_attenuation = np.clip(1.0 - params.hypoglycemia_insulin_attenuation * hypo_guard, 0.05, 1.0)
        insulin_clearance = exercise_multiplier * insulin_effect * insulin_attenuation * (glucose_excess / glucose_scale)
        hepatic_base = params.hepatic_production_gain / (1.0 + np.exp((g_blood - baseline_glucose) / 25.0))
        hepatic = hepatic_base * np.clip(
            1.0 - params.hepatic_suppression_gain * (x_rapid + x_basal) / max(params.insulin_effect_saturation, 1e-6),
            0.30,
            1.20,
        )
        hepatic += params.hypoglycemia_hepatic_support * hypo_guard
        basal_pull = params.glucose_effectiveness * (g_blood - baseline_glucose)
        exercise_clearance = params.exercise_clearance_gain * ex_state * insulin_attenuation * (glucose_excess / glucose_scale)
        dg_blood = (
            params.meal_gain_mgdl_per_g * params.gut_absorption_rate * q_gut
            + hepatic
            + params.basal_recovery_rate * max(baseline_glucose - g_blood, 0.0)
            - insulin_clearance
            - basal_pull
            - exercise_clearance
            + params.endogenous_production_offset
        )
        dg_cgm = params.cgm_delay_rate * (g_blood - g_cgm)
        return np.array([dq_stomach, dq_gut, di_rapid, di_basal, dx_rapid, dx_basal, dex, dg_blood, dg_cgm], dtype=float)

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_personal_model(window: dict, params: PersonalModelParams, dt_minutes: float = 1.0) -> dict:
    baseline = float(window["baseline_glucose_mg_dl"])
    duration = float(window["duration_minutes"])
    meal_events = list(window.get("meal_events", []))
    rapid_events = list(window.get("rapid_events", []))
    basal_events = list(window.get("basal_events", []))
    exercise_events = list(window.get("exercise_events", []))
    sample_times = np.asarray(window["sample_minutes"], dtype=float)
    max_time = max(duration, float(sample_times[-1]) if len(sample_times) else duration)
    grid = np.arange(0.0, max_time + dt_minutes, dt_minutes)
    grid = np.unique(np.concatenate([grid, sample_times, np.array([0.0, max_time])]))

    state = np.zeros(len(STATE_NAMES), dtype=float)
    state[7] = baseline
    state[8] = baseline
    states = np.zeros((len(grid), len(STATE_NAMES)), dtype=float)

    for index, current_time in enumerate(grid):
        states[index] = state
        if index == len(grid) - 1:
            continue
        next_time = float(grid[index + 1])
        dt = next_time - float(current_time)
        meal_input = sum(float(event["carbs_g"]) for event in meal_events if float(event["minutes"]) == float(current_time))
        rapid_input = sum(float(event["units"]) for event in rapid_events if float(event["minutes"]) == float(current_time))
        basal_input = sum(float(event["units"]) for event in basal_events if float(event["minutes"]) == float(current_time))
        exercise_input = sum(float(event["intensity"]) for event in exercise_events if float(event["minutes"]) == float(current_time))
        state = _rk4_step(
            state,
            dt,
            meal_input,
            rapid_input,
            basal_input,
            exercise_input,
            baseline,
            params,
        )

    return {
        "t_minutes": grid,
        "states": states,
        "state_names": list(STATE_NAMES),
        "glucose_mg_dl": states[:, 7],
        "cgm_glucose_mg_dl": states[:, 8],
    }


def build_personal_model_windows(glucose_points, events, horizon_minutes: int = 240, sample_step_minutes: int = 10) -> list[dict]:
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    windows = []

    def interp(ts):
        if ts < timestamps[0] or ts > timestamps[-1]:
            return None
        return float(np.interp(ts, timestamps, values))

    for event in events:
        if event.carbs_g <= 0.0 and event.rapid_units <= 0.0 and event.basal_units <= 0.0:
            continue
        window_start = event.timestamp - timedelta(minutes=45)
        window_end = event.timestamp + timedelta(minutes=horizon_minutes)
        start_s = window_start.timestamp()
        end_s = window_end.timestamp()
        mask = (timestamps >= start_s) & (timestamps <= end_s)
        if int(mask.sum()) < 16:
            continue
        baseline = interp(start_s)
        if baseline is None:
            continue
        sample_times = ((timestamps[mask] - event.timestamp.timestamp()) / 60.0).astype(float)
        sample_glucose = values[mask].astype(float)
        nonnegative_mask = sample_times >= 0.0
        if int(nonnegative_mask.sum()) < 10:
            continue
        sample_times = sample_times[nonnegative_mask]
        sample_glucose = sample_glucose[nonnegative_mask]
        if sample_step_minutes > 1 and len(sample_times):
            bucket_map = {}
            for minute, glucose in zip(sample_times, sample_glucose):
                bucket = int(round(float(minute) / float(sample_step_minutes))) * int(sample_step_minutes)
                bucket_map[bucket] = float(glucose)
            ordered_buckets = sorted(bucket_map.items())
            sample_times = np.asarray([float(item[0]) for item in ordered_buckets], dtype=float)
            sample_glucose = np.asarray([float(item[1]) for item in ordered_buckets], dtype=float)
            last_bucket = float(round(float(horizon_minutes) / sample_step_minutes) * sample_step_minutes)
            if sample_times[-1] != last_bucket:
                sample_times = np.append(sample_times, last_bucket)
                sample_glucose = np.append(sample_glucose, float(sample_glucose[-1]))
        meal_events = []
        rapid_events = []
        basal_events = []
        exercise_events = []
        for other in events:
            relative_minutes = (other.timestamp - event.timestamp).total_seconds() / 60.0
            if not (0.0 <= relative_minutes <= horizon_minutes):
                continue
            marker = float(round(relative_minutes))
            if other.carbs_g > 0.0:
                meal_events.append({"minutes": marker, "carbs_g": float(other.carbs_g)})
            if other.rapid_units > 0.0:
                rapid_events.append({"minutes": marker, "units": float(other.rapid_units)})
            if other.basal_units > 0.0:
                basal_events.append({"minutes": marker, "units": float(other.basal_units)})
        windows.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "hour_of_day": float(event.timestamp.hour + event.timestamp.minute / 60.0),
                "baseline_glucose_mg_dl": float(baseline),
                "duration_minutes": float(horizon_minutes),
                "sample_minutes": sample_times.tolist(),
                "sample_glucose_mg_dl": sample_glucose.tolist(),
                "meal_events": meal_events,
                "rapid_events": rapid_events,
                "basal_events": basal_events,
                "exercise_events": exercise_events,
                "carbs_g": float(event.carbs_g),
                "rapid_units": float(event.rapid_units),
                "basal_units": float(event.basal_units),
            }
        )
        windows[-1]["regime"] = classify_window_regime(windows[-1])
    return windows


def infer_active_parameters(windows: list[dict]) -> tuple[str, ...]:
    has_basal = any(window.get("basal_units", 0.0) > 0.0 or window.get("basal_events") for window in windows)
    has_exercise = any(window.get("exercise_events") for window in windows)
    active = [
        "stomach_emptying_rate",
        "gut_absorption_rate",
        "meal_gain_mgdl_per_g",
        "rapid_absorption_rate",
        "rapid_action_rate",
        "insulin_sensitivity_rapid",
        "glucose_effectiveness",
        "hepatic_production_gain",
        "cgm_delay_rate",
        "endogenous_production_offset",
    ]
    if has_basal:
        active.extend(["basal_absorption_rate", "basal_action_rate"])
    if has_exercise:
        active.extend(["exercise_decay_rate", "exercise_clearance_gain"])
    return tuple(active)


def _window_residuals(params_vector: np.ndarray, windows: list[dict], active_param_names: tuple[str, ...], base_params: PersonalModelParams) -> np.ndarray:
    params = _params_from_vector(params_vector, active_param_names, base_params)
    residuals = []
    for window in windows:
        result = simulate_personal_model(window, params)
        predicted = np.interp(window["sample_minutes"], result["t_minutes"], result["cgm_glucose_mg_dl"])
        observed = np.asarray(window["sample_glucose_mg_dl"], dtype=float)
        residuals.extend(((predicted - observed) / 25.0).tolist())
    return np.asarray(residuals, dtype=float)


def fit_personal_model(windows: list[dict], initial_params: PersonalModelParams | None = None) -> dict:
    base = initial_params or PersonalModelParams()
    if not windows:
        return {"status": "insufficient_data", "active_parameter_names": [], "params": asdict(base), "rmse_mg_dl": None, "mae_mg_dl": None}

    active_param_names = infer_active_parameters(windows)
    x0 = _vector_from_params(base, active_param_names)
    lower = np.array([PARAMETER_BOUNDS[name][0] for name in active_param_names], dtype=float)
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in active_param_names], dtype=float)
    result = least_squares(
        _window_residuals,
        x0=x0,
        bounds=(lower, upper),
        args=(windows, active_param_names, base),
        loss="soft_l1",
        max_nfev=20,
    )
    params = _params_from_vector(result.x, active_param_names, base)
    residuals = []
    for window in windows:
        sim = simulate_personal_model(window, params)
        predicted = np.interp(window["sample_minutes"], sim["t_minutes"], sim["cgm_glucose_mg_dl"])
        observed = np.asarray(window["sample_glucose_mg_dl"], dtype=float)
        residuals.extend((predicted - observed).tolist())
    residuals = np.asarray(residuals, dtype=float)
    return {
        "status": "fitted",
        "active_parameter_names": list(active_param_names),
        "params": asdict(params),
        "cost": float(result.cost),
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
    }


def evaluate_personal_model(windows: list[dict], params_payload: dict) -> dict:
    params = PersonalModelParams(**{name: params_payload[name] for name in PARAMETER_NAMES})
    residuals = []
    per_window = []
    for window in windows:
        sim = simulate_personal_model(window, params)
        predicted = np.interp(window["sample_minutes"], sim["t_minutes"], sim["cgm_glucose_mg_dl"])
        observed = np.asarray(window["sample_glucose_mg_dl"], dtype=float)
        errors = predicted - observed
        residuals.extend(errors.tolist())
        per_window.append(
            {
                "timestamp": window["timestamp"],
                "rmse_mg_dl": float(np.sqrt(np.mean(errors**2))),
                "mae_mg_dl": float(np.mean(np.abs(errors))),
                "carbs_g": float(window.get("carbs_g", 0.0)),
                "rapid_units": float(window.get("rapid_units", 0.0)),
                "basal_units": float(window.get("basal_units", 0.0)),
            }
        )
    residuals = np.asarray(residuals, dtype=float)
    return {
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "mae_mg_dl": float(np.mean(np.abs(residuals))) if len(residuals) else None,
        "windows": per_window,
    }

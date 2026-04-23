from dataclasses import dataclass

import numpy as np


@dataclass
class UltradianParams:
    Vp: float = 3.0
    Vi: float = 11.0
    Vg: float = 10.0
    E: float = 0.2
    tp: float = 6.0
    ti: float = 100.0
    td: float = 12.0
    Rm: float = 209.0
    a1: float = 6.67
    C1: float = 300.0
    C2: float = 144.0
    C3: float = 100.0
    C4: float = 80.0
    C5: float = 26.0
    Ub: float = 72.0
    U0: float = 4.0
    Um: float = 94.0
    Rg: float = 180.0
    alpha: float = 7.5
    beta: float = 1.77
    ka: float = 0.03
    insulin_bioavailability: float = 1.0
    meal_tau: float = 40.0
    meal_carb_scale: float = 1.0
    insulin_sensitivity_scale: float = 1.0
    insulin_rapid_tau: float = 75.0
    insulin_regular_tau: float = 150.0
    insulin_basal_tau: float = 360.0
    insulin_unknown_tau: float = 90.0
    insulin_basal_duration: float = 1440.0
    insulin_basal_onset: float = 90.0
    insulin_basal_tail: float = 180.0
    insulin_rapid_bio_scale: float = 1.0
    insulin_regular_bio_scale: float = 0.95
    insulin_basal_bio_scale: float = 1.0
    insulin_unknown_bio_scale: float = 1.0
    exercise_hr_baseline_bpm: float = 70.0
    exercise_fast_tau_minutes: float = 5.0
    exercise_slow_tau_minutes: float = 20.0
    exercise_recovery_tau_minutes: float = 90.0
    exercise_clearance_gain: float = 0.45
    exercise_sensitivity_gain: float = 0.30
    exercise_hepatic_suppression_gain: float = 0.12

    @property
    def kappa(self) -> float:
        return (1.0 / self.C4) * (1.0 / self.Vi + 1.0 / (self.E * self.ti))


def glucose_mgdl_to_total_mg(g_mg_dl, params: UltradianParams):
    return 10.0 * np.asarray(g_mg_dl) * params.Vg


def total_mg_to_glucose_mgdl(total_glucose, params: UltradianParams):
    return np.asarray(total_glucose) / params.Vg / 10.0


def total_insulin_to_uu_ml(total_insulin, volume_liters: float):
    return np.asarray(total_insulin) / volume_liters


def f1(glucose_total: float, params: UltradianParams) -> float:
    return params.Rm / (1.0 + np.exp(-glucose_total / (params.Vg * params.C1) + params.a1))


def f2(glucose_total: float, params: UltradianParams) -> float:
    return params.Ub * (1.0 - np.exp(-glucose_total / (params.C2 * params.Vg)))


def f3(remote_insulin_total: float, params: UltradianParams) -> float:
    remote_insulin_total = np.maximum(np.asarray(remote_insulin_total), 1e-12)
    x = params.kappa * remote_insulin_total
    base = (1.0 / (params.C3 * params.Vg)) * (
        params.U0 + (params.Um - params.U0) / (1.0 + x ** (-params.beta))
    )
    return params.insulin_sensitivity_scale * base


def f4(h3_total: float, params: UltradianParams) -> float:
    return params.Rg / (1.0 + np.exp(params.alpha * (h3_total / (params.C5 * params.Vp) - 1.0)))


def insulin_profile_parameters(insulin_type: str, params: UltradianParams) -> tuple[float, float]:
    normalized = str(insulin_type or "").strip().lower()
    if normalized == "basal":
        return params.insulin_basal_tau, params.insulin_basal_bio_scale
    if normalized == "regular":
        return params.insulin_regular_tau, params.insulin_regular_bio_scale
    if normalized == "rapida":
        return params.insulin_rapid_tau, params.insulin_rapid_bio_scale
    return params.insulin_unknown_tau, params.insulin_unknown_bio_scale


def _basal_glargine_release(elapsed: float, dose_milli_units: float, params: UltradianParams) -> float:
    if elapsed < 0.0 or elapsed > params.insulin_basal_duration:
        return 0.0
    ramp_up = min(elapsed / max(params.insulin_basal_onset, 1e-6), 1.0)
    if elapsed <= params.insulin_basal_duration - params.insulin_basal_tail:
        ramp_down = 1.0
    else:
        ramp_down = max((params.insulin_basal_duration - elapsed) / max(params.insulin_basal_tail, 1e-6), 0.0)
    effective_area_minutes = (
        params.insulin_basal_duration
        - 0.5 * params.insulin_basal_onset
        - 0.5 * params.insulin_basal_tail
    )
    flat_rate = dose_milli_units / max(effective_area_minutes, 1e-6)
    return flat_rate * ramp_up * ramp_down


def build_insulin_depot_input(insulin_events, params: UltradianParams):
    def insulin_depot_input(t: float) -> float:
        total = 0.0
        for event in insulin_events:
            if t < event.minutes:
                continue
            elapsed = t - event.minutes
            insulin_type = getattr(event, "insulin_type", "desconhecido")
            tau, bio_scale = insulin_profile_parameters(insulin_type, params)
            dose_milli_units = event.units * 1000.0 * bio_scale
            if insulin_type == "basal":
                total += _basal_glargine_release(elapsed, dose_milli_units, params)
            else:
                total += dose_milli_units * (elapsed / tau**2) * np.exp(-elapsed / tau)
        return total

    return insulin_depot_input


def build_meal_input(meal_events, params: UltradianParams, disturbance=None):
    def glucose_input(t: float) -> float:
        total = 0.0
        for event in meal_events:
            if t < event.minutes:
                continue
            elapsed = t - event.minutes
            meal_size_mg = event.carbs_g * 1000.0 * params.meal_carb_scale
            total += meal_size_mg * (elapsed / params.meal_tau**2) * np.exp(-elapsed / params.meal_tau)
        if disturbance is not None:
            cutoff = disturbance.get("cutoff_minutes")
            if cutoff is None or t <= cutoff:
                total += float(np.interp(t, disturbance["t"], disturbance["values"]))
        return total

    return glucose_input


def _exercise_filtered_component(
    elapsed_since_start: float,
    event_duration: float,
    hr_delta: float,
    rise_tau: float,
    fall_tau: float,
) -> float:
    if elapsed_since_start < 0.0:
        return 0.0
    if elapsed_since_start <= event_duration:
        return hr_delta * (1.0 - np.exp(-elapsed_since_start / max(rise_tau, 1e-6)))
    steady = hr_delta * (1.0 - np.exp(-event_duration / max(rise_tau, 1e-6)))
    return steady * np.exp(-(elapsed_since_start - event_duration) / max(fall_tau, 1e-6))


def build_exercise_effects(exercise_events, params: UltradianParams):
    def exercise_effects(t: float) -> tuple[float, float, float, float]:
        y_fast = 0.0
        z_slow = 0.0
        active_hr = params.exercise_hr_baseline_bpm
        for event in exercise_events:
            elapsed = t - event.minutes
            if elapsed < 0.0:
                continue
            hr_delta = max(float(event.heart_rate_bpm) - params.exercise_hr_baseline_bpm, 0.0)
            if hr_delta <= 0.0:
                continue
            y_fast += _exercise_filtered_component(
                elapsed_since_start=elapsed,
                event_duration=float(event.duration_minutes),
                hr_delta=hr_delta,
                rise_tau=params.exercise_fast_tau_minutes,
                fall_tau=params.exercise_fast_tau_minutes,
            )
            z_slow += _exercise_filtered_component(
                elapsed_since_start=elapsed,
                event_duration=float(event.duration_minutes),
                hr_delta=hr_delta,
                rise_tau=params.exercise_slow_tau_minutes,
                fall_tau=params.exercise_recovery_tau_minutes,
            )
            if 0.0 <= elapsed <= float(event.duration_minutes):
                active_hr = max(active_hr, float(event.heart_rate_bpm))

        normalized_fast = y_fast / max(params.exercise_hr_baseline_bpm, 1e-6)
        normalized_slow = z_slow / max(params.exercise_hr_baseline_bpm, 1e-6)
        clearance_scale = max(1.0, 1.0 + params.exercise_clearance_gain * normalized_fast)
        sensitivity_scale = max(1.0, 1.0 + params.exercise_sensitivity_gain * normalized_slow)
        hepatic_scale = max(0.7, 1.0 - params.exercise_hepatic_suppression_gain * normalized_fast)
        return clearance_scale, sensitivity_scale, hepatic_scale, active_hr

    return exercise_effects


def full_rhs(t: float, state: np.ndarray, params: UltradianParams, meal_input, insulin_depot_input, exercise_effects=None) -> np.ndarray:
    S, Ip, Ii, G, h1, h2, h3 = state
    glucose_input = meal_input(t)
    insulin_input = insulin_depot_input(t)
    if exercise_effects is None:
        clearance_scale = 1.0
        sensitivity_scale = 1.0
        hepatic_scale = 1.0
    else:
        clearance_scale, sensitivity_scale, hepatic_scale, _ = exercise_effects(t)

    dS = insulin_input - params.ka * S
    dIp = (
        f1(G, params)
        + params.ka * params.insulin_bioavailability * S
        - ((Ip / params.Vp) - (Ii / params.Vi)) * params.E
        - Ip / params.tp
    )
    dIi = ((Ip / params.Vp) - (Ii / params.Vi)) * params.E - Ii / params.ti
    dG = hepatic_scale * f4(h3, params) + glucose_input - clearance_scale * f2(G, params) - sensitivity_scale * f3(Ii, params) * G
    dh1 = (Ip - h1) / params.td
    dh2 = (h1 - h2) / params.td
    dh3 = (h2 - h3) / params.td
    return np.array([dS, dIp, dIi, dG, dh1, dh2, dh3], dtype=float)


def _rk4_step(t: float, state: np.ndarray, dt: float, params: UltradianParams, meal_input, insulin_depot_input, exercise_effects=None) -> np.ndarray:
    k1 = full_rhs(t, state, params, meal_input, insulin_depot_input, exercise_effects=exercise_effects)
    k2 = full_rhs(t + 0.5 * dt, state + 0.5 * dt * k1, params, meal_input, insulin_depot_input, exercise_effects=exercise_effects)
    k3 = full_rhs(t + 0.5 * dt, state + 0.5 * dt * k2, params, meal_input, insulin_depot_input, exercise_effects=exercise_effects)
    k4 = full_rhs(t + dt, state + dt * k3, params, meal_input, insulin_depot_input, exercise_effects=exercise_effects)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.maximum(next_state, 0.0)


def simulate_forward(
    params: UltradianParams,
    initial_state,
    start_time_minutes: float,
    end_time_minutes: float,
    meal_events,
    insulin_events,
    exercise_events=None,
    disturbance=None,
    dt_minutes: float = 1.0,
    extra_times=None,
):
    base_grid = np.arange(start_time_minutes, end_time_minutes + dt_minutes, dt_minutes)
    insulin_times = np.array(
        [event.minutes for event in insulin_events if start_time_minutes <= event.minutes <= end_time_minutes],
        dtype=float,
    )
    extras = np.array(extra_times or [], dtype=float)
    extras = extras[(start_time_minutes <= extras) & (extras <= end_time_minutes)]
    exercise_times = np.array(
        [
            marker
            for event in (exercise_events or [])
            for marker in (event.minutes, event.minutes + float(event.duration_minutes))
            if start_time_minutes <= marker <= end_time_minutes
        ],
        dtype=float,
    )
    grid = np.unique(np.concatenate([base_grid, insulin_times, exercise_times, extras, np.array([start_time_minutes, end_time_minutes])]))
    meal_input = build_meal_input(meal_events, params, disturbance=disturbance)
    insulin_depot_input = build_insulin_depot_input(insulin_events, params)
    exercise_effects = build_exercise_effects(exercise_events or [], params)
    state = np.asarray(initial_state, dtype=float)
    series = np.zeros((len(grid), len(state)), dtype=float)

    for index, current_time in enumerate(grid):
        series[index] = state
        if index == len(grid) - 1:
            continue
        next_time = grid[index + 1]
        state = _rk4_step(
            float(current_time),
            state,
            float(next_time - current_time),
            params,
            meal_input,
            insulin_depot_input,
            exercise_effects=exercise_effects,
        )

    glucose_mg_dl = total_mg_to_glucose_mgdl(series[:, 3], params)
    plasma_insulin = total_insulin_to_uu_ml(series[:, 1], params.Vp)
    remote_insulin = total_insulin_to_uu_ml(series[:, 2], params.Vi)
    exercise_clearance_scale = np.array([exercise_effects(float(value))[0] for value in grid], dtype=float)
    exercise_sensitivity_scale = np.array([exercise_effects(float(value))[1] for value in grid], dtype=float)
    exercise_hepatic_scale = np.array([exercise_effects(float(value))[2] for value in grid], dtype=float)
    exercise_hr_bpm = np.array([exercise_effects(float(value))[3] for value in grid], dtype=float)
    return {
        "t": grid,
        "states": series,
        "glucose_mg_dl": glucose_mg_dl,
        "plasma_insulin_uu_ml": plasma_insulin,
        "remote_insulin_uu_ml": remote_insulin,
        "exercise_clearance_scale": exercise_clearance_scale,
        "exercise_sensitivity_scale": exercise_sensitivity_scale,
        "exercise_hepatic_scale": exercise_hepatic_scale,
        "exercise_hr_bpm": exercise_hr_bpm,
    }

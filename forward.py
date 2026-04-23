import numpy as np

from model import simulate_forward


def compare_with_observed(forward_result, observed_minutes, observed_glucose_mg_dl):
    simulated_glucose = np.interp(observed_minutes, forward_result["t"], forward_result["glucose_mg_dl"])
    residuals = simulated_glucose - observed_glucose_mg_dl
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mape = float(np.mean(np.abs(residuals) / np.maximum(observed_glucose_mg_dl, 1.0)) * 100.0)
    return {
        "simulated_at_observation_mg_dl": simulated_glucose.tolist(),
        "residuals_mg_dl": residuals.tolist(),
        "mae_mg_dl": mae,
        "rmse_mg_dl": rmse,
        "mape_percent": mape,
    }


def run_forward(
    params,
    initial_state,
    start_time_minutes,
    end_time_minutes,
    meal_events,
    insulin_events,
    exercise_events,
    disturbance,
    dt_minutes,
    extra_times=None,
):
    return simulate_forward(
        params=params,
        initial_state=initial_state,
        start_time_minutes=start_time_minutes,
        end_time_minutes=end_time_minutes,
        meal_events=meal_events,
        insulin_events=insulin_events,
        exercise_events=exercise_events,
        disturbance=disturbance,
        dt_minutes=dt_minutes,
        extra_times=extra_times,
    )

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from calibration import apply_calibrated_params, load_calibrated_params
from config import load_config
from data_io import load_glucose_json, write_json
from forward import compare_with_observed, run_forward
from inputs import load_insulin_events, load_meal_events
from inverse import estimate_initial_state
from model import UltradianParams
from runner import _stabilize_forecast_anchor_state


def _parse_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tuner offline simples para hiperparâmetros do UKF")
    parser.add_argument("--alpha-grid", default="0.08,0.15,0.25")
    parser.add_argument("--measurement-noise-grid", default="4,6,8,10")
    parser.add_argument("--q-g-grid", default="8000,12000,18000")
    parser.add_argument("--q-ii-grid", default="1000,2500,5000")
    parser.add_argument("--q-h-grid", default="1500,3500,6000")
    parser.add_argument("--limit", type=int, default=20, help="Número máximo de combinações a avaliar")
    parser.add_argument("--output", default="outputs/ukf_tuning.json")
    parser.add_argument("--forecast-horizons", default="15,30,60")
    return parser


def _future_events(meal_events, insulin_events, anchor_minutes: float, horizon_minutes: float):
    future_meals = []
    for event in meal_events:
        relative = float(event.minutes - anchor_minutes)
        if 0.0 <= relative <= horizon_minutes:
            future_meals.append(type(event)(timestamp=event.timestamp, minutes=relative, carbs_g=event.carbs_g))

    future_insulin = []
    for event in insulin_events:
        relative = float(event.minutes - anchor_minutes)
        if 0.0 <= relative <= horizon_minutes:
            future_insulin.append(
                type(event)(
                    timestamp=event.timestamp,
                    minutes=relative,
                    units=event.units,
                    insulin_type=event.insulin_type,
                )
            )
    return future_meals, future_insulin


def _score_candidate(config, params, observed_minutes, observed_glucose, meal_events, insulin_events, inverse_result, measurement_rmse, horizons):
    observed_end_minutes = float(observed_minutes[-1])
    initial_state = np.asarray(inverse_result["states"][0], dtype=float)
    historical_result = run_forward(
        params=params,
        initial_state=initial_state,
        start_time_minutes=0.0,
        end_time_minutes=observed_end_minutes,
        meal_events=meal_events,
        insulin_events=insulin_events,
        exercise_events=[],
        disturbance={
            "t": inverse_result["t"],
            "values": inverse_result["inferred_glucose_input"],
            "cutoff_minutes": observed_end_minutes,
        },
        dt_minutes=config.forward_dt_minutes,
        extra_times=[observed_end_minutes],
    )
    forward_metrics = compare_with_observed(historical_result, observed_minutes, observed_glucose)

    horizon_errors = []
    for horizon in horizons:
        anchor_target = observed_end_minutes - horizon
        if anchor_target <= observed_minutes[0]:
            continue
        anchor_index = int(np.searchsorted(observed_minutes, anchor_target, side="right") - 1)
        anchor_minutes = float(observed_minutes[anchor_index])
        anchor_state = np.asarray(inverse_result["states"][anchor_index], dtype=float).copy()
        regularized_state, _ = _stabilize_forecast_anchor_state(
            anchor_state=anchor_state,
            observed_minutes=observed_minutes[: anchor_index + 1],
            observed_glucose=observed_glucose[: anchor_index + 1],
            params=params,
            config=config,
        )
        future_meals, future_insulin = _future_events(meal_events, insulin_events, anchor_minutes, horizon)
        forecast_result = run_forward(
            params=params,
            initial_state=regularized_state,
            start_time_minutes=0.0,
            end_time_minutes=horizon,
            meal_events=future_meals,
            insulin_events=future_insulin,
            exercise_events=[],
            disturbance=None,
            dt_minutes=config.forward_dt_minutes,
            extra_times=[horizon],
        )
        predicted = float(np.interp(horizon, forecast_result["t"], forecast_result["glucose_mg_dl"]))
        observed = float(np.interp(anchor_minutes + horizon, observed_minutes, observed_glucose))
        horizon_errors.append(predicted - observed)

    horizon_mae = float(np.mean(np.abs(horizon_errors))) if horizon_errors else None
    horizon_rmse = float(np.sqrt(np.mean(np.square(horizon_errors)))) if horizon_errors else None
    score = float(forward_metrics["rmse_mg_dl"] + 0.35 * measurement_rmse)
    if horizon_rmse is not None:
        score += 0.75 * horizon_rmse
    return {
        "forward_rmse_mg_dl": float(forward_metrics["rmse_mg_dl"]),
        "forward_mae_mg_dl": float(forward_metrics["mae_mg_dl"]),
        "holdout_forecast_mae_mg_dl": horizon_mae,
        "holdout_forecast_rmse_mg_dl": horizon_rmse,
        "holdout_forecast_errors_mg_dl": [float(value) for value in horizon_errors],
        "rank_score": score,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()

    observed_minutes, observed_glucose, timestamps = load_glucose_json(config.glucose_json_path)
    start_time = timestamps[0]
    meal_events = load_meal_events(config.meals_source, start_time)
    insulin_events = load_insulin_events(config.insulin_source, start_time)

    base_params = UltradianParams(
        ka=config.insulin_absorption_rate,
        insulin_bioavailability=config.insulin_bioavailability,
        meal_tau=config.meal_tau_minutes,
    )
    params = apply_calibrated_params(base_params, load_calibrated_params(config))
    forecast_horizons = _parse_list(args.forecast_horizons)

    alpha_grid = _parse_list(args.alpha_grid)
    measurement_noise_grid = _parse_list(args.measurement_noise_grid)
    q_g_grid = _parse_list(args.q_g_grid)
    q_ii_grid = _parse_list(args.q_ii_grid)
    q_h_grid = _parse_list(args.q_h_grid)

    results = []
    combinations = itertools.product(alpha_grid, measurement_noise_grid, q_g_grid, q_ii_grid, q_h_grid)
    for index, (alpha, measurement_noise_mgdl, q_g, q_ii, q_h) in enumerate(combinations):
        if index >= args.limit:
            break
        ukf_settings = {
            "alpha": alpha,
            "beta": config.ukf_beta,
            "kappa": config.ukf_kappa,
            "measurement_noise_mgdl": measurement_noise_mgdl,
            "q_s": config.ukf_process_noise_s,
            "q_ip": config.ukf_process_noise_ip,
            "q_ii": q_ii,
            "q_g": q_g,
            "q_h": q_h,
        }
        _, inverse_result, state_payload = estimate_initial_state(
            t_data=observed_minutes,
            glucose_data_mg_dl=observed_glucose,
            params=params,
            smoothing_factor=config.smoothing_factor,
            meal_events=meal_events,
            insulin_events=insulin_events,
            estimator_mode="ukf",
            ukf_settings=ukf_settings,
        )
        residuals = inverse_result["predicted_glucose_mg_dl"] - observed_glucose
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        innovation_rmse = float(np.sqrt(np.mean(np.asarray(inverse_result["innovations_mg_dl"], dtype=float) ** 2)))
        pipeline_metrics = _score_candidate(
            config=config,
            params=params,
            observed_minutes=observed_minutes,
            observed_glucose=observed_glucose,
            meal_events=meal_events,
            insulin_events=insulin_events,
            inverse_result=inverse_result,
            measurement_rmse=rmse,
            horizons=forecast_horizons,
        )
        results.append(
            {
                "rank_score": pipeline_metrics["rank_score"],
                "measurement_rmse_mg_dl": rmse,
                "measurement_mae_mg_dl": mae,
                "innovation_rmse_mg_dl": innovation_rmse,
                "forward_rmse_mg_dl": pipeline_metrics["forward_rmse_mg_dl"],
                "forward_mae_mg_dl": pipeline_metrics["forward_mae_mg_dl"],
                "holdout_forecast_mae_mg_dl": pipeline_metrics["holdout_forecast_mae_mg_dl"],
                "holdout_forecast_rmse_mg_dl": pipeline_metrics["holdout_forecast_rmse_mg_dl"],
                "holdout_forecast_errors_mg_dl": pipeline_metrics["holdout_forecast_errors_mg_dl"],
                "ukf_settings": state_payload["inverse"]["ukf_settings"],
            }
        )

    results.sort(key=lambda item: item["rank_score"])
    payload = {
        "evaluated": len(results),
        "best": results[0] if results else None,
        "results": results,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload["best"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

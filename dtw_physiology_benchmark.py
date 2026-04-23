import argparse
import json
import random
from pathlib import Path

import numpy as np

from external_history import load_external_history
from personal_model import (
    PARAMETER_BOUNDS,
    PersonalModelParams,
    build_personal_model_windows,
    evaluate_personal_model,
    simulate_personal_model,
)


TUNED_PARAM_NAMES = (
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark do modelo fisiológico com loss baseada em DTW")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/dtw_physiology_benchmark.json")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--train-windows", type=int, default=6)
    parser.add_argument("--window-samples", type=int, default=2)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_base_params() -> PersonalModelParams:
    path = Path("outputs/personal_model_prototype.json")
    if not path.exists():
        return PersonalModelParams()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload.get("full_fit", {}).get("params", {})
        filtered = {field: params[field] for field in PersonalModelParams.__dataclass_fields__ if field in params}
        return PersonalModelParams(**filtered)
    except Exception:
        return PersonalModelParams()


def _restricted_dtw(predicted: np.ndarray, observed: np.ndarray, window_samples: int = 2) -> dict:
    n = len(predicted)
    m = len(observed)
    window = max(int(window_samples), abs(n - m))
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf, dtype=float)
    cost[0, 0] = 0.0
    back = {}
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            local = abs(float(predicted[i - 1] - observed[j - 1]))
            options = (
                (cost[i - 1, j], (i - 1, j)),
                (cost[i, j - 1], (i, j - 1)),
                (cost[i - 1, j - 1], (i - 1, j - 1)),
            )
            prev_cost, prev_idx = min(options, key=lambda item: item[0])
            cost[i, j] = local + prev_cost
            back[(i, j)] = prev_idx
    path = []
    cursor = (n, m)
    while cursor in back:
        path.append(cursor)
        cursor = back[cursor]
        if cursor == (0, 0):
            break
    path.reverse()
    if not path:
        return {"distance": inf, "normalized_distance": inf, "mean_abs_lag_samples": None, "path_length": 0}
    lags = [abs(i - j) for i, j in path]
    total = float(cost[n, m])
    return {
        "distance": total,
        "normalized_distance": total / max(len(path), 1),
        "mean_abs_lag_samples": float(np.mean(lags)),
        "path_length": len(path),
    }


def _window_metrics(window: dict, params: PersonalModelParams, window_samples: int) -> dict:
    sim = simulate_personal_model(window, params)
    observed = np.asarray(window["sample_glucose_mg_dl"], dtype=float)
    predicted = np.interp(window["sample_minutes"], sim["t_minutes"], sim["cgm_glucose_mg_dl"]).astype(float)
    residuals = predicted - observed
    dtw = _restricted_dtw(predicted, observed, window_samples=window_samples)
    return {
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
        "dtw_distance": dtw["distance"],
        "dtw_normalized_distance": dtw["normalized_distance"],
        "mean_abs_lag_samples": dtw["mean_abs_lag_samples"],
    }


def _aggregate_metrics(windows: list[dict], params: PersonalModelParams, window_samples: int) -> dict:
    items = [_window_metrics(window, params, window_samples) for window in windows]
    return {
        "windows": len(windows),
        "rmse_mg_dl": float(np.mean([item["rmse_mg_dl"] for item in items])) if items else None,
        "mae_mg_dl": float(np.mean([item["mae_mg_dl"] for item in items])) if items else None,
        "dtw_normalized_distance": float(np.mean([item["dtw_normalized_distance"] for item in items])) if items else None,
        "mean_abs_lag_samples": float(np.mean([item["mean_abs_lag_samples"] for item in items if item["mean_abs_lag_samples"] is not None])) if items else None,
        "per_window": items,
    }


def _sample_candidate(best: PersonalModelParams, iteration: int, rng: random.Random) -> PersonalModelParams:
    payload = {field: getattr(best, field) for field in PersonalModelParams.__dataclass_fields__}
    stage = 0.35 if iteration < 40 else 0.20 if iteration < 80 else 0.10
    for name in TUNED_PARAM_NAMES:
        low, high = PARAMETER_BOUNDS[name]
        current = float(payload[name])
        span = max(high - low, 1e-6)
        noise = rng.uniform(-1.0, 1.0) * span * stage
        payload[name] = float(np.clip(current + noise, low, high))
    return PersonalModelParams(**payload)


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(int(args.seed))
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    windows = build_personal_model_windows(glucose_points, events, horizon_minutes=240)
    windows = sorted(windows, key=lambda item: item["timestamp"])
    if args.max_windows > 0:
        windows = windows[: int(args.max_windows)]
    train_size = min(int(args.train_windows), max(len(windows) - 1, 1))
    train_windows = windows[:train_size]
    valid_windows = windows[train_size:]

    base_params = _load_base_params()
    baseline_train = _aggregate_metrics(train_windows, base_params, int(args.window_samples))
    baseline_valid = _aggregate_metrics(valid_windows, base_params, int(args.window_samples))

    best_params = base_params
    best_train = baseline_train
    history = []
    for iteration in range(int(args.trials)):
        candidate = _sample_candidate(best_params, iteration, rng)
        metrics = _aggregate_metrics(train_windows, candidate, int(args.window_samples))
        score = (metrics["dtw_normalized_distance"] or 1e9, metrics["rmse_mg_dl"] or 1e9)
        best_score = (best_train["dtw_normalized_distance"] or 1e9, best_train["rmse_mg_dl"] or 1e9)
        if score < best_score:
            best_params = candidate
            best_train = metrics
            history.append(
                {
                    "iteration": iteration + 1,
                    "train_dtw_normalized_distance": metrics["dtw_normalized_distance"],
                    "train_rmse_mg_dl": metrics["rmse_mg_dl"],
                }
            )

    tuned_valid = _aggregate_metrics(valid_windows, best_params, int(args.window_samples))
    payload = {
        "csv_path": str(csv_path),
        "windows_total": len(windows),
        "train_windows": len(train_windows),
        "valid_windows": len(valid_windows),
        "window_samples": int(args.window_samples),
        "trials": int(args.trials),
        "baseline": {
            "params": {name: getattr(base_params, name) for name in PersonalModelParams.__dataclass_fields__},
            "train": baseline_train,
            "valid": baseline_valid,
        },
        "dtw_tuned": {
            "params": {name: getattr(best_params, name) for name in PersonalModelParams.__dataclass_fields__},
            "train": best_train,
            "valid": tuned_valid,
            "improvements": history[-10:],
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import numpy as np

from external_history import load_external_history
from personal_model import PersonalModelParams, simulate_personal_model
from takagi_sugeno_dynamic import build_dynamic_windows, rollout_dynamic_ts_window, train_dynamic_ts_model, build_dynamic_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark híbrido fisiológico + Takagi-Sugeno dinâmico")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/hybrid_reconstruction_benchmark.json")
    parser.add_argument("--model-output", default="outputs/takagi_sugeno_dynamic_model.json")
    parser.add_argument("--max-rows", type=int, default=1200)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--l2", type=float, default=8.0)
    return parser


def _load_personal_params() -> PersonalModelParams:
    path = Path("outputs/personal_model_prototype.json")
    if not path.exists():
        return PersonalModelParams()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload.get("full_fit", {}).get("params", {})
        if not params:
            return PersonalModelParams()
        filtered = {field: params[field] for field in PersonalModelParams.__dataclass_fields__ if field in params}
        return PersonalModelParams(**filtered)
    except Exception:
        return PersonalModelParams()


def _dynamic_window_to_personal(window: dict) -> dict:
    return {
        "baseline_glucose_mg_dl": float(window["observed_glucose"][0]),
        "duration_minutes": float(window["grid_minutes"][-1]),
        "sample_minutes": list(window["grid_minutes"]),
        "sample_glucose_mg_dl": list(window["observed_glucose"]),
        "meal_events": [
            {"minutes": float(item["minutes"]), "carbs_g": float(item["carbs_g"])}
            for item in window.get("events", [])
            if float(item.get("carbs_g", 0.0)) > 0.0
        ],
        "rapid_events": [
            {"minutes": float(item["minutes"]), "units": float(item["rapid_units"])}
            for item in window.get("events", [])
            if float(item.get("rapid_units", 0.0)) > 0.0
        ],
        "basal_events": [],
        "exercise_events": [],
    }


def _regime_weight(observed: np.ndarray, forecast_start_index: int) -> tuple[float, str]:
    current = float(observed[forecast_start_index])
    prev_30 = float(observed[max(forecast_start_index - 2, 0)])
    delta_30 = current - prev_30
    slope_30 = delta_30 / 30.0
    if delta_30 >= 25.0 or slope_30 >= 0.8:
        return 0.70, "alta_abrupta"
    if delta_30 <= -25.0 or slope_30 <= -0.8:
        return 0.65, "queda_abrupta"
    return 0.35, "estabilidade"


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)

    dynamic_rows = build_dynamic_rows(glucose_points, events)
    if args.max_rows > 0:
        dynamic_rows = dynamic_rows[-int(args.max_rows):]
    split_index = max(int(len(dynamic_rows) * 0.8), 32)
    split_index = min(split_index, len(dynamic_rows))
    ts_model = train_dynamic_ts_model(dynamic_rows[:split_index], Path(args.model_output), l2=float(args.l2))

    windows = build_dynamic_windows(glucose_points, events)
    if args.max_windows > 0:
        windows = windows[-int(args.max_windows):]
    valid_windows = windows[max(int(len(windows) * 0.8), 1):] if windows else []
    personal_params = _load_personal_params()

    phys_metrics = []
    ts_metrics = []
    hybrid_metrics = []
    examples = []

    for window in valid_windows:
        observed = np.asarray(window["observed_glucose"], dtype=float)
        grid = np.asarray(window["grid_minutes"], dtype=float)
        if len(observed) < 5:
            continue
        forecast_start_index = 4
        ts_rollout = rollout_dynamic_ts_window(ts_model, window)
        personal_window = _dynamic_window_to_personal(window)
        sim = simulate_personal_model(personal_window, personal_params, dt_minutes=1.0)
        phys_pred = np.interp(grid, sim["t_minutes"], sim["cgm_glucose_mg_dl"]).astype(float)
        ts_pred = np.asarray(ts_rollout["predicted_glucose"], dtype=float)
        weight_ts, regime = _regime_weight(observed, forecast_start_index)
        hybrid_pred = phys_pred.copy()
        hybrid_pred[: forecast_start_index + 1] = observed[: forecast_start_index + 1]
        hybrid_pred[forecast_start_index + 1 :] = (
            (1.0 - weight_ts) * phys_pred[forecast_start_index + 1 :]
            + weight_ts * ts_pred[forecast_start_index + 1 :]
        )
        suffix = slice(forecast_start_index + 1, None)
        phys_errors = phys_pred[suffix] - observed[suffix]
        ts_errors = ts_pred[suffix] - observed[suffix]
        hybrid_errors = hybrid_pred[suffix] - observed[suffix]
        phys_metrics.append((float(np.sqrt(np.mean(phys_errors**2))), float(np.mean(np.abs(phys_errors)))))
        ts_metrics.append((float(np.sqrt(np.mean(ts_errors**2))), float(np.mean(np.abs(ts_errors)))))
        hybrid_metrics.append((float(np.sqrt(np.mean(hybrid_errors**2))), float(np.mean(np.abs(hybrid_errors)))))
        examples.append(
            {
                "start_timestamp": window["start_timestamp"],
                "grid_minutes": grid.tolist(),
                "observed_glucose": observed.tolist(),
                "physiological_glucose": phys_pred.tolist(),
                "takagi_sugeno_glucose": ts_pred.tolist(),
                "hybrid_glucose": hybrid_pred.tolist(),
                "weight_ts": float(weight_ts),
                "regime": regime,
                "phys_rmse_mg_dl": phys_metrics[-1][0],
                "ts_rmse_mg_dl": ts_metrics[-1][0],
                "hybrid_rmse_mg_dl": hybrid_metrics[-1][0],
            }
        )

    payload = {
        "csv_path": str(csv_path),
        "windows_total": len(windows),
        "valid_windows": len(valid_windows),
        "model_status": ts_model.get("status"),
        "physiological_rollout": {
            "rmse_mg_dl": float(np.mean([item[0] for item in phys_metrics])) if phys_metrics else None,
            "mae_mg_dl": float(np.mean([item[1] for item in phys_metrics])) if phys_metrics else None,
        },
        "takagi_sugeno_rollout": {
            "rmse_mg_dl": float(np.mean([item[0] for item in ts_metrics])) if ts_metrics else None,
            "mae_mg_dl": float(np.mean([item[1] for item in ts_metrics])) if ts_metrics else None,
        },
        "hybrid_rollout": {
            "rmse_mg_dl": float(np.mean([item[0] for item in hybrid_metrics])) if hybrid_metrics else None,
            "mae_mg_dl": float(np.mean([item[1] for item in hybrid_metrics])) if hybrid_metrics else None,
        },
        "examples": examples[-2:],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

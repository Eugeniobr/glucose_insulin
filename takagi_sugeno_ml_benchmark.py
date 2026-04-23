import argparse
import json
from pathlib import Path

import numpy as np

from external_history import load_external_history
from predictor import predict_ridge_model, train_ridge_model
from takagi_sugeno_dynamic import (
    DYNAMIC_FEATURE_NAMES,
    build_dynamic_feature_payload,
    build_dynamic_rows,
    build_dynamic_windows,
    predict_dynamic_ts_delta,
    rollout_dynamic_ts_window,
    train_dynamic_ts_model,
)


RESIDUAL_FEATURE_NAMES = (
    "intercept",
    "current_glucose",
    "delta_15",
    "delta_30",
    "delta_60",
    "net_input_60",
    "mean_30",
    "std_30",
    "carbs_last_60",
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
    "ts_delta",
    "regime_alta_abrupta",
    "regime_queda_abrupta",
    "regime_estabilidade",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TS dinâmico com corretor residual por ML")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/takagi_sugeno_ml_benchmark.json")
    parser.add_argument("--ts-model-output", default="outputs/takagi_sugeno_dynamic_model.json")
    parser.add_argument("--max-rows", type=int, default=1200)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--ts-l2", type=float, default=8.0)
    parser.add_argument("--ml-l2", type=float, default=10.0)
    return parser


def _residual_feature_payload(feature_payload: dict, ts_delta: float) -> dict:
    regime = str(feature_payload.get("regime", "estabilidade"))
    return {
        "intercept": 1.0,
        "current_glucose": float(feature_payload.get("current_glucose", 0.0)),
        "delta_15": float(feature_payload.get("delta_15", 0.0)),
        "delta_30": float(feature_payload.get("delta_30", 0.0)),
        "delta_60": float(feature_payload.get("delta_60", 0.0)),
        "net_input_60": float(feature_payload.get("net_input_60", 0.0)),
        "mean_30": float(feature_payload.get("mean_30", 0.0)),
        "std_30": float(feature_payload.get("std_30", 0.0)),
        "carbs_last_60": float(feature_payload.get("carbs_last_60", 0.0)),
        "rapid_last_60": float(feature_payload.get("rapid_last_60", 0.0)),
        "basal_last_60": float(feature_payload.get("basal_last_60", 0.0)),
        "rapid_iob_60": float(feature_payload.get("rapid_iob_60", 0.0)),
        "rapid_action_60": float(feature_payload.get("rapid_action_60", 0.0)),
        "basal_iob_12h": float(feature_payload.get("basal_iob_12h", 0.0)),
        "basal_action_12h": float(feature_payload.get("basal_action_12h", 0.0)),
        "future_carbs_15": float(feature_payload.get("future_carbs_15", 0.0)),
        "future_rapid_15": float(feature_payload.get("future_rapid_15", 0.0)),
        "future_basal_15": float(feature_payload.get("future_basal_15", 0.0)),
        "future_rapid_action_15": float(feature_payload.get("future_rapid_action_15", 0.0)),
        "future_basal_action_15": float(feature_payload.get("future_basal_action_15", 0.0)),
        "minutes_since_meal": float(feature_payload.get("minutes_since_meal", -1.0)),
        "minutes_since_rapid": float(feature_payload.get("minutes_since_rapid", -1.0)),
        "minutes_since_basal": float(feature_payload.get("minutes_since_basal", -1.0)),
        "ts_delta": float(ts_delta),
        "regime_alta_abrupta": 1.0 if regime == "alta_abrupta" else 0.0,
        "regime_queda_abrupta": 1.0 if regime == "queda_abrupta" else 0.0,
        "regime_estabilidade": 1.0 if regime == "estabilidade" else 0.0,
    }


def _feature_vector(payload: dict) -> np.ndarray:
    return np.array([float(payload.get(name, 0.0)) for name in RESIDUAL_FEATURE_NAMES], dtype=float)


def _build_train_row(ts_model: dict, row: dict) -> tuple[np.ndarray, float]:
    feature_payload = {name: float(row.get(name, 0.0)) if name in DYNAMIC_FEATURE_NAMES else row.get(name) for name in row}
    ts_delta = predict_dynamic_ts_delta(ts_model, feature_payload)
    residual = float(row["target_delta_15"] - ts_delta)
    return _feature_vector(_residual_feature_payload(feature_payload, ts_delta)), residual


def _predict_ts_ml_delta(ts_model: dict, residual_model: dict, feature_payload: dict) -> float:
    ts_delta = predict_dynamic_ts_delta(ts_model, feature_payload)
    residual = predict_ridge_model(residual_model, _feature_vector(_residual_feature_payload(feature_payload, ts_delta)))
    return float(ts_delta + residual)


def _rollout_ts_ml_window(ts_model: dict, residual_model: dict, window: dict, step_minutes: int = 15) -> dict:
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
        predicted.append(current + _predict_ts_ml_delta(ts_model, residual_model, feature_payload))
    predicted = np.asarray(predicted, dtype=float)
    residuals = predicted - observed[: len(predicted)]
    return {
        "predicted_glucose": predicted.tolist(),
        "observed_glucose": observed[: len(predicted)].tolist(),
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
    }


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    rows = build_dynamic_rows(glucose_points, events)
    if args.max_rows > 0:
        rows = rows[-int(args.max_rows):]
    split_index = max(int(len(rows) * 0.8), 32)
    split_index = min(split_index, len(rows))
    train_rows = rows[:split_index]
    valid_rows = rows[split_index:]

    ts_model = train_dynamic_ts_model(train_rows, Path(args.ts_model_output), l2=float(args.ts_l2))
    train_pairs = [_build_train_row(ts_model, row) for row in train_rows]
    train_features = np.asarray([item[0] for item in train_pairs], dtype=float)
    train_targets = np.asarray([item[1] for item in train_pairs], dtype=float)
    residual_model = train_ridge_model(train_features, train_targets, float(args.ml_l2))

    ts_preds = []
    ts_ml_preds = []
    valid_targets = []
    for row in valid_rows:
      feature_payload = {name: float(row.get(name, 0.0)) if name in DYNAMIC_FEATURE_NAMES else row.get(name) for name in row}
      ts_delta = predict_dynamic_ts_delta(ts_model, feature_payload)
      ts_ml_delta = _predict_ts_ml_delta(ts_model, residual_model, feature_payload)
      ts_preds.append(float(row["current_glucose"]) + ts_delta)
      ts_ml_preds.append(float(row["current_glucose"]) + ts_ml_delta)
      valid_targets.append(float(row["current_glucose"] + row["target_delta_15"]))
    ts_preds = np.asarray(ts_preds, dtype=float)
    ts_ml_preds = np.asarray(ts_ml_preds, dtype=float)
    valid_targets = np.asarray(valid_targets, dtype=float)

    windows = build_dynamic_windows(glucose_points, events)
    if args.max_windows > 0:
        windows = windows[-int(args.max_windows):]
    valid_windows = windows[max(int(len(windows) * 0.8), 1):] if windows else []
    ts_rollouts = []
    ts_ml_rollouts = []
    examples = []
    for window in valid_windows:
        ts_result = rollout_dynamic_ts_window(ts_model, window)
        ts_ml_result = _rollout_ts_ml_window(ts_model, residual_model, window)
        if ts_result["rmse_mg_dl"] is None or ts_ml_result["rmse_mg_dl"] is None:
            continue
        ts_rollouts.append((ts_result["rmse_mg_dl"], ts_result["mae_mg_dl"]))
        ts_ml_rollouts.append((ts_ml_result["rmse_mg_dl"], ts_ml_result["mae_mg_dl"]))
        examples.append(
            {
                "start_timestamp": window["start_timestamp"],
                "grid_minutes": window["grid_minutes"],
                "observed_glucose": ts_result["observed_glucose"],
                "takagi_sugeno_glucose": ts_result["predicted_glucose"],
                "takagi_sugeno_ml_glucose": ts_ml_result["predicted_glucose"],
                "ts_rmse_mg_dl": ts_result["rmse_mg_dl"],
                "ts_ml_rmse_mg_dl": ts_ml_result["rmse_mg_dl"],
            }
        )

    payload = {
        "csv_path": str(csv_path),
        "rows_total": len(rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "windows_total": len(windows),
        "valid_windows": len(valid_windows),
        "ts_model_status": ts_model.get("status"),
        "one_step_valid": {
            "takagi_sugeno": {
                "rmse_mg_dl": float(np.sqrt(np.mean((ts_preds - valid_targets) ** 2))) if len(valid_targets) else None,
                "mae_mg_dl": float(np.mean(np.abs(ts_preds - valid_targets))) if len(valid_targets) else None,
            },
            "takagi_sugeno_ml": {
                "rmse_mg_dl": float(np.sqrt(np.mean((ts_ml_preds - valid_targets) ** 2))) if len(valid_targets) else None,
                "mae_mg_dl": float(np.mean(np.abs(ts_ml_preds - valid_targets))) if len(valid_targets) else None,
            },
        },
        "rollout_valid": {
            "takagi_sugeno": {
                "rmse_mg_dl": float(np.mean([item[0] for item in ts_rollouts])) if ts_rollouts else None,
                "mae_mg_dl": float(np.mean([item[1] for item in ts_rollouts])) if ts_rollouts else None,
            },
            "takagi_sugeno_ml": {
                "rmse_mg_dl": float(np.mean([item[0] for item in ts_ml_rollouts])) if ts_ml_rollouts else None,
                "mae_mg_dl": float(np.mean([item[1] for item in ts_ml_rollouts])) if ts_ml_rollouts else None,
            },
        },
        "examples": examples[-2:],
        "residual_model": {
            "feature_names": list(RESIDUAL_FEATURE_NAMES),
            "feature_means": residual_model["feature_means"].tolist(),
            "feature_scales": residual_model["feature_scales"].tolist(),
            "weights": residual_model["weights"].tolist(),
            "train_target_rmse_mg_dl": float(residual_model["rmse"]),
            "train_target_mae_mg_dl": float(residual_model["mae"]),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import numpy as np

from external_history import load_external_history
from takagi_sugeno_dynamic import (
    build_dynamic_rows,
    build_dynamic_windows,
    predict_dynamic_ts_delta,
    rollout_dynamic_ts_window,
    train_dynamic_ts_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark de reconstrução dinâmica com Takagi-Sugeno fuzzy")
    parser.add_argument("csv_path")
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--output", default="outputs/takagi_sugeno_dynamic_benchmark.json")
    parser.add_argument("--model-output", default="outputs/takagi_sugeno_dynamic_model.json")
    parser.add_argument("--max-rows", type=int, default=1600)
    parser.add_argument("--max-windows", type=int, default=12)
    return parser


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
    model = train_dynamic_ts_model(train_rows, Path(args.model_output), l2=float(args.l2))

    windows = build_dynamic_windows(glucose_points, events)
    if args.max_windows > 0:
        windows = windows[-int(args.max_windows):]
    valid_windows = windows[max(int(len(windows) * 0.8), 1):] if windows else []
    rollout_metrics = []
    rollout_examples = []
    for window in valid_windows:
        result = rollout_dynamic_ts_window(model, window)
        if result["rmse_mg_dl"] is not None:
            rollout_metrics.append(result)
            rollout_examples.append(
                {
                    "start_timestamp": window["start_timestamp"],
                    "grid_minutes": window["grid_minutes"],
                    "predicted_glucose": result["predicted_glucose"],
                    "observed_glucose": result["observed_glucose"],
                    "rmse_mg_dl": result["rmse_mg_dl"],
                    "mae_mg_dl": result["mae_mg_dl"],
                }
            )

    payload = {
        "csv_path": str(csv_path),
        "rows_total": len(rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "windows_total": len(windows),
        "valid_windows": len(valid_windows),
        "model_status": model.get("status"),
        "one_step_valid": {},
        "rollout_valid": {},
    }
    if valid_rows:
        predictions = np.asarray(
            [float(row["current_glucose"]) + predict_dynamic_ts_delta(model, row) for row in valid_rows],
            dtype=float,
        )
        targets = np.asarray([float(row["current_glucose"] + row["target_delta_15"]) for row in valid_rows], dtype=float)
        residuals = predictions - targets
        payload["one_step_valid"] = {
            "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
            "mae_mg_dl": float(np.mean(np.abs(residuals))),
        }
    if rollout_metrics:
        payload["rollout_valid"] = {
            "rmse_mg_dl": float(np.mean([item["rmse_mg_dl"] for item in rollout_metrics])),
            "mae_mg_dl": float(np.mean([item["mae_mg_dl"] for item in rollout_metrics])),
        }
        payload["rollout_examples"] = rollout_examples[-2:]
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

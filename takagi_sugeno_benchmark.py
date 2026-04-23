import argparse
import json
from pathlib import Path

import numpy as np

from external_history import build_direct_forecast_samples, load_external_history
from simple_predictor import SIMPLE_FEATURE_NAMES, train_simple_forecast_model
from takagi_sugeno_predictor import predict_takagi_sugeno_delta, train_takagi_sugeno_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark do modelo fuzzy Takagi-Sugeno")
    parser.add_argument("csv_path")
    parser.add_argument("--min-points", type=int, default=192)
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--output", default="outputs/takagi_sugeno_benchmark.json")
    parser.add_argument("--model-output", default="outputs/takagi_sugeno_model.json")
    parser.add_argument("--simple-model-output", default="outputs/simple_forecast_model_ts_compare.json")
    parser.add_argument("--max-rows-per-horizon", type=int, default=720)
    return parser


def _naive_metrics(rows: list[dict]) -> dict:
    current = np.asarray([float(row["current_glucose"]) for row in rows], dtype=float)
    target = np.asarray([float(row["target_glucose"]) for row in rows], dtype=float)
    residuals = current - target
    return {
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "mae_mg_dl": float(np.mean(np.abs(residuals))) if len(residuals) else None,
    }


def _eval_simple(horizon_payload: dict, rows: list[dict]) -> dict:
    if horizon_payload.get("status") != "trained":
        return {"status": horizon_payload.get("status", "unavailable"), "rmse_mg_dl": None, "mae_mg_dl": None}
    means = np.asarray(horizon_payload["feature_means"], dtype=float)
    scales = np.asarray(horizon_payload["feature_scales"], dtype=float)
    weights = np.asarray(horizon_payload["weights"], dtype=float)
    features = np.asarray([[float(row.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES] for row in rows], dtype=float)
    targets = np.asarray([float(row["target_delta"]) for row in rows], dtype=float)
    predictions = ((features - means) / scales) @ weights
    residuals = predictions - targets
    return {
        "status": "trained",
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "mae_mg_dl": float(np.mean(np.abs(residuals))) if len(residuals) else None,
    }


def _eval_ts(horizon_payload: dict, rows: list[dict]) -> dict:
    if horizon_payload.get("status") != "trained":
        return {"status": horizon_payload.get("status", "unavailable"), "rmse_mg_dl": None, "mae_mg_dl": None}
    predictions = np.asarray([predict_takagi_sugeno_delta(horizon_payload, row) for row in rows], dtype=float)
    targets = np.asarray([float(row["target_delta"]) for row in rows], dtype=float)
    residuals = predictions - targets
    return {
        "status": "trained",
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "mae_mg_dl": float(np.mean(np.abs(residuals))) if len(residuals) else None,
    }


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    samples = build_direct_forecast_samples(glucose_points, events)
    if args.max_rows_per_horizon > 0:
        filtered = []
        for horizon in (15, 30, 60, 120):
            rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
            filtered.extend(rows[-int(args.max_rows_per_horizon):])
        samples = sorted(filtered, key=lambda item: (int(item["horizon_minutes"]), item["timestamp"]))

    ts_model = train_takagi_sugeno_model(
        samples=samples,
        model_path=Path(args.model_output),
        min_points=int(args.min_points),
        l2=float(args.l2),
    )
    simple_model = train_simple_forecast_model(
        samples=samples,
        model_path=Path(args.simple_model_output),
        min_points=int(args.min_points),
        l2=float(args.l2),
    )

    benchmark = {
        "csv_path": str(csv_path),
        "samples_total": int(len(samples)),
        "min_points": int(args.min_points),
        "l2": float(args.l2),
        "max_rows_per_horizon": int(args.max_rows_per_horizon),
        "horizons": {},
    }
    for horizon in (15, 30, 60, 120):
        rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
        split_index = max(int(len(rows) * 0.8), int(args.min_points))
        split_index = min(split_index, max(len(rows) - 1, 0))
        valid_rows = rows[split_index:] if rows and split_index < len(rows) else []
        benchmark["horizons"][str(horizon)] = {
            "points_total": len(rows),
            "points_valid": len(valid_rows),
            "naive_persistence": _naive_metrics(valid_rows),
            "simple_model": _eval_simple(simple_model.get("horizons", {}).get(str(horizon), {}), valid_rows),
            "takagi_sugeno_fuzzy": _eval_ts(ts_model.get("horizons", {}).get(str(horizon), {}), valid_rows),
            "takagi_sugeno_summary": ts_model.get("horizons", {}).get(str(horizon), {}),
        }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import numpy as np

from config import load_config
from external_history import build_direct_forecast_samples, load_external_history
from simple_predictor import SIMPLE_FEATURE_NAMES, train_simple_forecast_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark walk-forward do previsor simples")
    parser.add_argument("csv_path")
    parser.add_argument("--min-points", type=int, default=192)
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--output", default="outputs/simple_forecast_benchmark.json")
    return parser


def _naive_metrics(rows: list[dict]) -> dict:
    current = np.asarray([float(row["current_glucose"]) for row in rows], dtype=float)
    target = np.asarray([float(row["target_glucose"]) for row in rows], dtype=float)
    residuals = current - target
    return {
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
    }


def _eval_model(horizon_payload: dict, rows: list[dict]) -> dict:
    if horizon_payload.get("status") != "trained":
        return {
            "status": horizon_payload.get("status", "unavailable"),
            "rmse_mg_dl": None,
            "mae_mg_dl": None,
        }
    means = np.asarray(horizon_payload["feature_means"], dtype=float)
    scales = np.asarray(horizon_payload["feature_scales"], dtype=float)
    weights = np.asarray(horizon_payload["weights"], dtype=float)
    features = np.asarray([[float(row.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES] for row in rows], dtype=float)
    target_delta = np.asarray([float(row["target_delta"]) for row in rows], dtype=float)
    normalized = (features - means) / scales
    predicted_delta = normalized @ weights
    residuals = predicted_delta - target_delta
    return {
        "status": "trained",
        "rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
        "mae_mg_dl": float(np.mean(np.abs(residuals))),
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    samples = build_direct_forecast_samples(glucose_points, events)
    model_payload = train_simple_forecast_model(
        samples=samples,
        model_path=config.simple_forecast_model_path,
        min_points=int(args.min_points),
        l2=float(args.l2),
    )

    benchmark = {
        "csv_path": str(csv_path),
        "samples_total": int(len(samples)),
        "min_points": int(args.min_points),
        "l2": float(args.l2),
        "horizons": {},
    }
    for horizon in (15, 30, 60, 120):
        rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
        split_index = max(int(len(rows) * 0.8), int(args.min_points))
        split_index = min(split_index, max(len(rows) - 1, 0))
        valid_rows = rows[split_index:] if rows and split_index < len(rows) else []
        naive = _naive_metrics(valid_rows) if valid_rows else {"rmse_mg_dl": None, "mae_mg_dl": None}
        simple = _eval_model(model_payload.get("horizons", {}).get(str(horizon), {}), valid_rows)
        benchmark["horizons"][str(horizon)] = {
            "points_total": len(rows),
            "points_valid": len(valid_rows),
            "naive_persistence": naive,
            "simple_model": simple,
            "model_summary": model_payload.get("horizons", {}).get(str(horizon), {}),
        }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

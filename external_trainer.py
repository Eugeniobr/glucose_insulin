import argparse
import json
from pathlib import Path

import numpy as np

from config import load_config
from data_io import write_json
from external_history import build_external_training_samples, load_external_history, summarize_external_history
from predictor import predict_ridge_model, train_ridge_model


EXTERNAL_FEATURE_NAMES = (
    "intercept",
    "current_glucose",
    "delta_15",
    "delta_30",
    "delta_60",
    "carbs_last_30",
    "carbs_last_60",
    "rapid_last_30",
    "rapid_last_60",
    "rapid_last_120",
    "future_carbs_total",
    "future_rapid_total",
    "minutes_since_event",
    "hour_sin",
    "hour_cos",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino offline com histórico externo")
    parser.add_argument("csv_path")
    parser.add_argument("--min-points", type=int, default=24)
    parser.add_argument("--l2", type=float, default=8.0)
    return parser


def _feature_vector(sample: dict) -> np.ndarray:
    angle = 2.0 * np.pi * float(sample["hour_of_day"]) / 24.0
    payload = {
        "intercept": 1.0,
        "current_glucose": float(sample["current_glucose"]),
        "delta_15": float(sample["delta_15"]),
        "delta_30": float(sample["delta_30"]),
        "delta_60": float(sample["delta_60"]),
        "carbs_last_30": float(sample["carbs_last_30"]),
        "carbs_last_60": float(sample["carbs_last_60"]),
        "rapid_last_30": float(sample["rapid_last_30"]),
        "rapid_last_60": float(sample["rapid_last_60"]),
        "rapid_last_120": float(sample["rapid_last_120"]),
        "future_carbs_total": float(sample["future_carbs_total"]),
        "future_rapid_total": float(sample["future_rapid_total"]),
        "minutes_since_event": float(sample["minutes_since_event"]),
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
    }
    return np.array([payload[name] for name in EXTERNAL_FEATURE_NAMES], dtype=float)


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    csv_path = Path(args.csv_path).resolve()

    glucose_points, events = load_external_history(csv_path)
    summary = summarize_external_history(glucose_points, events)
    samples = build_external_training_samples(glucose_points, events)

    model_payload = {
        "source_csv": str(csv_path),
        "updated_at": None,
        "feature_names": list(EXTERNAL_FEATURE_NAMES),
        "min_points": int(args.min_points),
        "l2": float(args.l2),
        "horizons": {},
    }

    horizon_report = {}
    for horizon in (15, 30, 60, 120):
        rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
        if len(rows) < args.min_points:
            model_payload["horizons"][str(horizon)] = {"status": "insufficient_data", "points_used": len(rows)}
            continue
        features = np.asarray([_feature_vector(sample) for sample in rows], dtype=float)
        targets = np.asarray([float(sample["target_glucose"]) for sample in rows], dtype=float)
        model = train_ridge_model(features, targets, args.l2)
        predictions = np.asarray([predict_ridge_model(model, vector) for vector in features], dtype=float)
        residuals = predictions - targets
        horizon_payload = {
            "status": "trained",
            "points_used": int(len(rows)),
            "feature_means": model["feature_means"].tolist(),
            "feature_scales": model["feature_scales"].tolist(),
            "weights": model["weights"].tolist(),
            "target_rmse_mg_dl": float(np.sqrt(np.mean(residuals**2))),
            "target_mae_mg_dl": float(np.mean(np.abs(residuals))),
        }
        model_payload["horizons"][str(horizon)] = horizon_payload
        horizon_report[str(horizon)] = {
            "points_used": int(len(rows)),
            "target_rmse_mg_dl": horizon_payload["target_rmse_mg_dl"],
            "target_mae_mg_dl": horizon_payload["target_mae_mg_dl"],
        }

    model_payload["updated_at"] = json.loads(json.dumps(summary.get("last_timestamp"))) if summary else None
    report_payload = {
        "csv_path": str(csv_path),
        "summary": summary,
        "training_samples": int(len(samples)),
        "horizon_report": horizon_report,
        "recommended_initial_cho_per_unit": summary.get("cho_per_unit_median") if summary else None,
    }

    write_json(config.external_history_report_path, report_payload)
    write_json(config.external_forecast_model_path, model_payload)
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

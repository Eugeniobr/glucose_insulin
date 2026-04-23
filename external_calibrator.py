import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from config import load_config
from data_io import write_json
from external_history import load_external_history, summarize_external_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibração fisiológica offline com histórico externo")
    parser.add_argument("csv_path")
    parser.add_argument("--reference-cho-per-unit", type=float, default=8.0)
    parser.add_argument("--min-paired-events", type=int, default=12)
    parser.add_argument("--min-isolated-rapid-events", type=int, default=3)
    parser.add_argument("--prior-blend", type=float, default=0.45)
    return parser


def _timestamps_and_values(glucose_points):
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    return timestamps, values


def _interp_glucose(glucose_timestamps: np.ndarray, glucose_values: np.ndarray, timestamp) -> float | None:
    target = timestamp.timestamp()
    if target < glucose_timestamps[0] or target > glucose_timestamps[-1]:
        return None
    return float(np.interp(target, glucose_timestamps, glucose_values))


def _window_mask(glucose_timestamps: np.ndarray, start_ts, end_ts):
    start_seconds = start_ts.timestamp()
    end_seconds = end_ts.timestamp()
    return (glucose_timestamps >= start_seconds) & (glucose_timestamps <= end_seconds)


def _paired_event_metrics(glucose_points, events):
    timestamps, values = _timestamps_and_values(glucose_points)
    paired = []
    for event in events:
        if event.carbs_g <= 0.0 or event.rapid_units <= 0.0:
            continue
        baseline_time = event.timestamp - timedelta(minutes=15)
        end_time = event.timestamp + timedelta(hours=4)
        baseline = _interp_glucose(timestamps, values, baseline_time)
        if baseline is None:
            baseline = _interp_glucose(timestamps, values, event.timestamp)
        if baseline is None:
            continue
        mask = _window_mask(timestamps, event.timestamp, end_time)
        if int(mask.sum()) < 10:
            continue
        window_times = timestamps[mask]
        window_values = values[mask]
        peak_index = int(np.argmax(window_values))
        peak_value = float(window_values[peak_index])
        peak_minutes = float((window_times[peak_index] - event.timestamp.timestamp()) / 60.0)
        paired.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "carbs_g": float(event.carbs_g),
                "rapid_units": float(event.rapid_units),
                "cho_per_unit": float(event.carbs_g / max(event.rapid_units, 1e-6)),
                "baseline_glucose_mg_dl": float(baseline),
                "peak_glucose_mg_dl": peak_value,
                "peak_increment_mg_dl": float(peak_value - baseline),
                "peak_minutes": peak_minutes,
                "glucose_2h_mg_dl": _interp_glucose(timestamps, values, event.timestamp + timedelta(hours=2)),
            }
        )
    return paired


def _isolated_rapid_metrics(glucose_points, events):
    timestamps, values = _timestamps_and_values(glucose_points)
    isolated = []
    for event in events:
        if event.rapid_units <= 0.0 or event.carbs_g > 0.0:
            continue
        has_nearby_carbs = any(
            other.carbs_g > 0.0 and abs((other.timestamp - event.timestamp).total_seconds()) <= 120 * 60
            for other in events
        )
        if has_nearby_carbs:
            continue
        baseline = _interp_glucose(timestamps, values, event.timestamp)
        if baseline is None:
            continue
        start_time = event.timestamp + timedelta(minutes=30)
        end_time = event.timestamp + timedelta(hours=4)
        mask = _window_mask(timestamps, start_time, end_time)
        if int(mask.sum()) < 8:
            continue
        window_times = timestamps[mask]
        window_values = values[mask]
        nadir_index = int(np.argmin(window_values))
        nadir_value = float(window_values[nadir_index])
        nadir_minutes = float((window_times[nadir_index] - event.timestamp.timestamp()) / 60.0)
        isolated.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "rapid_units": float(event.rapid_units),
                "baseline_glucose_mg_dl": float(baseline),
                "nadir_glucose_mg_dl": nadir_value,
                "drop_mg_dl": float(baseline - nadir_value),
                "drop_per_unit_mg_dl": float((baseline - nadir_value) / max(event.rapid_units, 1e-6)),
                "nadir_minutes": nadir_minutes,
            }
        )
    return isolated


def _derive_overrides(summary: dict, paired_metrics: list[dict], isolated_metrics: list[dict], args) -> dict:
    cho_per_unit = summary.get("cho_per_unit_median")
    paired_count = len(paired_metrics)
    isolated_count = len(isolated_metrics)
    prior_blend = float(np.clip(args.prior_blend, 0.0, 1.0))
    reference_ratio = max(float(args.reference_cho_per_unit), 1e-6)
    overrides = {}

    if cho_per_unit is not None and paired_count >= int(args.min_paired_events):
        patient_ratio = max(float(cho_per_unit), 1e-6)
        raw_meal_scale = float(np.clip(np.sqrt(reference_ratio / patient_ratio), 0.85, 1.25))
        raw_rapid_bio_scale = float(np.clip(np.sqrt(patient_ratio / reference_ratio), 0.8, 1.1))
        overrides["meal_carb_scale"] = float(1.0 + prior_blend * (raw_meal_scale - 1.0))
        overrides["insulin_rapid_bio_scale"] = float(1.0 + prior_blend * (raw_rapid_bio_scale - 1.0))
        overrides["insulin_unknown_bio_scale"] = overrides["insulin_rapid_bio_scale"]

    if paired_count >= int(args.min_paired_events):
        median_peak_minutes = float(np.median([item["peak_minutes"] for item in paired_metrics]))
        raw_meal_tau = float(np.clip(0.48 * median_peak_minutes, 25.0, 70.0))
        overrides["meal_tau"] = float(40.0 + prior_blend * (raw_meal_tau - 40.0))

    if isolated_count >= int(args.min_isolated_rapid_events):
        median_nadir_minutes = float(np.median([item["nadir_minutes"] for item in isolated_metrics]))
        raw_rapid_tau = float(np.clip(0.65 * median_nadir_minutes, 55.0, 110.0))
        overrides["insulin_rapid_tau"] = float(75.0 + prior_blend * (raw_rapid_tau - 75.0))
        median_drop_per_unit = float(np.median([item["drop_per_unit_mg_dl"] for item in isolated_metrics]))
        raw_sensitivity = float(np.clip(median_drop_per_unit / 30.0, 0.75, 1.2))
        overrides["insulin_sensitivity_scale"] = float(1.0 + 0.5 * prior_blend * (raw_sensitivity - 1.0))

    return overrides


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    summary = summarize_external_history(glucose_points, events)
    paired_metrics = _paired_event_metrics(glucose_points, events)
    isolated_metrics = _isolated_rapid_metrics(glucose_points, events)
    overrides = _derive_overrides(summary, paired_metrics, isolated_metrics, args)

    payload = {
        "csv_path": str(csv_path),
        "generated_at": summary.get("last_timestamp"),
        "summary": summary,
        "paired_event_analysis": {
            "count": len(paired_metrics),
            "median_peak_minutes": float(np.median([item["peak_minutes"] for item in paired_metrics])) if paired_metrics else None,
            "median_peak_increment_mg_dl": float(np.median([item["peak_increment_mg_dl"] for item in paired_metrics])) if paired_metrics else None,
        },
        "isolated_rapid_analysis": {
            "count": len(isolated_metrics),
            "median_nadir_minutes": float(np.median([item["nadir_minutes"] for item in isolated_metrics])) if isolated_metrics else None,
            "median_drop_per_unit_mg_dl": float(np.median([item["drop_per_unit_mg_dl"] for item in isolated_metrics])) if isolated_metrics else None,
        },
        "recommended_overrides": overrides,
        "settings": {
            "reference_cho_per_unit": float(args.reference_cho_per_unit),
            "min_paired_events": int(args.min_paired_events),
            "min_isolated_rapid_events": int(args.min_isolated_rapid_events),
            "prior_blend": float(args.prior_blend),
        },
    }
    write_json(config.external_physiology_priors_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

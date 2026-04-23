import argparse
import json
from pathlib import Path

import numpy as np

from config import load_config
from data_io import write_json
from external_history import load_external_history, summarize_external_history
from personal_model import (
    PersonalModelParams,
    build_personal_model_windows,
    build_walk_forward_splits,
    fit_personal_model,
    simulate_personal_model,
)
from predictor import predict_ridge_model, train_ridge_model


RESIDUAL_FEATURE_NAMES = (
    "intercept",
    "baseline_glucose",
    "carbs_g",
    "rapid_units",
    "basal_units",
    "cho_per_unit",
    "phys_pred",
    "phys_delta",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protótipo residual em cima do modelo fisiológico personalizado")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/personal_model_residual_prototype.json")
    parser.add_argument("--l2", type=float, default=24.0)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--min-train", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--min-regime-points", type=int, default=3)
    return parser


def _extract_horizon_rows(windows, params_payload: dict, horizon_minutes: int) -> list[dict]:
    params = PersonalModelParams(**params_payload)
    rows = []
    for window in windows:
        sim = simulate_personal_model(window, params)
        predicted = float(np.interp(horizon_minutes, sim["t_minutes"], sim["cgm_glucose_mg_dl"]))
        observed = float(np.interp(horizon_minutes, window["sample_minutes"], window["sample_glucose_mg_dl"]))
        baseline = float(window["baseline_glucose_mg_dl"])
        rapid_units = float(window.get("rapid_units", 0.0))
        rows.append(
            {
                "regime": str(window.get("regime", {}).get("name", "desconhecido")),
                "baseline_glucose": baseline,
                "carbs_g": float(window.get("carbs_g", 0.0)),
                "rapid_units": rapid_units,
                "basal_units": float(window.get("basal_units", 0.0)),
                "cho_per_unit": float(window.get("carbs_g", 0.0) / max(rapid_units, 1e-6)) if rapid_units > 0 else 0.0,
                "phys_pred": predicted,
                "phys_delta": predicted - baseline,
                "target_glucose": observed,
                "phys_residual": observed - predicted,
            }
        )
    return rows


def _feature_vector(row: dict) -> np.ndarray:
    return np.array([float(row.get(name, 0.0)) for name in RESIDUAL_FEATURE_NAMES], dtype=float)


def _train_regime_models(rows: list[dict], l2: float, min_regime_points: int) -> dict:
    payload = {"global": None, "by_regime": {}}
    if not rows:
        return payload
    features = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    targets = np.asarray([float(row["phys_residual"]) for row in rows], dtype=float)
    payload["global"] = train_ridge_model(features, targets, l2)
    regime_names = sorted({str(row["regime"]) for row in rows})
    for name in regime_names:
        subset = [row for row in rows if str(row["regime"]) == name]
        if len(subset) < int(min_regime_points):
            continue
        regime_features = np.asarray([_feature_vector(row) for row in subset], dtype=float)
        regime_targets = np.asarray([float(row["phys_residual"]) for row in subset], dtype=float)
        payload["by_regime"][name] = train_ridge_model(regime_features, regime_targets, l2)
    return payload


def _predict_corrected(row: dict, model_bundle: dict) -> float:
    model = model_bundle.get("by_regime", {}).get(str(row.get("regime"))) or model_bundle.get("global")
    if not model:
        return float(row["phys_pred"])
    residual = predict_ridge_model(model, _feature_vector(row))
    return float(row["phys_pred"] + residual)


def _evaluate_rows(rows: list[dict], model_bundle: dict) -> dict:
    if not rows:
        return {}
    observed = np.asarray([float(row["target_glucose"]) for row in rows], dtype=float)
    raw = np.asarray([float(row["phys_pred"]) for row in rows], dtype=float)
    corrected = np.asarray([_predict_corrected(row, model_bundle) for row in rows], dtype=float)
    raw_errors = raw - observed
    corrected_errors = corrected - observed
    by_regime = {}
    for regime in sorted({str(row["regime"]) for row in rows}):
        subset_idx = [index for index, row in enumerate(rows) if str(row["regime"]) == regime]
        subset_raw = raw_errors[subset_idx]
        subset_corrected = corrected_errors[subset_idx]
        by_regime[regime] = {
            "rows": len(subset_idx),
            "phys_rmse_mg_dl": float(np.sqrt(np.mean(subset_raw**2))),
            "residual_rmse_mg_dl": float(np.sqrt(np.mean(subset_corrected**2))),
            "phys_mae_mg_dl": float(np.mean(np.abs(subset_raw))),
            "residual_mae_mg_dl": float(np.mean(np.abs(subset_corrected))),
        }
    return {
        "rows": len(rows),
        "phys_rmse_mg_dl": float(np.sqrt(np.mean(raw_errors**2))),
        "phys_mae_mg_dl": float(np.mean(np.abs(raw_errors))),
        "residual_rmse_mg_dl": float(np.sqrt(np.mean(corrected_errors**2))),
        "residual_mae_mg_dl": float(np.mean(np.abs(corrected_errors))),
        "by_regime": by_regime,
    }


def _summarize_horizon_folds(folds: list[dict]) -> dict:
    phys = [float(item["valid"]["phys_rmse_mg_dl"]) for item in folds if item.get("valid", {}).get("phys_rmse_mg_dl") is not None]
    residual = [float(item["valid"]["residual_rmse_mg_dl"]) for item in folds if item.get("valid", {}).get("residual_rmse_mg_dl") is not None]
    phys_mae = [float(item["valid"]["phys_mae_mg_dl"]) for item in folds if item.get("valid", {}).get("phys_mae_mg_dl") is not None]
    residual_mae = [float(item["valid"]["residual_mae_mg_dl"]) for item in folds if item.get("valid", {}).get("residual_mae_mg_dl") is not None]
    regime_summary: dict[str, list[tuple[float, float, float, float]]] = {}
    for fold in folds:
        for name, metrics in fold.get("valid", {}).get("by_regime", {}).items():
            regime_summary.setdefault(name, []).append(
                (
                    float(metrics["phys_rmse_mg_dl"]),
                    float(metrics["residual_rmse_mg_dl"]),
                    float(metrics["phys_mae_mg_dl"]),
                    float(metrics["residual_mae_mg_dl"]),
                )
            )
    return {
        "folds": len(folds),
        "phys_rmse_mg_dl": float(np.mean(phys)) if phys else None,
        "residual_rmse_mg_dl": float(np.mean(residual)) if residual else None,
        "phys_mae_mg_dl": float(np.mean(phys_mae)) if phys_mae else None,
        "residual_mae_mg_dl": float(np.mean(residual_mae)) if residual_mae else None,
        "by_regime": {
            name: {
                "folds": len(values),
                "phys_rmse_mg_dl": float(np.mean([item[0] for item in values])),
                "residual_rmse_mg_dl": float(np.mean([item[1] for item in values])),
                "phys_mae_mg_dl": float(np.mean([item[2] for item in values])),
                "residual_mae_mg_dl": float(np.mean([item[3] for item in values])),
            }
            for name, values in regime_summary.items()
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    summary = summarize_external_history(glucose_points, events)
    windows = build_personal_model_windows(glucose_points, events, horizon_minutes=240)
    windows = sorted(windows, key=lambda item: item["timestamp"])
    if args.max_windows > 0:
        windows = windows[: int(args.max_windows)]

    splits = build_walk_forward_splits(
        windows,
        min_train_windows=int(args.min_train),
        validation_windows=int(args.valid_size),
        step_windows=int(args.step),
    )
    payload = {
        "csv_path": str(csv_path),
        "summary": summary,
        "windows_total": len(windows),
        "walk_forward": {
            "min_train_windows": int(args.min_train),
            "validation_windows": int(args.valid_size),
            "step_windows": int(args.step),
        },
        "l2": float(args.l2),
        "horizons": {},
    }

    horizons = (15, 30)
    horizon_folds: dict[int, list[dict]] = {horizon: [] for horizon in horizons}
    latest_fit = None
    for fold_index, split in enumerate(splits, start=1):
        fitted = fit_personal_model(split["train_windows"], initial_params=PersonalModelParams())
        latest_fit = fitted
        if fitted.get("status") != "fitted":
            continue
        for horizon in horizons:
            train_rows = _extract_horizon_rows(split["train_windows"], fitted["params"], horizon)
            valid_rows = _extract_horizon_rows(split["valid_windows"], fitted["params"], horizon)
            model_bundle = _train_regime_models(train_rows, float(args.l2), int(args.min_regime_points))
            horizon_folds[horizon].append(
                {
                    "fold": fold_index,
                    "train_windows": len(split["train_windows"]),
                    "valid_windows": len(split["valid_windows"]),
                    "train_start": split["train_start"],
                    "train_end": split["train_end"],
                    "valid_start": split["valid_start"],
                    "valid_end": split["valid_end"],
                    "train": _evaluate_rows(train_rows, model_bundle),
                    "valid": _evaluate_rows(valid_rows, model_bundle),
                    "trained_regimes": sorted(model_bundle.get("by_regime", {}).keys()),
                    "global_points": len(train_rows),
                }
            )

    payload["fit"] = latest_fit or {"status": "insufficient_data"}
    for horizon in horizons:
        payload["horizons"][str(horizon)] = {
            "summary": _summarize_horizon_folds(horizon_folds[horizon]),
            "folds": horizon_folds[horizon],
        }

    write_json(Path(args.output), payload)
    write_json(config.personal_model_residual_prototype_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

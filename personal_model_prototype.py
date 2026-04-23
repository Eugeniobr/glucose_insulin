import argparse
import json
from pathlib import Path

import numpy as np

from data_io import write_json
from external_history import load_external_history, summarize_external_history
from personal_model import (
    PARAMETER_NAMES,
    PersonalModelParams,
    build_personal_model_windows,
    build_walk_forward_splits,
    evaluate_personal_model,
    fit_personal_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protótipo de modelo fisiológico personalizado")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/personal_model_prototype.json")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--min-train", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2)
    parser.add_argument("--step", type=int, default=2)
    return parser


def _aggregate_regime_metrics(valid_windows: list[dict], params_payload: dict) -> dict:
    groups: dict[str, list[dict]] = {}
    for window in valid_windows:
        name = str(window.get("regime", {}).get("name", "desconhecido"))
        groups.setdefault(name, []).append(window)
    payload = {}
    for name, items in groups.items():
        metrics = evaluate_personal_model(items, params_payload)
        payload[name] = {
            "windows": len(items),
            "rmse_mg_dl": metrics.get("rmse_mg_dl"),
            "mae_mg_dl": metrics.get("mae_mg_dl"),
        }
    return payload


def _aggregate_folds(folds: list[dict]) -> dict:
    rmse_values = [float(fold["valid_evaluation"]["rmse_mg_dl"]) for fold in folds if fold.get("valid_evaluation", {}).get("rmse_mg_dl") is not None]
    mae_values = [float(fold["valid_evaluation"]["mae_mg_dl"]) for fold in folds if fold.get("valid_evaluation", {}).get("mae_mg_dl") is not None]
    regime_errors: dict[str, list[tuple[float, float]]] = {}
    for fold in folds:
        for name, metrics in fold.get("valid_by_regime", {}).items():
            if metrics.get("rmse_mg_dl") is None:
                continue
            regime_errors.setdefault(name, []).append((float(metrics["rmse_mg_dl"]), float(metrics["mae_mg_dl"])))
    regime_summary = {
        name: {
            "folds": len(values),
            "rmse_mg_dl": float(np.mean([item[0] for item in values])),
            "mae_mg_dl": float(np.mean([item[1] for item in values])),
        }
        for name, values in regime_errors.items()
    }
    return {
        "folds": len(folds),
        "rmse_mg_dl": float(np.mean(rmse_values)) if rmse_values else None,
        "mae_mg_dl": float(np.mean(mae_values)) if mae_values else None,
        "by_regime": regime_summary,
    }


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    summary = summarize_external_history(glucose_points, events)
    windows = build_personal_model_windows(glucose_points, events, horizon_minutes=240)
    windows = sorted(windows, key=lambda item: item["timestamp"])
    if args.max_windows > 0:
        windows = windows[: int(args.max_windows)]

    walk_forward = build_walk_forward_splits(
        windows,
        min_train_windows=int(args.min_train),
        validation_windows=int(args.valid_size),
        step_windows=int(args.step),
    )
    folds_payload = []
    for index, split in enumerate(walk_forward, start=1):
        fitted = fit_personal_model(split["train_windows"], initial_params=PersonalModelParams())
        valid_eval = evaluate_personal_model(split["valid_windows"], fitted["params"]) if fitted["status"] == "fitted" else {}
        folds_payload.append(
            {
                "fold": index,
                "train_windows": len(split["train_windows"]),
                "valid_windows": len(split["valid_windows"]),
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "valid_start": split["valid_start"],
                "valid_end": split["valid_end"],
                "fit": fitted,
                "valid_evaluation": valid_eval,
                "valid_by_regime": _aggregate_regime_metrics(split["valid_windows"], fitted["params"]) if fitted["status"] == "fitted" else {},
            }
        )

    full_fit = fit_personal_model(windows, initial_params=PersonalModelParams()) if windows else {
        "status": "insufficient_data",
        "active_parameter_names": [],
        "params": {},
    }
    full_eval = evaluate_personal_model(windows, full_fit["params"]) if windows and full_fit["status"] == "fitted" else {}
    payload = {
        "csv_path": str(csv_path),
        "summary": summary,
        "windows_total": len(windows),
        "parameter_names": list(PARAMETER_NAMES),
        "active_parameter_names": full_fit.get("active_parameter_names", []),
        "full_fit": full_fit,
        "full_evaluation": full_eval,
        "walk_forward": {
            "min_train_windows": int(args.min_train),
            "validation_windows": int(args.valid_size),
            "step_windows": int(args.step),
            "summary": _aggregate_folds(folds_payload),
            "folds": folds_payload,
        },
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

import numpy as np

from simple_predictor import SIMPLE_FEATURE_NAMES


TS_PREMISE_FEATURES = ("current_glucose", "slope_30")


def _feature_vector(sample: dict) -> np.ndarray:
    return np.array([float(sample.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES], dtype=float)


def _premise_vector(sample: dict) -> np.ndarray:
    return np.array([float(sample.get(name, 0.0)) for name in TS_PREMISE_FEATURES], dtype=float)


def _compute_centers(values: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(values, [0.2, 0.5, 0.8])
    return np.asarray(quantiles, dtype=float)


def _compute_widths(centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    widths = np.zeros_like(centers)
    spread = float(np.std(values)) if len(values) else 1.0
    fallback = max(spread * 0.75, 1e-3)
    for index, center in enumerate(centers):
        neighbors = []
        if index > 0:
            neighbors.append(abs(center - centers[index - 1]))
        if index < len(centers) - 1:
            neighbors.append(abs(centers[index + 1] - center))
        widths[index] = max(np.mean(neighbors) if neighbors else fallback, fallback)
    return widths


def _gaussian_membership(value: float, center: float, width: float) -> float:
    return float(np.exp(-0.5 * ((value - center) / max(width, 1e-6)) ** 2))


def _rule_weights(premise_values: np.ndarray, glucose_centers, glucose_widths, slope_centers, slope_widths) -> np.ndarray:
    glucose = float(premise_values[0])
    slope = float(premise_values[1])
    weights = []
    for g_idx, g_center in enumerate(glucose_centers):
        mu_g = _gaussian_membership(glucose, g_center, glucose_widths[g_idx])
        for s_idx, s_center in enumerate(slope_centers):
            mu_s = _gaussian_membership(slope, s_center, slope_widths[s_idx])
            weights.append(mu_g * mu_s)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def _weighted_ridge(features: np.ndarray, targets: np.ndarray, sample_weights: np.ndarray, l2: float) -> dict:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = (features - means) / scales
    sqrt_w = np.sqrt(np.clip(sample_weights, 1e-9, None))[:, None]
    weighted_x = normalized * sqrt_w
    weighted_y = targets * sqrt_w[:, 0]
    identity = np.eye(normalized.shape[1], dtype=float)
    identity[0, 0] = 0.0
    system = weighted_x.T @ weighted_x + float(l2) * identity
    rhs = weighted_x.T @ weighted_y
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ rhs
    predictions = normalized @ weights
    residuals = predictions - targets
    return {
        "feature_means": means,
        "feature_scales": scales,
        "weights": weights,
        "rmse": float(np.sqrt(np.mean((residuals**2) * np.clip(sample_weights, 0.0, None)))),
        "mae": float(np.mean(np.abs(residuals) * np.clip(sample_weights, 0.0, None))),
    }


def train_takagi_sugeno_model(samples: list[dict], model_path: Path, min_points: int = 96, l2: float = 6.0) -> dict:
    payload = {
        "feature_names": list(SIMPLE_FEATURE_NAMES),
        "premise_feature_names": list(TS_PREMISE_FEATURES),
        "horizons": {},
        "min_points": int(min_points),
        "l2": float(l2),
        "rules_per_axis": 3,
    }
    if not samples:
        model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    samples = sorted(samples, key=lambda item: item["timestamp"])
    for horizon in (15, 30, 60, 120):
        rows = [sample for sample in samples if int(sample["horizon_minutes"]) == horizon]
        if len(rows) < min_points:
            payload["horizons"][str(horizon)] = {"status": "insufficient_data", "points_used": len(rows)}
            continue

        split_index = max(int(len(rows) * 0.8), min_points)
        split_index = min(split_index, len(rows) - 1)
        train_rows = rows[:split_index]
        valid_rows = rows[split_index:]

        train_features = np.asarray([_feature_vector(row) for row in train_rows], dtype=float)
        train_targets = np.asarray([float(row["target_delta"]) for row in train_rows], dtype=float)
        train_premise = np.asarray([_premise_vector(row) for row in train_rows], dtype=float)

        glucose_centers = _compute_centers(train_premise[:, 0])
        slope_centers = _compute_centers(train_premise[:, 1])
        glucose_widths = _compute_widths(glucose_centers, train_premise[:, 0])
        slope_widths = _compute_widths(slope_centers, train_premise[:, 1])

        rule_models = []
        normalized_rule_weights = []
        for row in train_premise:
            normalized_rule_weights.append(
                _rule_weights(row, glucose_centers, glucose_widths, slope_centers, slope_widths)
            )
        normalized_rule_weights = np.asarray(normalized_rule_weights, dtype=float)

        for rule_index in range(normalized_rule_weights.shape[1]):
            local_weights = normalized_rule_weights[:, rule_index]
            rule_models.append(_weighted_ridge(train_features, train_targets, local_weights, l2))

        valid_rmse = None
        valid_mae = None
        if len(valid_rows):
            predictions = []
            targets = np.asarray([float(row["target_delta"]) for row in valid_rows], dtype=float)
            for row in valid_rows:
                predictions.append(predict_takagi_sugeno_delta(
                    {
                        "glucose_centers": glucose_centers.tolist(),
                        "glucose_widths": glucose_widths.tolist(),
                        "slope_centers": slope_centers.tolist(),
                        "slope_widths": slope_widths.tolist(),
                        "rule_models": [
                            {
                                "feature_means": model["feature_means"].tolist(),
                                "feature_scales": model["feature_scales"].tolist(),
                                "weights": model["weights"].tolist(),
                            }
                            for model in rule_models
                        ],
                    },
                    row,
                ))
            residuals = np.asarray(predictions, dtype=float) - targets
            valid_rmse = float(np.sqrt(np.mean(residuals**2)))
            valid_mae = float(np.mean(np.abs(residuals)))

        payload["horizons"][str(horizon)] = {
            "status": "trained",
            "points_used": int(len(rows)),
            "train_points": int(len(train_rows)),
            "valid_points": int(len(valid_rows)),
            "glucose_centers": glucose_centers.tolist(),
            "glucose_widths": glucose_widths.tolist(),
            "slope_centers": slope_centers.tolist(),
            "slope_widths": slope_widths.tolist(),
            "rule_models": [
                {
                    "feature_means": model["feature_means"].tolist(),
                    "feature_scales": model["feature_scales"].tolist(),
                    "weights": model["weights"].tolist(),
                }
                for model in rule_models
            ],
            "valid_target_rmse_mg_dl": valid_rmse,
            "valid_target_mae_mg_dl": valid_mae,
        }

    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def predict_takagi_sugeno_delta(horizon_payload: dict, sample: dict) -> float:
    feature_vector = _feature_vector(sample)
    premise = _premise_vector(sample)
    weights = _rule_weights(
        premise,
        horizon_payload["glucose_centers"],
        horizon_payload["glucose_widths"],
        horizon_payload["slope_centers"],
        horizon_payload["slope_widths"],
    )
    outputs = []
    for rule_model in horizon_payload["rule_models"]:
        means = np.asarray(rule_model["feature_means"], dtype=float)
        scales = np.asarray(rule_model["feature_scales"], dtype=float)
        coeffs = np.asarray(rule_model["weights"], dtype=float)
        normalized = (feature_vector - means) / scales
        outputs.append(float(normalized @ coeffs))
    return float(np.dot(weights, np.asarray(outputs, dtype=float)))


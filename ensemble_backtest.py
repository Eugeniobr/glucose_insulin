import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from external_history import build_direct_forecast_samples, load_external_history
from predictor import predict_ridge_model
from simple_predictor import SIMPLE_FEATURE_NAMES
from takagi_sugeno_predictor import TS_PREMISE_FEATURES
import takagi_sugeno_predictor as ts_predictor


HORIZONS = (15, 30, 60, 120)
TS_RULES_PER_AXIS = 3
SLOPE_REGIMES = ("low", "medium", "high", "jump")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest walk-forward com adaptação para trechos inclinados")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/backtest_ensemble_final_adaptive.json")
    parser.add_argument("--min-train-points", type=int, default=384)
    parser.add_argument("--l2-simple", type=float, default=8.0)
    parser.add_argument("--l2-ts", type=float, default=8.0)
    parser.add_argument("--retrain-every", type=int, default=120)
    parser.add_argument("--retrain-every-high-slope", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-rows-per-horizon", type=int, default=30000)
    parser.add_argument("--primary", choices=("simple", "ts", "naive", "blend", "agents", "physio_agents"), default="blend")
    parser.add_argument("--blend-ts-weight", type=float, default=0.45)
    parser.add_argument("--dynamic-blend", action="store_true", default=True)
    parser.add_argument("--blend-slope-gain", type=float, default=0.30)
    parser.add_argument("--blend-delta-gain", type=float, default=0.20)
    parser.add_argument("--blend-max-ts-weight", type=float, default=0.85)
    parser.add_argument("--bias-enabled", action="store_true", default=True)
    parser.add_argument("--bias-lookback-points", type=int, default=512)
    parser.add_argument("--bias-min-points", type=int, default=24)
    parser.add_argument("--bias-blend", type=float, default=0.45)
    parser.add_argument("--bias-max-adjustment", type=float, default=45.0)
    parser.add_argument("--slope-low-threshold", type=float, default=0.35)
    parser.add_argument("--slope-high-threshold", type=float, default=0.80)
    parser.add_argument("--jump-slope-threshold", type=float, default=1.10)
    parser.add_argument("--jump-delta15-threshold", type=float, default=18.0)
    parser.add_argument("--jump-ts-weight-bonus", type=float, default=0.20)
    parser.add_argument("--weight-high-slope-gain", type=float, default=2.0)
    parser.add_argument("--weight-jump-gain", type=float, default=1.5)
    parser.add_argument("--weight-max-multiplier", type=float, default=4.0)
    parser.add_argument("--slope-ref", type=float, default=1.0)
    parser.add_argument("--delta-ref", type=float, default=20.0)
    parser.add_argument("--agent-stable-std60-threshold", type=float, default=8.0)
    parser.add_argument("--agent-stable-range60-threshold", type=float, default=25.0)
    parser.add_argument("--agent-event-carbs-threshold", type=float, default=5.0)
    parser.add_argument("--agent-event-rapid-threshold", type=float, default=0.5)
    parser.add_argument("--agent-error-lookback-points", type=int, default=192)
    parser.add_argument("--agent-error-min-points", type=int, default=24)
    parser.add_argument("--physio-dominance-threshold", type=float, default=0.60)
    parser.add_argument("--physio-recent-event-minutes", type=float, default=90.0)
    parser.add_argument("--physio-committee-threshold", type=float, default=0.75)
    return parser


def _feature_vector(sample: dict) -> np.ndarray:
    return np.asarray([float(sample.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES], dtype=float)


def _premise_vector(sample: dict) -> np.ndarray:
    return np.asarray([float(sample.get(name, 0.0)) for name in TS_PREMISE_FEATURES], dtype=float)


def _slope_regime(sample: dict, low_threshold: float, high_threshold: float, jump_slope_threshold: float, jump_delta15_threshold: float) -> str:
    slope = abs(float(sample.get("slope_30", 0.0)))
    delta_15 = abs(float(sample.get("delta_15", 0.0)))
    if slope >= jump_slope_threshold or delta_15 >= jump_delta15_threshold:
        return "jump"
    if slope >= high_threshold:
        return "high"
    if slope >= low_threshold:
        return "medium"
    return "low"


def _sample_weight(sample: dict, gain: float, jump_gain: float, max_multiplier: float, slope_ref: float, regime: str) -> float:
    slope = abs(float(sample.get("slope_30", 0.0)))
    normalized = slope / max(float(slope_ref), 1e-6)
    weight = 1.0 + float(gain) * normalized
    if regime == "jump":
        weight += float(jump_gain)
    return float(np.clip(weight, 1.0, max(float(max_multiplier), 1.0)))


def _train_ridge_weighted(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, l2: float) -> dict:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = (features - means) / scales
    sqrt_w = np.sqrt(np.clip(weights, 1e-9, None))[:, None]
    weighted_x = normalized * sqrt_w
    weighted_y = targets * sqrt_w[:, 0]
    identity = np.eye(normalized.shape[1], dtype=float)
    identity[0, 0] = 0.0
    system = weighted_x.T @ weighted_x + float(l2) * identity
    rhs = weighted_x.T @ weighted_y
    try:
        coeffs = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(system) @ rhs
    predictions = normalized @ coeffs
    residuals = predictions - targets
    return {
        "feature_means": means,
        "feature_scales": scales,
        "weights": coeffs,
        "rmse": float(np.sqrt(np.mean((residuals**2) * np.clip(weights, 1e-9, None)))),
        "mae": float(np.mean(np.abs(residuals) * np.clip(weights, 1e-9, None))),
    }


def _train_simple_horizon(rows: list[dict], min_points: int, l2: float, args) -> dict:
    if len(rows) < min_points:
        return {"status": "insufficient_data", "points_used": len(rows)}
    x = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    y = np.asarray([float(row["target_delta"]) for row in rows], dtype=float)
    weights = np.asarray(
        [
            _sample_weight(
                row,
                gain=float(args.weight_high_slope_gain),
                jump_gain=float(args.weight_jump_gain),
                max_multiplier=float(args.weight_max_multiplier),
                slope_ref=float(args.slope_ref),
                regime=_slope_regime(
                    row,
                    float(args.slope_low_threshold),
                    float(args.slope_high_threshold),
                    float(args.jump_slope_threshold),
                    float(args.jump_delta15_threshold),
                ),
            )
            for row in rows
        ],
        dtype=float,
    )
    model = _train_ridge_weighted(x, y, weights, l2)
    return {
        "status": "trained",
        "points_used": len(rows),
        "feature_means": model["feature_means"],
        "feature_scales": model["feature_scales"],
        "weights": model["weights"],
        "train_weighted_rmse_mg_dl": model["rmse"],
        "train_weighted_mae_mg_dl": model["mae"],
    }


def _predict_simple(model: dict, row: dict) -> float | None:
    if model.get("status") != "trained":
        return None
    delta = predict_ridge_model(model, _feature_vector(row))
    return float(row["current_glucose"] + delta)


def _train_ts_horizon(rows: list[dict], min_points: int, l2: float, args) -> dict:
    if len(rows) < min_points:
        return {"status": "insufficient_data", "points_used": len(rows)}
    features = np.asarray([_feature_vector(row) for row in rows], dtype=float)
    targets = np.asarray([float(row["target_delta"]) for row in rows], dtype=float)
    premise = np.asarray([_premise_vector(row) for row in rows], dtype=float)
    sample_weights = np.asarray(
        [
            _sample_weight(
                row,
                gain=float(args.weight_high_slope_gain),
                jump_gain=float(args.weight_jump_gain),
                max_multiplier=float(args.weight_max_multiplier),
                slope_ref=float(args.slope_ref),
                regime=_slope_regime(
                    row,
                    float(args.slope_low_threshold),
                    float(args.slope_high_threshold),
                    float(args.jump_slope_threshold),
                    float(args.jump_delta15_threshold),
                ),
            )
            for row in rows
        ],
        dtype=float,
    )
    glucose_centers = ts_predictor._compute_centers(premise[:, 0])
    slope_centers = ts_predictor._compute_centers(premise[:, 1])
    glucose_widths = ts_predictor._compute_widths(glucose_centers, premise[:, 0])
    slope_widths = ts_predictor._compute_widths(slope_centers, premise[:, 1])
    memberships = np.asarray(
        [
            ts_predictor._rule_weights(
                prem,
                glucose_centers,
                glucose_widths,
                slope_centers,
                slope_widths,
            )
            for prem in premise
        ],
        dtype=float,
    )
    rule_models = []
    for rule_index in range(TS_RULES_PER_AXIS * TS_RULES_PER_AXIS):
        local_weights = memberships[:, rule_index] * sample_weights
        weighted = ts_predictor._weighted_ridge(features, targets, local_weights, l2)
        rule_models.append(
            {
                "feature_means": weighted["feature_means"].tolist(),
                "feature_scales": weighted["feature_scales"].tolist(),
                "weights": weighted["weights"].tolist(),
            }
        )
    return {
        "status": "trained",
        "points_used": len(rows),
        "glucose_centers": glucose_centers.tolist(),
        "glucose_widths": glucose_widths.tolist(),
        "slope_centers": slope_centers.tolist(),
        "slope_widths": slope_widths.tolist(),
        "rule_models": rule_models,
    }


def _predict_ts(model: dict, row: dict) -> float | None:
    if model.get("status") != "trained":
        return None
    delta = ts_predictor.predict_takagi_sugeno_delta(model, row)
    return float(row["current_glucose"] + delta)


def _bias_adjustment(
    residual_history: list[float],
    fallback_history: list[float],
    enabled: bool,
    lookback_points: int,
    min_points: int,
    blend: float,
    max_adjustment: float,
) -> tuple[float, dict]:
    if not enabled:
        return 0.0, {"status": "disabled", "points_used": len(residual_history)}
    local = residual_history[-max(lookback_points, 1):]
    source = "regime"
    used = local
    if len(local) < min_points:
        global_used = fallback_history[-max(lookback_points, 1):]
        if len(global_used) < min_points:
            return 0.0, {"status": "insufficient_data", "points_used": len(local), "global_points": len(global_used)}
        used = global_used
        source = "global"
    bias = float(np.mean(np.asarray(used, dtype=float)))
    adjustment = float(np.clip(blend * bias, -abs(max_adjustment), abs(max_adjustment)))
    return adjustment, {"status": "trained", "source": source, "points_used": len(used), "bias_mg_dl": bias, "adjustment_mg_dl": adjustment}


def _metrics_from_errors(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0, "rmse_mg_dl": None, "mae_mg_dl": None, "bias_mg_dl": None}
    arr = np.asarray(errors, dtype=float)
    return {
        "count": int(len(arr)),
        "rmse_mg_dl": float(np.sqrt(np.mean(arr**2))),
        "mae_mg_dl": float(np.mean(np.abs(arr))),
        "bias_mg_dl": float(np.mean(arr)),
    }


def _dynamic_ts_weight(base_weight: float, sample: dict, args) -> float:
    weight = float(np.clip(base_weight, 0.0, 1.0))
    if not bool(args.dynamic_blend):
        return weight
    slope = abs(float(sample.get("slope_30", 0.0)))
    delta_15 = abs(float(sample.get("delta_15", 0.0)))
    slope_component = float(args.blend_slope_gain) * min(slope / max(float(args.slope_ref), 1e-6), 1.0)
    delta_component = float(args.blend_delta_gain) * min(delta_15 / max(float(args.delta_ref), 1e-6), 1.0)
    weight = weight + slope_component + delta_component
    is_jump = slope >= float(args.jump_slope_threshold) or delta_15 >= float(args.jump_delta15_threshold)
    if is_jump:
        weight += float(args.jump_ts_weight_bonus)
    return float(np.clip(weight, 0.0, float(args.blend_max_ts_weight)))


def _agent_policy_choice(
    predictions: dict[str, float | None],
    row: dict,
    args,
    source_error_history: dict,
    regime: str,
) -> tuple[float, str, float]:
    available_simple = predictions.get("simple") is not None
    available_ts = predictions.get("ts") is not None
    available_naive = predictions.get("naive") is not None

    if not available_simple and not available_ts:
        if available_naive:
            return float(predictions["naive"]), "agents_naive_fallback", 0.0
        raise ValueError("Nenhuma fonte disponível para a política de agentes")

    votes = {"simple": 0, "ts": 0}
    slope = abs(float(row.get("slope_30", 0.0)))
    delta_15 = abs(float(row.get("delta_15", 0.0)))
    std_60 = float(row.get("std_60", 0.0))
    range_60 = float(row.get("range_60", 0.0))
    future_carbs = float(row.get("future_carbs_total", 0.0))
    future_rapid = float(row.get("future_rapid_total", 0.0))

    # Agente 1: salto/alta inclinação favorece TS
    if slope >= float(args.jump_slope_threshold) or delta_15 >= float(args.jump_delta15_threshold):
        votes["ts"] += 1
    else:
        votes["simple"] += 1

    # Agente 2: estabilidade favorece simple
    if std_60 <= float(args.agent_stable_std60_threshold) and range_60 <= float(args.agent_stable_range60_threshold):
        votes["simple"] += 1
    else:
        votes["ts"] += 1

    # Agente 3: evento iminente favorece TS
    if future_carbs >= float(args.agent_event_carbs_threshold) or future_rapid >= float(args.agent_event_rapid_threshold):
        votes["ts"] += 1
    else:
        votes["simple"] += 1

    # Agente 4: desempenho recente por regime favorece a menor MAE local.
    lookback = max(int(getattr(args, "agent_error_lookback_points", 192)), 1)
    min_points = max(int(getattr(args, "agent_error_min_points", 24)), 1)
    recent_simple = source_error_history["simple"][regime][-lookback:]
    recent_ts = source_error_history["ts"][regime][-lookback:]
    if len(recent_simple) < min_points:
        recent_simple = source_error_history["simple"]["global"][-lookback:]
    if len(recent_ts) < min_points:
        recent_ts = source_error_history["ts"]["global"][-lookback:]
    if len(recent_simple) >= min_points and len(recent_ts) >= min_points:
        simple_mae = float(np.mean(np.abs(np.asarray(recent_simple, dtype=float))))
        ts_mae = float(np.mean(np.abs(np.asarray(recent_ts, dtype=float))))
        if ts_mae < simple_mae:
            votes["ts"] += 1
        elif simple_mae < ts_mae:
            votes["simple"] += 1

    if not available_ts:
        return float(predictions["simple"]), "agents_simple_only", 0.0
    if not available_simple:
        return float(predictions["ts"]), "agents_ts_only", 1.0

    ts_votes = votes["ts"]
    simple_votes = votes["simple"]
    if ts_votes > simple_votes:
        return float(predictions["ts"]), "agents_vote_ts", 1.0
    if simple_votes > ts_votes:
        return float(predictions["simple"]), "agents_vote_simple", 0.0

    # Empate: blend com peso dinâmico, mas limitado por votos (50/50 base).
    dynamic_weight = _dynamic_ts_weight(0.5, row, args)
    value = (1.0 - dynamic_weight) * float(predictions["simple"]) + dynamic_weight * float(predictions["ts"])
    return float(value), "agents_vote_blend", float(dynamic_weight)


def _physio_agent_policy_choice(
    predictions: dict[str, float | None],
    row: dict,
    args,
    source_error_history: dict,
    regime: str,
) -> tuple[float, str, float]:
    available_simple = predictions.get("simple") is not None
    available_ts = predictions.get("ts") is not None
    available_naive = predictions.get("naive") is not None

    if not available_simple and not available_ts:
        if available_naive:
            return float(predictions["naive"]), "physio_agents_naive_fallback", 0.0
        raise ValueError("Nenhuma fonte disponível para a política physio_agents")

    slope = float(row.get("slope_30", 0.0))
    abs_slope = abs(slope)
    delta_15 = float(row.get("delta_15", 0.0))
    abs_delta_15 = abs(delta_15)
    std_60 = float(row.get("std_60", 0.0))
    range_60 = float(row.get("range_60", 0.0))
    future_carbs = float(row.get("future_carbs_total", 0.0))
    future_rapid = float(row.get("future_rapid_total", 0.0))
    carbs_last_30 = float(row.get("carbs_last_30", 0.0))
    rapid_last_30 = float(row.get("rapid_last_30", 0.0))
    mins_since_meal = float(row.get("minutes_since_meal", 1e9))
    mins_since_rapid = float(row.get("minutes_since_rapid", 1e9))

    carb_thr = max(float(args.agent_event_carbs_threshold), 1e-6)
    rapid_thr = max(float(args.agent_event_rapid_threshold), 1e-6)
    recent_window = max(float(args.physio_recent_event_minutes), 1.0)

    carbs_signal = (
        (future_carbs / carb_thr)
        + 0.5 * (carbs_last_30 / carb_thr)
        + (0.5 if mins_since_meal <= recent_window else 0.0)
    )
    rapid_signal = (
        (future_rapid / rapid_thr)
        + 0.5 * (rapid_last_30 / rapid_thr)
        + (0.5 if mins_since_rapid <= recent_window else 0.0)
    )
    dominance = carbs_signal - rapid_signal
    dominance_thr = float(args.physio_dominance_threshold)
    carb_active = carbs_signal >= 1.0
    rapid_active = rapid_signal >= 1.0
    calm_state = (
        std_60 <= float(args.agent_stable_std60_threshold)
        and range_60 <= float(args.agent_stable_range60_threshold)
        and not carb_active
        and not rapid_active
    )

    ts_votes = 0.0
    simple_votes = 0.0
    # Agente 1: curvatura extrema.
    if abs_slope >= float(args.jump_slope_threshold) or abs_delta_15 >= float(args.jump_delta15_threshold):
        ts_votes += 1.3
    else:
        simple_votes += 0.8

    # Agente 2: direção ascendente forte.
    if slope >= float(args.slope_high_threshold) or delta_15 >= float(args.slope_ref):
        ts_votes += 1.0
    else:
        simple_votes += 0.6

    # Agente 3: direção descendente forte.
    if slope <= -float(args.slope_high_threshold) or delta_15 <= -float(args.slope_ref):
        ts_votes += 1.0
    else:
        simple_votes += 0.6

    # Agente 4: domínio fisiológico carbo vs rápida.
    if dominance >= dominance_thr:
        if slope >= 0.0:
            ts_votes += 1.2
        else:
            simple_votes += 0.8
    elif dominance <= -dominance_thr:
        if slope <= 0.0:
            ts_votes += 1.2
        else:
            simple_votes += 0.8
    else:
        simple_votes += 0.7

    # Agente 5: estado calmo/homeostase.
    if calm_state:
        simple_votes += 1.1
    else:
        ts_votes += 0.9

    # Agente 6: recência de refeição/bolus.
    if mins_since_meal <= recent_window or mins_since_rapid <= recent_window:
        ts_votes += 0.9
    else:
        simple_votes += 0.6

    # Agente 7: desempenho recente por regime.
    lookback = max(int(getattr(args, "agent_error_lookback_points", 192)), 1)
    min_points = max(int(getattr(args, "agent_error_min_points", 24)), 1)
    recent_simple = source_error_history["simple"][regime][-lookback:]
    recent_ts = source_error_history["ts"][regime][-lookback:]
    if len(recent_simple) < min_points:
        recent_simple = source_error_history["simple"]["global"][-lookback:]
    if len(recent_ts) < min_points:
        recent_ts = source_error_history["ts"]["global"][-lookback:]
    if len(recent_simple) >= min_points and len(recent_ts) >= min_points:
        simple_mae = float(np.mean(np.abs(np.asarray(recent_simple, dtype=float))))
        ts_mae = float(np.mean(np.abs(np.asarray(recent_ts, dtype=float))))
        if ts_mae < simple_mae:
            ts_votes += 1.1
        elif simple_mae < ts_mae:
            simple_votes += 1.1

    if not available_ts:
        return float(predictions["simple"]), "physio_agents_simple_only", 0.0
    if not available_simple:
        return float(predictions["ts"]), "physio_agents_ts_only", 1.0

    margin = ts_votes - simple_votes
    committee_threshold = float(args.physio_committee_threshold)
    if margin >= committee_threshold:
        return float(predictions["ts"]), "physio_agents_vote_ts", 1.0
    if margin <= -committee_threshold:
        return float(predictions["simple"]), "physio_agents_vote_simple", 0.0

    base = 0.5 + 0.2 * np.tanh(0.5 * margin)
    weight = _dynamic_ts_weight(float(base), row, args)
    value = (1.0 - weight) * float(predictions["simple"]) + weight * float(predictions["ts"])
    return float(value), "physio_agents_vote_blend", float(weight)


def _choose_final_prediction(
    primary: str,
    base_blend_ts_weight: float,
    predictions: dict[str, float | None],
    row: dict,
    args,
    source_error_history: dict,
    regime: str,
) -> tuple[float, str, float]:
    if primary == "simple":
        if predictions.get("simple") is not None:
            return float(predictions["simple"]), "simple", 0.0
        if predictions.get("ts") is not None:
            return float(predictions["ts"]), "ts_fallback", 0.0
        return float(predictions["naive"]), "naive_fallback", 0.0
    if primary == "ts":
        if predictions.get("ts") is not None:
            return float(predictions["ts"]), "ts", 1.0
        if predictions.get("simple") is not None:
            return float(predictions["simple"]), "simple_fallback", 0.0
        return float(predictions["naive"]), "naive_fallback", 0.0
    if primary == "blend":
        simple = predictions.get("simple")
        ts = predictions.get("ts")
        if simple is not None and ts is not None:
            weight = _dynamic_ts_weight(base_blend_ts_weight, row, args)
            value = (1.0 - weight) * float(simple) + weight * float(ts)
            return float(value), "blend_simple_ts", float(weight)
        if simple is not None:
            return float(simple), "simple_fallback", 0.0
        if ts is not None:
            return float(ts), "ts_fallback", 1.0
        return float(predictions["naive"]), "naive_fallback", 0.0
    if primary == "agents":
        return _agent_policy_choice(
            predictions=predictions,
            row=row,
            args=args,
            source_error_history=source_error_history,
            regime=regime,
        )
    if primary == "physio_agents":
        return _physio_agent_policy_choice(
            predictions=predictions,
            row=row,
            args=args,
            source_error_history=source_error_history,
            regime=regime,
        )
    return float(predictions["naive"]), "naive", 0.0


def _backtest_horizon(rows: list[dict], horizon: int, args) -> dict:
    if not rows:
        return {
            "horizon_minutes": int(horizon),
            "rows_total": 0,
            "rows_tested": 0,
            "source_usage": {},
            "metrics": {},
            "metrics_by_regime": {},
            "bias_adaptation": {},
        }

    max_train_rows = max(int(args.max_train_rows), int(args.min_train_points))
    retrain_every_base = max(int(args.retrain_every), 1)
    retrain_every_high = max(int(args.retrain_every_high_slope), 1)
    min_train = int(args.min_train_points)
    primary = str(args.primary)
    blend_weight = float(args.blend_ts_weight)

    simple_model = {"status": "insufficient_data"}
    ts_model = {"status": "insufficient_data"}
    last_retrain_index = None

    source_error_history = {
        source: {"global": [], "low": [], "medium": [], "high": [], "jump": []}
        for source in ("naive", "simple", "ts", "final")
    }
    metrics_errors = {
        "naive_raw": [],
        "naive_corrected": [],
        "simple_raw": [],
        "simple_corrected": [],
        "ts_raw": [],
        "ts_corrected": [],
        "final": [],
    }
    metrics_errors_by_regime = {regime: {"final": []} for regime in SLOPE_REGIMES}
    source_usage = Counter()
    dynamic_weights_used = []

    start_index = min_train
    for index in range(start_index, len(rows)):
        row = rows[index]
        regime = _slope_regime(
            row,
            float(args.slope_low_threshold),
            float(args.slope_high_threshold),
            float(args.jump_slope_threshold),
            float(args.jump_delta15_threshold),
        )
        retrain_period = retrain_every_high if regime in {"high", "jump"} else retrain_every_base
        should_retrain = (
            index == start_index
            or last_retrain_index is None
            or (index - last_retrain_index) >= retrain_period
        )
        if should_retrain:
            train_start = max(0, index - max_train_rows)
            train_rows = rows[train_start:index]
            simple_model = _train_simple_horizon(train_rows, min_train, float(args.l2_simple), args)
            ts_model = _train_ts_horizon(train_rows, min_train, float(args.l2_ts), args)
            last_retrain_index = index

        target = float(row["target_glucose"])
        predictions_raw = {
            "naive": float(row["current_glucose"]),
            "simple": _predict_simple(simple_model, row),
            "ts": _predict_ts(ts_model, row),
        }

        predictions_corrected = {}
        for source_name in ("naive", "simple", "ts"):
            raw = predictions_raw[source_name]
            if raw is None:
                predictions_corrected[source_name] = None
                continue
            adjustment, _ = _bias_adjustment(
                residual_history=source_error_history[source_name][regime],
                fallback_history=source_error_history[source_name]["global"],
                enabled=bool(args.bias_enabled),
                lookback_points=int(args.bias_lookback_points),
                min_points=int(args.bias_min_points),
                blend=float(args.bias_blend),
                max_adjustment=float(args.bias_max_adjustment),
            )
            predictions_corrected[source_name] = float(raw - adjustment)

        final_prediction, source_used, used_ts_weight = _choose_final_prediction(
            primary=primary,
            base_blend_ts_weight=blend_weight,
            predictions=predictions_corrected,
            row=row,
            args=args,
            source_error_history=source_error_history,
            regime=regime,
        )
        source_usage[source_used] += 1
        dynamic_weights_used.append(float(used_ts_weight))

        naive_raw_error = float(predictions_raw["naive"] - target)
        naive_corrected_error = float(predictions_corrected["naive"] - target)
        metrics_errors["naive_raw"].append(naive_raw_error)
        metrics_errors["naive_corrected"].append(naive_corrected_error)
        source_error_history["naive"]["global"].append(naive_raw_error)
        source_error_history["naive"][regime].append(naive_raw_error)

        if predictions_raw["simple"] is not None:
            simple_raw_error = float(predictions_raw["simple"] - target)
            metrics_errors["simple_raw"].append(simple_raw_error)
            source_error_history["simple"]["global"].append(simple_raw_error)
            source_error_history["simple"][regime].append(simple_raw_error)
        if predictions_corrected["simple"] is not None:
            metrics_errors["simple_corrected"].append(float(predictions_corrected["simple"] - target))

        if predictions_raw["ts"] is not None:
            ts_raw_error = float(predictions_raw["ts"] - target)
            metrics_errors["ts_raw"].append(ts_raw_error)
            source_error_history["ts"]["global"].append(ts_raw_error)
            source_error_history["ts"][regime].append(ts_raw_error)
        if predictions_corrected["ts"] is not None:
            metrics_errors["ts_corrected"].append(float(predictions_corrected["ts"] - target))

        final_error = float(final_prediction - target)
        metrics_errors["final"].append(final_error)
        metrics_errors_by_regime[regime]["final"].append(final_error)
        source_error_history["final"]["global"].append(final_error)
        source_error_history["final"][regime].append(final_error)

    output_metrics = {name: _metrics_from_errors(values) for name, values in metrics_errors.items()}
    output_metrics_by_regime = {regime: {"final": _metrics_from_errors(payload["final"])} for regime, payload in metrics_errors_by_regime.items()}
    bias_payload = {
        "enabled": bool(args.bias_enabled),
        "lookback_points": int(args.bias_lookback_points),
        "min_points": int(args.bias_min_points),
        "blend": float(args.bias_blend),
        "max_adjustment_mg_dl": float(args.bias_max_adjustment),
        "regime_based": True,
    }
    dynamic_weight_payload = {
        "enabled": bool(args.dynamic_blend),
        "avg_ts_weight": float(np.mean(dynamic_weights_used)) if dynamic_weights_used else None,
        "p90_ts_weight": float(np.quantile(dynamic_weights_used, 0.9)) if dynamic_weights_used else None,
    }
    return {
        "horizon_minutes": int(horizon),
        "rows_total": int(len(rows)),
        "rows_tested": int(len(rows) - start_index),
        "source_usage": dict(source_usage),
        "metrics": output_metrics,
        "metrics_by_regime": output_metrics_by_regime,
        "bias_adaptation": bias_payload,
        "dynamic_blend": dynamic_weight_payload,
    }


def _slice_rows_by_horizon(samples: list[dict], max_rows_per_horizon: int) -> dict[int, list[dict]]:
    grouped = {h: [] for h in HORIZONS}
    for row in samples:
        horizon = int(row.get("horizon_minutes", -1))
        if horizon in grouped:
            grouped[horizon].append(row)
    for horizon in HORIZONS:
        grouped[horizon].sort(key=lambda item: item["timestamp"])
        if max_rows_per_horizon > 0:
            grouped[horizon] = grouped[horizon][-max_rows_per_horizon:]
    return grouped


def _aggregate_final_global(horizons_payload: dict) -> dict:
    count = 0
    sum_abs = 0.0
    sum_err = 0.0
    sum_sq = 0.0
    for horizon in HORIZONS:
        metrics = horizons_payload[str(horizon)]["metrics"]["final"]
        c = int(metrics["count"])
        if c == 0:
            continue
        rmse = float(metrics["rmse_mg_dl"])
        mae = float(metrics["mae_mg_dl"])
        bias = float(metrics["bias_mg_dl"])
        count += c
        sum_sq += (rmse**2) * c
        sum_abs += mae * c
        sum_err += bias * c
    if count == 0:
        return {"count": 0, "rmse_mg_dl": None, "mae_mg_dl": None, "bias_mg_dl": None}
    return {
        "count": count,
        "rmse_mg_dl": float(np.sqrt(sum_sq / count)),
        "mae_mg_dl": float(sum_abs / count),
        "bias_mg_dl": float(sum_err / count),
    }


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    samples = build_direct_forecast_samples(glucose_points, events)
    by_horizon = _slice_rows_by_horizon(samples, int(args.max_rows_per_horizon))

    horizons_payload = {}
    for horizon in HORIZONS:
        horizons_payload[str(horizon)] = _backtest_horizon(by_horizon[horizon], horizon, args)

    output = {
        "generated_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "samples_total": int(len(samples)),
        "horizons": horizons_payload,
        "final_global": _aggregate_final_global(horizons_payload),
        "settings": {
            "min_train_points": int(args.min_train_points),
            "l2_simple": float(args.l2_simple),
            "l2_ts": float(args.l2_ts),
            "retrain_every": int(args.retrain_every),
            "retrain_every_high_slope": int(args.retrain_every_high_slope),
            "max_train_rows": int(args.max_train_rows),
            "max_rows_per_horizon": int(args.max_rows_per_horizon),
            "primary": str(args.primary),
            "blend_ts_weight": float(args.blend_ts_weight),
            "dynamic_blend": bool(args.dynamic_blend),
            "blend_slope_gain": float(args.blend_slope_gain),
            "blend_delta_gain": float(args.blend_delta_gain),
            "blend_max_ts_weight": float(args.blend_max_ts_weight),
            "bias_enabled": bool(args.bias_enabled),
            "bias_lookback_points": int(args.bias_lookback_points),
            "bias_min_points": int(args.bias_min_points),
            "bias_blend": float(args.bias_blend),
            "bias_max_adjustment": float(args.bias_max_adjustment),
            "slope_low_threshold": float(args.slope_low_threshold),
            "slope_high_threshold": float(args.slope_high_threshold),
            "jump_slope_threshold": float(args.jump_slope_threshold),
            "jump_delta15_threshold": float(args.jump_delta15_threshold),
            "jump_ts_weight_bonus": float(args.jump_ts_weight_bonus),
            "weight_high_slope_gain": float(args.weight_high_slope_gain),
            "weight_jump_gain": float(args.weight_jump_gain),
            "weight_max_multiplier": float(args.weight_max_multiplier),
            "slope_ref": float(args.slope_ref),
            "delta_ref": float(args.delta_ref),
            "agent_stable_std60_threshold": float(args.agent_stable_std60_threshold),
            "agent_stable_range60_threshold": float(args.agent_stable_range60_threshold),
            "agent_event_carbs_threshold": float(args.agent_event_carbs_threshold),
            "agent_event_rapid_threshold": float(args.agent_event_rapid_threshold),
            "agent_error_lookback_points": int(args.agent_error_lookback_points),
            "agent_error_min_points": int(args.agent_error_min_points),
            "physio_dominance_threshold": float(args.physio_dominance_threshold),
            "physio_recent_event_minutes": float(args.physio_recent_event_minutes),
            "physio_committee_threshold": float(args.physio_committee_threshold),
        },
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

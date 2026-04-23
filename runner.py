import os
import subprocess
import sys
import time
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from calibration import apply_calibrated_params, calibrate_online_params, load_calibrated_params, save_initial_calibration_snapshot
from config import AppConfig
from data_io import ensure_output_dirs, load_glucose_json, write_json
from external_history import build_direct_forecast_samples, load_external_history
from forward import compare_with_observed, run_forward
from inputs import load_exercise_events, load_insulin_events, load_meal_events, serialize_events
from inverse import estimate_initial_state
from model import UltradianParams, f2, f3, f4
from predictor import (
    blend_forecast_predictions,
    load_external_forecast_model,
    predict_with_external_forecast_model,
    predict_with_forecast_model,
    train_forecast_model,
)
from regimes import detect_regime, regime_parameter_overrides
from simple_predictor import (
    build_simple_feature_payload,
    load_simple_forecast_model,
    predict_with_simple_model,
    train_simple_forecast_model,
)
from takagi_sugeno_ml import TS_ML_HORIZONS, load_takagi_sugeno_ml_model, predict_with_takagi_sugeno_ml

FORECAST_HORIZONS_MINUTES = (15, 30, 60, 120)
FORECAST_DELTA_WINDOW_HOURS = 2


def _prepare_environment(config: AppConfig) -> None:
    ensure_output_dirs(config.outputs_dir, config.plots_dir)
    mpl_config_dir = config.outputs_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _ukf_settings_from_config(config: AppConfig) -> dict:
    return {
        "alpha": float(config.ukf_alpha),
        "beta": float(config.ukf_beta),
        "kappa": float(config.ukf_kappa),
        "measurement_noise_mgdl": float(config.ukf_measurement_noise_mgdl),
        "q_s": float(config.ukf_process_noise_s),
        "q_ip": float(config.ukf_process_noise_ip),
        "q_ii": float(config.ukf_process_noise_ii),
        "q_g": float(config.ukf_process_noise_g),
        "q_h": float(config.ukf_process_noise_h),
    }


def _bayes_settings_from_config(config: AppConfig) -> dict:
    return {
        "particle_count": int(config.bayes_particle_count),
        "measurement_noise_mgdl": float(config.bayes_measurement_noise_mgdl),
        "state_noise_scale": float(config.bayes_state_noise_scale),
        "param_noise_scale": float(config.bayes_param_noise_scale),
    }


def run_extract_script(config: AppConfig) -> None:
    subprocess.run(
        [sys.executable, str(config.extract_script)],
        cwd=str(config.base_dir),
        check=True,
    )


def _resolve_external_history_csv(config: AppConfig) -> Path | None:
    explicit_path = str(getattr(config, "external_history_csv_path", "") or "").strip()
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = (config.base_dir / candidate).resolve()
        return candidate if candidate.exists() else None

    candidates = sorted(
        config.base_dir.glob("*glucose*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _refresh_takagi_sugeno_dashboard_assets(config: AppConfig) -> None:
    if not getattr(config, "takagi_sugeno_ml_enabled", False):
        return
    csv_path = _resolve_external_history_csv(config)
    if csv_path is None:
        return
    script_path = (config.base_dir / "takagi_sugeno_ml_benchmark.py").resolve()
    if not script_path.exists():
        return
    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(csv_path),
                "--output",
                str(config.takagi_sugeno_ml_benchmark_path),
                "--ts-model-output",
                str(config.takagi_sugeno_dynamic_model_path),
            ],
            cwd=str(config.base_dir),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[{datetime.now().isoformat()}] aviso: falha ao atualizar benchmark TS+ML: {exc}", file=sys.stderr)


def _refresh_simple_forecast_model_assets(config: AppConfig) -> None:
    if not getattr(config, "simple_forecast_retrain_from_external_enabled", False):
        return
    if not getattr(config, "simple_forecast_model_enabled", False):
        return
    csv_path = _resolve_external_history_csv(config)
    if csv_path is None:
        return
    try:
        glucose_points, events = load_external_history(csv_path)
        samples = build_direct_forecast_samples(glucose_points, events)
        max_samples = int(getattr(config, "simple_forecast_max_samples", 0))
        if max_samples > 0:
            samples = samples[-max_samples:]
        train_simple_forecast_model(
            samples=samples,
            model_path=config.simple_forecast_model_path,
            min_points=int(getattr(config, "simple_forecast_min_points", 96)),
            l2=float(getattr(config, "simple_forecast_l2", 6.0)),
        )
    except Exception as exc:
        print(f"[{datetime.now().isoformat()}] aviso: falha ao atualizar modelo simples: {exc}", file=sys.stderr)


def _prediction_key_for_source(source: str) -> str:
    mapping = {
        "final": "predicted_glucose_mg_dl",
        "simple": "simple_predicted_glucose_mg_dl",
        "ts_ml": "takagi_sugeno_ml_predicted_glucose_mg_dl",
        "ts": "takagi_sugeno_predicted_glucose_mg_dl",
        "hybrid": "hybrid_predicted_glucose_mg_dl",
        "physiological": "physiological_predicted_glucose_mg_dl",
    }
    return mapping.get(source, "predicted_glucose_mg_dl")


def _estimate_online_source_bias(
    config: AppConfig,
    history: list[dict],
    generated_at: str,
    horizon_minutes: int,
    source: str,
) -> dict:
    if not getattr(config, "adaptive_forecast_bias_enabled", False):
        return {"status": "disabled", "source": source, "horizon_minutes": int(horizon_minutes)}
    lookback_hours = int(getattr(config, "adaptive_forecast_bias_lookback_hours", 24))
    min_points = int(getattr(config, "adaptive_forecast_bias_min_points", 6))
    cutoff = datetime.fromisoformat(generated_at) - timedelta(hours=max(lookback_hours, 1))
    prediction_key = _prediction_key_for_source(source)
    residuals = []
    for item in history:
        if not item.get("is_resolved"):
            continue
        if int(item.get("horizon_minutes", -1)) != int(horizon_minutes):
            continue
        if datetime.fromisoformat(item["generated_at"]) < cutoff:
            continue
        if item.get("observed_glucose_mg_dl") is None:
            continue
        if item.get(prediction_key) is None:
            continue
        residuals.append(float(item[prediction_key]) - float(item["observed_glucose_mg_dl"]))
    if len(residuals) < min_points:
        return {
            "status": "insufficient_data",
            "source": source,
            "horizon_minutes": int(horizon_minutes),
            "lookback_hours": lookback_hours,
            "points_used": len(residuals),
            "min_points": min_points,
            "bias_mg_dl": None,
        }
    bias = float(np.mean(np.asarray(residuals, dtype=float)))
    return {
        "status": "trained",
        "source": source,
        "horizon_minutes": int(horizon_minutes),
        "lookback_hours": lookback_hours,
        "points_used": len(residuals),
        "min_points": min_points,
        "bias_mg_dl": bias,
    }


def _apply_online_bias_correction(
    config: AppConfig,
    history: list[dict],
    generated_at: str,
    horizon_minutes: int,
    source: str,
    prediction_mg_dl: float,
) -> tuple[float, dict]:
    summary = _estimate_online_source_bias(
        config=config,
        history=history,
        generated_at=generated_at,
        horizon_minutes=horizon_minutes,
        source=source,
    )
    base_payload = {
        "source": source,
        "horizon_minutes": int(horizon_minutes),
        "raw_prediction_mg_dl": float(prediction_mg_dl),
    }
    if summary.get("status") != "trained":
        payload = dict(base_payload)
        payload.update(summary)
        payload["corrected_prediction_mg_dl"] = float(prediction_mg_dl)
        payload["applied_adjustment_mg_dl"] = 0.0
        return float(prediction_mg_dl), payload

    bias = float(summary["bias_mg_dl"])
    blend = float(np.clip(getattr(config, "adaptive_forecast_bias_blend", 0.45), 0.0, 1.0))
    max_adjustment = abs(float(getattr(config, "adaptive_forecast_bias_max_adjustment_mg_dl", 45.0)))
    adjustment = float(np.clip(blend * bias, -max_adjustment, max_adjustment))
    corrected = float(prediction_mg_dl - adjustment)
    payload = dict(base_payload)
    payload.update(summary)
    payload["blend"] = blend
    payload["max_adjustment_mg_dl"] = max_adjustment
    payload["applied_adjustment_mg_dl"] = adjustment
    payload["corrected_prediction_mg_dl"] = corrected
    return corrected, payload


def _is_jump_detected_from_simple_features(config: AppConfig, simple_feature_payload: dict) -> bool:
    slope_30 = abs(float(simple_feature_payload.get("slope_30", 0.0)))
    delta_15 = abs(float(simple_feature_payload.get("delta_15", 0.0)))
    return (
        slope_30 >= float(getattr(config, "jump_slope_threshold", 1.1))
        or delta_15 >= float(getattr(config, "jump_delta15_threshold", 18.0))
    )


def _compute_jump_ts_weight(config: AppConfig, simple_feature_payload: dict, jump_detected: bool) -> float:
    base_weight = float(np.clip(getattr(config, "jump_blend_base_ts_weight", 0.45), 0.0, 1.0))
    slope_30 = abs(float(simple_feature_payload.get("slope_30", 0.0)))
    delta_15 = abs(float(simple_feature_payload.get("delta_15", 0.0)))
    slope_ref = max(float(getattr(config, "jump_slope_ref", 1.0)), 1e-6)
    delta_ref = max(float(getattr(config, "jump_delta15_ref", 20.0)), 1e-6)
    slope_component = float(getattr(config, "jump_blend_slope_gain", 0.35)) * min(slope_30 / slope_ref, 1.0)
    delta_component = float(getattr(config, "jump_blend_delta_gain", 0.25)) * min(delta_15 / delta_ref, 1.0)
    bonus = float(getattr(config, "jump_ts_weight_bonus", 0.22)) if jump_detected else 0.0
    raw_weight = base_weight + slope_component + delta_component + bonus
    return float(np.clip(raw_weight, 0.0, float(getattr(config, "jump_blend_max_ts_weight", 0.9))))


def _annotate_time_series(ax, x_values, y_values, color, max_labels=4):
    if len(x_values) == 0:
        return
    sample_indexes = np.linspace(0, len(x_values) - 1, min(max_labels, len(x_values)), dtype=int)
    seen = set()
    for index in sample_indexes:
        if int(index) in seen:
            continue
        seen.add(int(index))
        ax.scatter([x_values[index]], [y_values[index]], color=color, s=20, zorder=5)
        ax.annotate(
            f"{float(y_values[index]):.0f}",
            (x_values[index], y_values[index]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )


def _load_forecast_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return []
    for item in payload:
        item.setdefault("horizon_minutes", 120)
    return payload


def _load_external_history_prior(config: AppConfig) -> dict:
    default_payload = {
        "enabled": bool(getattr(config, "external_physiology_priors_enabled", False)),
        "status": "disabled" if not getattr(config, "external_physiology_priors_enabled", False) else "unavailable",
        "source": str(config.external_physiology_priors_path),
        "applied_overrides": {},
    }
    if not getattr(config, "external_physiology_priors_enabled", False):
        return default_payload
    if config.external_physiology_priors_path.exists():
        with open(config.external_physiology_priors_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        overrides = payload.get("recommended_overrides", {}) if isinstance(payload, dict) else {}
        if overrides:
            return {
                "enabled": True,
                "status": "applied",
                "source": str(config.external_physiology_priors_path),
                "summary": payload.get("summary", {}),
                "paired_event_analysis": payload.get("paired_event_analysis", {}),
                "isolated_rapid_analysis": payload.get("isolated_rapid_analysis", {}),
                "settings": payload.get("settings", {}),
                "applied_overrides": overrides,
            }
    if not getattr(config, "external_history_priors_enabled", False):
        default_payload["status"] = "missing_physiology_priors"
        return default_payload
    if not config.external_history_report_path.exists():
        default_payload["source"] = str(config.external_history_report_path)
        return default_payload

    with open(config.external_history_report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    paired_events = int(summary.get("paired_meal_bolus_events", 0) or 0)
    recommended_ratio = payload.get("recommended_initial_cho_per_unit")
    if recommended_ratio is None:
        recommended_ratio = summary.get("cho_per_unit_median")
    if recommended_ratio is None:
        default_payload["status"] = "missing_ratio"
        return default_payload
    if paired_events < int(getattr(config, "external_history_min_paired_events", 12)):
        default_payload["status"] = "insufficient_pairs"
        default_payload["paired_meal_bolus_events"] = paired_events
        default_payload["recommended_cho_per_unit"] = float(recommended_ratio)
        return default_payload

    reference_ratio = max(float(getattr(config, "external_history_ratio_reference_g_per_unit", 8.0)), 1e-6)
    patient_ratio = max(float(recommended_ratio), 1e-6)
    ratio_scale = patient_ratio / reference_ratio
    prior_blend = float(np.clip(getattr(config, "external_history_prior_blend", 0.35), 0.0, 1.0))
    raw_meal_scale = float(np.clip(np.sqrt(reference_ratio / patient_ratio), 0.85, 1.25))
    raw_rapid_bio_scale = float(np.clip(np.sqrt(ratio_scale), 0.8, 1.1))
    meal_carb_scale = float(1.0 + prior_blend * (raw_meal_scale - 1.0))
    rapid_bio_scale = float(1.0 + prior_blend * (raw_rapid_bio_scale - 1.0))
    return {
        "enabled": True,
        "status": "applied",
        "source": str(config.external_history_report_path),
        "paired_meal_bolus_events": paired_events,
        "recommended_cho_per_unit": float(patient_ratio),
        "reference_cho_per_unit": float(reference_ratio),
        "prior_blend": prior_blend,
        "ratio_scale": float(ratio_scale),
        "applied_overrides": {
            "meal_carb_scale": meal_carb_scale,
            "insulin_rapid_bio_scale": rapid_bio_scale,
            "insulin_unknown_bio_scale": rapid_bio_scale,
        },
    }


def _update_forecast_history(
    config: AppConfig,
    generated_at: str,
    horizon_predictions: list[dict],
    anchor_state: np.ndarray,
    future_meals: list[dict],
    future_insulin: list[dict],
    regime_payload: dict,
    observed_timestamps: list[datetime],
    observed_glucose: np.ndarray,
) -> list[dict]:
    history = _load_forecast_history(config.forecast_history_path)
    observed_seconds = np.array([timestamp.timestamp() for timestamp in observed_timestamps], dtype=float)
    observed_values = np.asarray(observed_glucose, dtype=float)

    for item in history:
        target_timestamp = datetime.fromisoformat(item["forecast_target_timestamp"])
        target_seconds = target_timestamp.timestamp()
        if observed_seconds[0] <= target_seconds <= observed_seconds[-1]:
            observed_at_target = float(np.interp(target_seconds, observed_seconds, observed_values))
            item["observed_glucose_mg_dl"] = observed_at_target
            item["delta_mg_dl"] = float(item["predicted_glucose_mg_dl"] - observed_at_target)
            if item.get("physiological_predicted_glucose_mg_dl") is not None:
                item["physiological_delta_mg_dl"] = float(item["physiological_predicted_glucose_mg_dl"] - observed_at_target)
            if item.get("hybrid_predicted_glucose_mg_dl") is not None:
                item["hybrid_delta_mg_dl"] = float(item["hybrid_predicted_glucose_mg_dl"] - observed_at_target)
            if item.get("simple_predicted_glucose_mg_dl") is not None:
                item["simple_delta_mg_dl"] = float(item["simple_predicted_glucose_mg_dl"] - observed_at_target)
            if item.get("takagi_sugeno_ml_predicted_glucose_mg_dl") is not None:
                item["takagi_sugeno_ml_delta_mg_dl"] = float(item["takagi_sugeno_ml_predicted_glucose_mg_dl"] - observed_at_target)
            if item.get("takagi_sugeno_predicted_glucose_mg_dl") is not None:
                item["takagi_sugeno_delta_mg_dl"] = float(item["takagi_sugeno_predicted_glucose_mg_dl"] - observed_at_target)
            item["is_resolved"] = True

    for prediction in horizon_predictions:
        current_entry = {
            "generated_at": generated_at,
            "horizon_minutes": int(prediction["horizon_minutes"]),
            "forecast_target_timestamp": prediction["forecast_target_timestamp"],
            "predicted_glucose_mg_dl": float(prediction["predicted_glucose_mg_dl"]),
            "physiological_predicted_glucose_mg_dl": float(
                prediction.get("physiological_predicted_glucose_mg_dl", prediction["predicted_glucose_mg_dl"])
            ),
            "hybrid_predicted_glucose_mg_dl": float(
                prediction.get("hybrid_predicted_glucose_mg_dl", prediction["predicted_glucose_mg_dl"])
            ),
            "simple_predicted_glucose_mg_dl": float(
                prediction.get("simple_predicted_glucose_mg_dl", prediction["predicted_glucose_mg_dl"])
            ),
            "takagi_sugeno_predicted_glucose_mg_dl": float(
                prediction.get("takagi_sugeno_predicted_glucose_mg_dl", prediction["predicted_glucose_mg_dl"])
            ),
            "takagi_sugeno_ml_predicted_glucose_mg_dl": float(
                prediction.get("takagi_sugeno_ml_predicted_glucose_mg_dl", prediction["predicted_glucose_mg_dl"])
            ),
            "hybrid_prediction_status": prediction.get("hybrid_prediction_status", "unavailable"),
            "hybrid_predicted_residual_mg_dl": float(prediction.get("hybrid_predicted_residual_mg_dl", 0.0)),
            "final_prediction_source": prediction.get("final_prediction_source", "unknown"),
            "anchor_state": [float(value) for value in anchor_state.tolist()],
            "future_meals": future_meals,
            "future_insulin": future_insulin,
            "regime": _serialize_regime_payload(regime_payload),
            "observed_glucose_mg_dl": None,
            "delta_mg_dl": None,
            "is_resolved": False,
        }
        duplicate_index = next(
            (
                index
                for index, item in enumerate(history)
                if item["generated_at"] == generated_at
                and item["forecast_target_timestamp"] == prediction["forecast_target_timestamp"]
                and int(item.get("horizon_minutes", -1)) == int(prediction["horizon_minutes"])
            ),
            None,
        )
        if duplicate_index is None:
            history.append(current_entry)
        else:
            history[duplicate_index] = current_entry

    cutoff = datetime.fromisoformat(generated_at) - timedelta(days=7)
    history = [item for item in history if datetime.fromisoformat(item["generated_at"]) >= cutoff]
    write_json(config.forecast_history_path, history)
    return history


def _build_forecast_history_window(history: list[dict], generated_at: str, hours: int, horizon_minutes: int | None = None) -> list[dict]:
    cutoff = datetime.fromisoformat(generated_at) - timedelta(hours=hours)
    return [
        item
        for item in history
        if item.get("is_resolved") and datetime.fromisoformat(item["generated_at"]) >= cutoff
        and (horizon_minutes is None or int(item.get("horizon_minutes", -1)) == int(horizon_minutes))
    ]


def _serialize_relative_future_events(meal_events, insulin_events, forecast_anchor_minutes: float, forecast_minutes: int):
    meal_payload = []
    for event in meal_events:
        relative_minutes = float(event.minutes - forecast_anchor_minutes)
        if 0.0 <= relative_minutes <= forecast_minutes:
            meal_payload.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "minutes": relative_minutes,
                    "carbs_g": float(event.carbs_g),
                }
            )
    insulin_payload = []
    for event in insulin_events:
        relative_minutes = float(event.minutes - forecast_anchor_minutes)
        if 0.0 <= relative_minutes <= forecast_minutes:
            insulin_payload.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "minutes": relative_minutes,
                    "units": float(event.units),
                    "insulin_type": event.insulin_type,
                }
            )
    return meal_payload, insulin_payload


def _serialize_regime_payload(regime_payload: dict) -> dict:
    return {
        "name": regime_payload.get("name", "estavel"),
        "reason": regime_payload.get("reason", "default"),
        "current_hour": float(regime_payload.get("current_hour", 0.0)),
        "current_glucose_mg_dl": float(regime_payload.get("current_glucose_mg_dl", 0.0)),
        "recent_slope_mgdl_per_min": float(regime_payload.get("recent_slope_mgdl_per_min", 0.0)),
        "recent_variability_mgdl": float(regime_payload.get("recent_variability_mgdl", 0.0)),
        "minutes_since_meal": regime_payload.get("minutes_since_meal"),
        "minutes_since_insulin": regime_payload.get("minutes_since_insulin"),
        "recent_carbs_g": float(regime_payload.get("recent_carbs_g", 0.0)),
        "recent_insulin_units": float(regime_payload.get("recent_insulin_units", 0.0)),
        "recent_rapid_units": float(regime_payload.get("recent_rapid_units", 0.0)),
        "flags": dict(regime_payload.get("flags", {})),
    }


def _load_previous_state_snapshot(config: AppConfig) -> dict | None:
    if not config.state_path.exists():
        return None
    with open(config.state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _reuse_forecast_anchor_state(previous_state: dict | None, latest_timestamp_iso: str, current_state: np.ndarray) -> np.ndarray:
    if not previous_state:
        return current_state
    if previous_state.get("latest_timestamp") != latest_timestamp_iso:
        return current_state
    persisted = previous_state.get("latest_observation_state")
    if not persisted:
        return current_state
    vector = np.array(
        [
            persisted.get("S", 0.0),
            persisted.get("Ip", 0.0),
            persisted.get("Ii", 0.0),
            persisted.get("G", 0.0),
            persisted.get("h1", 0.0),
            persisted.get("h2", 0.0),
            persisted.get("h3", 0.0),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(vector)) or vector.shape != current_state.shape:
        return current_state
    return 0.5 * current_state + 0.5 * vector


def _estimate_recent_glucose_slope(observed_minutes: np.ndarray, observed_glucose: np.ndarray, window_minutes: float) -> float:
    end_time = float(observed_minutes[-1])
    start_time = max(float(observed_minutes[0]), end_time - float(window_minutes))
    mask = observed_minutes >= start_time
    window_t = observed_minutes[mask]
    window_g = observed_glucose[mask]
    if len(window_t) < 2:
        return float((observed_glucose[-1] - observed_glucose[-2]) / max(observed_minutes[-1] - observed_minutes[-2], 1e-6))
    centered_t = window_t - window_t.mean()
    slope, _ = np.polyfit(centered_t, window_g, 1)
    return float(slope)


def _glucose_rate_without_inputs(state: np.ndarray, params: UltradianParams) -> float:
    glucose_total = float(state[3])
    remote_insulin = float(state[2])
    delayed_state = float(state[6])
    return float(f4(delayed_state, params) - f2(glucose_total, params) - f3(remote_insulin, params) * glucose_total)


def _stabilize_forecast_anchor_state(
    anchor_state: np.ndarray,
    observed_minutes: np.ndarray,
    observed_glucose: np.ndarray,
    params: UltradianParams,
    config: AppConfig,
) -> tuple[np.ndarray, dict]:
    state = np.asarray(anchor_state, dtype=float).copy()
    recent_slope_mgdl_per_min = _estimate_recent_glucose_slope(
        observed_minutes=observed_minutes,
        observed_glucose=observed_glucose,
        window_minutes=config.forecast_anchor_slope_window_minutes,
    )
    target_floor_total_mg_per_min = 10.0 * params.Vg * max(
        recent_slope_mgdl_per_min,
        -float(config.forecast_anchor_max_drop_mgdl_per_min),
    )
    baseline_rate = _glucose_rate_without_inputs(state, params)
    if baseline_rate >= target_floor_total_mg_per_min:
        return state, {
            "applied": False,
            "reason": "within_floor",
            "recent_slope_mgdl_per_min": recent_slope_mgdl_per_min,
            "target_floor_total_mg_per_min": target_floor_total_mg_per_min,
            "baseline_rate_total_mg_per_min": baseline_rate,
            "adjusted_rate_total_mg_per_min": baseline_rate,
            "remote_state_scale": 1.0,
        }

    lower = float(config.forecast_anchor_min_scale)
    upper = 1.0
    adjusted = state.copy()
    for _ in range(32):
        scale = 0.5 * (lower + upper)
        candidate = state.copy()
        candidate[1] *= scale
        candidate[2] *= scale
        candidate[4] *= scale
        candidate[5] *= scale
        candidate[6] *= scale
        candidate_rate = _glucose_rate_without_inputs(candidate, params)
        if candidate_rate >= target_floor_total_mg_per_min:
            adjusted = candidate
            lower = scale
        else:
            upper = scale

    if np.allclose(adjusted, state):
        adjusted = state.copy()
        adjusted[1] *= float(config.forecast_anchor_min_scale)
        adjusted[2] *= float(config.forecast_anchor_min_scale)
        adjusted[4] *= float(config.forecast_anchor_min_scale)
        adjusted[5] *= float(config.forecast_anchor_min_scale)
        adjusted[6] *= float(config.forecast_anchor_min_scale)

    adjusted_rate = _glucose_rate_without_inputs(adjusted, params)
    remote_scale = float(adjusted[2] / max(state[2], 1e-9)) if state[2] > 0 else 1.0
    return adjusted, {
        "applied": True,
        "reason": "clamped_remote_state",
        "recent_slope_mgdl_per_min": recent_slope_mgdl_per_min,
        "target_floor_total_mg_per_min": target_floor_total_mg_per_min,
        "baseline_rate_total_mg_per_min": baseline_rate,
        "adjusted_rate_total_mg_per_min": adjusted_rate,
        "remote_state_scale": remote_scale,
    }


def _merge_forward_segments(historical_result: dict, forecast_result: dict) -> dict:
    return {
        "t": np.concatenate([historical_result["t"], forecast_result["t"][1:]]),
        "states": np.vstack([historical_result["states"], forecast_result["states"][1:]]),
        "glucose_mg_dl": np.concatenate([historical_result["glucose_mg_dl"], forecast_result["glucose_mg_dl"][1:]]),
        "plasma_insulin_uu_ml": np.concatenate(
            [historical_result["plasma_insulin_uu_ml"], forecast_result["plasma_insulin_uu_ml"][1:]]
        ),
        "remote_insulin_uu_ml": np.concatenate(
            [historical_result["remote_insulin_uu_ml"], forecast_result["remote_insulin_uu_ml"][1:]]
        ),
        "exercise_clearance_scale": np.concatenate(
            [historical_result.get("exercise_clearance_scale", np.array([])), forecast_result.get("exercise_clearance_scale", np.array([]))[1:]]
        ),
        "exercise_sensitivity_scale": np.concatenate(
            [historical_result.get("exercise_sensitivity_scale", np.array([])), forecast_result.get("exercise_sensitivity_scale", np.array([]))[1:]]
        ),
        "exercise_hepatic_scale": np.concatenate(
            [historical_result.get("exercise_hepatic_scale", np.array([])), forecast_result.get("exercise_hepatic_scale", np.array([]))[1:]]
        ),
        "exercise_hr_bpm": np.concatenate(
            [historical_result.get("exercise_hr_bpm", np.array([])), forecast_result.get("exercise_hr_bpm", np.array([]))[1:]]
        ),
    }


def _summarize_horizon_errors(history: list[dict]) -> dict:
    summary = {}
    for horizon in FORECAST_HORIZONS_MINUTES:
        resolved = [item["delta_mg_dl"] for item in history if item.get("is_resolved") and int(item.get("horizon_minutes", -1)) == horizon]
        if resolved:
            values = np.asarray(resolved, dtype=float)
            summary[f"plus_{horizon}_min"] = {
                "count": int(len(values)),
                "mae_mg_dl": float(np.mean(np.abs(values))),
                "rmse_mg_dl": float(np.sqrt(np.mean(values**2))),
                "mean_delta_mg_dl": float(np.mean(values)),
            }
        else:
            summary[f"plus_{horizon}_min"] = {
                "count": 0,
                "mae_mg_dl": None,
                "rmse_mg_dl": None,
                "mean_delta_mg_dl": None,
            }
    return summary


def save_plot(config: AppConfig, observed_minutes, observed_glucose, smooth_glucose, forward_result, meal_events, insulin_events, exercise_events, start_time) -> Path:
    display_start_minutes = max(float(observed_minutes[-1]) - config.display_window_hours * 60.0, 0.0)
    observed_mask = observed_minutes >= display_start_minutes
    forward_mask = forward_result["t"] >= display_start_minutes
    meal_events_window = [event for event in meal_events if display_start_minutes <= event.minutes <= forward_result["t"][-1]]
    insulin_events_window = [event for event in insulin_events if display_start_minutes <= event.minutes <= forward_result["t"][-1]]
    exercise_events_window = [event for event in exercise_events if display_start_minutes <= event.minutes <= forward_result["t"][-1]]

    observed_minutes_window = observed_minutes[observed_mask]
    observed_glucose_window = observed_glucose[observed_mask]
    smooth_glucose_window = smooth_glucose[observed_mask]
    forward_minutes_window = forward_result["t"][forward_mask]
    forward_glucose_window = forward_result["glucose_mg_dl"][forward_mask]
    plasma_window = forward_result["plasma_insulin_uu_ml"][forward_mask]
    remote_window = forward_result["remote_insulin_uu_ml"][forward_mask]
    exercise_hr_window = np.asarray(forward_result.get("exercise_hr_bpm", []), dtype=float)[forward_mask]

    observed_times = [start_time + timedelta(minutes=float(value)) for value in observed_minutes_window]
    forward_times = [start_time + timedelta(minutes=float(value)) for value in forward_minutes_window]
    forecast_start_time = start_time + timedelta(minutes=float(observed_minutes[-1]))

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(forward_times, forward_glucose_window, label="Forward", linewidth=2, color="#1d4ed8")
    axes[0].plot(observed_times, observed_glucose_window, "o", label="Observado", markersize=4, color="#b45309")
    axes[0].plot(observed_times, smooth_glucose_window, "--", label="Suavizado", linewidth=1.5, color="#7c2d12")
    axes[0].axvspan(forecast_start_time, forward_times[-1], color="#0f766e", alpha=0.08)
    axes[0].axvline(forecast_start_time, color="#0f766e", linestyle="--", linewidth=1.5)
    for event in meal_events_window:
        axes[0].axvline(start_time + timedelta(minutes=float(event.minutes)), color="#d97706", alpha=0.25, linewidth=1)
    for event in insulin_events_window:
        axes[0].axvline(start_time + timedelta(minutes=float(event.minutes)), color="#2563eb", alpha=0.2, linewidth=1)
    axes[0].set_ylabel("Glicose (mg/dL)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    _annotate_time_series(axes[0], observed_times, observed_glucose_window, "#b45309")
    _annotate_time_series(axes[0], forward_times, forward_glucose_window, "#1d4ed8")

    axes[1].plot(forward_times, plasma_window, label="Insulina plasmática", color="#2563eb")
    axes[1].plot(forward_times, remote_window, label="Insulina remota", color="#0891b2")
    axes[1].axvspan(forecast_start_time, forward_times[-1], color="#0f766e", alpha=0.08)
    axes[1].axvline(forecast_start_time, color="#0f766e", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Insulina (uU/mL)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _annotate_time_series(axes[1], forward_times, plasma_window, "#2563eb")
    _annotate_time_series(axes[1], forward_times, remote_window, "#0891b2")

    meal_minutes = [event.minutes for event in meal_events_window]
    meal_carbs = [event.carbs_g for event in meal_events_window]
    insulin_minutes = [event.minutes for event in insulin_events_window]
    insulin_units = [event.units for event in insulin_events_window]
    meal_times = [start_time + timedelta(minutes=float(value)) for value in meal_minutes]
    insulin_times = [start_time + timedelta(minutes=float(value)) for value in insulin_minutes]
    axes[2].bar(meal_times, meal_carbs, width=0.012, alpha=0.6, label="Carboidratos (g)", color="#f59e0b")
    axes[2].bar(insulin_times, insulin_units, width=0.01, alpha=0.6, label="Insulina (U)", color="#2563eb")
    if len(exercise_hr_window):
        exercise_axis = axes[2].twinx()
        exercise_axis.plot(forward_times, exercise_hr_window, color="#7c3aed", linewidth=1.8, label="FC (bpm)")
        exercise_axis.set_ylabel("FC (bpm)")
        for event in exercise_events_window:
            start = start_time + timedelta(minutes=float(event.minutes))
            end = start_time + timedelta(minutes=float(event.minutes + event.duration_minutes))
            axes[2].axvspan(start, end, color="#7c3aed", alpha=0.08)
        exercise_axis.legend(loc="upper right")
    axes[2].axvspan(forecast_start_time, forward_times[-1], color="#0f766e", alpha=0.08)
    axes[2].axvline(forecast_start_time, color="#0f766e", linestyle="--", linewidth=1.5)
    axes[2].set_ylabel("Entradas")
    axes[2].set_xlabel("Horário")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    for event_time, carbs in zip(meal_times, meal_carbs):
        axes[2].annotate(f"{carbs:.0f}", (event_time, carbs), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for event_time, units in zip(insulin_times, insulin_units):
        axes[2].annotate(f"{units:.1f}", (event_time, units), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", labelbottom=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))

    figure.tight_layout()
    plot_path = config.plots_dir / "latest_comparison.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def run_pipeline(config: AppConfig, skip_extract: bool = False) -> dict:
    _prepare_environment(config)
    if not skip_extract:
        run_extract_script(config)

    generated_at = datetime.now().isoformat()

    observed_minutes, observed_glucose, timestamps = load_glucose_json(config.glucose_json_path)
    start_time = timestamps[0]

    meal_events = load_meal_events(config.meals_source, start_time)
    insulin_events = load_insulin_events(config.insulin_source, start_time)
    exercise_events = load_exercise_events(
        config.exercise_source,
        start_time,
        baseline_hr_bpm=config.exercise_hr_baseline_bpm,
    )
    external_history_prior = _load_external_history_prior(config)

    base_param_updates = {
        "ka": config.insulin_absorption_rate,
        "insulin_bioavailability": config.insulin_bioavailability,
        "meal_tau": config.meal_tau_minutes,
        "exercise_hr_baseline_bpm": config.exercise_hr_baseline_bpm,
        "exercise_fast_tau_minutes": config.exercise_fast_tau_minutes,
        "exercise_slow_tau_minutes": config.exercise_slow_tau_minutes,
        "exercise_recovery_tau_minutes": config.exercise_recovery_tau_minutes,
        "exercise_clearance_gain": config.exercise_clearance_gain,
        "exercise_sensitivity_gain": config.exercise_sensitivity_gain,
        "exercise_hepatic_suppression_gain": config.exercise_hepatic_suppression_gain,
    }
    base_param_updates.update(external_history_prior.get("applied_overrides", {}))
    base_params = UltradianParams(**base_param_updates)
    previous_state = _load_previous_state_snapshot(config)
    forecast_history_existing = _load_forecast_history(config.forecast_history_path)
    forecast_model = train_forecast_model(config, forecast_history_existing, generated_at)
    _refresh_simple_forecast_model_assets(config)
    _refresh_takagi_sugeno_dashboard_assets(config)
    simple_forecast_model = load_simple_forecast_model(config)
    takagi_sugeno_ml_model = load_takagi_sugeno_ml_model(config)
    external_forecast_model = load_external_forecast_model(config)
    persisted_overrides = load_calibrated_params(config)
    seeded_params = apply_calibrated_params(base_params, persisted_overrides)
    params, calibration_info = calibrate_online_params(
        config=config,
        base_params=seeded_params,
        history=forecast_history_existing,
        generated_at=generated_at,
    )
    save_initial_calibration_snapshot(config, params, generated_at)
    regime_payload = detect_regime(
        config=config,
        observed_minutes=observed_minutes,
        observed_glucose=observed_glucose,
        timestamps=timestamps,
        meal_events=meal_events,
        insulin_events=insulin_events,
    )
    regime_overrides = regime_parameter_overrides(params, regime_payload) if config.regime_model_enabled else {}
    if regime_overrides:
        params = replace(params, **regime_overrides)
    regime_payload["applied_overrides"] = {key: float(value) for key, value in regime_overrides.items()}
    regime_payload["enabled"] = bool(config.regime_model_enabled)

    initial_state, inverse_result, state_payload = estimate_initial_state(
        t_data=observed_minutes,
        glucose_data_mg_dl=observed_glucose,
        params=params,
        smoothing_factor=config.smoothing_factor,
        meal_events=meal_events,
        insulin_events=insulin_events,
        estimator_mode=config.state_estimator_mode,
        ukf_settings=_ukf_settings_from_config(config),
        bayes_settings=_bayes_settings_from_config(config),
    )
    if config.state_estimator_mode == "bayes" and inverse_result.get("adapted_params"):
        params = replace(params, **inverse_result["adapted_params"])
        state_payload["params"] = asdict(params)

    observed_end_minutes = float(observed_minutes[-1])
    historical_result = run_forward(
        params=params,
        initial_state=initial_state,
        start_time_minutes=0.0,
        end_time_minutes=observed_end_minutes,
        meal_events=meal_events,
        insulin_events=insulin_events,
        exercise_events=exercise_events,
        disturbance={
            "t": inverse_result["t"],
            "values": inverse_result["inferred_glucose_input"],
            "cutoff_minutes": observed_end_minutes,
        },
        dt_minutes=config.forward_dt_minutes,
        extra_times=[observed_end_minutes],
    )
    if config.state_estimator_mode == "ukf":
        latest_observation_state = np.asarray(inverse_result["states"][-1], dtype=float).copy()
    else:
        latest_observation_state = historical_result["states"][-1].copy()
    forecast_anchor_state = _reuse_forecast_anchor_state(
        previous_state=previous_state,
        latest_timestamp_iso=timestamps[-1].isoformat(),
        current_state=latest_observation_state,
    )
    forecast_initial_state, anchor_regularization = _stabilize_forecast_anchor_state(
        anchor_state=forecast_anchor_state,
        observed_minutes=observed_minutes,
        observed_glucose=observed_glucose,
        params=params,
        config=config,
    )
    forecast_result = run_forward(
        params=params,
        initial_state=forecast_initial_state,
        start_time_minutes=observed_end_minutes,
        end_time_minutes=float(observed_end_minutes + config.forecast_minutes),
        meal_events=meal_events,
        insulin_events=insulin_events,
        exercise_events=exercise_events,
        disturbance=None,
        dt_minutes=config.forward_dt_minutes,
        extra_times=[observed_end_minutes + horizon for horizon in FORECAST_HORIZONS_MINUTES],
    )
    forward_result = _merge_forward_segments(historical_result, forecast_result)

    metrics = compare_with_observed(forward_result, observed_minutes, observed_glucose)
    plot_path = save_plot(
        config=config,
        observed_minutes=observed_minutes,
        observed_glucose=observed_glucose,
        smooth_glucose=inverse_result["glucose_smooth_mg_dl"],
        forward_result=forward_result,
        meal_events=meal_events,
        insulin_events=insulin_events,
        exercise_events=exercise_events,
        start_time=start_time,
    )

    simulated_timestamps = [
        (start_time + timedelta(minutes=float(minutes))).isoformat()
        for minutes in forward_result["t"]
    ]
    forecast_start_minutes = float(observed_minutes[-1])
    forecast_end_minutes = float(forward_result["t"][-1])
    forecast_mask = forward_result["t"] >= forecast_start_minutes
    forecast_glucose = forward_result["glucose_mg_dl"][forecast_mask]
    future_meals_payload, future_insulin_payload = _serialize_relative_future_events(
        meal_events=meal_events,
        insulin_events=insulin_events,
        forecast_anchor_minutes=observed_end_minutes,
        forecast_minutes=config.forecast_minutes,
    )
    takagi_sugeno_ml_predictions = predict_with_takagi_sugeno_ml(
        model_payload=takagi_sugeno_ml_model,
        timestamps=timestamps,
        observed_glucose=observed_glucose,
        meal_events=meal_events,
        insulin_events=insulin_events,
    )
    horizon_predictions = []
    hybrid_horizon_predictions = []
    simple_horizon_predictions = []
    ts_ml_horizon_predictions = []
    online_bias_corrections = []
    jump_adaptation_horizons = []
    for horizon in FORECAST_HORIZONS_MINUTES:
        horizon_minutes = observed_end_minutes + horizon
        horizon_timestamp = (timestamps[-1] + timedelta(minutes=horizon)).isoformat()
        predicted = float(np.interp(horizon_minutes, forward_result["t"], forward_result["glucose_mg_dl"]))
        simple_feature_payload = build_simple_feature_payload(
            observed_timestamps=timestamps,
            observed_glucose=observed_glucose,
            future_meals=future_meals_payload,
            future_insulin=future_insulin_payload,
            horizon_minutes=horizon,
        )
        simple_prediction = predict_with_simple_model(
            model_payload=simple_forecast_model,
            feature_payload=simple_feature_payload,
            horizon_minutes=horizon,
        )
        ts_ml_prediction = takagi_sugeno_ml_predictions.get(int(horizon), {})
        ts_ml_prediction_value = float(ts_ml_prediction.get("takagi_sugeno_ml_predicted_glucose_mg_dl", predicted))
        ts_prediction_value = float(ts_ml_prediction.get("takagi_sugeno_predicted_glucose_mg_dl", predicted))
        ts_ml_applied = int(horizon) in TS_ML_HORIZONS and bool(ts_ml_prediction)
        internal_hybrid_prediction = predict_with_forecast_model(
            model_payload=forecast_model,
            anchor_state=forecast_initial_state,
            future_meals=future_meals_payload,
            future_insulin=future_insulin_payload,
            physiological_prediction=predicted,
            horizon_minutes=horizon,
            regime_payload=regime_payload,
        )
        external_prediction = predict_with_external_forecast_model(
            model_payload=external_forecast_model,
            anchor_state=forecast_initial_state,
            future_meals=future_meals_payload,
            future_insulin=future_insulin_payload,
            horizon_minutes=horizon,
        )
        hybrid_prediction = blend_forecast_predictions(
            config=config,
            physiological_prediction=predicted,
            internal_prediction=internal_hybrid_prediction,
            external_prediction=external_prediction,
        )
        corrected_physiological, physiological_bias_payload = _apply_online_bias_correction(
            config=config,
            history=forecast_history_existing,
            generated_at=generated_at,
            horizon_minutes=horizon,
            source="physiological",
            prediction_mg_dl=float(predicted),
        )
        corrected_hybrid, hybrid_bias_payload = _apply_online_bias_correction(
            config=config,
            history=forecast_history_existing,
            generated_at=generated_at,
            horizon_minutes=horizon,
            source="hybrid",
            prediction_mg_dl=float(hybrid_prediction["prediction_mg_dl"]),
        )
        corrected_simple = float(simple_prediction.get("prediction_mg_dl", predicted))
        simple_bias_payload = {
            "status": "unavailable",
            "source": "simple",
            "horizon_minutes": int(horizon),
            "raw_prediction_mg_dl": corrected_simple,
            "corrected_prediction_mg_dl": corrected_simple,
            "applied_adjustment_mg_dl": 0.0,
        }
        if simple_prediction.get("applied"):
            corrected_simple, simple_bias_payload = _apply_online_bias_correction(
                config=config,
                history=forecast_history_existing,
                generated_at=generated_at,
                horizon_minutes=horizon,
                source="simple",
                prediction_mg_dl=float(simple_prediction.get("prediction_mg_dl", predicted)),
            )

        corrected_ts_ml = ts_ml_prediction_value
        ts_ml_bias_payload = {
            "status": "unavailable",
            "source": "ts_ml",
            "horizon_minutes": int(horizon),
            "raw_prediction_mg_dl": corrected_ts_ml,
            "corrected_prediction_mg_dl": corrected_ts_ml,
            "applied_adjustment_mg_dl": 0.0,
        }
        if ts_ml_applied:
            corrected_ts_ml, ts_ml_bias_payload = _apply_online_bias_correction(
                config=config,
                history=forecast_history_existing,
                generated_at=generated_at,
                horizon_minutes=horizon,
                source="ts_ml",
                prediction_mg_dl=ts_ml_prediction_value,
            )

        jump_aware_enabled = bool(getattr(config, "jump_aware_forecast_enabled", False))
        jump_detected = _is_jump_detected_from_simple_features(config, simple_feature_payload) if jump_aware_enabled else False
        jump_ts_weight = None
        jump_blend_prediction = None
        jump_blend_applied = False
        if jump_aware_enabled and ts_ml_applied and simple_prediction.get("applied"):
            jump_ts_weight = _compute_jump_ts_weight(config, simple_feature_payload, jump_detected)
            jump_blend_prediction = float((1.0 - jump_ts_weight) * corrected_simple + jump_ts_weight * corrected_ts_ml)

        online_bias_corrections.append(
            {
                "horizon_minutes": int(horizon),
                "physiological": physiological_bias_payload,
                "hybrid": hybrid_bias_payload,
                "simple": simple_bias_payload,
                "ts_ml": ts_ml_bias_payload,
            }
        )
        jump_adaptation_horizons.append(
            {
                "horizon_minutes": int(horizon),
                "enabled": jump_aware_enabled,
                "jump_detected": bool(jump_detected),
                "force_blend_when_detected": bool(getattr(config, "jump_force_blend_when_detected", True)),
                "ts_weight": jump_ts_weight,
                "blend_prediction_mg_dl": jump_blend_prediction,
                "simple_slope_30_mgdl_per_min": float(simple_feature_payload.get("slope_30", 0.0)),
                "simple_delta_15_mg_dl": float(simple_feature_payload.get("delta_15", 0.0)),
            }
        )
        final_prediction = float(corrected_hybrid)
        final_source = ",".join(hybrid_prediction["sources"]) if hybrid_prediction["sources"] else "physiological"
        if (
            jump_aware_enabled
            and jump_detected
            and bool(getattr(config, "jump_force_blend_when_detected", True))
            and jump_blend_prediction is not None
        ):
            final_prediction = float(jump_blend_prediction)
            jump_blend_applied = True
            final_source = "jump_blend(simple,ts_ml)"
        elif getattr(config, "takagi_sugeno_ml_enabled", False) and getattr(config, "takagi_sugeno_ml_as_primary", False) and ts_ml_applied:
            final_prediction = float(corrected_ts_ml)
            final_source = "ts_ml+online_bias" if ts_ml_bias_payload.get("status") == "trained" else "ts_ml"
        elif config.simple_forecast_model_enabled and config.simple_forecast_as_primary and simple_prediction.get("applied"):
            final_prediction = float(corrected_simple)
            final_source = "simple+online_bias" if simple_bias_payload.get("status") == "trained" else "simple"
        elif final_source == "physiological":
            final_prediction = float(corrected_physiological)
            if physiological_bias_payload.get("status") == "trained":
                final_source = "physiological+online_bias"
        elif hybrid_bias_payload.get("status") == "trained":
            final_source = f"{final_source}+online_bias"
        horizon_predictions.append(
            {
                "horizon_minutes": horizon,
                "forecast_target_timestamp": horizon_timestamp,
                "predicted_glucose_mg_dl": final_prediction,
                "physiological_predicted_glucose_mg_dl": float(corrected_physiological),
                "physiological_raw_predicted_glucose_mg_dl": predicted,
                "hybrid_predicted_glucose_mg_dl": float(corrected_hybrid),
                "hybrid_raw_predicted_glucose_mg_dl": float(hybrid_prediction["prediction_mg_dl"]),
                "hybrid_prediction_status": ",".join(hybrid_prediction["sources"]) if hybrid_prediction["sources"] else "unavailable",
                "hybrid_predicted_residual_mg_dl": float(corrected_hybrid - corrected_physiological),
                "simple_predicted_glucose_mg_dl": float(corrected_simple),
                "simple_raw_predicted_glucose_mg_dl": float(simple_prediction.get("prediction_mg_dl", predicted)),
                "simple_prediction_status": simple_prediction.get("status", "unavailable"),
                "takagi_sugeno_predicted_glucose_mg_dl": ts_prediction_value,
                "takagi_sugeno_ml_predicted_glucose_mg_dl": float(corrected_ts_ml),
                "takagi_sugeno_ml_raw_predicted_glucose_mg_dl": ts_ml_prediction_value,
                "takagi_sugeno_ml_prediction_status": "trained" if ts_ml_applied else "unavailable",
                "final_prediction_source": final_source,
                "internal_hybrid_prediction_mg_dl": float(hybrid_prediction["internal_prediction_mg_dl"]),
                "external_hybrid_prediction_mg_dl": float(hybrid_prediction["external_prediction_mg_dl"]),
                "online_bias": {
                    "physiological": physiological_bias_payload,
                    "hybrid": hybrid_bias_payload,
                    "simple": simple_bias_payload,
                    "ts_ml": ts_ml_bias_payload,
                },
                "jump_aware": {
                    "enabled": jump_aware_enabled,
                    "jump_detected": bool(jump_detected),
                    "jump_blend_applied": bool(jump_blend_applied),
                    "jump_ts_weight": jump_ts_weight,
                    "jump_blend_prediction_mg_dl": jump_blend_prediction,
                    "jump_slope_30_mgdl_per_min": float(simple_feature_payload.get("slope_30", 0.0)),
                    "jump_delta_15_mg_dl": float(simple_feature_payload.get("delta_15", 0.0)),
                },
            }
        )
        hybrid_horizon_predictions.append(
            {
                "horizon_minutes": horizon,
                "forecast_target_timestamp": horizon_timestamp,
                "predicted_glucose_mg_dl": float(corrected_hybrid),
                "physiological_predicted_glucose_mg_dl": float(corrected_physiological),
                "prediction_status": ",".join(hybrid_prediction["sources"]) if hybrid_prediction["sources"] else "unavailable",
                "predicted_residual_mg_dl": float(corrected_hybrid - corrected_physiological),
                "internal_prediction_mg_dl": float(hybrid_prediction["internal_prediction_mg_dl"]),
                "external_prediction_mg_dl": float(hybrid_prediction["external_prediction_mg_dl"]),
                "online_bias": hybrid_bias_payload,
            }
        )
        simple_horizon_predictions.append(
            {
                "horizon_minutes": horizon,
                "forecast_target_timestamp": horizon_timestamp,
                "predicted_glucose_mg_dl": float(corrected_simple),
                "prediction_status": simple_prediction.get("status", "unavailable"),
                "predicted_delta_mg_dl": float(simple_prediction.get("predicted_delta_mg_dl", 0.0)),
                "online_bias": simple_bias_payload,
            }
        )
        ts_ml_horizon_predictions.append(
            {
                "horizon_minutes": horizon,
                "forecast_target_timestamp": horizon_timestamp,
                "predicted_glucose_mg_dl": float(corrected_ts_ml),
                "takagi_sugeno_predicted_glucose_mg_dl": ts_prediction_value,
                "prediction_status": "trained" if ts_ml_applied else "unavailable",
                "online_bias": ts_ml_bias_payload,
            }
        )
    forecast_history = _update_forecast_history(
        config=config,
        generated_at=generated_at,
        horizon_predictions=horizon_predictions,
        anchor_state=forecast_initial_state,
        future_meals=future_meals_payload,
        future_insulin=future_insulin_payload,
        regime_payload=regime_payload,
        observed_timestamps=timestamps,
        observed_glucose=observed_glucose,
    )
    forecast_history_window = _build_forecast_history_window(
        history=forecast_history,
        generated_at=generated_at,
        hours=FORECAST_DELTA_WINDOW_HOURS,
        horizon_minutes=None,
    )
    forecast_history_window_12h = _build_forecast_history_window(
        history=forecast_history,
        generated_at=generated_at,
        hours=max(int(getattr(config, "display_window_hours", 12)), 2),
        horizon_minutes=None,
    )
    horizon_metrics = _summarize_horizon_errors(forecast_history)
    metrics_payload = {
        "generated_at": generated_at,
        "glucose_points": int(len(observed_glucose)),
        "window_minutes": float(observed_minutes[-1]),
        "forecast_minutes": int(config.forecast_minutes),
        "metrics": metrics,
        "forecast_horizon_metrics": horizon_metrics,
        "calibration": calibration_info,
        "forecast_model": forecast_model,
        "simple_forecast_model": simple_forecast_model,
        "takagi_sugeno_ml_model": takagi_sugeno_ml_model.get("benchmark", {}),
        "external_forecast_model": external_forecast_model,
        "external_history_prior": external_history_prior,
        "regime": _serialize_regime_payload(regime_payload),
        "state_estimator_mode": config.state_estimator_mode,
        "forecast_anchor_regularization": anchor_regularization,
        "forecast": {
            "start_timestamp": timestamps[-1].isoformat(),
            "end_timestamp": simulated_timestamps[-1],
            "glucose_now_mg_dl": float(observed_glucose[-1]),
            "glucose_in_2h_mg_dl": float(horizon_predictions[-1]["predicted_glucose_mg_dl"]),
            "physiological_glucose_in_2h_mg_dl": float(forward_result["glucose_mg_dl"][-1]),
            "hybrid_glucose_in_2h_mg_dl": float(hybrid_horizon_predictions[-1]["predicted_glucose_mg_dl"]),
            "simple_glucose_in_2h_mg_dl": float(simple_horizon_predictions[-1]["predicted_glucose_mg_dl"]),
            "takagi_sugeno_ml_glucose_in_1h_mg_dl": float(
                next(
                    item["predicted_glucose_mg_dl"]
                    for item in ts_ml_horizon_predictions
                    if int(item["horizon_minutes"]) == 60
                )
            ),
            "forecast_primary_source": horizon_predictions[-1]["final_prediction_source"],
            "forecast_min_mg_dl": float(forecast_glucose.min()),
            "forecast_max_mg_dl": float(forecast_glucose.max()),
            "horizons": horizon_predictions,
            "hybrid_horizons": hybrid_horizon_predictions,
            "simple_horizons": simple_horizon_predictions,
            "takagi_sugeno_ml_horizons": ts_ml_horizon_predictions,
        },
        "forecast_history_summary": {
            "window_hours": FORECAST_DELTA_WINDOW_HOURS,
            "resolved_points_last_2h": len(forecast_history_window),
            "resolved_points_last_12h": len(forecast_history_window_12h),
            "mean_delta_mg_dl": float(np.mean([item["delta_mg_dl"] for item in forecast_history_window]))
            if forecast_history_window
            else None,
        },
        "online_bias_adaptation": {
            "enabled": bool(getattr(config, "adaptive_forecast_bias_enabled", False)),
            "lookback_hours": int(getattr(config, "adaptive_forecast_bias_lookback_hours", 24)),
            "min_points": int(getattr(config, "adaptive_forecast_bias_min_points", 6)),
            "blend": float(getattr(config, "adaptive_forecast_bias_blend", 0.45)),
            "max_adjustment_mg_dl": float(getattr(config, "adaptive_forecast_bias_max_adjustment_mg_dl", 45.0)),
            "horizons": online_bias_corrections,
        },
        "jump_adaptation": {
            "enabled": bool(getattr(config, "jump_aware_forecast_enabled", False)),
            "force_blend_when_detected": bool(getattr(config, "jump_force_blend_when_detected", True)),
            "jump_slope_threshold": float(getattr(config, "jump_slope_threshold", 1.1)),
            "jump_delta15_threshold": float(getattr(config, "jump_delta15_threshold", 18.0)),
            "jump_blend_base_ts_weight": float(getattr(config, "jump_blend_base_ts_weight", 0.45)),
            "jump_blend_slope_gain": float(getattr(config, "jump_blend_slope_gain", 0.35)),
            "jump_blend_delta_gain": float(getattr(config, "jump_blend_delta_gain", 0.25)),
            "jump_blend_max_ts_weight": float(getattr(config, "jump_blend_max_ts_weight", 0.9)),
            "jump_ts_weight_bonus": float(getattr(config, "jump_ts_weight_bonus", 0.22)),
            "horizons": jump_adaptation_horizons,
        },
        "inputs": {
            "meals_count": len(meal_events),
            "insulin_count": len(insulin_events),
            "exercise_count": len(exercise_events),
        },
        "files": {
            "plot": str(plot_path),
            "glucose_json": str(config.glucose_json_path),
        },
    }
    state_payload.update(
        {
            "generated_at": generated_at,
            "latest_glucose_mg_dl": float(observed_glucose[-1]),
            "latest_timestamp": timestamps[-1].isoformat(),
            "forecast_minutes": int(config.forecast_minutes),
            "forecast_end_timestamp": simulated_timestamps[-1],
            "calibration": calibration_info,
            "forecast_model": forecast_model,
            "simple_forecast_model": simple_forecast_model,
            "takagi_sugeno_ml_model": takagi_sugeno_ml_model.get("benchmark", {}),
            "external_forecast_model": external_forecast_model,
            "external_history_prior": external_history_prior,
            "jump_adaptation": {
                "enabled": bool(getattr(config, "jump_aware_forecast_enabled", False)),
                "force_blend_when_detected": bool(getattr(config, "jump_force_blend_when_detected", True)),
                "jump_slope_threshold": float(getattr(config, "jump_slope_threshold", 1.1)),
                "jump_delta15_threshold": float(getattr(config, "jump_delta15_threshold", 18.0)),
            },
            "regime": _serialize_regime_payload(regime_payload),
            "state_estimator_mode": config.state_estimator_mode,
            "forecast_anchor_regularization": anchor_regularization,
            "latest_observation_state": {
                "S": float(latest_observation_state[0]),
                "Ip": float(latest_observation_state[1]),
                "Ii": float(latest_observation_state[2]),
                "G": float(latest_observation_state[3]),
                "h1": float(latest_observation_state[4]),
                "h2": float(latest_observation_state[5]),
                "h3": float(latest_observation_state[6]),
            },
            "forecast_initial_state": {
                "S": float(forecast_initial_state[0]),
                "Ip": float(forecast_initial_state[1]),
                "Ii": float(forecast_initial_state[2]),
                "G": float(forecast_initial_state[3]),
                "h1": float(forecast_initial_state[4]),
                "h2": float(forecast_initial_state[5]),
                "h3": float(forecast_initial_state[6]),
            },
            "meals": serialize_events(meal_events),
            "insulin": serialize_events(insulin_events),
            "exercise": serialize_events(exercise_events),
        }
    )
    series_payload = {
        "generated_at": generated_at,
        "t_minutes": forward_result["t"].tolist(),
        "timestamps": simulated_timestamps,
        "forecast_start_minutes": forecast_start_minutes,
        "forecast_end_minutes": forecast_end_minutes,
        "forecast_start_timestamp": timestamps[-1].isoformat(),
        "forecast_end_timestamp": simulated_timestamps[-1],
        "glucose_mg_dl": forward_result["glucose_mg_dl"].tolist(),
        "plasma_insulin_uu_ml": forward_result["plasma_insulin_uu_ml"].tolist(),
        "remote_insulin_uu_ml": forward_result["remote_insulin_uu_ml"].tolist(),
        "exercise_hr_bpm": forward_result.get("exercise_hr_bpm", np.array([])).tolist(),
        "exercise_clearance_scale": forward_result.get("exercise_clearance_scale", np.array([])).tolist(),
        "exercise_sensitivity_scale": forward_result.get("exercise_sensitivity_scale", np.array([])).tolist(),
        "inferred_glucose_input_mg_min": inverse_result["inferred_glucose_input"].tolist(),
        "observed_glucose_mg_dl": observed_glucose.tolist(),
        "observed_t_minutes": observed_minutes.tolist(),
        "observed_timestamps": [timestamp.isoformat() for timestamp in timestamps],
        "hybrid_forecast_horizons": hybrid_horizon_predictions,
        "simple_forecast_horizons": simple_horizon_predictions,
        "takagi_sugeno_ml_forecast_horizons": ts_ml_horizon_predictions,
        "takagi_sugeno_ml_rollout": takagi_sugeno_ml_predictions.get("rollout", {}),
        "jump_adaptation_horizons": jump_adaptation_horizons,
        "forecast_history": forecast_history_window,
        "regime": _serialize_regime_payload(regime_payload),
    }

    write_json(config.metrics_path, metrics_payload)
    write_json(config.state_path, state_payload)
    write_json(config.series_path, series_payload)

    return {
        "generated_at": generated_at,
        "metrics_path": str(config.metrics_path),
        "state_path": str(config.state_path),
        "series_path": str(config.series_path),
        "plot_path": str(plot_path),
        "rmse_mg_dl": metrics["rmse_mg_dl"],
        "mae_mg_dl": metrics["mae_mg_dl"],
    }


def sleep_until_next_run(refresh_minutes: int) -> None:
    now = datetime.now()
    interval_seconds = refresh_minutes * 60
    current_bucket = int(now.timestamp() // interval_seconds)
    next_run = datetime.fromtimestamp((current_bucket + 1) * interval_seconds)
    delay = max((next_run - now).total_seconds(), 1.0)
    time.sleep(delay)


def run_forever(config: AppConfig, skip_extract: bool = False) -> None:
    while True:
        try:
            summary = run_pipeline(config, skip_extract=skip_extract)
            print(f"[{summary['generated_at']}] pipeline concluído | RMSE={summary['rmse_mg_dl']:.2f} mg/dL")
        except Exception as exc:
            print(f"[{datetime.now().isoformat()}] erro no pipeline: {exc}", file=sys.stderr)
        sleep_until_next_run(config.refresh_minutes)

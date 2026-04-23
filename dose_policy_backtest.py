import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from external_history import build_direct_forecast_samples, load_external_history
from simple_predictor import SIMPLE_FEATURE_NAMES


HORIZONS = (15, 30, 60, 120)
REGIMES = ("low", "medium", "high", "jump")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest de política de dose (simulação/surrogate, não clínico)")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/backtest_dose_policy.json")
    parser.add_argument("--max-rows-per-horizon", type=int, default=12000)
    parser.add_argument("--min-train-points", type=int, default=384)
    parser.add_argument("--retrain-every", type=int, default=120)
    parser.add_argument("--retrain-every-high-slope", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--l2", type=float, default=8.0)
    parser.add_argument("--target-min", type=float, default=70.0)
    parser.add_argument("--target-max", type=float, default=180.0)
    parser.add_argument("--target-mid", type=float, default=110.0)
    parser.add_argument("--base-isf", type=float, default=45.0)
    parser.add_argument("--isf-lookback", type=int, default=288)
    parser.add_argument("--isf-min-points", type=int, default=16)
    parser.add_argument("--isf-min", type=float, default=15.0)
    parser.add_argument("--isf-max", type=float, default=120.0)
    parser.add_argument("--min-insulin-units-observed", type=float, default=0.5)
    parser.add_argument("--max-bolus", type=float, default=4.0)
    parser.add_argument("--iob-threshold", type=float, default=3.0)
    parser.add_argument("--slope-low-threshold", type=float, default=0.35)
    parser.add_argument("--slope-high-threshold", type=float, default=0.80)
    parser.add_argument("--jump-slope-threshold", type=float, default=1.10)
    parser.add_argument("--jump-delta15-threshold", type=float, default=18.0)
    parser.add_argument("--carb-ratio-g-per-u", type=float, default=10.0)
    parser.add_argument("--carb-coverage-fraction", type=float, default=0.20)
    parser.add_argument("--uptrend-dose-gain", type=float, default=0.35)
    parser.add_argument("--event-recent-minutes", type=float, default=90.0)
    parser.add_argument("--committee-threshold", type=float, default=0.20)
    parser.add_argument("--pk-rapid-tau-min", type=float, default=75.0)
    parser.add_argument("--pk-basal-tau-min", type=float, default=360.0)
    return parser


def _feature_vector(sample: dict) -> np.ndarray:
    return np.asarray([float(sample.get(name, 0.0)) for name in SIMPLE_FEATURE_NAMES], dtype=float)


def _train_ridge(features: np.ndarray, targets: np.ndarray, l2: float) -> dict:
    means = features.mean(axis=0)
    scales = np.where(features.std(axis=0) < 1e-9, 1.0, features.std(axis=0))
    normalized = (features - means) / scales
    identity = np.eye(normalized.shape[1], dtype=float)
    identity[0, 0] = 0.0
    system = normalized.T @ normalized + float(l2) * identity
    rhs = normalized.T @ targets
    try:
        coeffs = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.pinv(system) @ rhs
    return {"feature_means": means, "feature_scales": scales, "weights": coeffs}


def _predict_ridge(model: dict, row: dict) -> float:
    vec = _feature_vector(row)
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    normalized = (vec - means) / np.where(np.abs(scales) < 1e-9, 1.0, scales)
    delta = float(normalized @ weights)
    return float(row["current_glucose"] + delta)


def _regime(row: dict, args) -> str:
    slope = abs(float(row.get("slope_30", 0.0)))
    delta_15 = abs(float(row.get("delta_15", 0.0)))
    if slope >= float(args.jump_slope_threshold) or delta_15 >= float(args.jump_delta15_threshold):
        return "jump"
    if slope >= float(args.slope_high_threshold):
        return "high"
    if slope >= float(args.slope_low_threshold):
        return "medium"
    return "low"


def _estimate_isf(row: dict, history: list[float], args) -> tuple[float, str]:
    lookback = max(int(args.isf_lookback), 1)
    recent = history[-lookback:]
    source = "base"
    if len(recent) >= int(args.isf_min_points):
        isf_emp = float(np.median(np.asarray(recent, dtype=float)))
        source = "adaptive"
    else:
        isf_emp = float(args.base_isf)
    glucose = float(row.get("current_glucose", 0.0))
    std_60 = float(row.get("std_60", 0.0))
    resistance_factor = 1.0
    if glucose >= 180.0:
        resistance_factor *= 0.92
    if glucose >= 220.0:
        resistance_factor *= 0.90
    variability_factor = 1.0 - min(std_60 / 100.0, 0.15)
    isf = isf_emp * resistance_factor * variability_factor
    isf = float(np.clip(isf, float(args.isf_min), float(args.isf_max)))
    return isf, source


def _horizon_action_fraction(horizon_min: int) -> float:
    if horizon_min <= 15:
        return 0.25
    if horizon_min <= 30:
        return 0.45
    if horizon_min <= 60:
        return 0.75
    return 1.0


def _pk_active_insulin_proxy_u(row: dict, args) -> float:
    mins_since_rapid = float(row.get("minutes_since_rapid", 1e9))
    mins_since_meal = float(row.get("minutes_since_meal", 1e9))
    rapid_last_30 = float(row.get("rapid_last_30", 0.0))
    rapid_last_60 = float(row.get("rapid_last_60", 0.0))
    rapid_last_120 = float(row.get("rapid_last_120", 0.0))
    tau_rapid = max(float(args.pk_rapid_tau_min), 1e-6)
    tau_basal = max(float(args.pk_basal_tau_min), 1e-6)

    rapid_decay = np.exp(-max(mins_since_rapid, 0.0) / tau_rapid) if mins_since_rapid < 1e8 else 0.0
    meal_decay = np.exp(-max(mins_since_meal, 0.0) / tau_basal) if mins_since_meal < 1e8 else 0.0
    # Proxy de IOB ativo ponderando janelas recentes e decaimento temporal.
    active_u = (
        0.55 * rapid_last_30
        + 0.30 * rapid_last_60
        + 0.15 * rapid_last_120
    ) * rapid_decay + 0.10 * meal_decay
    return float(max(active_u, 0.0))


def _recommend_dose_u(row: dict, forecast_glucose: float, isf: float, args) -> tuple[float, dict]:
    target_min = float(args.target_min)
    target_max = float(args.target_max)
    target_mid = float(args.target_mid)
    slope = float(row.get("slope_30", 0.0))
    delta_15 = float(row.get("delta_15", 0.0))
    future_carbs = float(row.get("future_carbs_total", 0.0))
    carbs_last_30 = float(row.get("carbs_last_30", 0.0))
    rapid_last_120 = float(row.get("rapid_last_120", 0.0))
    mins_since_meal = float(row.get("minutes_since_meal", 1e9))
    mins_since_rapid = float(row.get("minutes_since_rapid", 1e9))
    std_60 = float(row.get("std_60", 0.0))
    range_60 = float(row.get("range_60", 0.0))

    current_glucose = float(row.get("current_glucose", forecast_glucose))
    # Fórmula pedida: dose_total = sensibilidade*CHO + correção; correção = (glicemia - 140)/30.
    sensitivity_u_per_g = float(np.clip(3.4 / max(isf, 1e-6), 0.02, 0.20))
    cho_g = max(0.0, future_carbs)
    cho_u = sensitivity_u_per_g * cho_g
    correction_formula_u = float((current_glucose - 140.0) / 30.0)
    formula_dose_u = max(0.0, cho_u + correction_formula_u)

    correction_u = max(0.0, (forecast_glucose - target_mid) / max(isf, 1e-6))

    up_signal = max(0.0, slope) + max(0.0, delta_15 / 10.0)
    trend_u = float(args.uptrend_dose_gain) * up_signal / max(isf / 45.0, 0.5)

    carb_load = future_carbs + float(args.carb_coverage_fraction) * carbs_last_30
    carb_u = max(0.0, carb_load / max(float(args.carb_ratio_g_per_u), 1e-6))

    anti_hypo_guard = 1.0
    if forecast_glucose <= target_min or (slope < -0.35 and delta_15 < 0.0):
        anti_hypo_guard = 0.0
    elif forecast_glucose < target_mid:
        anti_hypo_guard = 0.35

    recent_event_guard = 1.0
    if mins_since_meal <= float(args.event_recent_minutes) and mins_since_rapid <= float(args.event_recent_minutes):
        recent_event_guard = 0.85

    uncertainty_guard = float(np.clip(1.0 - 0.004 * std_60 - 0.0015 * range_60, 0.35, 1.0))

    # Comitê de microagentes usa a fórmula como âncora e ajusta por contexto.
    agent_outputs = {
        "formula": formula_dose_u,
        "correction": correction_u,
        "trend": trend_u,
        "carb": 0.6 * carb_u,
        "safety": min(formula_dose_u, 0.8),
    }
    weights = {"formula": 0.50, "correction": 0.15, "trend": 0.15, "carb": 0.10, "safety": 0.10}
    base_u = float(sum(weights[k] * agent_outputs[k] for k in agent_outputs))

    # Voto direcional simples do comitê.
    votes = 0.0
    votes += 1.0 if correction_u > 0 else -0.5
    votes += 1.0 if (slope > 0.2 or delta_15 > 4.0) else -0.5
    votes += 1.0 if future_carbs > 0 else -0.5
    votes += -1.0 if forecast_glucose < target_mid else 0.5
    if votes < float(args.committee_threshold):
        base_u *= 0.55

    dose_u = base_u * anti_hypo_guard * recent_event_guard * uncertainty_guard

    # Farmacocinética: atenua dose adicional quando IOB ativo estimado já é alto.
    pk_active_u = _pk_active_insulin_proxy_u(row, args)
    pk_attenuation = float(np.clip(pk_active_u / 6.0, 0.0, 0.65))
    dose_u *= (1.0 - pk_attenuation)

    # Cap de IOB: se já há insulina rápida recente alta, limita dose adicional.
    max_bolus = float(args.max_bolus)
    if rapid_last_120 >= float(args.iob_threshold) or pk_active_u >= float(args.iob_threshold):
        max_bolus = min(max_bolus, 1.0)
    dose_u = float(np.clip(dose_u, 0.0, max_bolus))

    return dose_u, {
        "formula_dose_u": float(formula_dose_u),
        "sensitivity_u_per_g": float(sensitivity_u_per_g),
        "cho_u": float(cho_u),
        "correction_formula_u": float(correction_formula_u),
        "correction_u": float(correction_u),
        "trend_u": float(trend_u),
        "carb_u": float(carb_u),
        "anti_hypo_guard": float(anti_hypo_guard),
        "uncertainty_guard": float(uncertainty_guard),
        "pk_active_u": float(pk_active_u),
        "pk_attenuation": float(pk_attenuation),
    }


def _metrics(errors: list[float], values: list[float], target_min: float, target_max: float) -> dict:
    if not errors:
        return {
            "count": 0,
            "rmse_mg_dl": None,
            "mae_mg_dl": None,
            "bias_mg_dl": None,
            "tir_pct": None,
            "tbr_pct": None,
            "tar_pct": None,
        }
    err = np.asarray(errors, dtype=float)
    val = np.asarray(values, dtype=float)
    tir = float(np.mean((val >= target_min) & (val <= target_max)) * 100.0)
    tbr = float(np.mean(val < target_min) * 100.0)
    tar = float(np.mean(val > target_max) * 100.0)
    return {
        "count": int(len(err)),
        "rmse_mg_dl": float(np.sqrt(np.mean(err**2))),
        "mae_mg_dl": float(np.mean(np.abs(err))),
        "bias_mg_dl": float(np.mean(err)),
        "tir_pct": tir,
        "tbr_pct": tbr,
        "tar_pct": tar,
    }


def _slice_rows_by_horizon(samples: list[dict], max_rows_per_horizon: int) -> dict[int, list[dict]]:
    grouped = {h: [] for h in HORIZONS}
    for row in samples:
        h = int(row.get("horizon_minutes", -1))
        if h in grouped:
            grouped[h].append(row)
    for h in HORIZONS:
        grouped[h].sort(key=lambda item: item["timestamp"])
        if max_rows_per_horizon > 0:
            grouped[h] = grouped[h][-max_rows_per_horizon:]
    return grouped


def _backtest_horizon(rows: list[dict], horizon: int, args) -> dict:
    if len(rows) <= int(args.min_train_points):
        return {"horizon_minutes": int(horizon), "rows_total": len(rows), "rows_tested": 0}

    min_train = int(args.min_train_points)
    max_train_rows = max(int(args.max_train_rows), min_train)
    retrain_base = max(int(args.retrain_every), 1)
    retrain_high = max(int(args.retrain_every_high_slope), 1)
    start_idx = min_train

    model = None
    last_retrain = None

    forecast_errors = []
    forecast_values = []
    policy_errors = []
    policy_values = []
    policy_doses = []
    isf_values = []
    isf_sources = defaultdict(int)
    regime_counts = defaultdict(int)
    regime_policy_errors = {r: [] for r in REGIMES}
    regime_policy_values = {r: [] for r in REGIMES}
    empirical_isf_history = []

    for idx in range(start_idx, len(rows)):
        row = rows[idx]
        rg = _regime(row, args)
        retrain_period = retrain_high if rg in {"high", "jump"} else retrain_base
        if model is None or last_retrain is None or (idx - last_retrain) >= retrain_period:
            train_rows = rows[max(0, idx - max_train_rows):idx]
            x = np.asarray([_feature_vector(r) for r in train_rows], dtype=float)
            y = np.asarray([float(r["target_delta"]) for r in train_rows], dtype=float)
            model = _train_ridge(x, y, float(args.l2))
            last_retrain = idx

        target = float(row["target_glucose"])
        forecast = _predict_ridge(model, row)

        isf, source = _estimate_isf(row, empirical_isf_history, args)
        dose_u, _ = _recommend_dose_u(row, forecast, isf, args)
        frac = _horizon_action_fraction(horizon)
        policy_pred = float(forecast - dose_u * isf * frac)

        forecast_err = float(forecast - target)
        policy_err = float(policy_pred - target)
        forecast_errors.append(forecast_err)
        forecast_values.append(forecast)
        policy_errors.append(policy_err)
        policy_values.append(policy_pred)
        policy_doses.append(float(dose_u))
        isf_values.append(float(isf))
        isf_sources[source] += 1
        regime_counts[rg] += 1
        regime_policy_errors[rg].append(policy_err)
        regime_policy_values[rg].append(policy_pred)

        rapid_last_120 = float(row.get("rapid_last_120", 0.0))
        delta_120 = float(row.get("delta_120", 0.0))
        if rapid_last_120 >= float(args.min_insulin_units_observed) and delta_120 < 0.0:
            empirical = float((-delta_120) / max(rapid_last_120, 1e-6))
            empirical = float(np.clip(empirical, float(args.isf_min), float(args.isf_max)))
            empirical_isf_history.append(empirical)

    out = {
        "horizon_minutes": int(horizon),
        "rows_total": int(len(rows)),
        "rows_tested": int(len(rows) - start_idx),
        "forecast_baseline": _metrics(
            forecast_errors, forecast_values, float(args.target_min), float(args.target_max)
        ),
        "policy_surrogate": _metrics(
            policy_errors, policy_values, float(args.target_min), float(args.target_max)
        ),
        "dose_recommendation": {
            "total_units": float(np.sum(np.asarray(policy_doses, dtype=float))),
            "mean_units": float(np.mean(np.asarray(policy_doses, dtype=float))) if policy_doses else 0.0,
            "p90_units": float(np.quantile(np.asarray(policy_doses, dtype=float), 0.9)) if policy_doses else 0.0,
            "nonzero_pct": float(np.mean(np.asarray(policy_doses, dtype=float) > 0.0) * 100.0) if policy_doses else 0.0,
        },
        "adaptivity": {
            "isf_mean": float(np.mean(np.asarray(isf_values, dtype=float))) if isf_values else None,
            "isf_p10": float(np.quantile(np.asarray(isf_values, dtype=float), 0.1)) if isf_values else None,
            "isf_p90": float(np.quantile(np.asarray(isf_values, dtype=float), 0.9)) if isf_values else None,
            "isf_source_counts": dict(isf_sources),
            "empirical_isf_samples": int(len(empirical_isf_history)),
        },
        "policy_by_regime": {
            rg: _metrics(
                regime_policy_errors[rg],
                regime_policy_values[rg],
                float(args.target_min),
                float(args.target_max),
            )
            for rg in REGIMES
        },
        "regime_counts": dict(regime_counts),
    }
    return out


def _aggregate_global(horizons_payload: dict) -> dict:
    agg = {
        "baseline": {"count": 0, "sum_sq": 0.0, "sum_abs": 0.0, "sum_bias": 0.0},
        "policy": {"count": 0, "sum_sq": 0.0, "sum_abs": 0.0, "sum_bias": 0.0},
    }
    for h in HORIZONS:
        payload = horizons_payload[str(h)]
        for key, source in (("baseline", "forecast_baseline"), ("policy", "policy_surrogate")):
            m = payload[source]
            c = int(m.get("count", 0) or 0)
            if c == 0:
                continue
            agg[key]["count"] += c
            agg[key]["sum_sq"] += (float(m["rmse_mg_dl"]) ** 2) * c
            agg[key]["sum_abs"] += float(m["mae_mg_dl"]) * c
            agg[key]["sum_bias"] += float(m["bias_mg_dl"]) * c
    out = {}
    for key in ("baseline", "policy"):
        c = agg[key]["count"]
        if c == 0:
            out[key] = {"count": 0, "rmse_mg_dl": None, "mae_mg_dl": None, "bias_mg_dl": None}
            continue
        out[key] = {
            "count": int(c),
            "rmse_mg_dl": float(np.sqrt(agg[key]["sum_sq"] / c)),
            "mae_mg_dl": float(agg[key]["sum_abs"] / c),
            "bias_mg_dl": float(agg[key]["sum_bias"] / c),
        }
    return out


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path).resolve()
    glucose_points, events = load_external_history(csv_path)
    samples = build_direct_forecast_samples(glucose_points, events)
    rows_by_horizon = _slice_rows_by_horizon(samples, int(args.max_rows_per_horizon))

    horizons_payload = {}
    for h in HORIZONS:
        horizons_payload[str(h)] = _backtest_horizon(rows_by_horizon[h], h, args)

    output = {
        "generated_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "samples_total": int(len(samples)),
        "horizons": horizons_payload,
        "final_global": _aggregate_global(horizons_payload),
        "settings": vars(args),
        "disclaimer": "Backtest surrogate de política de dose para pesquisa. Não usar como recomendação clínica.",
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

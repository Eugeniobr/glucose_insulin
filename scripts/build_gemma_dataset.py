#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dose_engine import DoseLimits, compute_dose, simulate_scenarios


@dataclass
class GlucosePoint:
    timestamp: datetime
    glucose_mg_dl: float


@dataclass
class EventPoint:
    timestamp: datetime
    carbs_g: float
    insulin_rapid_u: float
    insulin_long_u: float
    insulin_total_u: float


def _parse_dt(value: str) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    for fmt in ("%m-%d-%Y %I:%M %p", "%d-%m-%Y %I:%M %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _to_float(raw: Any) -> float:
    if raw is None:
        return 0.0
    text = str(raw).strip().replace(",", ".")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _load_csv(path: Path) -> tuple[list[GlucosePoint], list[EventPoint]]:
    glucose: list[GlucosePoint] = []
    events: list[EventPoint] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = _parse_dt(row.get("Carimbo de data/hora do dispositivo", ""))
            if ts is None:
                continue

            g_hist = _to_float(row.get("Histórico de glicose mg/dL"))
            g_scan = _to_float(row.get("Analisar glicose mg/dL"))
            g_strip = _to_float(row.get("Tira de glicose mg/dL"))
            g = g_hist or g_scan or g_strip
            if g > 0:
                glucose.append(GlucosePoint(timestamp=ts, glucose_mg_dl=g))

            carbs = _to_float(row.get("Carboidratos (gramas)"))
            rapid = _to_float(row.get("Insulina de ação rápida (unidades)"))
            long_u = _to_float(row.get("Insulina de ação prolongada (unidades)"))
            meal_u = _to_float(row.get("Insulina de refeição (unidades)"))
            corr_u = _to_float(row.get("Insulina de correção (unidades)"))
            user_u = _to_float(row.get("Insulina da alteração do usuário (unidades)"))
            rapid_total = rapid + meal_u + corr_u + user_u
            insulin_total = rapid_total + long_u
            if carbs > 0 or insulin_total > 0:
                events.append(
                    EventPoint(
                        timestamp=ts,
                        carbs_g=carbs,
                        insulin_rapid_u=rapid_total,
                        insulin_long_u=long_u,
                        insulin_total_u=insulin_total,
                    )
                )

    glucose.sort(key=lambda x: x.timestamp)
    events.sort(key=lambda x: x.timestamp)
    return glucose, events


def _nearest_glucose(points: list[GlucosePoint], t: datetime, max_delta_min: int = 30) -> float | None:
    if not points:
        return None
    best: GlucosePoint | None = None
    best_delta = timedelta(days=999)
    for p in points:
        d = abs(p.timestamp - t)
        if d < best_delta:
            best = p
            best_delta = d
    if best is None:
        return None
    if best_delta > timedelta(minutes=max_delta_min):
        return None
    return best.glucose_mg_dl


def _recent_glucose(points: list[GlucosePoint], t: datetime, lookback_min: int = 120) -> list[dict[str, Any]]:
    start = t - timedelta(minutes=lookback_min)
    selected = [p for p in points if start <= p.timestamp <= t]
    selected = selected[-12:]
    return [{"t": p.timestamp.isoformat(), "glucose_mg_dl": round(p.glucose_mg_dl, 1)} for p in selected]


def _trend_label(series: list[dict[str, Any]]) -> str:
    if len(series) < 2:
        return "indefinida"
    delta = float(series[-1]["glucose_mg_dl"]) - float(series[0]["glucose_mg_dl"])
    if delta > 20:
        return "subindo"
    if delta < -20:
        return "caindo"
    return "estavel"


def _risk_label(glucose: float) -> str:
    if glucose < 70:
        return "hipo_alta"
    if glucose < 90:
        return "hipo_moderada"
    if glucose > 250:
        return "hiper_alta"
    if glucose > 180:
        return "hiper_moderada"
    return "baixo"


def _build_example(
    event: EventPoint,
    glucose_now: float,
    recent_series: list[dict[str, Any]],
    sensitivity_u_per_g: float,
    limits: DoseLimits,
) -> dict[str, Any]:
    # All insulin terms are explicitly in U.
    iob_u = max(0.0, 0.45 * event.insulin_rapid_u + 0.20 * event.insulin_long_u)
    dose = compute_dose(
        glucose_mg_dl=glucose_now,
        cho_g=event.carbs_g,
        sensitivity_u_per_g=sensitivity_u_per_g,
        pk_active_u=iob_u,
        limits=limits,
    ).to_dict()
    sims = simulate_scenarios(
        glucose_mg_dl=glucose_now,
        cho_effective_g=event.carbs_g,
        sensitivity_u_per_g=sensitivity_u_per_g,
        pk_active_u=iob_u,
        limits=limits,
    )

    trend = _trend_label(recent_series)
    risk = _risk_label(glucose_now)
    preferred = "base"
    if trend == "caindo" or glucose_now < 100:
        preferred = "conservative"
    elif trend == "subindo" and glucose_now > 180:
        preferred = "aggressive"

    prompt_context = {
        "timestamp": event.timestamp.isoformat(),
        "glucose_now_mg_dl": round(glucose_now, 1),
        "trend": trend,
        "recent_glucose": recent_series,
        "event": {
            "carbs_g": round(event.carbs_g, 1),
            "insulin_rapid_u": round(event.insulin_rapid_u, 2),
            "insulin_long_u": round(event.insulin_long_u, 2),
            "insulin_total_u": round(event.insulin_total_u, 2),
        },
        "deterministic_dose": dose,
        "simulations": sims,
        "units": {
            "glucose": "mg/dL",
            "insulin": "U",
            "carbs": "g",
        },
    }

    target = {
        "resumo": f"Glicose {int(round(glucose_now))} mg/dL com tendência {trend}.",
        "risco_2h": risk,
        "cenario_preferido": preferred,
        "dose_sugerida_u": round(float((sims.get(preferred) or {}).get("final_u", dose.get("final_u", 0.0))), 2),
        "justificativa": [
            "Baseado em CHO, glicose atual e insulina ativa (IOB em U).",
            "Cenário preferido selecionado por tendência glicêmica e risco imediato.",
        ],
        "alertas": [
            "Saída para apoio analítico, não substitui decisão clínica.",
        ],
    }

    return {
        "instruction": (
            "Analise o contexto glicêmico e os cenários de dose determinísticos em U. "
            "Responda APENAS JSON com chaves: resumo, risco_2h, cenario_preferido, dose_sugerida_u, justificativa, alertas."
        ),
        "input": json.dumps(prompt_context, ensure_ascii=False),
        "output": json.dumps(target, ensure_ascii=False),
        "metadata": {
            "timestamp": event.timestamp.isoformat(),
            "glucose_now_mg_dl": round(glucose_now, 1),
            "carbs_g": round(event.carbs_g, 1),
            "insulin_total_u": round(event.insulin_total_u, 2),
        },
    }


def _augment_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
    payload = json.loads(example["input"])
    out = json.loads(example["output"])
    sims = payload.get("simulations") or {}
    base = sims.get("base") or {}
    cons = sims.get("conservative") or {}
    aggr = sims.get("aggressive") or {}

    mode = idx % 4
    if mode == 0:
        chosen = "conservative"
        risk = "hipo_moderada" if payload.get("glucose_now_mg_dl", 120) < 110 else "baixo"
    elif mode == 1:
        chosen = "base"
        risk = out.get("risco_2h", "baixo")
    elif mode == 2:
        chosen = "aggressive"
        risk = "hiper_moderada" if payload.get("glucose_now_mg_dl", 120) > 170 else "baixo"
    else:
        chosen = "base"
        risk = "baixo"

    chosen_dose = (sims.get(chosen) or {}).get("final_u", base.get("final_u", 0.0))
    if mode == 3:
        chosen_dose = 0.5 * float(cons.get("final_u", 0.0)) + 0.5 * float(base.get("final_u", 0.0))

    out["cenario_preferido"] = chosen
    out["risco_2h"] = risk
    out["dose_sugerida_u"] = round(float(chosen_dose), 2)
    out["justificativa"] = [
        "Variação sintética para robustez de treinamento em cenários próximos.",
        "Dose em U sempre derivada do motor determinístico e dos cenários simulados.",
    ]
    out["alertas"] = ["Amostra aumentada sinteticamente; uso para fine-tuning supervisionado."]

    return {
        "instruction": example["instruction"],
        "input": json.dumps(payload, ensure_ascii=False),
        "output": json.dumps(out, ensure_ascii=False),
        "metadata": {**(example.get("metadata") or {}), "augmented": True, "augmentation_id": idx},
    }


def build_dataset(
    *,
    csv_path: Path,
    output_path: Path,
    min_carbs_g: float,
    min_insulin_u: float,
    sensitivity_u_per_g: float,
    augment_factor: int,
    max_samples: int | None,
) -> dict[str, Any]:
    glucose, events = _load_csv(csv_path)
    limits = DoseLimits()

    examples: list[dict[str, Any]] = []
    for ev in events:
        if ev.carbs_g < min_carbs_g and ev.insulin_total_u < min_insulin_u:
            continue
        glucose_now = _nearest_glucose(glucose, ev.timestamp)
        if glucose_now is None:
            continue
        recent_series = _recent_glucose(glucose, ev.timestamp)
        if len(recent_series) < 3:
            continue
        ex = _build_example(ev, glucose_now, recent_series, sensitivity_u_per_g, limits)
        examples.append(ex)
        if max_samples is not None and len(examples) >= max_samples:
            break

    if augment_factor > 1 and examples:
        augmented: list[dict[str, Any]] = []
        for ex in examples:
            augmented.append(ex)
            for j in range(augment_factor - 1):
                augmented.append(_augment_example(ex, j))
        examples = augmented

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "csv_path": str(csv_path),
        "output_path": str(output_path),
        "num_glucose_points": len(glucose),
        "num_events": len(events),
        "num_examples": len(examples),
        "units": {"glucose": "mg/dL", "insulin": "U", "carbs": "g"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera dataset JSONL para fine-tuning Gemma (glicemia/bolus/CHO)")
    parser.add_argument("--csv-path", default="EugênioSilva Rezende_glucose_4-19-2026.csv")
    parser.add_argument("--output-path", default="data/gemma_train.jsonl")
    parser.add_argument("--min-carbs-g", type=float, default=10.0)
    parser.add_argument("--min-insulin-u", type=float, default=0.5)
    parser.add_argument("--sensitivity-u-per-g", type=float, default=0.10)
    parser.add_argument("--augment-factor", type=int, default=1, help="1 = sem augmentação; 2+ replica com variações")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = sem limite")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_dataset(
        csv_path=(ROOT / args.csv_path).resolve(),
        output_path=(ROOT / args.output_path).resolve(),
        min_carbs_g=float(args.min_carbs_g),
        min_insulin_u=float(args.min_insulin_u),
        sensitivity_u_per_g=float(args.sensitivity_u_per_g),
        augment_factor=max(1, int(args.augment_factor)),
        max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

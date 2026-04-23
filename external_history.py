import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


CSV_DATETIME_FORMAT = "%m-%d-%Y %I:%M %p"


@dataclass
class ExternalGlucosePoint:
    timestamp: datetime
    glucose_mg_dl: float


@dataclass
class ExternalEvent:
    timestamp: datetime
    carbs_g: float
    rapid_units: float
    basal_units: float


def _parse_float(raw_value: str) -> float | None:
    text = str(raw_value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_external_history(csv_path: Path):
    glucose_points: list[ExternalGlucosePoint] = []
    event_map: dict[datetime, ExternalEvent] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp_raw = str(row.get("Carimbo de data/hora do dispositivo", "")).strip()
            if not timestamp_raw:
                continue
            timestamp = datetime.strptime(timestamp_raw, CSV_DATETIME_FORMAT)
            glucose = _parse_float(row.get("Histórico de glicose mg/dL", ""))
            carbs = _parse_float(row.get("Carboidratos (gramas)", ""))
            rapid = _parse_float(row.get("Insulina de ação rápida (unidades)", ""))
            basal = _parse_float(row.get("Insulina de ação prolongada (unidades)", ""))
            if glucose is not None:
                glucose_points.append(ExternalGlucosePoint(timestamp=timestamp, glucose_mg_dl=glucose))
            if carbs is not None or rapid is not None or basal is not None:
                existing = event_map.get(timestamp)
                if existing is None:
                    event_map[timestamp] = ExternalEvent(
                        timestamp=timestamp,
                        carbs_g=float(carbs or 0.0),
                        rapid_units=float(rapid or 0.0),
                        basal_units=float(basal or 0.0),
                    )
                else:
                    existing.carbs_g += float(carbs or 0.0)
                    existing.rapid_units += float(rapid or 0.0)
                    existing.basal_units += float(basal or 0.0)
    glucose_points.sort(key=lambda item: item.timestamp)
    events = sorted(event_map.values(), key=lambda item: item.timestamp)
    return glucose_points, events


def summarize_external_history(glucose_points: list[ExternalGlucosePoint], events: list[ExternalEvent]) -> dict:
    if not glucose_points:
        return {}

    first = glucose_points[0].timestamp
    last = glucose_points[-1].timestamp
    paired_events = [event for event in events if event.carbs_g > 0.0 and event.rapid_units > 0.0]
    isolated_rapid = []
    for event in events:
        if event.rapid_units <= 0.0 or event.carbs_g > 0.0:
            continue
        if all(abs((other.timestamp - event.timestamp).total_seconds()) > 120 * 60 or other.carbs_g <= 0.0 for other in events):
            isolated_rapid.append(event)

    cho_per_unit = [event.carbs_g / event.rapid_units for event in paired_events if event.rapid_units > 0.0]
    return {
        "rows_glucose": len(glucose_points),
        "rows_events": len(events),
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "covered_days": int((last - first).days),
        "paired_meal_bolus_events": len(paired_events),
        "isolated_rapid_events": len(isolated_rapid),
        "carb_events": int(sum(1 for event in events if event.carbs_g > 0.0)),
        "rapid_events": int(sum(1 for event in events if event.rapid_units > 0.0)),
        "basal_events": int(sum(1 for event in events if event.basal_units > 0.0)),
        "total_carbs_g": float(sum(event.carbs_g for event in events)),
        "total_rapid_units": float(sum(event.rapid_units for event in events)),
        "total_basal_units": float(sum(event.basal_units for event in events)),
        "cho_per_unit_mean": float(np.mean(cho_per_unit)) if cho_per_unit else None,
        "cho_per_unit_median": float(np.median(cho_per_unit)) if cho_per_unit else None,
    }


def _interpolate_glucose(glucose_points: list[ExternalGlucosePoint], timestamp: datetime) -> float | None:
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    target = timestamp.timestamp()
    if target < timestamps[0] or target > timestamps[-1]:
        return None
    return float(np.interp(target, timestamps, values))


def build_external_training_samples(glucose_points: list[ExternalGlucosePoint], events: list[ExternalEvent], horizons=(15, 30, 60, 120)) -> list[dict]:
    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    glucose_values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    samples: list[dict] = []
    if len(glucose_points) < 10:
        return samples

    event_days = {event.timestamp.date() for event in events if event.carbs_g > 0.0 or event.rapid_units > 0.0 or event.basal_units > 0.0}
    for point in glucose_points:
        if point.timestamp.date() not in event_days:
            continue
        now = point.timestamp
        now_seconds = now.timestamp()
        if now_seconds - timestamps[0] < 90 * 60:
            continue
        for horizon in horizons:
            future_time = now + timedelta(minutes=int(horizon))
            future_seconds = future_time.timestamp()
            if future_seconds > timestamps[-1]:
                continue

            glucose_now = float(point.glucose_mg_dl)
            glucose_15 = float(np.interp(now_seconds - 15 * 60, timestamps, glucose_values))
            glucose_30 = float(np.interp(now_seconds - 30 * 60, timestamps, glucose_values))
            glucose_60 = float(np.interp(now_seconds - 60 * 60, timestamps, glucose_values))
            future_glucose = float(np.interp(future_seconds, timestamps, glucose_values))

            recent_events = [event for event in events if 0.0 <= (now - event.timestamp).total_seconds() <= 120 * 60]
            future_events = [event for event in events if 0.0 <= (event.timestamp - now).total_seconds() <= horizon * 60]
            last_event_minutes = min(
                [((now - event.timestamp).total_seconds() / 60.0) for event in recent_events],
                default=-1.0,
            )
            samples.append(
                {
                    "timestamp": now.isoformat(),
                    "horizon_minutes": int(horizon),
                    "current_glucose": glucose_now,
                    "delta_15": glucose_now - glucose_15,
                    "delta_30": glucose_now - glucose_30,
                    "delta_60": glucose_now - glucose_60,
                    "carbs_last_30": float(sum(event.carbs_g for event in recent_events if (now - event.timestamp).total_seconds() <= 30 * 60)),
                    "carbs_last_60": float(sum(event.carbs_g for event in recent_events if (now - event.timestamp).total_seconds() <= 60 * 60)),
                    "rapid_last_30": float(sum(event.rapid_units for event in recent_events if (now - event.timestamp).total_seconds() <= 30 * 60)),
                    "rapid_last_60": float(sum(event.rapid_units for event in recent_events if (now - event.timestamp).total_seconds() <= 60 * 60)),
                    "rapid_last_120": float(sum(event.rapid_units for event in recent_events)),
                    "future_carbs_total": float(sum(event.carbs_g for event in future_events)),
                    "future_rapid_total": float(sum(event.rapid_units for event in future_events)),
                    "minutes_since_event": float(last_event_minutes),
                    "hour_of_day": float(now.hour + now.minute / 60.0),
                    "target_glucose": future_glucose,
                }
            )
    return samples


def build_direct_forecast_samples(
    glucose_points: list[ExternalGlucosePoint],
    events: list[ExternalEvent],
    horizons=(15, 30, 60, 120),
    sample_step_minutes: int = 15,
) -> list[dict]:
    samples: list[dict] = []
    if len(glucose_points) < 32:
        return samples

    timestamps = np.array([point.timestamp.timestamp() for point in glucose_points], dtype=float)
    glucose_values = np.array([point.glucose_mg_dl for point in glucose_points], dtype=float)
    step_seconds = max(int(sample_step_minutes), 1) * 60

    event_records = [
        {
            "timestamp": event.timestamp.timestamp(),
            "carbs_g": float(event.carbs_g),
            "rapid_units": float(event.rapid_units),
        }
        for event in events
    ]

    def glucose_at(offset_minutes: float, now_seconds: float) -> float:
        return float(np.interp(now_seconds + offset_minutes * 60.0, timestamps, glucose_values))

    first_seconds = timestamps[0]
    last_seconds = timestamps[-1]
    cursor = int(first_seconds + 180 * 60)
    end_limit = int(last_seconds - max(horizons) * 60)

    while cursor <= end_limit:
        current_glucose = glucose_at(0.0, cursor)
        prev_15 = glucose_at(-15.0, cursor)
        prev_30 = glucose_at(-30.0, cursor)
        prev_60 = glucose_at(-60.0, cursor)
        prev_120 = glucose_at(-120.0, cursor)
        window_30_mask = (timestamps >= cursor - 30 * 60) & (timestamps <= cursor)
        window_60_mask = (timestamps >= cursor - 60 * 60) & (timestamps <= cursor)
        window_120_mask = (timestamps >= cursor - 120 * 60) & (timestamps <= cursor)
        values_30 = glucose_values[window_30_mask]
        values_60 = glucose_values[window_60_mask]
        values_120 = glucose_values[window_120_mask]
        slope_30 = float((current_glucose - prev_30) / 30.0)
        slope_60 = float((current_glucose - prev_60) / 60.0)

        recent_events_30 = [event for event in event_records if 0.0 <= cursor - event["timestamp"] <= 30 * 60]
        recent_events_60 = [event for event in event_records if 0.0 <= cursor - event["timestamp"] <= 60 * 60]
        recent_events_120 = [event for event in event_records if 0.0 <= cursor - event["timestamp"] <= 120 * 60]
        last_meal_minutes = min(
            [float((cursor - event["timestamp"]) / 60.0) for event in event_records if event["carbs_g"] > 0.0 and event["timestamp"] <= cursor],
            default=-1.0,
        )
        last_rapid_minutes = min(
            [float((cursor - event["timestamp"]) / 60.0) for event in event_records if event["rapid_units"] > 0.0 and event["timestamp"] <= cursor],
            default=-1.0,
        )
        hour = datetime.fromtimestamp(cursor).hour + datetime.fromtimestamp(cursor).minute / 60.0
        angle = 2.0 * np.pi * hour / 24.0

        shared = {
            "timestamp": datetime.fromtimestamp(cursor).isoformat(),
            "current_glucose": current_glucose,
            "delta_15": current_glucose - prev_15,
            "delta_30": current_glucose - prev_30,
            "delta_60": current_glucose - prev_60,
            "delta_120": current_glucose - prev_120,
            "slope_30": slope_30,
            "slope_60": slope_60,
            "mean_30": float(np.mean(values_30)) if len(values_30) else current_glucose,
            "mean_60": float(np.mean(values_60)) if len(values_60) else current_glucose,
            "std_30": float(np.std(values_30)) if len(values_30) else 0.0,
            "std_60": float(np.std(values_60)) if len(values_60) else 0.0,
            "range_60": float(np.max(values_60) - np.min(values_60)) if len(values_60) else 0.0,
            "carbs_last_30": float(sum(event["carbs_g"] for event in recent_events_30)),
            "carbs_last_60": float(sum(event["carbs_g"] for event in recent_events_60)),
            "carbs_last_120": float(sum(event["carbs_g"] for event in recent_events_120)),
            "rapid_last_30": float(sum(event["rapid_units"] for event in recent_events_30)),
            "rapid_last_60": float(sum(event["rapid_units"] for event in recent_events_60)),
            "rapid_last_120": float(sum(event["rapid_units"] for event in recent_events_120)),
            "minutes_since_meal": last_meal_minutes,
            "minutes_since_rapid": last_rapid_minutes,
            "hour_sin": float(np.sin(angle)),
            "hour_cos": float(np.cos(angle)),
        }

        for horizon in horizons:
            future_events = [event for event in event_records if 0.0 < event["timestamp"] - cursor <= horizon * 60]
            future_glucose = glucose_at(float(horizon), cursor)
            sample = dict(shared)
            sample.update(
                {
                    "horizon_minutes": int(horizon),
                    "future_carbs_total": float(sum(event["carbs_g"] for event in future_events)),
                    "future_rapid_total": float(sum(event["rapid_units"] for event in future_events)),
                    "target_glucose": future_glucose,
                    "target_delta": float(future_glucose - current_glucose),
                }
            )
            samples.append(sample)

        cursor += step_seconds

    return samples

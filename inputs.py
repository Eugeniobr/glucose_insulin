import csv
import io
import sys
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class MealEvent:
    timestamp: datetime
    minutes: float
    carbs_g: float


@dataclass
class InsulinEvent:
    timestamp: datetime
    minutes: float
    units: float
    insulin_type: str


@dataclass
class ExerciseEvent:
    timestamp: datetime
    minutes: float
    duration_minutes: float
    heart_rate_bpm: float


def normalize_insulin_type(raw_value: str) -> str:
    value = _normalize_header(raw_value)
    if not value:
        return "desconhecido"

    rapid_aliases = (
        "rapida",
        "ultrarrapida",
        "ultra rapida",
        "ultra-rapida",
        "asparte",
        "rapid",
        "bolus",
        "correcao",
        "correction",
        "humalog",
        "novorapid",
        "novo rapid",
        "fiasp",
        "apidra",
        "lispro",
        "aspart",
        "glulisina",
    )
    regular_aliases = (
        "regular",
        "r",
        "insulina regular",
        "actrapid",
        "humulin r",
    )
    basal_aliases = (
        "basal",
        "lenta",
        "longa",
        "ultralenta",
        "ultra lenta",
        "intermediaria",
        "intermediaria/nph",
        "nph",
        "glargina",
        "lantus",
        "detemir",
        "levemir",
        "degludeca",
        "tresiba",
    )

    if any(alias in value for alias in rapid_aliases):
        return "rapida"
    if any(alias in value for alias in regular_aliases):
        return "regular"
    if any(alias in value for alias in basal_aliases):
        return "basal"
    return "desconhecido"


def _normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.strip().lower()


def _read_source_text(source: str) -> str:
    if not source:
        return ""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            with urlopen(source, timeout=20) as response:
                return response.read().decode("utf-8-sig")
        return Path(source).read_text(encoding="utf-8-sig")
    except (URLError, OSError, TimeoutError, UnicodeDecodeError) as exc:
        print(f"[inputs] Aviso: não foi possível carregar origem '{source}': {exc}", file=sys.stderr)
        return ""


def _parse_csv_rows(source: str) -> Iterable[dict]:
    text = _read_source_text(source)
    if not text.strip():
        return []
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except csv.Error:
        dialect = csv.get_dialect("excel")
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return list(reader)


def _first_present(row: dict, aliases: list[str]):
    normalized = {_normalize_header(key): value for key, value in row.items()}
    for alias in aliases:
        if alias in normalized and str(normalized[alias]).strip():
            return str(normalized[alias]).strip()
    return ""


def _parse_datetime(raw_value: str, reference_date: date) -> datetime:
    raw_value = raw_value.strip()
    for parser in (datetime.fromisoformat,):
        try:
            return parser(raw_value)
        except ValueError:
            pass

    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw_value, fmt)
        except ValueError:
            pass

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(raw_value, fmt).time()
            return datetime.combine(reference_date, parsed_time)
        except ValueError:
            pass

    raise ValueError(f"Timestamp inválido: {raw_value}")


def _parse_insulin_units_and_type(units_raw: str, insulin_type_raw: str) -> tuple[float, str]:
    normalized_type = normalize_insulin_type(insulin_type_raw)
    raw = (units_raw or "").strip()
    compact = raw.replace(" ", "").replace(",", ".")
    suffix_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)", compact)

    if suffix_match:
        units_value = float(suffix_match.group(1))
        suffix = suffix_match.group(2).lower()
        if suffix == "b":
            return units_value, "basal"
        if suffix == "r":
            return units_value, "rapida"
        return units_value, normalized_type

    numeric_match = re.search(r"[0-9]+(?:\.[0-9]+)?", compact)
    if not numeric_match:
        raise ValueError(f"Dose de insulina inválida: {units_raw}")
    inferred_type = normalized_type if normalized_type != "desconhecido" else "rapida"
    return float(numeric_match.group(0)), inferred_type


def load_meal_events(source: str, start_time: datetime) -> list[MealEvent]:
    events = []
    for row in _parse_csv_rows(source):
        timestamp_raw = _first_present(
            row,
            ["timestamp", "horario", "horario da refeicao", "carimbo de data/hora"],
        )
        carbs_raw = _first_present(
            row,
            ["carboidratos (g)", "carboidratos", "carbs", "carbohydrate", "quantas gramas de cho ?"],
        )
        if not timestamp_raw or not carbs_raw:
            continue
        timestamp = _parse_datetime(timestamp_raw, start_time.date())
        minutes = (timestamp - start_time).total_seconds() / 60.0
        events.append(MealEvent(timestamp=timestamp, minutes=minutes, carbs_g=float(carbs_raw.replace(",", "."))))
    return sorted(events, key=lambda event: event.timestamp)


def load_insulin_events(source: str, start_time: datetime) -> list[InsulinEvent]:
    events = []
    for row in _parse_csv_rows(source):
        timestamp_raw = _first_present(
            row,
            ["timestamp", "horario", "horario da insulina", "carimbo de data/hora"],
        )
        units_raw = _first_present(
            row,
            ["unidades", "units", "dose", "quantas unidades ?"],
        )
        insulin_type_raw = _first_present(row, ["tipo", "type"])
        if not timestamp_raw or not units_raw:
            continue
        units, insulin_type = _parse_insulin_units_and_type(units_raw, insulin_type_raw)
        timestamp = _parse_datetime(timestamp_raw, start_time.date())
        minutes = (timestamp - start_time).total_seconds() / 60.0
        events.append(
            InsulinEvent(
                timestamp=timestamp,
                minutes=minutes,
                units=units,
                insulin_type=insulin_type,
            )
        )
    return sorted(events, key=lambda event: event.timestamp)


def load_exercise_events(source: str, start_time: datetime, baseline_hr_bpm: float = 70.0) -> list[ExerciseEvent]:
    if not source:
        return []
    events = []
    for row in _parse_csv_rows(source):
        timestamp_raw = _first_present(
            row,
            ["timestamp", "inicio", "horario", "horario da atividade", "carimbo de data/hora", "start"],
        )
        duration_raw = _first_present(
            row,
            ["duracao", "duracao (min)", "duration", "duration_minutes", "minutos", "tempo"],
        )
        heart_rate_raw = _first_present(
            row,
            ["frequencia cardiaca", "frequencia cardiaca media", "fc", "hr", "heart rate", "bpm"],
        )
        intensity_raw = _first_present(
            row,
            ["intensidade", "intensity"],
        )
        if not timestamp_raw or not duration_raw:
            continue
        timestamp = _parse_datetime(timestamp_raw, start_time.date())
        minutes = (timestamp - start_time).total_seconds() / 60.0
        duration_minutes = float(duration_raw.replace(",", "."))
        if heart_rate_raw:
            heart_rate_bpm = float(heart_rate_raw.replace(",", "."))
        else:
            normalized_intensity = _normalize_header(intensity_raw)
            if "moder" in normalized_intensity:
                heart_rate_bpm = baseline_hr_bpm * 2.0
            elif "leve" in normalized_intensity or "mild" in normalized_intensity:
                heart_rate_bpm = baseline_hr_bpm * 1.5
            else:
                heart_rate_bpm = baseline_hr_bpm * 1.6
        events.append(
            ExerciseEvent(
                timestamp=timestamp,
                minutes=minutes,
                duration_minutes=duration_minutes,
                heart_rate_bpm=heart_rate_bpm,
            )
        )
    return sorted(events, key=lambda event: event.timestamp)


def serialize_events(events: list) -> list[dict]:
    serialized = []
    for event in events:
        payload = asdict(event)
        payload["timestamp"] = event.timestamp.isoformat()
        serialized.append(payload)
    return serialized

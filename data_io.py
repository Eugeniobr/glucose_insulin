import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_glucose_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    timestamps = []
    glucose_values = []

    for row in data:
        timestamp = row.get("timestamp")
        glucose = row.get("value_in_mg_per_dl", row.get("value"))
        if timestamp is None or glucose is None:
            continue
        timestamps.append(datetime.fromisoformat(str(timestamp)))
        glucose_values.append(float(glucose))

    if len(timestamps) < 5:
        raise ValueError("JSON de glicose com poucos pontos válidos.")

    order = np.argsort(np.array(timestamps, dtype="datetime64[ns]"))
    timestamps = [timestamps[idx] for idx in order]
    glucose_values = np.asarray(glucose_values, dtype=float)[order]

    start_time = timestamps[0]
    time_minutes = np.asarray(
        [(timestamp - start_time).total_seconds() / 60.0 for timestamp in timestamps],
        dtype=float,
    )
    return time_minutes, glucose_values, timestamps


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

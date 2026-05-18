"""Preprocessing and feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLS = ["patient_id", "timestamp", "glucose", "bolus", "basal", "carbs"]
LIBRE_TS_COL = "Carimbo de data/hora do dispositivo"


@dataclass
class NormStats:
    mean: float
    std: float


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            if col in {"bolus", "basal", "carbs"}:
                df[col] = 0.0
            else:
                raise ValueError(f"Missing required column: {col}")
    return df


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)


def _convert_libre_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Libre-style Portuguese export into the expected twin schema."""
    out = pd.DataFrame()
    out["patient_id"] = (
        df.get("Número de série", pd.Series(["unknown"] * len(df))).astype(str).replace({"": "unknown"})
    )
    out["timestamp"] = pd.to_datetime(df.get(LIBRE_TS_COL), errors="coerce", format="%m-%d-%Y %I:%M %p")
    missing = out["timestamp"].isna()
    if missing.any():
        out.loc[missing, "timestamp"] = pd.to_datetime(df.loc[missing, LIBRE_TS_COL], errors="coerce", dayfirst=True)

    g_hist = _to_float_series(df.get("Histórico de glicose mg/dL", pd.Series([0.0] * len(df))))
    g_scan = _to_float_series(df.get("Analisar glicose mg/dL", pd.Series([0.0] * len(df))))
    g_strip = _to_float_series(df.get("Tira de glicose mg/dL", pd.Series([0.0] * len(df))))
    glucose = g_hist.where(g_hist > 0, g_scan)
    glucose = glucose.where(glucose > 0, g_strip)
    out["glucose"] = glucose

    rapid = _to_float_series(df.get("Insulina de ação rápida (unidades)", pd.Series([0.0] * len(df))))
    meal_u = _to_float_series(df.get("Insulina de refeição (unidades)", pd.Series([0.0] * len(df))))
    corr_u = _to_float_series(df.get("Insulina de correção (unidades)", pd.Series([0.0] * len(df))))
    user_u = _to_float_series(df.get("Insulina da alteração do usuário (unidades)", pd.Series([0.0] * len(df))))
    out["bolus"] = rapid + meal_u + corr_u + user_u
    out["basal"] = _to_float_series(df.get("Insulina de ação prolongada (unidades)", pd.Series([0.0] * len(df))))
    out["carbs"] = _to_float_series(df.get("Carboidratos (gramas)", pd.Series([0.0] * len(df))))
    return out


def read_and_align(csv_path: str, freq_minutes: int = 5) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    if set(REQUIRED_COLS).issubset(df_raw.columns):
        df = ensure_columns(df_raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif LIBRE_TS_COL in df_raw.columns:
        df = _convert_libre_schema(df_raw)
    else:
        raise ValueError(
            "Input CSV schema not recognized. Expected required columns "
            f"{REQUIRED_COLS} or Libre export column '{LIBRE_TS_COL}'."
        )
    df = df.dropna(subset=["patient_id", "timestamp"]).copy()

    num_cols = ["glucose", "bolus", "basal", "carbs"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    out = []
    freq = f"{freq_minutes}min"
    for pid, g in df.groupby("patient_id", sort=False):
        # Consolidate duplicate timestamps before reindexing.
        g = (
            g.groupby("timestamp", as_index=False)
            .agg(
                glucose=("glucose", "mean"),
                bolus=("bolus", "sum"),
                basal=("basal", "mean"),
                carbs=("carbs", "sum"),
            )
            .set_index("timestamp")[num_cols]
            .sort_index()
        )
        idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g = g.reindex(idx)
        g.index.name = "timestamp"

        g["bolus"] = g["bolus"].fillna(0.0)
        g["basal"] = g["basal"].fillna(0.0)
        g["carbs"] = g["carbs"].fillna(0.0)
        g["glucose"] = g["glucose"].interpolate(limit_direction="both").ffill().bfill()

        g["patient_id"] = pid
        out.append(g.reset_index())

    return pd.concat(out, ignore_index=True)


def _minutes_since_last_event(series: pd.Series, freq_minutes: int) -> pd.Series:
    out = np.zeros(len(series), dtype=float)
    last = None
    for i, v in enumerate(series.to_numpy()):
        if v > 0:
            last = i
            out[i] = 0.0
        elif last is None:
            out[i] = np.inf
        else:
            out[i] = (i - last) * freq_minutes
    max_clip = 24 * 60
    out[np.isinf(out)] = max_clip
    return pd.Series(np.clip(out, 0, max_clip), index=series.index)


def add_features(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow"] = df["timestamp"].dt.dayofweek.astype(float)

    out = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").copy()

        g["glucose_lag_1"] = g["glucose"].shift(1)
        g["glucose_lag_3"] = g["glucose"].shift(3)
        g["glucose_lag_6"] = g["glucose"].shift(6)

        g["bolus_sum_30m"] = g["bolus"].rolling(6, min_periods=1).sum()
        g["bolus_sum_60m"] = g["bolus"].rolling(12, min_periods=1).sum()
        g["carbs_sum_30m"] = g["carbs"].rolling(6, min_periods=1).sum()
        g["carbs_sum_60m"] = g["carbs"].rolling(12, min_periods=1).sum()
        g["basal_mean_30m"] = g["basal"].rolling(6, min_periods=1).mean()

        g["mins_since_meal"] = _minutes_since_last_event(g["carbs"], freq_minutes)
        g["mins_since_bolus"] = _minutes_since_last_event(g["bolus"], freq_minutes)

        g = g.bfill().ffill()
        out.append(g)

    return pd.concat(out, ignore_index=True)


def feature_columns() -> list[str]:
    return [
        "glucose",
        "bolus",
        "basal",
        "carbs",
        "hour_sin",
        "hour_cos",
        "dow",
        "glucose_lag_1",
        "glucose_lag_3",
        "glucose_lag_6",
        "bolus_sum_30m",
        "bolus_sum_60m",
        "carbs_sum_30m",
        "carbs_sum_60m",
        "basal_mean_30m",
        "mins_since_meal",
        "mins_since_bolus",
    ]


def fit_patient_normalizers(df_train: pd.DataFrame, cols: list[str]) -> dict[tuple[str, str], NormStats]:
    norms: dict[tuple[str, str], NormStats] = {}
    for pid, g in df_train.groupby("patient_id"):
        for c in cols:
            mean = float(g[c].mean())
            std = float(g[c].std())
            if std < 1e-6:
                std = 1.0
            norms[(str(pid), c)] = NormStats(mean, std)
    return norms


def apply_patient_normalizers(df: pd.DataFrame, cols: list[str], norms: dict[tuple[str, str], NormStats]) -> pd.DataFrame:
    df = df.copy()
    global_means = {c: float(df[c].mean()) for c in cols}
    global_stds = {c: float(df[c].std()) if float(df[c].std()) > 1e-6 else 1.0 for c in cols}

    for idx, row in df.iterrows():
        pid = str(row["patient_id"])
        for c in cols:
            st = norms.get((pid, c), NormStats(global_means[c], global_stds[c]))
            df.at[idx, c] = (float(row[c]) - st.mean) / st.std
    return df

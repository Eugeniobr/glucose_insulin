"""PyTorch dataset for metabolic twin windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class WindowMeta:
    patient_id: str
    timestamp: str


class TwinWindowDataset(Dataset):
    """Builds sequence windows and multihorizon targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        lookback_steps: int,
        horizon_steps: list[int],
    ) -> None:
        self.feature_cols = feature_cols
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        metas: list[WindowMeta] = []

        max_h = max(horizon_steps)
        for pid, g in df.groupby("patient_id", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            feat = g[feature_cols].to_numpy(dtype=np.float32)
            glucose = g["glucose"].to_numpy(dtype=np.float32)
            ts = g["timestamp"].astype(str).to_numpy()

            start = lookback_steps
            end = len(g) - max_h
            for t in range(start, end):
                xs.append(feat[t - lookback_steps : t])
                ys.append(np.array([glucose[t + h] for h in horizon_steps], dtype=np.float32))
                metas.append(WindowMeta(str(pid), ts[t]))

        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32)
        self.meta = metas

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def get_meta(self, idx: int) -> WindowMeta:
        return self.meta[idx]

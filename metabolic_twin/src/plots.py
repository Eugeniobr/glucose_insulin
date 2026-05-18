"""Plot helpers for training evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_real_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], out_path: Path) -> None:
    plt.figure(figsize=(12, 4))
    for i, h in enumerate(horizons):
        plt.plot(y_true[:, i], label=f"real {h}m", alpha=0.7)
        plt.plot(y_pred[:, i], label=f"pred {h}m", alpha=0.7)
        if i == 0:
            break
    plt.title("Glucose: Real vs Predicted")
    plt.xlabel("Samples")
    plt.ylabel("Glucose")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], out_path: Path) -> None:
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    x = np.arange(len(horizons))
    w = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - w / 2, mae, width=w, label="MAE")
    plt.bar(x + w / 2, rmse, width=w, label="RMSE")
    plt.xticks(x, [f"{h}m" for h in horizons])
    plt.title("Error by Horizon")
    plt.ylabel("mg/dL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residual = (y_pred - y_true).reshape(-1)
    plt.figure(figsize=(8, 4))
    plt.hist(residual, bins=40, alpha=0.8)
    plt.title("Residual Distribution")
    plt.xlabel("Prediction error (mg/dL)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_patient_example(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: list[str],
    horizons: list[int],
    out_path: Path,
) -> None:
    if not patient_ids:
        return
    pid = patient_ids[0]
    idx = [i for i, p in enumerate(patient_ids) if p == pid][:120]
    if not idx:
        return

    t = np.array(idx)
    plt.figure(figsize=(12, 4))
    plt.plot(t, y_true[idx, 0], label=f"real {horizons[0]}m")
    plt.plot(t, y_pred[idx, 0], label=f"pred {horizons[0]}m")
    plt.title(f"Patient example: {pid}")
    plt.xlabel("Sample index")
    plt.ylabel("mg/dL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

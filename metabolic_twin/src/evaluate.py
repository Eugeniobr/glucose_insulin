"""Evaluation script with plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TwinWindowDataset
from model import MetabolicTwin
from plots import plot_error_by_horizon, plot_patient_example, plot_real_vs_pred, plot_residual_distribution
from preprocessing import add_features, apply_patient_normalizers, feature_columns, read_and_align
from train import split_df, to_df_if_records


def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_pred.append(model(xb).cpu().numpy())
            y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artifacts", default="metabolic_twin/artifacts")
    ap.add_argument("--plots-dir", default="metabolic_twin/artifacts/plots")
    ap.add_argument("--split", choices=["patient", "time"], default="patient")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    cols = meta["feature_cols"]

    df = read_and_align(args.csv, 5)
    df = add_features(df, 5)
    train_df, val_df, test_df = split_df(df, args.split, 0.15, 0.15)
    test_df = to_df_if_records(test_df)

    norms_raw = meta["norms"]
    norms = {}
    from preprocessing import NormStats

    for k, v in norms_raw.items():
        pid, c = k.split("::", 1)
        norms[(pid, c)] = NormStats(v["mean"], v["std"])

    test_df = apply_patient_normalizers(test_df, cols, norms)

    ds_test = TwinWindowDataset(test_df, cols, int(meta["lookback_steps"]), list(meta["horizon_steps"]))
    dl_test = DataLoader(ds_test, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetabolicTwin(
        input_dim=len(cols),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        output_horizons=len(meta["horizon_steps"]),
        rnn_type=str(meta.get("model_type", "gru")),
    ).to(device)
    state = torch.load(artifacts / "best_model.pt", map_location=device)
    model.load_state_dict(state)

    y_true, y_pred = predict(model, dl_test, device)

    horizons = list(meta["horizons_minutes"])
    plot_real_vs_pred(y_true, y_pred, horizons, plots_dir / "real_vs_pred.png")
    plot_error_by_horizon(y_true, y_pred, horizons, plots_dir / "error_by_horizon.png")
    plot_residual_distribution(y_true, y_pred, plots_dir / "residuals.png")

    patient_ids = [ds_test.get_meta(i).patient_id for i in range(len(ds_test))]
    plot_patient_example(y_true, y_pred, patient_ids, horizons, plots_dir / "patient_example.png")

    print(f"saved plots to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

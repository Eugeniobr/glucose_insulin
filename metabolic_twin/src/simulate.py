"""Counterfactual simulation script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import TwinWindowDataset
from model import MetabolicTwin
from preprocessing import NormStats, add_features, apply_patient_normalizers, read_and_align
from train import split_df, to_df_if_records


def simulate_counterfactual(
    model,
    window,
    bolus_multiplier: float = 1.0,
    carbs_multiplier: float = 1.0,
    bolus_idx: int = 1,
    carbs_idx: int = 3,
):
    """Compare base vs counterfactual prediction by scaling bolus/carbs in input window."""
    model.eval()
    x_base = window.clone()
    x_cf = window.clone()
    x_cf[:, :, bolus_idx] = x_cf[:, :, bolus_idx] * bolus_multiplier
    x_cf[:, :, carbs_idx] = x_cf[:, :, carbs_idx] * carbs_multiplier

    with torch.no_grad():
        pred_base = model(x_base).cpu().numpy()
        pred_cf = model(x_cf).cpu().numpy()
    return pred_base, pred_cf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artifacts", default="metabolic_twin/artifacts")
    ap.add_argument("--split", choices=["patient", "time"], default="patient")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--bolus-mult", type=float, default=1.2)
    ap.add_argument("--carbs-mult", type=float, default=1.0)
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    meta = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))
    cols = list(meta["feature_cols"])

    df = read_and_align(args.csv, 5)
    df = add_features(df, 5)
    train_df, _, test_df = split_df(df, args.split, 0.15, 0.15)
    train_df = to_df_if_records(train_df)
    test_df = to_df_if_records(test_df)

    norms_raw = meta["norms"]
    norms = {}
    for k, v in norms_raw.items():
        pid, c = k.split("::", 1)
        norms[(pid, c)] = NormStats(v["mean"], v["std"])

    test_df = apply_patient_normalizers(test_df, cols, norms)
    ds = TwinWindowDataset(test_df, cols, int(meta["lookback_steps"]), list(meta["horizon_steps"]))

    idx = max(0, min(args.sample_idx, len(ds) - 1))
    x, y = ds[idx]
    x = x.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetabolicTwin(
        input_dim=len(cols),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        output_horizons=len(meta["horizon_steps"]),
        rnn_type=str(meta.get("model_type", "gru")),
    ).to(device)
    model.load_state_dict(torch.load(artifacts / "best_model.pt", map_location=device))

    pred_base, pred_cf = simulate_counterfactual(
        model,
        x.to(device),
        bolus_multiplier=args.bolus_mult,
        carbs_multiplier=args.carbs_mult,
        bolus_idx=cols.index("bolus"),
        carbs_idx=cols.index("carbs"),
    )

    horizons = meta["horizons_minutes"]
    print("DISCLAIMER: experimental simulation only, not medical advice.")
    print("target_true:", y.numpy().tolist())
    print("pred_base:", pred_base[0].tolist())
    print("pred_counterfactual:", pred_cf[0].tolist())
    for h, a, b in zip(horizons, pred_base[0], pred_cf[0]):
        print(f"h={h}m delta_cf_minus_base={float(b-a):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

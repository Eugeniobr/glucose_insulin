"""Training script for the metabolic twin MVP."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from config import TwinConfig
from dataset import TwinWindowDataset
from model import MetabolicTwin
from preprocessing import add_features, apply_patient_normalizers, feature_columns, fit_patient_normalizers, read_and_align


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_df(df, strategy: str, val_frac: float, test_frac: float):
    if strategy == "patient":
        patients = sorted(df["patient_id"].unique())
        if len(patients) < 3:
            # Fallback for single/few-subject data.
            strategy = "time"
        else:
            rng = np.random.default_rng(42)
            rng.shuffle(patients)
            n = len(patients)
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))
            test_ids = set(patients[:n_test])
            val_ids = set(patients[n_test : n_test + n_val])
            train_ids = set(patients[n_test + n_val :])

            return (
                df[df["patient_id"].isin(train_ids)].copy(),
                df[df["patient_id"].isin(val_ids)].copy(),
                df[df["patient_id"].isin(test_ids)].copy(),
            )

    out_train, out_val, out_test = [], [], []
    for _, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        out_train.append(g.iloc[: n - n_test - n_val])
        out_val.append(g.iloc[n - n_test - n_val : n - n_test])
        out_test.append(g.iloc[n - n_test :])
    return (
        np.concatenate([x.to_records(index=False) for x in out_train]),
        np.concatenate([x.to_records(index=False) for x in out_val]),
        np.concatenate([x.to_records(index=False) for x in out_test]),
    )


def to_df_if_records(x):
    import pandas as pd

    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame.from_records(x)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_h = np.mean(np.abs(y_true - y_pred), axis=0).tolist()
    rmse_h = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).tolist()
    return {"mae": float(mae), "rmse": float(rmse), "mae_h": mae_h, "rmse_h": rmse_h}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="metabolic_twin/artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--split", choices=["patient", "time"], default="patient")
    parser.add_argument("--model", choices=["gru", "lstm"], default="gru")
    args = parser.parse_args()

    cfg = TwinConfig(data_csv=Path(args.csv), output_dir=Path(args.out), max_epochs=args.epochs, split_strategy=args.split, model_type=args.model)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    df = read_and_align(str(cfg.data_csv), cfg.freq_minutes)
    df = add_features(df, cfg.freq_minutes)

    train_df, val_df, test_df = split_df(df, cfg.split_strategy, cfg.val_frac, cfg.test_frac)
    train_df = to_df_if_records(train_df)
    val_df = to_df_if_records(val_df)
    test_df = to_df_if_records(test_df)

    cols = feature_columns()
    norms = fit_patient_normalizers(train_df, cols)
    train_df = apply_patient_normalizers(train_df, cols, norms)
    val_df = apply_patient_normalizers(val_df, cols, norms)
    test_df = apply_patient_normalizers(test_df, cols, norms)

    ds_train = TwinWindowDataset(train_df, cols, cfg.lookback_steps, cfg.horizon_steps)
    ds_val = TwinWindowDataset(val_df, cols, cfg.lookback_steps, cfg.horizon_steps)
    ds_test = TwinWindowDataset(test_df, cols, cfg.lookback_steps, cfg.horizon_steps)
    if len(ds_train) == 0 or len(ds_val) == 0 or len(ds_test) == 0:
        raise RuntimeError(
            "Insufficient windows after preprocessing/split. "
            "Use more data, reduce lookback, or change split strategy."
        )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetabolicTwin(input_dim=len(cols), hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, output_horizons=len(cfg.horizon_steps), rnn_type=cfg.model_type).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = run_epoch(model, dl_train, criterion, optim, device)
        val_metrics = evaluate(model, dl_val, device)
        val_loss = val_metrics["rmse"]

        print(f"epoch={epoch:03d} train_loss={tr_loss:.4f} val_rmse={val_metrics['rmse']:.3f} val_mae={val_metrics['mae']:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(best_state, cfg.output_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= cfg.patience:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, dl_test, device)
    print("test", test_metrics)

    meta = {
        "feature_cols": cols,
        "lookback_steps": cfg.lookback_steps,
        "horizon_steps": cfg.horizon_steps,
        "horizons_minutes": cfg.horizons_minutes,
        "model_type": cfg.model_type,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "norms": {f"{pid}::{c}": {"mean": st.mean, "std": st.std} for (pid, c), st in norms.items()},
        "test_metrics": test_metrics,
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

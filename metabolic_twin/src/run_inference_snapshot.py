"""Generate a digital twin snapshot JSON for downstream apps/bots.

Experimental-only output. Not medical advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import TwinWindowDataset
from model import MetabolicTwin
from preprocessing import NormStats, add_features, apply_patient_normalizers, feature_columns, read_and_align
from simulate import simulate_counterfactual


def _load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_norms(meta: dict) -> dict[tuple[str, str], NormStats]:
    norms = {}
    for k, v in (meta.get("norms") or {}).items():
        pid, c = k.split("::", 1)
        norms[(pid, c)] = NormStats(float(v["mean"]), float(v["std"]))
    return norms


def _latest_per_patient(df):
    out = []
    for _, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp")
        out.append(g.iloc[[-1]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artifacts", default="metabolic_twin/artifacts")
    ap.add_argument("--output", default="outputs/twin_simulation.json")
    ap.add_argument("--bolus-up", type=float, default=1.2)
    ap.add_argument("--bolus-down", type=float, default=0.8)
    ap.add_argument("--carbs-up", type=float, default=1.2)
    ap.add_argument("--carbs-down", type=float, default=0.8)
    ap.add_argument("--max-patients", type=int, default=20)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    csv_path = (root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv).resolve()
    artifacts = (root / args.artifacts).resolve() if not Path(args.artifacts).is_absolute() else Path(args.artifacts).resolve()
    out_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()

    meta = _load_metadata(artifacts / "metadata.json")
    cols = meta["feature_cols"]
    lookback_steps = int(meta["lookback_steps"])
    horizon_steps = list(meta["horizon_steps"])
    horizons_minutes = list(meta["horizons_minutes"])

    df = read_and_align(str(csv_path), 5)
    df = add_features(df, 5)
    norms = _build_norms(meta)
    df_n = apply_patient_normalizers(df, cols, norms)

    ds = TwinWindowDataset(df_n, cols, lookback_steps, horizon_steps)
    if len(ds) == 0:
        raise RuntimeError("No windows available for inference snapshot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetabolicTwin(
        input_dim=len(cols),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        output_horizons=len(horizon_steps),
        rnn_type=str(meta.get("model_type", "gru")),
    ).to(device)
    model.load_state_dict(torch.load(artifacts / "best_model.pt", map_location=device))
    model.eval()

    records = []
    seen = set()
    for i in range(len(ds) - 1, -1, -1):
        m = ds.get_meta(i)
        pid = str(m.patient_id)
        if pid in seen:
            continue
        seen.add(pid)

        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            base = model(x).cpu().numpy()[0]
        _, cf_bolus_up = simulate_counterfactual(
            model,
            x,
            bolus_multiplier=float(args.bolus_up),
            carbs_multiplier=1.0,
            bolus_idx=cols.index("bolus"),
            carbs_idx=cols.index("carbs"),
        )
        _, cf_bolus_down = simulate_counterfactual(
            model,
            x,
            bolus_multiplier=float(args.bolus_down),
            carbs_multiplier=1.0,
            bolus_idx=cols.index("bolus"),
            carbs_idx=cols.index("carbs"),
        )
        _, cf_carbs_up = simulate_counterfactual(
            model,
            x,
            bolus_multiplier=1.0,
            carbs_multiplier=float(args.carbs_up),
            bolus_idx=cols.index("bolus"),
            carbs_idx=cols.index("carbs"),
        )
        _, cf_carbs_down = simulate_counterfactual(
            model,
            x,
            bolus_multiplier=1.0,
            carbs_multiplier=float(args.carbs_down),
            bolus_idx=cols.index("bolus"),
            carbs_idx=cols.index("carbs"),
        )

        records.append(
            {
                "patient_id": pid,
                "timestamp": str(m.timestamp),
                "horizons_minutes": horizons_minutes,
                "baseline": [float(v) for v in base.tolist()],
                "counterfactual": {
                    "bolus_x_up": [float(v) for v in cf_bolus_up[0].tolist()],
                    "bolus_x_down": [float(v) for v in cf_bolus_down[0].tolist()],
                    "carbs_x_up": [float(v) for v in cf_carbs_up[0].tolist()],
                    "carbs_x_down": [float(v) for v in cf_carbs_down[0].tolist()],
                },
                "target_true": [float(v) for v in y.numpy().tolist()],
            }
        )
        if len(records) >= max(1, int(args.max_patients)):
            break

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "disclaimer": "Experimental simulation only. Not medical advice.",
        "model": {
            "type": str(meta.get("model_type", "gru")),
            "lookback_steps": lookback_steps,
            "horizons_minutes": horizons_minutes,
        },
        "records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

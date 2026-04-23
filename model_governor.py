import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


HORIZONS = ("15", "30", "60", "120")


@dataclass
class Candidate:
    name: str
    args: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governador de melhorias contínuas do modelo (walk-forward)")
    parser.add_argument("--csv-path", default="", help="CSV externo; se vazio usa o mais recente *glucose*.csv")
    parser.add_argument("--max-rows-per-horizon", type=int, default=6000)
    parser.add_argument("--min-train-points", type=int, default=384)
    parser.add_argument("--retrain-every", type=int, default=120)
    parser.add_argument("--retrain-every-high-slope", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--jump-weight", type=float, default=0.25, help="Peso extra do RMSE em regime jump no score")
    parser.add_argument("--min-improvement-pct", type=float, default=0.20, help="Mínimo % para trocar campeão")
    parser.add_argument("--state-path", default="outputs/model_governor_state.json")
    parser.add_argument("--runs-dir", default="outputs/governor_runs")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def _find_latest_csv(base_dir: Path) -> Path | None:
    candidates = sorted(base_dir.glob("*glucose*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _jump_rmse(payload: dict) -> float | None:
    total = 0
    sum_sq = 0.0
    for h in HORIZONS:
        try:
            m = payload["horizons"][h]["metrics_by_regime"]["jump"]["final"]
            c = int(m.get("count") or 0)
            rmse = m.get("rmse_mg_dl")
            if c > 0 and rmse is not None:
                total += c
                sum_sq += (float(rmse) ** 2) * c
        except Exception:
            continue
    if total == 0:
        return None
    return float(np.sqrt(sum_sq / total))


def _score(global_rmse: float, jump_rmse: float | None, jump_weight: float) -> float:
    if jump_rmse is None:
        return float(global_rmse)
    return float(global_rmse + float(jump_weight) * jump_rmse)


def _run_candidate(base_dir: Path, candidate: Candidate, csv_path: Path, args) -> dict:
    runs_dir = (base_dir / args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = runs_dir / f"{candidate.name}.json"
    cmd = [
        args.python_bin,
        "ensemble_backtest.py",
        str(csv_path),
        "--output",
        str(output_path),
        "--max-rows-per-horizon",
        str(args.max_rows_per_horizon),
        "--min-train-points",
        str(args.min_train_points),
        "--retrain-every",
        str(args.retrain_every),
        "--retrain-every-high-slope",
        str(args.retrain_every_high_slope),
        "--max-train-rows",
        str(args.max_train_rows),
        "--bias-enabled",
        "--dynamic-blend",
    ] + candidate.args
    proc = subprocess.run(
        cmd,
        cwd=str(base_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "name": candidate.name,
            "status": "failed",
            "command": cmd,
            "stderr": (proc.stderr or proc.stdout or "").strip()[:1200],
            "output_path": str(output_path),
        }
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    global_rmse = float(payload["final_global"]["rmse_mg_dl"])
    global_mae = float(payload["final_global"]["mae_mg_dl"])
    jump_rmse = _jump_rmse(payload)
    score = _score(global_rmse, jump_rmse, args.jump_weight)
    return {
        "name": candidate.name,
        "status": "ok",
        "command": cmd,
        "output_path": str(output_path),
        "global_rmse_mg_dl": global_rmse,
        "global_mae_mg_dl": global_mae,
        "jump_rmse_mg_dl": jump_rmse,
        "score": score,
    }


def _default_candidates() -> list[Candidate]:
    return [
        Candidate("blend_default", ["--primary", "blend"]),
        Candidate(
            "blend_jump_bias",
            [
                "--primary",
                "blend",
                "--blend-ts-weight",
                "0.45",
                "--blend-slope-gain",
                "0.35",
                "--blend-delta-gain",
                "0.25",
                "--blend-max-ts-weight",
                "0.90",
                "--jump-ts-weight-bonus",
                "0.22",
            ],
        ),
        Candidate(
            "physio_agents_default",
            [
                "--primary",
                "physio_agents",
                "--agent-stable-std60-threshold",
                "6",
                "--agent-stable-range60-threshold",
                "18",
                "--agent-event-carbs-threshold",
                "2",
                "--agent-event-rapid-threshold",
                "0.2",
                "--agent-error-lookback-points",
                "160",
                "--agent-error-min-points",
                "20",
                "--physio-dominance-threshold",
                "0.55",
                "--physio-recent-event-minutes",
                "100",
                "--physio-committee-threshold",
                "0.7",
            ],
        ),
        Candidate(
            "physio_agents_jump_focus",
            [
                "--primary",
                "physio_agents",
                "--agent-stable-std60-threshold",
                "5",
                "--agent-stable-range60-threshold",
                "16",
                "--agent-event-carbs-threshold",
                "1.5",
                "--agent-event-rapid-threshold",
                "0.15",
                "--agent-error-lookback-points",
                "220",
                "--agent-error-min-points",
                "20",
                "--physio-dominance-threshold",
                "0.45",
                "--physio-recent-event-minutes",
                "120",
                "--physio-committee-threshold",
                "0.55",
            ],
        ),
        Candidate(
            "agents_recent_error",
            [
                "--primary",
                "agents",
                "--agent-stable-std60-threshold",
                "6",
                "--agent-stable-range60-threshold",
                "18",
                "--agent-event-carbs-threshold",
                "2",
                "--agent-event-rapid-threshold",
                "0.2",
                "--agent-error-lookback-points",
                "160",
                "--agent-error-min-points",
                "20",
            ],
        ),
    ]


def main() -> int:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent

    csv_path = Path(args.csv_path).resolve() if args.csv_path else _find_latest_csv(base_dir)
    if csv_path is None or not csv_path.exists():
        raise SystemExit("CSV externo não encontrado. Use --csv-path.")

    state_path = (base_dir / args.state_path).resolve()
    old_state = {}
    if state_path.exists():
        try:
            old_state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_state = {}

    candidates = _default_candidates()
    results = [_run_candidate(base_dir, c, csv_path, args) for c in candidates]
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "csv_path": str(csv_path),
            "status": "failed",
            "results": results,
            "message": "Nenhum candidato completou com sucesso.",
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    ranked = sorted(ok_results, key=lambda r: float(r["score"]))
    best = ranked[0]

    incumbent = old_state.get("champion", {})
    incumbent_score = incumbent.get("score")
    promoted = True
    if incumbent_score is not None:
        threshold = float(incumbent_score) * (1.0 - float(args.min_improvement_pct) / 100.0)
        promoted = float(best["score"]) <= threshold

    champion = best if promoted else incumbent
    history = old_state.get("history", [])
    history.append(
        {
            "generated_at": datetime.now().isoformat(),
            "csv_path": str(csv_path),
            "best_this_round": best,
            "promoted": bool(promoted),
            "champion_name": champion.get("name"),
        }
    )
    history = history[-50:]

    payload = {
        "generated_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "status": "ok",
        "ranking": ranked,
        "champion": champion,
        "promoted": bool(promoted),
        "settings": {
            "max_rows_per_horizon": int(args.max_rows_per_horizon),
            "min_train_points": int(args.min_train_points),
            "retrain_every": int(args.retrain_every),
            "retrain_every_high_slope": int(args.retrain_every_high_slope),
            "max_train_rows": int(args.max_train_rows),
            "jump_weight": float(args.jump_weight),
            "min_improvement_pct": float(args.min_improvement_pct),
        },
        "results": results,
        "history": history,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

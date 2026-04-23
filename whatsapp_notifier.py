import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Envia resumo do dashboard via WhatsApp Cloud API")
    parser.add_argument("--send-now", action="store_true", help="Envia uma mensagem imediatamente")
    parser.add_argument("--watch", action="store_true", help="Monitora atualizações e envia automaticamente")
    parser.add_argument("--interval-sec", type=int, default=120, help="Intervalo de checagem no modo watch")
    parser.add_argument("--dry-run", action="store_true", help="Não envia na API, apenas imprime a mensagem")
    parser.add_argument("--metrics-path", default="", help="Override para outputs/metrics.json")
    parser.add_argument("--state-path", default="", help="Override para outputs/state_latest.json")
    parser.add_argument("--series-path", default="", help="Override para outputs/forward_series.json")
    parser.add_argument("--max-history-items", type=int, default=3, help="Quantidade de itens recentes de erro no texto")
    return parser


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def _build_message(metrics: dict[str, Any], state: dict[str, Any], series: dict[str, Any], max_history_items: int) -> str:
    generated_at = metrics.get("generated_at") or state.get("generated_at") or datetime.now().isoformat()
    latest_glucose = state.get("latest_glucose_mg_dl")
    rmse = (metrics.get("metrics") or {}).get("rmse_mg_dl")
    mae = (metrics.get("metrics") or {}).get("mae_mg_dl")

    forecast = metrics.get("forecast") or {}
    simple_2h = forecast.get("simple_glucose_in_2h_mg_dl", forecast.get("glucose_in_2h_mg_dl"))
    ts_1h = forecast.get("takagi_sugeno_ml_glucose_in_1h_mg_dl", None)
    source_2h = forecast.get("forecast_primary_source", "--")

    jump = metrics.get("jump_adaptation") or {}
    jump_detected = jump.get("jump_detected", False)
    jump_weight = jump.get("jump_blend_ts_weight", None)

    history = (series.get("forecast_history") or [])[-max(max_history_items, 0):]
    history_lines = []
    for item in history:
        if not item.get("is_resolved"):
            continue
        ts = item.get("generated_at", "")
        delta = item.get("delta_mg_dl")
        if ts:
            history_lines.append(f"- {ts[-8:-3]}: { _fmt_num(delta, 1) } mg/dL")
    if not history_lines:
        history_lines.append("- sem projeções resolvidas recentes")

    lines = [
        "Dashboard Glicose-Insulina",
        f"Atualização: {generated_at}",
        f"Glicose atual: {_fmt_num(latest_glucose, 0)} mg/dL",
        f"RMSE/MAE: {_fmt_num(rmse, 2)} / {_fmt_num(mae, 2)} mg/dL",
        f"Previsão simples +2h: {_fmt_num(simple_2h, 0)} mg/dL",
        f"Previsão TS+ML +1h: {_fmt_num(ts_1h, 0)} mg/dL",
        f"Fonte +2h: {source_2h}",
        f"Jump detectado: {'sim' if jump_detected else 'não'} | peso TS: {_fmt_num(jump_weight, 2)}",
        "Erro recente (projeção - observado):",
        *history_lines,
    ]
    return "\n".join(lines)


def _send_whatsapp_text(access_token: str, phone_number_id: str, to_number: str, body: str, timeout_sec: int = 20) -> dict[str, Any]:
    endpoint = f"https://graph.facebook.com/v22.0/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": body[:4096]},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {"status": "ok"}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc


def _resolve_paths(args) -> tuple[Path, Path, Path]:
    config = load_config()
    metrics_path = Path(args.metrics_path).resolve() if args.metrics_path else config.metrics_path
    state_path = Path(args.state_path).resolve() if args.state_path else config.state_path
    series_path = Path(args.series_path).resolve() if args.series_path else config.series_path
    return metrics_path, state_path, series_path


def _send_from_files(args) -> tuple[str, dict[str, Any] | None]:
    metrics_path, state_path, series_path = _resolve_paths(args)
    metrics = _load_json(metrics_path)
    state = _load_json(state_path)
    series = _load_json(series_path)
    message = _build_message(metrics, state, series, max_history_items=int(args.max_history_items))

    token = _env("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = _env("WHATSAPP_PHONE_NUMBER_ID")
    to_number = _env("WHATSAPP_TO")

    if args.dry_run:
        print(message)
        return metrics.get("generated_at", ""), None

    missing = [name for name, value in (
        ("WHATSAPP_ACCESS_TOKEN", token),
        ("WHATSAPP_PHONE_NUMBER_ID", phone_number_id),
        ("WHATSAPP_TO", to_number),
    ) if not value]
    if missing:
        raise RuntimeError(f"Variáveis ausentes: {', '.join(missing)}")

    result = _send_whatsapp_text(
        access_token=token,
        phone_number_id=phone_number_id,
        to_number=to_number,
        body=message,
    )
    print(json.dumps({"sent_at": datetime.now().isoformat(), "response": result}, ensure_ascii=False))
    return metrics.get("generated_at", ""), result


def main() -> int:
    args = build_parser().parse_args()
    mode_count = int(args.send_now) + int(args.watch)
    if mode_count != 1:
        raise SystemExit("Use exatamente um modo: --send-now ou --watch")

    if args.send_now:
        _send_from_files(args)
        return 0

    interval_sec = max(int(args.interval_sec), 15)
    print(f"Monitorando dashboard a cada {interval_sec}s...")
    last_generated_at = ""
    while True:
        try:
            generated_at, _ = _send_from_files(args)
            if generated_at and generated_at == last_generated_at:
                # Evita spam quando não houve update.
                pass
            else:
                last_generated_at = generated_at
        except Exception as exc:
            print(f"[{datetime.now().isoformat()}] erro: {exc}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())

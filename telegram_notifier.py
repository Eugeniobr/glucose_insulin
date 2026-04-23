import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notificador Telegram para métricas do dashboard")
    parser.add_argument("--send-now", action="store_true", help="Envia mensagem agora")
    parser.add_argument("--watch", action="store_true", help="Monitora e envia quando houver update")
    parser.add_argument("--interval-sec", type=int, default=120, help="Intervalo no modo watch")
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime, não envia")
    parser.add_argument("--send-photo", action="store_true", help="Envia também foto do gráfico")
    parser.add_argument("--metrics-path", default="")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--series-path", default="")
    parser.add_argument("--plot-path", default="outputs/plots/latest_comparison.png")
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


def _resolve_paths(args) -> tuple[Path, Path, Path, Path]:
    config = load_config()
    metrics_path = Path(args.metrics_path).resolve() if args.metrics_path else config.metrics_path
    state_path = Path(args.state_path).resolve() if args.state_path else config.state_path
    series_path = Path(args.series_path).resolve() if args.series_path else config.series_path
    plot_path = Path(args.plot_path).resolve() if args.plot_path else (config.base_dir / "outputs/plots/latest_comparison.png").resolve()
    return metrics_path, state_path, series_path, plot_path


def _build_message(metrics: dict[str, Any], state: dict[str, Any], series: dict[str, Any]) -> str:
    generated_at = metrics.get("generated_at") or state.get("generated_at") or datetime.now().isoformat()
    latest_glucose = state.get("latest_glucose_mg_dl")
    rmse = (metrics.get("metrics") or {}).get("rmse_mg_dl")
    mae = (metrics.get("metrics") or {}).get("mae_mg_dl")
    forecast = metrics.get("forecast") or {}
    simple_2h = forecast.get("simple_glucose_in_2h_mg_dl", forecast.get("glucose_in_2h_mg_dl"))
    ts_1h = forecast.get("takagi_sugeno_ml_glucose_in_1h_mg_dl", None)
    source_2h = forecast.get("forecast_primary_source", "--")

    history = (series.get("forecast_history") or [])[-3:]
    history_lines = []
    for item in history:
        if not item.get("is_resolved"):
            continue
        ts = item.get("generated_at", "")
        delta = item.get("delta_mg_dl")
        history_lines.append(f"- {ts[-8:-3]}: {_fmt_num(delta, 1)} mg/dL")
    if not history_lines:
        history_lines.append("- sem projeções resolvidas recentes")

    lines = [
        "Dashboard Glicose-Insulina",
        f"Atualização: {generated_at}",
        f"Glicose atual: {_fmt_num(latest_glucose, 0)} mg/dL",
        f"RMSE/MAE: {_fmt_num(rmse, 2)} / {_fmt_num(mae, 2)} mg/dL",
        f"Simples +2h: {_fmt_num(simple_2h, 0)} mg/dL",
        f"TS+ML +1h: {_fmt_num(ts_1h, 0)} mg/dL",
        f"Fonte +2h: {source_2h}",
        "Erro recente (projeção - observado):",
        *history_lines,
    ]
    return "\n".join(lines)


def _telegram_api(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int = 20) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content else {}


def _post_multipart(url: str, fields: dict[str, str], file_field: str, file_path: Path, timeout_sec: int = 30) -> dict[str, Any]:
    boundary = "----telegram-boundary-7d4f2a9c"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{value}\r\n".encode("utf-8"))
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    chunks.append(b"Content-Type: image/png\r\n\r\n")
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content else {}


def _send_text(token: str, chat_id: str, text: str) -> dict[str, Any]:
    return _post_json(
        _telegram_api(token, "sendMessage"),
        {"chat_id": chat_id, "text": text[:4096], "disable_web_page_preview": True},
    )


def _send_photo(token: str, chat_id: str, photo_path: Path, caption: str) -> dict[str, Any]:
    if not photo_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {photo_path}")
    return _post_multipart(
        _telegram_api(token, "sendPhoto"),
        fields={"chat_id": chat_id, "caption": caption[:1024]},
        file_field="photo",
        file_path=photo_path,
    )


def _read_updates_and_guess_chat_id(token: str) -> str:
    url = _telegram_api(token, "getUpdates")
    with request.urlopen(url, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    results = payload.get("result") or []
    for item in reversed(results):
        msg = item.get("message") or item.get("edited_message") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is not None:
            return str(cid)
    return ""


def _send_once(args, token: str, chat_id: str) -> tuple[str, dict[str, Any] | None]:
    metrics_path, state_path, series_path, plot_path = _resolve_paths(args)
    metrics = _load_json(metrics_path)
    state = _load_json(state_path)
    series = _load_json(series_path)
    text = _build_message(metrics, state, series)
    generated_at = str(metrics.get("generated_at") or "")

    if args.dry_run:
        print(text)
        if args.send_photo:
            print(f"[dry-run] foto: {plot_path}")
        return generated_at, None

    resp_text = _send_text(token, chat_id, text)
    if args.send_photo:
        _send_photo(token, chat_id, plot_path, "Plot atualizado do pipeline")
    return generated_at, resp_text


def main() -> int:
    args = build_parser().parse_args()
    if int(args.send_now) + int(args.watch) != 1:
        raise SystemExit("Use exatamente um modo: --send-now ou --watch")

    token = _env("TELEGRAM_BOT_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("Defina TELEGRAM_BOT_TOKEN")
    if args.dry_run and not token:
        token = "dry-run-token"
    chat_id = _env("TELEGRAM_CHAT_ID")

    if not chat_id and not args.dry_run:
        try:
            chat_id = _read_updates_and_guess_chat_id(token)
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Falha ao ler getUpdates (HTTP {exc.code}): {raw[:400]}") from exc
        if not chat_id:
            raise SystemExit("TELEGRAM_CHAT_ID ausente e nenhum chat encontrado em getUpdates. Envie uma mensagem ao bot e tente novamente.")

    if args.send_now:
        _, response = _send_once(args, token, chat_id)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False))
        return 0

    interval = max(int(args.interval_sec), 15)
    print(f"Monitorando a cada {interval}s...")
    last_generated = ""
    while True:
        try:
            generated, _ = _send_once(args, token, chat_id)
            if generated:
                if generated == last_generated:
                    pass
                else:
                    last_generated = generated
        except Exception as exc:
            print(f"[{datetime.now().isoformat()}] erro: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

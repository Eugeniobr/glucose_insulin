import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, parse, request

from config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webhook WhatsApp + DeepSeek para assistente do dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verify-token", default=os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip())
    parser.add_argument("--dry-run", action="store_true", help="Não envia mensagens no WhatsApp")
    parser.add_argument("--manual-meals-csv", default="outputs/whatsapp_manual_meals.csv")
    parser.add_argument("--manual-insulin-csv", default="outputs/whatsapp_manual_insulin.csv")
    return parser


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fmt(v, d=1) -> str:
    try:
        if v is None:
            return "--"
        return f"{float(v):.{d}f}"
    except Exception:
        return "--"


def _send_whatsapp(access_token: str, phone_number_id: str, to_number: str, body: str, dry_run: bool) -> dict:
    if dry_run:
        print(f"[dry-run] -> {to_number}: {body}")
        return {"status": "dry-run"}
    endpoint = f"https://graph.facebook.com/v22.0/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": body[:4096]},
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_deepseek(api_key: str, model: str, user_text: str, context_text: str) -> str:
    endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    system_prompt = (
        "Você é um assistente técnico do projeto de glicose-insulina. "
        "Responda em português de forma objetiva. "
        "NUNCA inclua links, URLs, http, https, www ou referências clicáveis. "
        "A resposta deve ser somente texto puro. "
        "Sempre tratar recomendações de dose como simulação de pesquisa, não orientação clínica. "
        "Se faltar dado, diga exatamente o que faltou."
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto do dashboard:\n{context_text}\n\nPergunta:\n{user_text}"},
        ],
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return "Não consegui gerar resposta do DeepSeek agora."
            raw = str(((choices[0].get("message") or {}).get("content") or "")).strip()
            return _sanitize_whatsapp_text(raw) or "Sem conteúdo retornado."
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return f"Falha DeepSeek (HTTP {exc.code}): {raw[:600]}"
    except Exception as exc:
        return f"Falha DeepSeek: {exc}"


def _dashboard_context(config) -> str:
    metrics = _read_json(config.metrics_path)
    state = _read_json(config.state_path)
    series = _read_json(config.series_path)
    forecast = metrics.get("forecast", {})
    hist = series.get("forecast_history", [])
    resolved = [x for x in hist[-6:] if x.get("is_resolved")]
    history_lines = []
    for item in resolved[-3:]:
        history_lines.append(f"{item.get('generated_at','')}: delta={_fmt(item.get('delta_mg_dl'),1)}")
    return (
        f"generated_at={metrics.get('generated_at')}\n"
        f"rmse={_fmt((metrics.get('metrics') or {}).get('rmse_mg_dl'),2)} mae={_fmt((metrics.get('metrics') or {}).get('mae_mg_dl'),2)}\n"
        f"latest_glucose={_fmt(state.get('latest_glucose_mg_dl'),0)}\n"
        f"simple_2h={_fmt(forecast.get('simple_glucose_in_2h_mg_dl', forecast.get('glucose_in_2h_mg_dl')),0)}\n"
        f"ts_ml_1h={_fmt(forecast.get('takagi_sugeno_ml_glucose_in_1h_mg_dl'),0)}\n"
        f"source_2h={forecast.get('forecast_primary_source')}\n"
        f"recent_deltas={'; '.join(history_lines) if history_lines else 'none'}"
    )


def _compute_pk_active_from_state(state: dict) -> float:
    insulin = state.get("insulin") or []
    if not insulin:
        return 0.0
    now = datetime.fromisoformat(state.get("latest_timestamp")) if state.get("latest_timestamp") else datetime.now()
    active = 0.0
    for ev in insulin:
        ts_raw = ev.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        elapsed_min = max((now - ts).total_seconds() / 60.0, 0.0)
        units = float(ev.get("units", 0.0) or 0.0)
        kind = str(ev.get("insulin_type", "desconhecido")).lower()
        tau = 360.0 if kind == "basal" else 75.0
        active += units * pow(2.718281828, -elapsed_min / tau)
    return max(active, 0.0)


def _dose_formula(glucose: float, cho_g: float, sensitivity_u_per_g: float, pk_active_u: float) -> dict:
    cho_dose = max(0.0, sensitivity_u_per_g * max(0.0, cho_g))
    correction = (glucose - 140.0) / 30.0
    raw = max(0.0, cho_dose + correction)
    pk_att = max(0.0, min(0.65, pk_active_u / 6.0))
    post_pk = raw * (1.0 - pk_att)
    final = min(4.0, max(0.0, post_pk))
    return {
        "cho_u": cho_dose,
        "correction_u": correction,
        "raw_u": raw,
        "pk_active_u": pk_active_u,
        "pk_att": pk_att,
        "final_u": final,
    }


def _ensure_csv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _append_manual_meal(path: Path, timestamp: str, cho_g: float) -> None:
    _ensure_csv(path, ["timestamp", "carboidratos (g)"])
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, f"{cho_g:.1f}"])


def _append_manual_insulin(path: Path, timestamp: str, units: float, insulin_type: str) -> None:
    _ensure_csv(path, ["timestamp", "unidades", "tipo"])
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, f"{units:.2f}", insulin_type])


def _run_pipeline_with_manual_inputs(base_dir: Path, meals_csv: Path, insulin_csv: Path) -> tuple[bool, str]:
    env = os.environ.copy()
    env["MEALS_CSV_URL"] = str(meals_csv.resolve())
    env["INSULIN_CSV_URL"] = str(insulin_csv.resolve())
    cmd = [sys.executable, "main.py", "once", "--skip-extract"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(base_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()[:700]
        return True, "Pipeline atualizado com inputs manuais."
    except Exception as exc:
        return False, str(exc)


def _run_model_governor(base_dir: Path) -> tuple[bool, str]:
    cmd = [sys.executable, "model_governor.py"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()[:900]
        return True, (proc.stdout or "ok").strip()[:900]
    except Exception as exc:
        return False, str(exc)


def _governor_status(base_dir: Path) -> str:
    path = (base_dir / "outputs/model_governor_state.json").resolve()
    if not path.exists():
        return "Governador ainda não executou. Envie: melhorar modelo"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Estado do governador inválido."
    champion = payload.get("champion") or {}
    ranked = payload.get("ranking") or []
    top_lines = []
    for item in ranked[:3]:
        top_lines.append(
            f"- {item.get('name')}: score {_fmt(item.get('score'),3)} | rmse {_fmt(item.get('global_rmse_mg_dl'),3)} | jump {_fmt(item.get('jump_rmse_mg_dl'),3)}"
        )
    return (
        "Status melhoria contínua:\n"
        f"- Última execução: {payload.get('generated_at','--')}\n"
        f"- Campeão: {champion.get('name','--')}\n"
        f"- Score campeão: {_fmt(champion.get('score'),3)}\n"
        f"- Promovido nesta rodada: {'sim' if payload.get('promoted') else 'não'}\n"
        "Top candidatos:\n"
        + ("\n".join(top_lines) if top_lines else "- sem ranking")
    )


def _parse_command(text: str) -> dict:
    msg = (text or "").strip()
    low = msg.lower()

    if low in {"metricas", "métricas", "status"}:
        return {"type": "metrics"}
    if low in {"forecast", "previsao", "previsão"}:
        return {"type": "forecast"}

    m_dose = re.search(r"dose\s+cho\s*=?\s*([0-9]+(?:[.,][0-9]+)?)\s+glicemia\s*=?\s*([0-9]+(?:[.,][0-9]+)?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?", low)
    if m_dose:
        cho = float(m_dose.group(1).replace(",", "."))
        glucose = float(m_dose.group(2).replace(",", "."))
        sens = m_dose.group(3)
        sensitivity = float(sens.replace(",", ".")) if sens else None
        return {"type": "dose", "cho": cho, "glucose": glucose, "sensitivity": sensitivity}

    m_cho = re.search(r"(?:cho|carbo)\s+([0-9]+(?:[.,][0-9]+)?)", low)
    if m_cho and ("registr" in low or "add" in low or low.startswith("cho ")):
        return {"type": "add_cho", "cho": float(m_cho.group(1).replace(",", "."))}

    m_ins = re.search(r"(?:insulina|insulin)\s+([0-9]+(?:[.,][0-9]+)?)(?:\s+(rapida|rápida|basal|regular))?", low)
    if m_ins and ("registr" in low or "add" in low or low.startswith("insulina ")):
        insulin_type = m_ins.group(2) or "rapida"
        insulin_type = "rapida" if insulin_type in {"rapida", "rápida"} else insulin_type
        return {"type": "add_insulin", "units": float(m_ins.group(1).replace(",", ".")), "insulin_type": insulin_type}

    if low in {"atualizar", "update"}:
        return {"type": "update"}
    if low in {"melhorar modelo", "otimizar modelo", "improve model"}:
        return {"type": "improve_model"}
    if low in {"status melhoria", "status do modelo", "governador status"}:
        return {"type": "governor_status"}

    return {"type": "llm", "text": msg}


def _sanitize_whatsapp_text(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "[link removido]", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwww\.\S+\b", "[link removido]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{3,}", "  ", cleaned)
    return cleaned.strip()


def _build_metrics_reply(config) -> str:
    metrics = _read_json(config.metrics_path)
    state = _read_json(config.state_path)
    forecast = metrics.get("forecast") or {}
    return (
        "Métricas gerais:\n"
        f"- RMSE: {_fmt((metrics.get('metrics') or {}).get('rmse_mg_dl'),2)} mg/dL\n"
        f"- MAE: {_fmt((metrics.get('metrics') or {}).get('mae_mg_dl'),2)} mg/dL\n"
        f"- Glicose atual: {_fmt(state.get('latest_glucose_mg_dl'),0)} mg/dL\n"
        f"- Fonte forecast +2h: {forecast.get('forecast_primary_source','--')}"
    )


def _build_forecast_reply(config) -> str:
    metrics = _read_json(config.metrics_path)
    forecast = metrics.get("forecast") or {}
    return (
        "Forecast atual:\n"
        f"- Simples +2h: {_fmt(forecast.get('simple_glucose_in_2h_mg_dl', forecast.get('glucose_in_2h_mg_dl')),0)} mg/dL\n"
        f"- TS+ML +1h: {_fmt(forecast.get('takagi_sugeno_ml_glucose_in_1h_mg_dl'),0)} mg/dL\n"
        f"- Primário +2h: {_fmt(forecast.get('glucose_in_2h_mg_dl'),0)} mg/dL"
    )


class BotHandler(BaseHTTPRequestHandler):
    verify_token = ""
    dry_run = False
    config = None
    access_token = ""
    phone_number_id = ""
    deepseek_api_key = ""
    deepseek_model = "deepseek-chat"
    manual_meals_csv = Path("outputs/whatsapp_manual_meals.csv")
    manual_insulin_csv = Path("outputs/whatsapp_manual_insulin.csv")

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = parse.urlparse(self.path)
        if parsed.path != "/webhook":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        qs = parse.parse_qs(parsed.query)
        mode = (qs.get("hub.mode") or [""])[0]
        token = (qs.get("hub.verify_token") or [""])[0]
        challenge = (qs.get("hub.challenge") or [""])[0]
        if mode == "subscribe" and token == self.verify_token:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            self.wfile.write(challenge.encode("utf-8"))
            return
        self.send_error(HTTPStatus.FORBIDDEN, "Invalid verify token")

    def do_POST(self):
        parsed = parse.urlparse(self.path)
        if parsed.path != "/webhook":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

        messages = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    messages.append(msg)
        for msg in messages:
            threading.Thread(target=self._handle_message, args=(msg,), daemon=True).start()

    def _handle_message(self, msg: dict) -> None:
        from_number = str(msg.get("from", "")).strip()
        text = (((msg.get("text") or {}).get("body")) or "").strip()
        if not from_number or not text:
            return
        cmd = _parse_command(text)
        response = _sanitize_whatsapp_text(self._dispatch_command(cmd, from_number))
        try:
            _send_whatsapp(self.access_token, self.phone_number_id, from_number, response, self.dry_run)
        except Exception as exc:
            print(f"falha envio whatsapp: {exc}")

    def _dispatch_command(self, cmd: dict, from_number: str) -> str:
        if cmd["type"] == "metrics":
            return _build_metrics_reply(self.config)
        if cmd["type"] == "forecast":
            return _build_forecast_reply(self.config)
        if cmd["type"] == "dose":
            state = _read_json(self.config.state_path)
            cho = float(cmd["cho"])
            glucose = float(cmd["glucose"])
            if cmd.get("sensitivity") is not None:
                sensitivity = float(cmd["sensitivity"])
            else:
                scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                adaptive_isf = 45.0 * scale
                sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
            dose = _dose_formula(
                glucose=glucose,
                cho_g=cho,
                sensitivity_u_per_g=sensitivity,
                pk_active_u=_compute_pk_active_from_state(state),
            )
            return (
                "Dose (simulação, não clínica):\n"
                f"- Sensibilidade: {_fmt(sensitivity,3)} U/g\n"
                f"- CHO: {_fmt(cho,0)} g | Glicemia: {_fmt(glucose,0)} mg/dL\n"
                f"- CHO dose: {_fmt(dose['cho_u'],2)} U\n"
                f"- Correção: {_fmt(dose['correction_u'],2)} U\n"
                f"- Bruta: {_fmt(dose['raw_u'],2)} U\n"
                f"- PK ativo: {_fmt(dose['pk_active_u'],2)} U (aten. {_fmt(dose['pk_att']*100,0)}%)\n"
                f"- Final limitada: {_fmt(dose['final_u'],2)} U"
            )
        if cmd["type"] == "add_cho":
            now_iso = datetime.now().replace(microsecond=0).isoformat()
            _append_manual_meal(self.manual_meals_csv, now_iso, float(cmd["cho"]))
            threading.Thread(target=self._update_pipeline_and_notify, args=(from_number,), daemon=True).start()
            return f"CHO registrado: {_fmt(cmd['cho'],0)} g. Atualizando modelo..."
        if cmd["type"] == "add_insulin":
            now_iso = datetime.now().replace(microsecond=0).isoformat()
            _append_manual_insulin(self.manual_insulin_csv, now_iso, float(cmd["units"]), str(cmd["insulin_type"]))
            threading.Thread(target=self._update_pipeline_and_notify, args=(from_number,), daemon=True).start()
            return f"Insulina registrada: {_fmt(cmd['units'],2)} U ({cmd['insulin_type']}). Atualizando modelo..."
        if cmd["type"] == "update":
            threading.Thread(target=self._update_pipeline_and_notify, args=(from_number,), daemon=True).start()
            return "Atualização do modelo iniciada."
        if cmd["type"] == "improve_model":
            threading.Thread(target=self._improve_model_and_notify, args=(from_number,), daemon=True).start()
            return "Ciclo de melhoria contínua iniciado (backtests comparativos)."
        if cmd["type"] == "governor_status":
            return _governor_status(self.config.base_dir)

        context = _dashboard_context(self.config)
        if not self.deepseek_api_key:
            return "DeepSeek API key ausente. Defina DEEPSEEK_API_KEY."
        return _call_deepseek(
            api_key=self.deepseek_api_key,
            model=self.deepseek_model,
            user_text=cmd["text"],
            context_text=context,
        )

    def _update_pipeline_and_notify(self, to_number: str) -> None:
        ok, detail = _run_pipeline_with_manual_inputs(
            base_dir=self.config.base_dir,
            meals_csv=self.manual_meals_csv,
            insulin_csv=self.manual_insulin_csv,
        )
        text = f"{'Atualização concluída.' if ok else 'Falha na atualização.'}\n{detail}"
        try:
            _send_whatsapp(self.access_token, self.phone_number_id, to_number, text, self.dry_run)
        except Exception as exc:
            print(f"falha envio pós-atualização: {exc}")

    def _improve_model_and_notify(self, to_number: str) -> None:
        ok, detail = _run_model_governor(self.config.base_dir)
        status = _governor_status(self.config.base_dir)
        text = f"{'Melhoria concluída.' if ok else 'Falha na melhoria.'}\n{detail[:350]}\n\n{status}"
        text = _sanitize_whatsapp_text(text)
        try:
            _send_whatsapp(self.access_token, self.phone_number_id, to_number, text, self.dry_run)
        except Exception as exc:
            print(f"falha envio pós-melhoria: {exc}")


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()

    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    if not access_token and not args.dry_run:
        raise SystemExit("Defina WHATSAPP_ACCESS_TOKEN ou use --dry-run")
    if not phone_number_id and not args.dry_run:
        raise SystemExit("Defina WHATSAPP_PHONE_NUMBER_ID ou use --dry-run")
    if not args.verify_token:
        raise SystemExit("Defina WHATSAPP_VERIFY_TOKEN (env ou --verify-token)")

    handler = BotHandler
    handler.verify_token = args.verify_token
    handler.dry_run = bool(args.dry_run)
    handler.config = config
    handler.access_token = access_token
    handler.phone_number_id = phone_number_id
    handler.deepseek_api_key = deepseek_api_key
    handler.deepseek_model = deepseek_model
    handler.manual_meals_csv = (config.base_dir / args.manual_meals_csv).resolve()
    handler.manual_insulin_csv = (config.base_dir / args.manual_insulin_csv).resolve()

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Webhook rodando em http://{args.host}:{args.port}/webhook")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

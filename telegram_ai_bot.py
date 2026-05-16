import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from config import load_config
from dose_engine import DoseLimits, compute_dose, simulate_scenarios

try:
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    torch = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bot Telegram conversacional com DeepSeek")
    parser.add_argument("--poll-timeout-sec", type=int, default=25, help="Timeout de long polling no getUpdates")
    parser.add_argument("--loop-sleep-sec", type=float, default=1.0, help="Pausa entre ciclos")
    parser.add_argument("--offset-path", default="outputs/telegram_ai_offset.json")
    parser.add_argument("--manual-meals-csv", default="outputs/telegram_manual_meals.csv")
    parser.add_argument("--manual-insulin-csv", default="outputs/telegram_manual_insulin.csv")
    parser.add_argument(
        "--food-table-path",
        default=os.getenv("FOOD_GI_TABLE_PATH", "").strip(),
        help="CSV/JSON com tabela de alimentos e índice glicêmico",
    )
    parser.add_argument("--allowed-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", "").strip())
    parser.add_argument("--dry-run", action="store_true", help="Não envia para Telegram, apenas imprime")
    return parser


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _fmt(v: Any, d: int = 1) -> str:
    try:
        if v is None:
            return "--"
        return f"{float(v):.{d}f}"
    except Exception:
        return "--"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _norm_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _pick_field(row: dict[str, Any], aliases: list[str]) -> Any:
    normalized = {_norm_text(key): value for key, value in row.items()}
    for alias in aliases:
        val = normalized.get(_norm_text(alias))
        if val is None:
            continue
        if str(val).strip() == "":
            continue
        return val
    return None


def _extract_amount_g(raw_food_text: str) -> tuple[str, float | None]:
    text = (raw_food_text or "").strip()
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:g|gramas?)\b", text, flags=re.IGNORECASE)
    if not m:
        return text, None
    grams = float(m.group(1).replace(",", "."))
    cleaned = (text[: m.start()] + " " + text[m.end() :]).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, grams


def _extract_pdf_rows(path: Path) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("Dependência ausente: pypdf") from exc

    reader = PdfReader(str(path))
    rows: list[dict[str, Any]] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            clean = re.sub(r"\s{2,}", " ", line).strip(" -\t")
            if not clean:
                continue
            low = _norm_text(clean)
            if low in {"alimento", "ig", "cg", "indice glicemico", "carga glicemica"}:
                continue
            if len(clean) < 4:
                continue
            # Heurística: linha de tabela com alimento + IG (inteiro 10-130).
            m = re.match(r"^(.+?)\s+([0-9]{1,3})(?:\s+([0-9]{1,3}(?:[.,][0-9]+)?))?$", clean)
            if not m:
                continue
            food = m.group(1).strip(" -;:")
            if not food or len(food) < 2:
                continue
            gi = float(m.group(2))
            if gi < 10 or gi > 130:
                continue
            cg_val = _parse_float(m.group(3)) if m.group(3) else None
            rows.append(
                {
                    "alimento": food,
                    "ig": gi,
                    "cg": cg_val,
                }
            )
    return rows


def _load_food_table(base_dir: Path, configured_path: str) -> dict[str, Any]:
    candidates: list[Path] = []
    if configured_path:
        p = Path(configured_path)
        candidates.append(p if p.is_absolute() else (base_dir / p))
    else:
        candidates.extend(
            [
                base_dir / "food_gi_table.csv",
                base_dir / "food_gi_table.json",
                base_dir / "food_gi_table.pdf",
                base_dir / "foods_gi.csv",
                base_dir / "foods_gi.json",
                base_dir / "foods_gi.pdf",
                base_dir / "indice_glicemico.csv",
                base_dir / "indice_glicemico.json",
                base_dir / "indice_glicemico.pdf",
                base_dir / "tabela_indice_glicemico.csv",
                base_dir / "tabela_indice_glicemico.json",
                base_dir / "tabela_indice_glicemico.pdf",
                base_dir / "outputs/food_gi_table.csv",
                base_dir / "outputs/food_gi_table.json",
                base_dir / "outputs/food_gi_table.pdf",
            ]
        )
        for cand in base_dir.glob("*"):
            if not cand.is_file():
                continue
            if cand.suffix.lower() not in {".csv", ".json", ".pdf"}:
                continue
            if re.search(r"(ig|indice|glicem|glyc|food|alimento)", cand.name, flags=re.IGNORECASE):
                candidates.append(cand)

    path: Path | None = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            path = cand.resolve()
            break
    if path is None:
        return {
            "path": "",
            "entries": [],
            "status": "missing",
            "message": "Tabela de alimentos/IG não encontrada",
        }

    try:
        rows: list[dict[str, Any]]
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("JSON precisa ser lista de objetos")
            rows = [row for row in payload if isinstance(row, dict)]
        elif path.suffix.lower() == ".pdf":
            rows = _extract_pdf_rows(path)
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
    except Exception as exc:
        return {
            "path": str(path),
            "entries": [],
            "status": "error",
            "message": f"Falha ao ler tabela: {exc}",
        }

    entries: list[dict[str, Any]] = []
    for row in rows:
        food_name_raw = _pick_field(
            row,
            [
                "alimento",
                "food",
                "nome",
                "item",
                "descricao",
                "descrição",
            ],
        )
        if food_name_raw is None:
            continue
        food_name = str(food_name_raw).strip()
        if not food_name:
            continue
        gi = _parse_float(
            _pick_field(
                row,
                [
                    "ig",
                    "gi",
                    "indice_glicemico",
                    "índice_glicêmico",
                    "indice glicemico",
                    "glycemic_index",
                ],
            )
        )
        glycemic_load = _parse_float(
            _pick_field(
                row,
                [
                    "cg",
                    "carga_glicemica",
                    "carga glicemica",
                    "glycemic_load",
                ],
            )
        )
        carbs_per_portion = _parse_float(
            _pick_field(
                row,
                [
                    "cho_g",
                    "carboidratos_g",
                    "carboidratos (g)",
                    "carbs_g",
                    "carboidratos",
                    "carbs",
                ],
            )
        )
        if carbs_per_portion is None and glycemic_load is not None and gi is not None and gi > 0:
            carbs_per_portion = max(0.0, float(glycemic_load) * 100.0 / float(gi))
        carbs_per_100g = _parse_float(
            _pick_field(
                row,
                [
                    "cho_100g",
                    "carboidratos_100g",
                    "carbs_100g",
                ],
            )
        )
        portion_g = _parse_float(
            _pick_field(
                row,
                [
                    "porcao_g",
                    "porção_g",
                    "porcao",
                    "porção",
                    "serving_g",
                    "gramas_por_porcao",
                ],
            )
        )
        entries.append(
            {
                "food_name": food_name,
                "norm_name": _norm_text(food_name),
                "gi": gi,
                "glycemic_load": glycemic_load,
                "carbs_per_portion_g": carbs_per_portion,
                "carbs_per_100g": carbs_per_100g,
                "portion_g": portion_g,
            }
        )

    if not entries:
        return {
            "path": str(path),
            "entries": [],
            "status": "empty",
            "message": "Tabela sem linhas válidas (coluna alimento obrigatória).",
        }

    return {
        "path": str(path),
        "entries": entries,
        "status": "ok",
        "message": f"Tabela carregada com {len(entries)} alimentos.",
    }


def _lookup_food(food_query: str, table: dict[str, Any]) -> dict[str, Any] | None:
    entries = table.get("entries") or []
    if not entries:
        return None
    q = _norm_text(food_query)
    if not q:
        return None

    exact = [entry for entry in entries if entry["norm_name"] == q]
    if exact:
        return exact[0]
    starts = [entry for entry in entries if entry["norm_name"].startswith(q)]
    if starts:
        return starts[0]
    contains = [entry for entry in entries if q in entry["norm_name"]]
    if contains:
        return contains[0]

    q_tokens = set(q.split())
    best_entry: dict[str, Any] | None = None
    best_score = 0
    for entry in entries:
        score = len(q_tokens.intersection(set(entry["norm_name"].split())))
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_score > 0:
        return best_entry
    return None


def _estimate_carbs_from_entry(entry: dict[str, Any], grams: float | None) -> float | None:
    carbs_100 = entry.get("carbs_per_100g")
    carbs_portion = entry.get("carbs_per_portion_g")
    portion_g = entry.get("portion_g")
    if grams is not None:
        if carbs_100 is not None:
            return max(0.0, float(carbs_100) * float(grams) / 100.0)
        if carbs_portion is not None and portion_g:
            return max(0.0, float(carbs_portion) * float(grams) / float(portion_g))
    if carbs_portion is not None:
        return max(0.0, float(carbs_portion))
    return None


def _gi_impact_factor(gi: float | None) -> float:
    if gi is None:
        return 1.0
    return max(0.85, min(1.20, 1.0 + (float(gi) - 55.0) * 0.004))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _telegram_api(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int = 30) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content else {}


def _send_telegram_message(token: str, chat_id: str, text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] -> {chat_id}: {text}")
        return
    _post_json(
        _telegram_api(token, "sendMessage"),
        {
            "chat_id": chat_id,
            "text": text[:4096],
            "disable_web_page_preview": True,
        },
    )


def _send_telegram_photo(token: str, chat_id: str, photo_path: Path, caption: str, dry_run: bool) -> None:
    if not photo_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {photo_path}")
    if dry_run:
        print(f"[dry-run] photo -> {chat_id}: {photo_path} | {caption}")
        return
    boundary = "----telegram-photo-boundary-2a3f"
    chunks: list[bytes] = []
    fields = {"chat_id": str(chat_id), "caption": caption[:1024]}
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{value}\r\n".encode("utf-8"))
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(f'Content-Disposition: form-data; name="photo"; filename="{photo_path.name}"\r\n'.encode("utf-8"))
    chunks.append(b"Content-Type: image/png\r\n\r\n")
    chunks.append(photo_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    req = request.Request(
        _telegram_api(token, "sendPhoto"),
        data=b"".join(chunks),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=45):
        pass


def _get_updates(token: str, offset: int | None, timeout_sec: int) -> list[dict[str, Any]]:
    params = {"timeout": str(max(timeout_sec, 1))}
    if offset is not None:
        params["offset"] = str(offset)
    url = _telegram_api(token, "getUpdates") + "?" + parse.urlencode(params)
    with request.urlopen(url, timeout=max(timeout_sec + 10, 15)) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("result") or []


def _sanitize_text(text: str) -> str:
    cleaned = re.sub(r"\s{3,}", "  ", (text or "").strip())
    return cleaned[:4000]


def _dashboard_context(config) -> str:
    metrics = _read_json(config.metrics_path)
    state = _read_json(config.state_path)
    series = _read_json(config.series_path)
    forecast = metrics.get("forecast") or {}
    hist = series.get("forecast_history") or []
    resolved = [x for x in hist[-8:] if x.get("is_resolved")]
    hist_lines = []
    for item in resolved[-4:]:
        hist_lines.append(f"{item.get('generated_at','')}: delta={_fmt(item.get('delta_mg_dl'),1)}")
    return (
        f"generated_at={metrics.get('generated_at')}\n"
        f"rmse={_fmt((metrics.get('metrics') or {}).get('rmse_mg_dl'),2)} "
        f"mae={_fmt((metrics.get('metrics') or {}).get('mae_mg_dl'),2)}\n"
        f"latest_glucose={_fmt(state.get('latest_glucose_mg_dl'),0)}\n"
        f"simple_2h={_fmt(forecast.get('simple_glucose_in_2h_mg_dl', forecast.get('glucose_in_2h_mg_dl')),0)}\n"
        f"ts_ml_1h={_fmt(forecast.get('takagi_sugeno_ml_glucose_in_1h_mg_dl'),0)}\n"
        f"source_2h={forecast.get('forecast_primary_source')}\n"
        f"deltas={'; '.join(hist_lines) if hist_lines else 'none'}"
    )


def _dose_limits_from_env() -> DoseLimits:
    return DoseLimits(
        target_mg_dl=float(os.getenv("DOSE_TARGET_MGDL", "110") or 110),
        correction_divisor_mg_dl_per_u=float(os.getenv("DOSE_CORRECTION_DIVISOR_MGDL_PER_U", "30") or 30),
        max_bolus_u=float(os.getenv("DOSE_MAX_BOLUS_U", "4.0") or 4.0),
        max_iob_reduction=float(os.getenv("DOSE_MAX_IOB_REDUCTION", "0.65") or 0.65),
        min_glucose_block_mg_dl=float(os.getenv("DOSE_MIN_GLUCOSE_BLOCK_MGDL", "90") or 90),
        low_glucose_reduction_mg_dl=float(os.getenv("DOSE_LOW_GLUCOSE_REDUCTION_MGDL", "110") or 110),
    )


def _call_deepseek(api_key: str, model: str, user_text: str, context: str) -> str:
    endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é assistente técnico do projeto de glicose-insulina. "
                    "Responda em português, objetivo e prático. "
                    "Quando falar de dose, trate como simulação/apoio analítico e não orientação clínica."
                ),
            },
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta:\n{user_text}"},
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
        with request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return "Não consegui obter resposta do DeepSeek agora."
        content = str(((choices[0].get("message") or {}).get("content")) or "").strip()
        return _sanitize_text(content or "Sem conteúdo retornado.")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return _sanitize_text(f"Falha DeepSeek HTTP {exc.code}: {raw[:600]}")
    except Exception as exc:
        return _sanitize_text(f"Falha DeepSeek: {exc}")


def _call_ollama(model: str, user_text: str, context: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{base_url}/api/chat"
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "80") or 80)
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2") or 0.2)
    payload = {
        "model": model,
        "stream": False,
        "options": {"num_predict": max(32, min(num_predict, 512)), "temperature": max(0.0, min(temperature, 1.0))},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é assistente técnico do projeto de glicose-insulina. "
                    "Responda em português, objetivo e prático. "
                    "Quando falar de dose, trate como simulação/apoio analítico e não orientação clínica."
                ),
            },
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta:\n{user_text}"},
        ],
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        return _sanitize_text(content or "Sem conteúdo retornado pelo Ollama.")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return _sanitize_text(f"Falha Ollama HTTP {exc.code}: {raw[:600]}")
    except Exception as exc:
        return _sanitize_text(f"Falha Ollama: {exc}")


_GEMMA_LORA_RUNTIME: dict[str, Any] = {}


def _load_gemma_lora_runtime(base_model: str, adapter_path: str) -> dict[str, Any]:
    cache_key = f"{base_model}::{adapter_path}"
    existing = _GEMMA_LORA_RUNTIME.get(cache_key)
    if isinstance(existing, dict):
        return existing
    if AutoTokenizer is None or AutoModelForCausalLM is None or PeftModel is None:
        raise RuntimeError("Dependências de inferência local ausentes (transformers/peft/torch).")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if (torch is not None and torch.cuda.is_available()) else torch.float32,  # type: ignore[arg-type]
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    runtime = {"tokenizer": tok, "model": model}
    _GEMMA_LORA_RUNTIME[cache_key] = runtime
    return runtime


def _call_gemma_lora(base_model: str, adapter_path: str, user_text: str, context: str) -> str:
    runtime = _load_gemma_lora_runtime(base_model, adapter_path)
    tok = runtime["tokenizer"]
    model = runtime["model"]
    max_new_tokens = int(os.getenv("GEMMA_LORA_MAX_NEW_TOKENS", "120") or 120)
    temperature = float(os.getenv("GEMMA_LORA_TEMPERATURE", "0.2") or 0.2)

    prompt = (
        "<start_of_turn>user\n"
        "Você é assistente técnico do projeto de glicose-insulina. "
        "Responda em português, objetivo e prático. "
        "Quando falar de dose, trate como simulação/apoio analítico e não orientação clínica.\n\n"
        f"Contexto:\n{context}\n\nPergunta:\n{user_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    inputs = tok(prompt, return_tensors="pt")
    if torch is not None:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  # type: ignore[union-attr]
        out = model.generate(
            **inputs,
            max_new_tokens=max(32, min(max_new_tokens, 512)),
            do_sample=temperature > 0.0,
            temperature=max(0.0, min(temperature, 1.0)),
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    return _sanitize_text(text or "Sem conteúdo retornado pelo Gemma LoRA.")


def _call_llm(
    provider: str,
    deepseek_key: str,
    deepseek_model: str,
    ollama_model: str,
    user_text: str,
    context: str,
) -> str:
    mode = (provider or "auto").strip().lower()
    if mode == "deepseek":
        if not deepseek_key:
            return "DEEPSEEK_API_KEY ausente. Defina a variável ou use LLM_PROVIDER=ollama."
        return _call_deepseek(deepseek_key, deepseek_model, user_text, context)
    if mode == "ollama":
        return _call_ollama(ollama_model, user_text, context)
    if mode == "gemma_lora":
        base_model = _env("GEMMA_LORA_BASE_MODEL", "google/gemma-2-2b-it")
        adapter_path = _env("GEMMA_LORA_ADAPTER_PATH", "outputs/gemma_lora")
        return _call_gemma_lora(base_model, adapter_path, user_text, context)

    # auto: tenta DeepSeek primeiro, fallback para Ollama.
    if deepseek_key:
        reply = _call_deepseek(deepseek_key, deepseek_model, user_text, context)
        if not reply.lower().startswith("falha deepseek"):
            return reply
    return _call_ollama(ollama_model, user_text, context)


def _dose_formula(glucose: float, cho_g: float, sensitivity_u_per_g: float, pk_active_u: float) -> dict[str, Any]:
    result = compute_dose(
        glucose_mg_dl=float(glucose),
        cho_g=float(cho_g),
        sensitivity_u_per_g=float(sensitivity_u_per_g),
        pk_active_u=float(pk_active_u),
        limits=_dose_limits_from_env(),
    ).to_dict()
    return {
        "cho_u": float(result["cho_u"]),
        "correction_u": float(result["correction_u"]),
        "raw_u": float(result["raw_u"]),
        "pk_active_u": float(result["pk_active_u"]),
        "pk_att": float(result["pk_att"]),
        "final_u": float(result["final_u"]),
        "blocked": bool(result.get("blocked", False)),
        "reasons": list(result.get("reasons") or []),
    }


def _compute_pk_active_from_state(state: dict[str, Any]) -> float:
    insulin = state.get("insulin") or []
    if not insulin:
        return 0.0
    now_raw = state.get("latest_timestamp")
    try:
        now = datetime.fromisoformat(now_raw) if now_raw else datetime.now()
    except ValueError:
        now = datetime.now()
    total = 0.0
    for ev in insulin:
        ts_raw = ev.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        elapsed = max((now - ts).total_seconds() / 60.0, 0.0)
        units = float(ev.get("units", 0.0) or 0.0)
        kind = str(ev.get("insulin_type", "rapida")).lower()
        tau = 360.0 if kind == "basal" else 75.0
        total += units * np_exp(-elapsed / tau)
    return max(total, 0.0)


def np_exp(x: float) -> float:
    # Evita dependência de numpy só para exp.
    return pow(2.718281828459045, x)


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
    # Garante arquivos mínimos para evitar falha quando só CHO ou só insulina foi registrado.
    _ensure_csv(meals_csv, ["timestamp", "carboidratos (g)"])
    _ensure_csv(insulin_csv, ["timestamp", "unidades", "tipo"])
    env = os.environ.copy()
    env["MEALS_CSV_URL"] = str(meals_csv.resolve())
    env["INSULIN_CSV_URL"] = str(insulin_csv.resolve())

    def _run(skip_extract: bool) -> tuple[bool, str]:
        cmd = [sys.executable, "main.py", "once"]
        if skip_extract:
            cmd.append("--skip-extract")
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
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    force_skip_extract = os.getenv("BOT_FORCE_SKIP_EXTRACT", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_skip_extract:
        ok, detail = _run(skip_extract=True)
        if ok:
            return True, "Pipeline atualizado (skip extract forçado)."
        return False, detail

    ok, detail = _run(skip_extract=False)
    if ok:
        return True, "Pipeline atualizado."

    ok_fallback, detail_fallback = _run(skip_extract=True)
    if ok_fallback:
        return True, "Pipeline atualizado sem extração nova (fallback)."
    return False, f"{detail[:320]} | fallback --skip-extract: {detail_fallback[:320]}"


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


def _build_records_reply(meals_csv: Path, insulin_csv: Path) -> str:
    def tail_lines(path: Path, max_lines: int = 3) -> list[str]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= 1:
            return []
        return lines[-max_lines:]

    meal_lines = tail_lines(meals_csv, 3)
    insulin_lines = tail_lines(insulin_csv, 3)
    meal_count = max((len(path_lines := meal_lines)), 0)
    insulin_count = max((len(path2_lines := insulin_lines)), 0)
    # contagens totais reais (sem header)
    total_meals = 0
    total_insulin = 0
    if meals_csv.exists():
        total_meals = max(len(meals_csv.read_text(encoding="utf-8").splitlines()) - 1, 0)
    if insulin_csv.exists():
        total_insulin = max(len(insulin_csv.read_text(encoding="utf-8").splitlines()) - 1, 0)
    meal_preview = "\n".join(f"- {line}" for line in meal_lines) if meal_lines else "- sem registros"
    insulin_preview = "\n".join(f"- {line}" for line in insulin_lines) if insulin_lines else "- sem registros"
    return (
        "Registros manuais locais:\n"
        f"- CHO total: {total_meals}\n"
        f"- Insulina total: {total_insulin}\n"
        "Últimos CHO:\n"
        f"{meal_preview}\n"
        "Últimas insulinas:\n"
        f"{insulin_preview}"
    )


def _plot_pipeline_png(config) -> Path:
    path = (config.base_dir / "outputs/plots/latest_comparison.png").resolve()
    if not path.exists():
        raise FileNotFoundError("Plot principal não encontrado em outputs/plots/latest_comparison.png")
    return path


def _plot_delta_png(config) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib indisponível para gerar gráfico de delta")
    series = _read_json(config.series_path)
    history = series.get("forecast_history") or []
    points = []
    for item in history[-120:]:
        if not item.get("is_resolved"):
            continue
        ts = item.get("generated_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        points.append((dt, float(item.get("delta_mg_dl", 0.0) or 0.0)))
    if not points:
        raise RuntimeError("Sem pontos de erro resolvido para plotar.")
    out_dir = (config.base_dir / "outputs/telegram_plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecast_delta_recent.png"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#be123c" if y >= 0 else "#0f766e" for y in ys]
    ax.bar(xs, ys, color=colors, width=0.01)
    ax.axhline(0.0, color="#1f2933", linewidth=1.0, linestyle="--")
    ax.set_title("Erro recente: projeção - observado")
    ax.set_ylabel("mg/dL")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_ts_png(config) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib indisponível para gerar gráfico TS")
    series = _read_json(config.series_path)
    rollout = series.get("takagi_sugeno_ml_rollout") or {}
    start_ts = rollout.get("start_timestamp")
    grid = rollout.get("grid_minutes") or []
    observed = rollout.get("observed_glucose") or []
    ts_curve = rollout.get("takagi_sugeno_glucose") or []
    ts_ml_curve = rollout.get("takagi_sugeno_ml_glucose") or []
    if not (start_ts and grid and observed and ts_curve):
        raise RuntimeError("Rollout TS dinâmico indisponível nos dados atuais.")
    try:
        start = datetime.fromisoformat(start_ts)
    except ValueError as exc:
        raise RuntimeError("Timestamp inválido no rollout TS.") from exc
    xs = [start.timestamp() + float(m) * 60.0 for m in grid]
    xdt = [datetime.fromtimestamp(v) for v in xs]
    out_dir = (config.base_dir / "outputs/telegram_plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ts_dynamic_recent.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xdt, observed, label="Observado", color="#b45309", linewidth=2.4)
    ax.plot(xdt, ts_curve, label="TS dinâmico", color="#7c3aed", linewidth=2.2, linestyle="--")
    if ts_ml_curve and len(ts_ml_curve) == len(xdt):
        ax.plot(xdt, ts_ml_curve, label="TS+ML", color="#dc2626", linewidth=2.0, linestyle=":")
    ax.set_title("Reconstrução Takagi-Sugeno dinâmica")
    ax.set_ylabel("mg/dL")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _parse_command(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    low = raw.lower()
    if low in {"ajuda", "/help", "help"}:
        return {"type": "help"}
    if low in {"metricas", "métricas", "status"}:
        return {"type": "metrics"}
    if low in {"forecast", "previsao", "previsão"}:
        return {"type": "forecast"}
    if low in {"registros", "registro", "logs", "historico", "histórico"}:
        return {"type": "records"}
    if low in {"atualizar", "update"}:
        return {"type": "update"}
    if low in {"grafico", "gráfico", "plot", "grafico pipeline", "gráfico pipeline", "plot pipeline"}:
        return {"type": "plot_pipeline"}
    if low in {"grafico erro", "gráfico erro", "plot erro", "grafico delta", "gráfico delta", "plot delta"}:
        return {"type": "plot_delta"}
    if low in {"grafico ts", "gráfico ts", "plot ts", "grafico reconstrucao", "gráfico reconstrução"}:
        return {"type": "plot_ts"}

    m_food_lookup = re.search(r"^(?:buscar\s+)?alimento\s+(.+)$", raw, flags=re.IGNORECASE)
    if m_food_lookup:
        return {"type": "food_lookup", "food_query": m_food_lookup.group(1).strip()}

    m_add_food = re.search(r"^(?:registrar|add)\s+alimento\s+(.+)$", raw, flags=re.IGNORECASE)
    if m_add_food:
        return {"type": "add_food", "food_query": m_add_food.group(1).strip()}

    m_dose_food_glucose = re.search(
        r"^dose\s+alimento\s+(.+?)\s+glicemia\s*=?\s*([0-9]+(?:[.,][0-9]+)?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_dose_food_glucose:
        return {
            "type": "dose_food",
            "food_query": m_dose_food_glucose.group(1).strip(),
            "glucose": float(m_dose_food_glucose.group(2).replace(",", ".")),
            "sensitivity": float(m_dose_food_glucose.group(3).replace(",", ".")) if m_dose_food_glucose.group(3) else None,
        }
    m_dose_food_auto = re.search(
        r"^dose\s+alimento\s+(.+?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_dose_food_auto:
        return {
            "type": "dose_food_auto_glucose",
            "food_query": m_dose_food_auto.group(1).strip(),
            "sensitivity": float(m_dose_food_auto.group(2).replace(",", ".")) if m_dose_food_auto.group(2) else None,
        }

    m_dose = re.search(
        r"dose\s+cho\s*=?\s*([0-9]+(?:[.,][0-9]+)?)\s+glicemia\s*=?\s*([0-9]+(?:[.,][0-9]+)?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?",
        low,
    )
    if m_dose:
        return {
            "type": "dose",
            "cho": float(m_dose.group(1).replace(",", ".")),
            "glucose": float(m_dose.group(2).replace(",", ".")),
            "sensitivity": float(m_dose.group(3).replace(",", ".")) if m_dose.group(3) else None,
        }
    m_dose_auto = re.search(
        r"dose\s+cho\s*=?\s*([0-9]+(?:[.,][0-9]+)?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?",
        low,
    )
    if m_dose_auto:
        return {
            "type": "dose_auto_glucose",
            "cho": float(m_dose_auto.group(1).replace(",", ".")),
            "sensitivity": float(m_dose_auto.group(2).replace(",", ".")) if m_dose_auto.group(2) else None,
        }
    m_sim_dose = re.search(
        r"simular\s+dose\s+cho\s*=?\s*([0-9]+(?:[.,][0-9]+)?)\s+glicemia\s*=?\s*([0-9]+(?:[.,][0-9]+)?)(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?",
        low,
    )
    if m_sim_dose:
        return {
            "type": "simulate_dose",
            "cho": float(m_sim_dose.group(1).replace(",", ".")),
            "glucose": float(m_sim_dose.group(2).replace(",", ".")),
            "sensitivity": float(m_sim_dose.group(3).replace(",", ".")) if m_sim_dose.group(3) else None,
        }
    m_sim_food = re.search(
        r"^simular\s+dose\s+alimento\s+(.+?)(?:\s+glicemia\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?(?:\s+sens\s*=?\s*([0-9]+(?:[.,][0-9]+)?))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_sim_food:
        return {
            "type": "simulate_food_dose",
            "food_query": m_sim_food.group(1).strip(),
            "glucose": float(m_sim_food.group(2).replace(",", ".")) if m_sim_food.group(2) else None,
            "sensitivity": float(m_sim_food.group(3).replace(",", ".")) if m_sim_food.group(3) else None,
        }

    m_cho = re.search(r"(?:cho|carbo)\s+([0-9]+(?:[.,][0-9]+)?)", low)
    if m_cho and ("registr" in low or "add" in low or low.startswith("cho ")):
        return {"type": "add_cho", "cho": float(m_cho.group(1).replace(",", "."))}

    m_ins = re.search(r"(?:insulina|insulin)\s+([0-9]+(?:[.,][0-9]+)?)(?:\s+(rapida|rápida|basal|regular))?", low)
    if m_ins and ("registr" in low or "add" in low or low.startswith("insulina ")):
        kind = m_ins.group(2) or "rapida"
        kind = "rapida" if kind in {"rapida", "rápida"} else kind
        return {"type": "add_insulin", "units": float(m_ins.group(1).replace(",", ".")), "insulin_type": kind}

    return {"type": "llm", "text": raw}


def _help_text() -> str:
    return (
        "Comandos:\n"
        "- métricas\n"
        "- forecast\n"
        "- registros (últimos CHO/insulina salvos)\n"
        "- dose cho 40 glicemia 180 [sens 0.10]\n"
        "- dose cho 10 [sens 0.10] (usa glicose atual)\n"
        "- alimento banana\n"
        "- registrar alimento banana\n"
        "- registrar alimento arroz 120g\n"
        "- dose alimento banana glicemia 180 [sens 0.10]\n"
        "- dose alimento arroz 120g [sens 0.10]\n"
        "- simular dose cho 40 glicemia 180 [sens 0.10]\n"
        "- simular dose alimento arroz 120g [glicemia 180] [sens 0.10]\n"
        "- registrar cho 30\n"
        "- registrar insulina 2.5 rapida\n"
        "- atualizar\n"
        "- gráfico (pipeline)\n"
        "- gráfico erro (delta projeção-observado)\n"
        "- gráfico ts (reconstrução TS dinâmica)\n"
        "- ou pergunte livremente para o Gemma/DeepSeek."
    )


def main() -> int:
    args = build_parser().parse_args()
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Defina TELEGRAM_BOT_TOKEN")
    deepseek_key = _env("DEEPSEEK_API_KEY")
    deepseek_model = _env("DEEPSEEK_MODEL", "deepseek-chat")
    llm_provider = _env("LLM_PROVIDER", "auto")
    ollama_model = _env("OLLAMA_MODEL", "llama3.1:8b")

    config = load_config()
    offset_path = (config.base_dir / args.offset_path).resolve()
    meals_csv = (config.base_dir / args.manual_meals_csv).resolve()
    insulin_csv = (config.base_dir / args.manual_insulin_csv).resolve()
    food_table = _load_food_table(config.base_dir, args.food_table_path)

    offset_state = _read_json(offset_path)
    next_offset = offset_state.get("next_offset")
    if not isinstance(next_offset, int):
        next_offset = None

    allowed_chat = str(args.allowed_chat_id).strip() if args.allowed_chat_id else ""
    if food_table.get("status") == "ok":
        print(f"Tabela de alimentos ativa: {food_table.get('path')}")
    else:
        print(f"Tabela de alimentos indisponível: {food_table.get('message')}")
    print("Telegram AI bot em execução (polling).")
    while True:
        try:
            updates = _get_updates(token, next_offset, timeout_sec=int(args.poll_timeout_sec))
            for upd in updates:
                update_id = int(upd.get("update_id", 0))
                next_offset = update_id + 1
                _write_json(offset_path, {"next_offset": next_offset, "updated_at": datetime.now().isoformat()})

                msg = upd.get("message") or upd.get("edited_message") or {}
                text = (((msg.get("text") or "")).strip())
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", "")).strip()
                from_user = msg.get("from") or {}
                is_bot = bool(from_user.get("is_bot"))
                if is_bot or not text or not chat_id:
                    continue
                if allowed_chat and chat_id != allowed_chat:
                    _send_telegram_message(token, chat_id, "Chat não autorizado para este bot.", args.dry_run)
                    continue

                cmd = _parse_command(text)
                try:
                    if cmd["type"] == "help":
                        reply = _help_text()
                    elif cmd["type"] == "metrics":
                        reply = _build_metrics_reply(config)
                    elif cmd["type"] == "forecast":
                        reply = _build_forecast_reply(config)
                    elif cmd["type"] == "records":
                        reply = _build_records_reply(meals_csv, insulin_csv)
                    elif cmd["type"] == "food_lookup":
                        if food_table.get("status") != "ok":
                            reply = (
                                f"Tabela de alimentos indisponível: {food_table.get('message')}.\n"
                                "Defina FOOD_GI_TABLE_PATH no .env para CSV/JSON/PDF com alimentos e IG."
                            )
                        else:
                            food_query, grams = _extract_amount_g(str(cmd["food_query"]))
                            entry = _lookup_food(food_query, food_table)
                            if entry is None:
                                reply = f"Alimento não encontrado na tabela: {food_query}"
                            else:
                                cho_est = _estimate_carbs_from_entry(entry, grams)
                                gi = entry.get("gi")
                                cg = entry.get("glycemic_load")
                                gi_factor = _gi_impact_factor(gi)
                                qty_msg = f"{_fmt(grams,0)} g" if grams is not None else "porção padrão"
                                reply = (
                                    "Lookup alimento:\n"
                                    f"- Nome: {entry.get('food_name')}\n"
                                    f"- Quantidade: {qty_msg}\n"
                                    f"- IG: {_fmt(gi,0)}\n"
                                    f"- CG: {_fmt(cg,1)}\n"
                                    f"- CHO estimado: {_fmt(cho_est,1)} g\n"
                                    f"- Fator IG dose: {_fmt(gi_factor,2)}x"
                                )
                    elif cmd["type"] == "add_food":
                        if food_table.get("status") != "ok":
                            reply = (
                                f"Tabela de alimentos indisponível: {food_table.get('message')}.\n"
                                "Defina FOOD_GI_TABLE_PATH no .env para CSV/JSON/PDF com alimentos e IG."
                            )
                        else:
                            food_query, grams = _extract_amount_g(str(cmd["food_query"]))
                            entry = _lookup_food(food_query, food_table)
                            if entry is None:
                                reply = f"Alimento não encontrado na tabela: {food_query}"
                            else:
                                cho_est = _estimate_carbs_from_entry(entry, grams)
                                if cho_est is None:
                                    reply = (
                                        f"Encontrei {entry.get('food_name')}, mas sem CHO na tabela para registrar.\n"
                                        "Use registrar cho <gramas> ou adicione colunas de CHO na tabela."
                                    )
                                else:
                                    now_iso = datetime.now().replace(microsecond=0).isoformat()
                                    _append_manual_meal(meals_csv, now_iso, float(cho_est))
                                    ok, detail = _run_pipeline_with_manual_inputs(config.base_dir, meals_csv, insulin_csv)
                                    gi = entry.get("gi")
                                    reply = (
                                        f"Alimento registrado: {entry.get('food_name')} | CHO {_fmt(cho_est,1)} g | IG {_fmt(gi,0)}.\n"
                                        f"{'OK' if ok else 'Falha'}: {detail}"
                                    )
                    elif cmd["type"] == "dose":
                        state = _read_json(config.state_path)
                        if cmd.get("sensitivity") is not None:
                            sensitivity = float(cmd["sensitivity"])
                        else:
                            scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                            adaptive_isf = 45.0 * scale
                            sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                        dose = _dose_formula(
                            glucose=float(cmd["glucose"]),
                            cho_g=float(cmd["cho"]),
                            sensitivity_u_per_g=sensitivity,
                            pk_active_u=_compute_pk_active_from_state(state),
                        )
                        reply = (
                            "Dose (simulação, não clínica):\n"
                            f"- Sensibilidade: {_fmt(sensitivity,3)} U/g\n"
                            f"- CHO dose: {_fmt(dose['cho_u'],2)} U\n"
                            f"- Correção: {_fmt(dose['correction_u'],2)} U\n"
                            f"- Bruta: {_fmt(dose['raw_u'],2)} U\n"
                            f"- PK ativo: {_fmt(dose['pk_active_u'],2)} U (aten. {_fmt(dose['pk_att']*100,0)}%)\n"
                            f"- Final limitada: {_fmt(dose['final_u'],2)} U\n"
                            f"- Bloqueio segurança: {'sim' if dose.get('blocked') else 'não'}\n"
                            f"- Guardrails: {', '.join(dose.get('reasons') or ['none'])}"
                        )
                    elif cmd["type"] == "dose_food":
                        if food_table.get("status") != "ok":
                            reply = (
                                f"Tabela de alimentos indisponível: {food_table.get('message')}.\n"
                                "Defina FOOD_GI_TABLE_PATH no .env para CSV/JSON/PDF com alimentos e IG."
                            )
                        else:
                            state = _read_json(config.state_path)
                            food_query, grams = _extract_amount_g(str(cmd["food_query"]))
                            entry = _lookup_food(food_query, food_table)
                            if entry is None:
                                reply = f"Alimento não encontrado na tabela: {food_query}"
                            else:
                                cho_est = _estimate_carbs_from_entry(entry, grams)
                                if cho_est is None:
                                    reply = (
                                        f"Encontrei {entry.get('food_name')}, mas sem CHO para calcular dose.\n"
                                        "Use: dose cho <gramas> glicemia <valor>."
                                    )
                                else:
                                    if cmd.get("sensitivity") is not None:
                                        sensitivity = float(cmd["sensitivity"])
                                    else:
                                        scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                                        adaptive_isf = 45.0 * scale
                                        sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                                    gi = entry.get("gi")
                                    gi_factor = _gi_impact_factor(gi)
                                    cho_effective = float(cho_est) * gi_factor
                                    dose = _dose_formula(
                                        glucose=float(cmd["glucose"]),
                                        cho_g=cho_effective,
                                        sensitivity_u_per_g=sensitivity,
                                        pk_active_u=_compute_pk_active_from_state(state),
                                    )
                                    reply = (
                                        "Dose por alimento (simulação, não clínica):\n"
                                        f"- Alimento: {entry.get('food_name')}\n"
                                        f"- CHO base: {_fmt(cho_est,1)} g\n"
                                        f"- IG: {_fmt(gi,0)} | fator IG: {_fmt(gi_factor,2)}x\n"
                                        f"- CHO efetivo: {_fmt(cho_effective,1)} g\n"
                                        f"- Sensibilidade: {_fmt(sensitivity,3)} U/g\n"
                                        f"- CHO dose: {_fmt(dose['cho_u'],2)} U\n"
                                        f"- Correção: {_fmt(dose['correction_u'],2)} U\n"
                                        f"- Bruta: {_fmt(dose['raw_u'],2)} U\n"
                                        f"- PK ativo: {_fmt(dose['pk_active_u'],2)} U (aten. {_fmt(dose['pk_att']*100,0)}%)\n"
                                        f"- Final limitada: {_fmt(dose['final_u'],2)} U\n"
                                        f"- Bloqueio segurança: {'sim' if dose.get('blocked') else 'não'}\n"
                                        f"- Guardrails: {', '.join(dose.get('reasons') or ['none'])}"
                                    )
                    elif cmd["type"] == "dose_food_auto_glucose":
                        if food_table.get("status") != "ok":
                            reply = (
                                f"Tabela de alimentos indisponível: {food_table.get('message')}.\n"
                                "Defina FOOD_GI_TABLE_PATH no .env para CSV/JSON/PDF com alimentos e IG."
                            )
                        else:
                            state = _read_json(config.state_path)
                            food_query, grams = _extract_amount_g(str(cmd["food_query"]))
                            entry = _lookup_food(food_query, food_table)
                            if entry is None:
                                reply = f"Alimento não encontrado na tabela: {food_query}"
                            else:
                                cho_est = _estimate_carbs_from_entry(entry, grams)
                                if cho_est is None:
                                    reply = (
                                        f"Encontrei {entry.get('food_name')}, mas sem CHO para calcular dose.\n"
                                        "Use: dose cho <gramas>."
                                    )
                                else:
                                    current_glucose = float(state.get("latest_glucose_mg_dl", 140.0) or 140.0)
                                    if cmd.get("sensitivity") is not None:
                                        sensitivity = float(cmd["sensitivity"])
                                    else:
                                        scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                                        adaptive_isf = 45.0 * scale
                                        sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                                    gi = entry.get("gi")
                                    gi_factor = _gi_impact_factor(gi)
                                    cho_effective = float(cho_est) * gi_factor
                                    dose = _dose_formula(
                                        glucose=current_glucose,
                                        cho_g=cho_effective,
                                        sensitivity_u_per_g=sensitivity,
                                        pk_active_u=_compute_pk_active_from_state(state),
                                    )
                                    reply = (
                                        "Dose por alimento (simulação, não clínica):\n"
                                        f"- Alimento: {entry.get('food_name')}\n"
                                        f"- Glicose atual: {_fmt(current_glucose,0)} mg/dL\n"
                                        f"- CHO base: {_fmt(cho_est,1)} g\n"
                                        f"- IG: {_fmt(gi,0)} | fator IG: {_fmt(gi_factor,2)}x\n"
                                        f"- CHO efetivo: {_fmt(cho_effective,1)} g\n"
                                        f"- Sensibilidade: {_fmt(sensitivity,3)} U/g\n"
                                        f"- CHO dose: {_fmt(dose['cho_u'],2)} U\n"
                                        f"- Correção: {_fmt(dose['correction_u'],2)} U\n"
                                        f"- Bruta: {_fmt(dose['raw_u'],2)} U\n"
                                        f"- PK ativo: {_fmt(dose['pk_active_u'],2)} U (aten. {_fmt(dose['pk_att']*100,0)}%)\n"
                                        f"- Final limitada: {_fmt(dose['final_u'],2)} U\n"
                                        f"- Bloqueio segurança: {'sim' if dose.get('blocked') else 'não'}\n"
                                        f"- Guardrails: {', '.join(dose.get('reasons') or ['none'])}"
                                    )
                    elif cmd["type"] == "dose_auto_glucose":
                        state = _read_json(config.state_path)
                        current_glucose = float(state.get("latest_glucose_mg_dl", 140.0) or 140.0)
                        if cmd.get("sensitivity") is not None:
                            sensitivity = float(cmd["sensitivity"])
                        else:
                            scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                            adaptive_isf = 45.0 * scale
                            sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                        dose = _dose_formula(
                            glucose=current_glucose,
                            cho_g=float(cmd["cho"]),
                            sensitivity_u_per_g=sensitivity,
                            pk_active_u=_compute_pk_active_from_state(state),
                        )
                        reply = (
                            "Dose (simulação, não clínica):\n"
                            f"- Glicose atual: {_fmt(current_glucose,0)} mg/dL\n"
                            f"- Sensibilidade: {_fmt(sensitivity,3)} U/g\n"
                            f"- CHO dose: {_fmt(dose['cho_u'],2)} U\n"
                            f"- Correção: {_fmt(dose['correction_u'],2)} U\n"
                            f"- Bruta: {_fmt(dose['raw_u'],2)} U\n"
                            f"- PK ativo: {_fmt(dose['pk_active_u'],2)} U (aten. {_fmt(dose['pk_att']*100,0)}%)\n"
                            f"- Final limitada: {_fmt(dose['final_u'],2)} U\n"
                            f"- Bloqueio segurança: {'sim' if dose.get('blocked') else 'não'}\n"
                            f"- Guardrails: {', '.join(dose.get('reasons') or ['none'])}"
                        )
                    elif cmd["type"] == "simulate_dose":
                        state = _read_json(config.state_path)
                        if cmd.get("sensitivity") is not None:
                            sensitivity = float(cmd["sensitivity"])
                        else:
                            scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                            adaptive_isf = 45.0 * scale
                            sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                        pk_active_u = _compute_pk_active_from_state(state)
                        sims = simulate_scenarios(
                            glucose_mg_dl=float(cmd["glucose"]),
                            cho_effective_g=float(cmd["cho"]),
                            sensitivity_u_per_g=sensitivity,
                            pk_active_u=pk_active_u,
                            limits=_dose_limits_from_env(),
                        )
                        sim_context = (
                            _dashboard_context(config)
                            + "\n\nSIMULACAO_DETERMINISTICA_JSON:\n"
                            + json.dumps(sims, ensure_ascii=False)
                        )
                        llm_reply = _call_llm(
                            provider=llm_provider,
                            deepseek_key=deepseek_key,
                            deepseek_model=deepseek_model,
                            ollama_model=ollama_model,
                            user_text=(
                                "Analise os cenários conservative/base/aggressive e responda em português com: "
                                "resumo, riscos, cenário preferido e alertas. Apoio analítico, não clínico."
                            ),
                            context=sim_context,
                        )
                        base = sims.get("base") or {}
                        reply = (
                            "Simulação de dose (U):\n"
                            f"- Base final: {_fmt(base.get('final_u'),2)} U\n"
                            f"- Conservadora: {_fmt((sims.get('conservative') or {}).get('final_u'),2)} U\n"
                            f"- Agressiva: {_fmt((sims.get('aggressive') or {}).get('final_u'),2)} U\n"
                            f"- IOB ativo: {_fmt(pk_active_u,2)} U\n\n"
                            "Análise Gemma:\n"
                            f"{llm_reply}"
                        )
                    elif cmd["type"] == "simulate_food_dose":
                        if food_table.get("status") != "ok":
                            reply = (
                                f"Tabela de alimentos indisponível: {food_table.get('message')}.\n"
                                "Defina FOOD_GI_TABLE_PATH no .env para CSV/JSON/PDF com alimentos e IG."
                            )
                        else:
                            state = _read_json(config.state_path)
                            food_query, grams = _extract_amount_g(str(cmd["food_query"]))
                            entry = _lookup_food(food_query, food_table)
                            if entry is None:
                                reply = f"Alimento não encontrado na tabela: {food_query}"
                            else:
                                cho_est = _estimate_carbs_from_entry(entry, grams)
                                if cho_est is None:
                                    reply = f"Encontrei {entry.get('food_name')}, mas sem CHO para simular dose."
                                else:
                                    current_glucose = float(
                                        cmd["glucose"] if cmd.get("glucose") is not None else state.get("latest_glucose_mg_dl", 140.0)
                                    )
                                    if cmd.get("sensitivity") is not None:
                                        sensitivity = float(cmd["sensitivity"])
                                    else:
                                        scale = float((state.get("params") or {}).get("insulin_sensitivity_scale", 1.0) or 1.0)
                                        adaptive_isf = 45.0 * scale
                                        sensitivity = max(0.02, min(0.20, 3.4 / max(adaptive_isf, 1e-6)))
                                    gi = entry.get("gi")
                                    cho_effective = float(cho_est) * _gi_impact_factor(gi)
                                    pk_active_u = _compute_pk_active_from_state(state)
                                    sims = simulate_scenarios(
                                        glucose_mg_dl=current_glucose,
                                        cho_effective_g=cho_effective,
                                        sensitivity_u_per_g=sensitivity,
                                        pk_active_u=pk_active_u,
                                        limits=_dose_limits_from_env(),
                                    )
                                    sim_context = (
                                        _dashboard_context(config)
                                        + "\n\nALIMENTO:\n"
                                        + json.dumps(
                                            {
                                                "nome": entry.get("food_name"),
                                                "gramas": grams,
                                                "ig": gi,
                                                "cho_base_g": cho_est,
                                                "cho_efetivo_g": cho_effective,
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\nSIMULACAO_DETERMINISTICA_JSON:\n"
                                        + json.dumps(sims, ensure_ascii=False)
                                    )
                                    llm_reply = _call_llm(
                                        provider=llm_provider,
                                        deepseek_key=deepseek_key,
                                        deepseek_model=deepseek_model,
                                        ollama_model=ollama_model,
                                        user_text=(
                                            "Analise os cenários conservative/base/aggressive e responda em português com: "
                                            "resumo, riscos, cenário preferido e alertas. Apoio analítico, não clínico."
                                        ),
                                        context=sim_context,
                                    )
                                    base = sims.get("base") or {}
                                    reply = (
                                        f"Simulação por alimento: {entry.get('food_name')}\n"
                                        f"- Dose base: {_fmt(base.get('final_u'),2)} U\n"
                                        f"- Conservadora: {_fmt((sims.get('conservative') or {}).get('final_u'),2)} U\n"
                                        f"- Agressiva: {_fmt((sims.get('aggressive') or {}).get('final_u'),2)} U\n"
                                        f"- IOB ativo: {_fmt(pk_active_u,2)} U\n\n"
                                        "Análise Gemma:\n"
                                        f"{llm_reply}"
                                    )
                    elif cmd["type"] == "add_cho":
                        now_iso = datetime.now().replace(microsecond=0).isoformat()
                        _append_manual_meal(meals_csv, now_iso, float(cmd["cho"]))
                        ok, detail = _run_pipeline_with_manual_inputs(config.base_dir, meals_csv, insulin_csv)
                        reply = f"CHO registrado: {_fmt(cmd['cho'],0)} g.\n{'OK' if ok else 'Falha'}: {detail}"
                    elif cmd["type"] == "add_insulin":
                        now_iso = datetime.now().replace(microsecond=0).isoformat()
                        _append_manual_insulin(insulin_csv, now_iso, float(cmd["units"]), str(cmd["insulin_type"]))
                        ok, detail = _run_pipeline_with_manual_inputs(config.base_dir, meals_csv, insulin_csv)
                        reply = f"Insulina registrada: {_fmt(cmd['units'],2)} U ({cmd['insulin_type']}).\n{'OK' if ok else 'Falha'}: {detail}"
                    elif cmd["type"] == "update":
                        ok, detail = _run_pipeline_with_manual_inputs(config.base_dir, meals_csv, insulin_csv)
                        reply = f"{'Atualização concluída.' if ok else 'Falha na atualização.'}\n{detail}"
                    elif cmd["type"] == "plot_pipeline":
                        path = _plot_pipeline_png(config)
                        _send_telegram_photo(token, chat_id, path, "Gráfico principal do pipeline", args.dry_run)
                        reply = "Enviei o gráfico principal."
                    elif cmd["type"] == "plot_delta":
                        path = _plot_delta_png(config)
                        _send_telegram_photo(token, chat_id, path, "Erro recente (projeção - observado)", args.dry_run)
                        reply = "Enviei o gráfico de erro recente."
                    elif cmd["type"] == "plot_ts":
                        path = _plot_ts_png(config)
                        _send_telegram_photo(token, chat_id, path, "Reconstrução Takagi-Sugeno dinâmica", args.dry_run)
                        reply = "Enviei o gráfico TS dinâmico."
                    else:
                        reply = _call_llm(
                            provider=llm_provider,
                            deepseek_key=deepseek_key,
                            deepseek_model=deepseek_model,
                            ollama_model=ollama_model,
                            user_text=cmd["text"],
                            context=_dashboard_context(config),
                        )
                except Exception as cmd_exc:
                    reply = f"Falha ao processar comando: {cmd_exc}"

                _send_telegram_message(token, chat_id, _sanitize_text(reply), args.dry_run)
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            print(f"[{datetime.now().isoformat()}] HTTP error: {exc.code} {raw[:400]}")
        except Exception as exc:
            print(f"[{datetime.now().isoformat()}] erro: {exc}")
        time.sleep(max(float(args.loop_sleep_sec), 0.2))


if __name__ == "__main__":
    raise SystemExit(main())

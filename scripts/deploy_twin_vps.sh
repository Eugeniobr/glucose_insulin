#!/usr/bin/env bash
set -euo pipefail

cd /root/glucose_insulin

echo "[1/5] Pull latest code"
git pull --rebase origin production

echo "[2/5] Build and restart telegram bot"
docker compose up -d --build --force-recreate telegram_bot

echo "[3/5] Check bot env (twin)"
docker compose exec telegram_bot env | grep -E 'TWIN_DATA_CSV|TWIN_ARTIFACTS_DIR|TWIN_SNAPSHOT_PATH|LLM_PROVIDER|OLLAMA_BASE_URL|OLLAMA_MODEL' || true

echo "[4/5] Generate digital twin snapshot"
docker compose exec telegram_bot python3 metabolic_twin/src/run_inference_snapshot.py \
  --csv "${TWIN_DATA_CSV:-EugênioSilva Rezende_glucose_4-19-2026.csv}" \
  --artifacts "${TWIN_ARTIFACTS_DIR:-metabolic_twin/artifacts}" \
  --output "${TWIN_SNAPSHOT_PATH:-outputs/twin_simulation.json}"

echo "[5/5] Tail logs"
docker compose logs --tail=100 telegram_bot

echo "Done. Test in Telegram with: twin / twin atualizar"

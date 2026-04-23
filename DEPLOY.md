# Deploy de Produção

## 1. Preparação
1. Copie `.env.example` para `.env` e preencha os valores:
```bash
cp .env.example .env
```
2. Garanta que `dados_glicose_final.json` exista no diretório raiz.

## 2. Subir serviços
```bash
docker compose up -d --build
```

Serviços:
- `pipeline`: atualiza previsões continuamente (`main.py loop --skip-extract`)
- `dashboard`: web em `http://<host>:8000`
- `telegram_bot`: bot conversacional no Telegram

## 3. Logs
```bash
docker compose logs -f pipeline
docker compose logs -f dashboard
docker compose logs -f telegram_bot
```

## 4. Variantes de LLM
### Ollama (recomendado local/servidor próprio)
No `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### DeepSeek
No `.env`:
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=<sua_chave>
DEEPSEEK_MODEL=deepseek-chat
```

### Auto fallback
No `.env`:
```env
LLM_PROVIDER=auto
```
(Tenta DeepSeek e cai para Ollama em falha.)

## 5. Comandos úteis no bot Telegram
- `métricas`
- `forecast`
- `registros`
- `dose cho 10`
- `dose cho 40 glicemia 180`
- `registrar cho 20`
- `registrar insulina 2.0 rapida`
- `gráfico`
- `gráfico erro`
- `gráfico ts`

## 6. Atualização
```bash
git pull
docker compose up -d --build
```

## 7. Parar
```bash
docker compose down
```

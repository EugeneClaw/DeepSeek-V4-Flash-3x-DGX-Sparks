#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a
PORT="${VLLM_PORT:-8888}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
echo "POST $URL model=$MODEL"
curl -fsS --max-time 180 "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words.\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}"
echo

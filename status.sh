#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
set -a; source "$ROOT/.env"; set +a
PORT="${VLLM_PORT:-8888}"
code=$(curl -sS -o /tmp/dsv4-3n-models.json -w "%{http_code}" --max-time 5 "http://127.0.0.1:${PORT}/v1/models" || echo 000)
echo "api=$code"
if [ "$code" = "200" ]; then
  python3 - "$PORT" <<'PY'
import json
print(json.load(open("/tmp/dsv4-3n-models.json")))
PY
fi
docker compose -p "${PROJECT:-dsv4-0731-3n}" --env-file "$ROOT/.env.3n" -f "$ROOT/docker-compose.yml" ps || true

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
set -a; source "$ROOT/.env"; set +a
PORT="${VLLM_PORT:-8888}"
# DSPARK_API_KEYS auth (begin). Same rules as the compose entrypoint so a
# probe never disagrees with the server about which variable it honoured.
AUTH_HEADER_ARGS=()
case "${DSPARK_API_KEYS:-}" in
  *[$'\r\n\v\f']*)
    echo "error: DSPARK_API_KEYS must be a single-line space-separated list" >&2
    exit 2
    ;;
  *\\*)
    echo "error: DSPARK_API_KEYS must not contain backslashes" >&2
    exit 2
    ;;
esac
_dspark_keys_set=0
case "${DSPARK_API_KEYS:-}" in
  *[!$' \t']*) _dspark_keys_set=1 ;;
esac
if [ -n "${VLLM_API_KEY:-}" ] && [ "$_dspark_keys_set" = "1" ]; then
  echo "error: VLLM_API_KEY and DSPARK_API_KEYS are both set; set exactly one of them" >&2
  exit 2
fi
if [ "$_dspark_keys_set" = "1" ]; then
  _dspark_keys=()
  # shellcheck disable=SC2206
  read -r -a _dspark_keys <<< "${DSPARK_API_KEYS}"
  for _dspark_key in "${_dspark_keys[@]}"; do
    case "$_dspark_key" in
      -*) echo "error: DSPARK_API_KEYS contains a token beginning with '-'" >&2; exit 2 ;;
    esac
  done
  # Probe with the first parsed key: without this the poll never sees a 200
  # against a keyed server and waits out its full timeout on a healthy cluster.
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${_dspark_keys[0]}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi
# DSPARK_API_KEYS auth (end)

code=$(curl -sS -o /tmp/dsv4-3n-models.json -w "%{http_code}" --max-time 5 "${AUTH_HEADER_ARGS[@]}" "http://127.0.0.1:${PORT}/v1/models" || echo 000)
echo "api=$code"
if [ "$code" = "200" ]; then
  python3 - "$PORT" <<'PY'
import json
print(json.load(open("/tmp/dsv4-3n-models.json")))
PY
fi
docker compose -p "${PROJECT:-dsv4-0731-3n}" --env-file "$ROOT/.env.3n" -f "$ROOT/docker-compose.yml" ps || true

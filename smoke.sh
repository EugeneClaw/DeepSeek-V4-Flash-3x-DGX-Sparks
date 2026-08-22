#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a
PORT="${VLLM_PORT:-8888}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
CONCURRENCY="${CONCURRENCY:-1}"
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

echo "POST $URL model=$MODEL x${CONCURRENCY}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
fail=0
for i in $(seq 1 "$CONCURRENCY"); do
  (
    curl -fsS --max-time 180 "${AUTH_HEADER_ARGS[@]}" "$URL" \
  -H "Content-Type: application/json" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK and the number '"$i"'."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"thinking":false}}' \
      >"$tmpdir/$i.json"
  ) &
done
wait || fail=1
for i in $(seq 1 "$CONCURRENCY"); do
  if [ -s "$tmpdir/$i.json" ]; then
    echo "[$i] $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$tmpdir/$i.json" 2>/dev/null || echo BAD-JSON)"
  else
    echo "[$i] FAILED (no response)"
    fail=1
  fi
done
[ "$fail" = 0 ] || { echo "smoke FAILED" >&2; exit 1; }
echo "smoke OK (x${CONCURRENCY})"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
set -a; source "$ROOT/.env"; set +a
exec docker compose -p "${PROJECT:-dsv4-0731-3n}" --env-file "$ROOT/.env.3n" -f "$ROOT/docker-compose.yml" logs --tail "${1:-80}" -f

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
PROJECT="${PROJECT:-dsv4-0731-3n}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
SSH="ssh $SSH_OPTS -p $SSH_PORT"
ssh_spec() { case "$1" in *@*) printf '%s' "$1" ;; *) printf '%s@%s' "$SSH_USER" "$1" ;; esac; }
NODE0_DIR="${NODE0_DIR:-$ROOT}"
NODE1_DIR="${NODE1_DIR:-$ROOT}"
NODE2_DIR="${NODE2_DIR:-$ROOT}"
down() { $SSH "$1" "cd $2 && docker compose -p $PROJECT --env-file .env.3n -f docker-compose.yml down --remove-orphans" || true; }
down "$(ssh_spec "$NODE2_HOST")" "$NODE2_DIR"
down "$(ssh_spec "$NODE1_HOST")" "$NODE1_DIR"
(cd "$NODE0_DIR" && docker compose -p "$PROJECT" --env-file .env.3n -f docker-compose.yml down --remove-orphans) || true
echo stopped

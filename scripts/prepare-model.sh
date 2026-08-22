#!/usr/bin/env bash
# Download the HF snapshot onto node0, then rsync the hub dir to the workers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
set -a; source "$ROOT/.env"; set +a

if [ "${ABLITERATED:-1}" = "1" ]; then
  MODEL="${DSPARK_MODEL_ABLITERATED}"
  REV="${DSPARK_REVISION_ABLITERATED:-}"
else
  MODEL="${DSPARK_MODEL_OFFICIAL}"
  REV="${DSPARK_REVISION_OFFICIAL:-}"
fi

echo "Downloading $MODEL ${REV:+@ $REV} on node0 (forces online)..."
export HF_HUB_OFFLINE=0
# huggingface_hub defaults both timeouts to 10s — short enough that a slow or
# proxied link kills a multi-GB shard mid-transfer (Mia #97). Ride it out.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
if command -v huggingface-cli >/dev/null 2>&1; then
  if [ -n "$REV" ]; then
    huggingface-cli download "$MODEL" --revision "$REV"
  else
    huggingface-cli download "$MODEL"
  fi
else
  REV_ARG=""
  [ -n "$REV" ] && REV_ARG="--revision $REV"
  python3 -m huggingface_hub.commands.huggingface_cli download "$MODEL" $REV_ARG
fi

HUB="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}"
if [ ! -d "$HUB" ]; then
  echo "Expected hub dir $HUB after download." >&2
  exit 1
fi

for var in NODE1_HOST NODE2_HOST; do
  eval "val=\${$var-}"
  if [ -z "$val" ] || [[ "$val" == *CHANGE_ME* ]]; then
    echo "$var is unset or still CHANGE_ME. Edit .env." >&2
    exit 2
  fi
done

SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
ssh_spec() { case "$1" in *@*) printf '%s' "$1" ;; *) printf '%s@%s' "$SSH_USER" "$1" ;; esac; }

echo "Syncing hub snapshot to workers..."
for host in "$NODE1_HOST" "$NODE2_HOST"; do
  spec=$(ssh_spec "$host")
  remote_home=$(ssh $SSH_OPTS -p "$SSH_PORT" "$spec" 'printf %s "$HOME"')
  ssh $SSH_OPTS -p "$SSH_PORT" "$spec" "mkdir -p '${remote_home}/.cache/huggingface/hub'"
  # rsync does not expand $HOME on the remote; resolve it first.
  rsync -az -e "ssh $SSH_OPTS -p $SSH_PORT" \
    "$HUB/" "$spec:${remote_home}/.cache/huggingface/hub/models--${MODEL//\//--}/"
done
echo "Caches warm. Keep HF_HUB_OFFLINE=1 in .env."

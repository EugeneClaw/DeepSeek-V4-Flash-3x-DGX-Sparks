#!/usr/bin/env bash
# Launch the 3-node EP + mesh + DSpark stack. Run on node0 (head).
# Worker-first, same pattern as the Mia 2x recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
PROJECT="${PROJECT:-dsv4-0731-3n}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE — copy .env.example to .env and fill NODE*_HOST." >&2
  exit 1
fi

# Normalise the operator's env file once into a private 0600 snapshot
# (BOM stripped, CRLF -> LF); the operator's file stays byte-identical.
# Port of Mia #98. One snapshot feeds this script and every .env.3n publish.
_cleanup_dspark_env() { [ -z "${_dspark_env_clean:-}" ] || rm -f -- "$_dspark_env_clean"; }
trap _cleanup_dspark_env EXIT
_dspark_env_clean="$(mktemp)"
chmod 600 "$_dspark_env_clean"
# DSPARK_API_KEYS ambient guard (begin): the key list must come from .env only.
_dspark_ambient_has=0
_dspark_ambient_keys=""
if [ -n "${DSPARK_API_KEYS+x}" ]; then
  _dspark_ambient_has=1
  _dspark_ambient_keys="$DSPARK_API_KEYS"
fi
unset DSPARK_API_KEYS
sed $'1s/^\xEF\xBB\xBF//; s/\r$//' "$ENV_FILE" > "$_dspark_env_clean"
set -a
# shellcheck disable=RC1090
source "$_dspark_env_clean" || exit
set +a
if [ "$_dspark_ambient_has" = "1" ] && [ "$_dspark_ambient_keys" != "${DSPARK_API_KEYS:-}" ]; then
  echo "error: DSPARK_API_KEYS is set in the environment but does not match .env; set it only in .env" >&2
  exit 2
fi
# DSPARK_API_KEYS ambient guard (end)

for var in NODE0_HOST NODE1_HOST NODE2_HOST NODE0_IP NODE1_IP NODE2_IP GLOO_IFACE; do
  eval "val=\${$var-}"
  if [ -z "$val" ] || [[ "$val" == *CHANGE_ME* ]]; then
    echo "$var is unset or still CHANGE_ME. Edit .env." >&2
    exit 2
  fi
done

# DSPARK_API_KEYS / VLLM_API_KEY validation (port of Mia PR #89). Identical
# rules to the compose entrypoint so the launcher fails the same way, before
# any side effect. Probes use the first parsed key.
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
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${_dspark_keys[0]}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi
# Keyed starts require the startup-log redaction hotfix (it is synced to the
# workers with the tree, so checking the head copy is sufficient).
if { [ "$_dspark_keys_set" = "1" ] || [ -n "${VLLM_API_KEY:-}" ]; } && [ ! -f "$ROOT/patches/hotfix-vllm-redact-api-key-log.sh" ]; then
  echo "error: API keys are configured but patches/hotfix-vllm-redact-api-key-log.sh is missing; keyed starts require the startup-log redaction hotfix" >&2
  exit 1
fi

SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
# shellcheck disable=SC2086
SSH="ssh $SSH_OPTS -p $SSH_PORT"
# shellcheck disable=SC2086
RSYNC="rsync -az -e \"ssh $SSH_OPTS -p $SSH_PORT\""

ssh_spec() {
  local host="$1"
  case "$host" in
    *@*) printf '%s' "$host" ;;
    *) printf '%s@%s' "$SSH_USER" "$host" ;;
  esac
}

NODE0_DIR="${NODE0_DIR:-$ROOT}"
NODE1_DIR="${NODE1_DIR:-$ROOT}"
NODE2_DIR="${NODE2_DIR:-$ROOT}"

if [ "${ABLITERATED:-1}" = "1" ]; then
  DSPARK_MODEL="${DSPARK_MODEL_ABLITERATED:-Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX}"
  DSPARK_REVISION="${DSPARK_REVISION_ABLITERATED:-2a7dd6a12d46f752e290c6255140a42600897864}"
else
  DSPARK_MODEL="${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
  DSPARK_REVISION="${DSPARK_REVISION_OFFICIAL:-}"
fi
export DSPARK_MODEL DSPARK_REVISION

MASTER_ADDR="${MASTER_ADDR:-$NODE0_IP}"
EPLB_JSON="${EPLB_CONFIG:-{\"num_redundant_experts\":2}}"
export MASTER_ADDR EPLB_JSON
export NCCL_NET=Mesh
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=mesh
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME="=${GLOO_IFACE}"
export TP_SOCKET_IFNAME="$GLOO_IFACE"
export GLOO_SOCKET_IFNAME="$GLOO_IFACE"

write_env() {
  local rank="$1" repo="$2" ip="$3" hf_home="$4"
  cat <<EOF
DSPARK_VLLM_IMAGE=${DSPARK_VLLM_IMAGE}
DSPARK_MODEL=/cache/huggingface/dsv4-0731-tp3-model
DSPARK_REVISION=
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}
HF_CACHE=${hf_home}/.cache/huggingface
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_XET=1
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8888}
VLLM_HOST_IP=${ip}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-25200}
NODE_RANK=${rank}
HEADLESS=$([ "$rank" = 0 ] && echo || echo 1)
TENSOR_PARALLEL_SIZE=3
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DATA_PARALLEL_RPC_PORT=13345
NNODES=3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
LONG_PREFILL_TOKEN_THRESHOLD=${LONG_PREFILL_TOKEN_THRESHOLD:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.835}
MTP_NUM_TOKENS=${MTP_NUM_TOKENS:-5}
DEFAULT_THINKING=${DEFAULT_THINKING:-off}
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
MOE_BACKEND=flashinfer_cutlass
ENABLE_EP=1
ENABLE_EPLB=1
EPLB_CONFIG=${EPLB_JSON}
ALL2ALL_BACKEND=allgather_reducescatter
NCCL_NET=Mesh
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME==${GLOO_IFACE}
TP_SOCKET_IFNAME=${GLOO_IFACE}
GLOO_SOCKET_IFNAME=${GLOO_IFACE}
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_NET_PLUGIN=mesh
NCCL_ALGO=Ring
NCCL_CUMEM_ENABLE=0
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NCCL_NVLS_ENABLE=0
CUTE_DSL_ARCH=sm_121a
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_CUDA_ARCH_LIST=12.1a
DSPARK_SKIP_HOTFIX=0
DSPARK_SKIP_ISSUE22_HOTFIX=0
DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=0
DSPARK_SUPPRESS_STOPS_IN_REASONING=1
VLLM_DSV4_TP_PAD=heads
VLLM_DSV4_PAD_HEADS=${VLLM_DSV4_PAD_HEADS:-72}
VLLM_DSV4_PAD_GROUPS=${VLLM_DSV4_PAD_GROUPS:-9}
NCCL_MESH_PLUGIN_DIR=${repo}/nccl-mesh
TP_PAD_ROOT=${repo}/overlay
DSPARK_PATCHES_DIR=${repo}/patches
VLLM_API_KEY="${VLLM_API_KEY:-}"
DSPARK_API_KEYS="${DSPARK_API_KEYS:-}"
DSPARK_MAX_INFLIGHT_PREFILLS=${DSPARK_MAX_INFLIGHT_PREFILLS:-2}
DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}
EOF
}

# Publish .env.3n atomically and 0600 everywhere: a failed transfer must
# never leave a truncated or world-readable credentials file behind.
# Port of Mia #98/#5410f88, adapted to three nodes.
publish_env_local() {
  local dest="$1" tmp="${1}.tmp.$$"
  umask 077
  cat > "$tmp"
  chmod 600 "$tmp"
  mv -f -- "$tmp" "$dest"
}
publish_env_ssh() {
  local spec="$1" dest="$2"
  $SSH "$spec" "
    set -eu
    _env_final='$dest'
    _env_tmp=\"\${_env_final}.tmp.\$\$\"
    _cleanup_remote_env() { [ -z \"\$_env_tmp\" ] || rm -f -- \"\$_env_tmp\"; }
    trap _cleanup_remote_env EXIT HUP INT TERM
    umask 077
    cat > \"\$_env_tmp\"
    chmod 600 \"\$_env_tmp\"
    mv -f -- \"\$_env_tmp\" \"\$_env_final\"
    _env_tmp=
    trap - EXIT HUP INT TERM
  "
}

sync_tree() {
  local spec="$1" dest="$2"
  $SSH "$spec" "mkdir -p $dest"
  rsync -az -e "ssh $SSH_OPTS -p $SSH_PORT" \
    --exclude '.git' --exclude '.env' --exclude '.env.3n' --exclude 'nccl-mesh-plugin' \
    "$NODE0_DIR/" "$spec:$dest/"
}

compose_up() {
  local spec="$1" dir="$2"
  $SSH "$spec" "cd $dir && docker compose -p $PROJECT --env-file .env.3n -f docker-compose.yml up -d"
}

if docker ps --format '{{.Names}}' | grep -qx "${PROJECT}-vllm-dspark-1"; then
  echo "DSpark head container already exists for project $PROJECT. Stop it first (./stop.sh) or use PROJECT=..." >&2
  echo "This is not a failed start: dockerd likely restored ranks after a reboot (restart: unless-stopped). The cluster may already be serving. Run ./stop.sh only if you want a cold start. Supervisors: treat exit 3 as already-up (systemd SuccessExitStatus=3)." >&2
  exit 3
fi

echo "== 3x EP mesh MTP=${MTP_NUM_TOKENS:-5} util=${GPU_MEMORY_UTILIZATION:-0.835} model=$DSPARK_MODEL =="

if ! docker image inspect "${DSPARK_VLLM_IMAGE}" >/dev/null 2>&1; then
  echo "Image ${DSPARK_VLLM_IMAGE} missing on node0. Run ./scripts/build-image.sh on every node." >&2
  exit 1
fi
if [ ! -f "$NODE0_DIR/nccl-mesh/libnccl-net-mesh.so" ] && [ ! -f "$NODE0_DIR/nccl-mesh/libnccl-net.so" ]; then
  echo "NCCL mesh plugin missing. Run ./scripts/build-mesh-plugin.sh on node0 (it copies the .so)." >&2
  exit 1
fi

N1=$(ssh_spec "$NODE1_HOST")
N2=$(ssh_spec "$NODE2_HOST")

for spec in "$N1" "$N2"; do
  if ! $SSH "$spec" "docker image inspect ${DSPARK_VLLM_IMAGE} >/dev/null 2>&1"; then
    echo "Image ${DSPARK_VLLM_IMAGE} missing on ${spec}. Run ./scripts/build-image.sh on every node." >&2
    exit 1
  fi
done

SNAP_REL="hub/models--${DSPARK_MODEL//\//--}/snapshots/${DSPARK_REVISION}"
SNAP="${HOME}/.cache/huggingface/${SNAP_REL}"
if [ ! -f "${SNAP}/config.json" ]; then
  echo "Missing snapshot ${SNAP} — run ./scripts/prepare-model.sh first." >&2
  exit 1
fi

echo "== sync repo to workers =="
sync_tree "$N1" "$NODE1_DIR"
sync_tree "$N2" "$NODE2_DIR"

echo "== padded model dir =="
PAD_MODE=heads bash "$NODE0_DIR/scripts/prepare-tp3-model-dir.sh" \
  "${HOME}/.cache/huggingface/${SNAP_REL}" \
  "${HOME}/.cache/huggingface/dsv4-0731-tp3-model"
$SSH "$N1" "PAD_MODE=heads bash $NODE1_DIR/scripts/prepare-tp3-model-dir.sh \$HOME/.cache/huggingface/${SNAP_REL} \$HOME/.cache/huggingface/dsv4-0731-tp3-model"
$SSH "$N2" "PAD_MODE=heads bash $NODE2_DIR/scripts/prepare-tp3-model-dir.sh \$HOME/.cache/huggingface/${SNAP_REL} \$HOME/.cache/huggingface/dsv4-0731-tp3-model"

H1=$($SSH "$N1" 'printf %s "$HOME"')
H2=$($SSH "$N2" 'printf %s "$HOME"')
write_env 0 "$NODE0_DIR" "$NODE0_IP" "$HOME" | publish_env_local "$NODE0_DIR/.env.3n"
write_env 1 "$NODE1_DIR" "$NODE1_IP" "$H1" | publish_env_ssh "$N1" "$NODE1_DIR/.env.3n"
write_env 2 "$NODE2_DIR" "$NODE2_IP" "$H2" | publish_env_ssh "$N2" "$NODE2_DIR/.env.3n"

echo "== start rank2 =="
compose_up "$N2" "$NODE2_DIR"
sleep 5
echo "== start rank1 =="
compose_up "$N1" "$NODE1_DIR"
sleep 5
echo "== start rank0 =="
# env -u: the hub-id DSPARK_MODEL/DSPARK_REVISION exports (used for cache
# prep above) must NOT override .env.3n's container-local model path in
# compose interpolation — shell env wins over --env-file, and the hub id
# would make the head load the raw 64-head config (not divisible by TP=3)
# while the workers load the padded 72-head dir. Workers get a clean env
# via ssh; the head needs this scrub.
(cd "$NODE0_DIR" && env -u DSPARK_MODEL -u DSPARK_REVISION docker compose -p "$PROJECT" --env-file .env.3n -f docker-compose.yml up -d)

echo "== waiting for /v1/models =="
ok=0
for i in $(seq 1 80); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 3 "${AUTH_HEADER_ARGS[@]}" "http://127.0.0.1:${VLLM_PORT:-8888}/v1/models" || echo 000)
  if [ "$code" = "200" ]; then
    echo "READY after ${i} polls"
    ok=1
    break
  fi
  echo "  t=$((i*15))s api=$code"
  sleep 15
done
if [ "$ok" != 1 ]; then
  echo "API did not come up. Check: docker compose -p $PROJECT logs --tail 80" >&2
  exit 1
fi
echo "API: http://${NODE0_IP}:${VLLM_PORT:-8888}/v1"
echo "Next: ./smoke.sh"

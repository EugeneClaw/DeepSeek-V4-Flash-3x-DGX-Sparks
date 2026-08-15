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

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

for var in NODE0_HOST NODE1_HOST NODE2_HOST NODE0_IP NODE1_IP NODE2_IP GLOO_IFACE; do
  eval "val=\${$var-}"
  if [ -z "$val" ] || [[ "$val" == *CHANGE_ME* ]]; then
    echo "$var is unset or still CHANGE_ME. Edit .env." >&2
    exit 2
  fi
done

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
EOF
}

sync_tree() {
  local spec="$1" dest="$2"
  $SSH "$spec" "mkdir -p $dest"
  rsync -az -e "ssh $SSH_OPTS -p $SSH_PORT" \
    --exclude '.git' --exclude '.env' --exclude 'nccl-mesh-plugin' \
    "$NODE0_DIR/" "$spec:$dest/"
}

compose_up() {
  local spec="$1" dir="$2"
  $SSH "$spec" "cd $dir && docker compose -p $PROJECT --env-file .env.3n -f docker-compose.yml up -d"
}

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
write_env 0 "$NODE0_DIR" "$NODE0_IP" "$HOME" > "$NODE0_DIR/.env.3n"
write_env 1 "$NODE1_DIR" "$NODE1_IP" "$H1" | $SSH "$N1" "cat > $NODE1_DIR/.env.3n"
write_env 2 "$NODE2_DIR" "$NODE2_IP" "$H2" | $SSH "$N2" "cat > $NODE2_DIR/.env.3n"

echo "== start rank2 =="
compose_up "$N2" "$NODE2_DIR"
sleep 5
echo "== start rank1 =="
compose_up "$N1" "$NODE1_DIR"
sleep 5
echo "== start rank0 =="
(cd "$NODE0_DIR" && docker compose -p "$PROJECT" --env-file .env.3n -f docker-compose.yml up -d)

echo "== waiting for /v1/models =="
ok=0
for i in $(seq 1 80); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:${VLLM_PORT:-8888}/v1/models" || echo 000)
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

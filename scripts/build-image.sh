#!/usr/bin/env bash
# Build the thin Cutlass-EP image on this node. Repeat on all three Sparks
# (or build once and `docker save | ssh docker load`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a
BASE="${DSPARK_VLLM_BASE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
TAG="${DSPARK_VLLM_IMAGE:-dspark-vllm-gx10:0.1.1-cutlass-ep}"
echo "Pulling $BASE"
docker pull "$BASE"
echo "Building $TAG"
docker build -t "$TAG" -f "$ROOT/image/Dockerfile" "$ROOT/image"
echo "ok $TAG"

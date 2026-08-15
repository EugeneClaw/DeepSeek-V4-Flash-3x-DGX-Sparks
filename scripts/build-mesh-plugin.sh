#!/usr/bin/env bash
# Build autoscriptlabs/nccl-mesh-plugin and install libnccl-net-mesh.so
# into ./nccl-mesh (bind-mounted into the container).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${NCCL_MESH_SRC:-$ROOT/nccl-mesh-plugin}"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/autoscriptlabs/nccl-mesh-plugin.git "$SRC"
fi
make -C "$SRC" -j"$(nproc)"
mkdir -p "$ROOT/nccl-mesh"
cp -a "$SRC/libnccl-net.so" "$ROOT/nccl-mesh/libnccl-net.so"
ln -sfn libnccl-net.so "$ROOT/nccl-mesh/libnccl-net-mesh.so"
echo "Installed $ROOT/nccl-mesh/libnccl-net-mesh.so"
echo "Copy that directory onto every node (start.sh rsyncs the repo, including this)."

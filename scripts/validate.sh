#!/usr/bin/env bash
# CPU-only sanity check. Does not measure tok/s.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for f in start.sh stop.sh status.sh logs.sh smoke.sh \
         scripts/prepare-model.sh scripts/prepare-tp3-model-dir.sh \
         scripts/build-image.sh scripts/build-mesh-plugin.sh; do
  bash -n "$f"
  echo "ok $f"
done
python3 -c "from pathlib import Path; Path('docker-compose.yml').read_text(); print('ok compose')"
echo "validate ok"

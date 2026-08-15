#!/usr/bin/env bash
# Build a local model dir with padded config.json and symlinks to SuperDeepseek shards.
# Does not mutate the HF snapshot (2x restore stays clean).
set -euo pipefail
SNAP="${1:?snapshot dir}"
OUT="${2:?output dir}"
python3 - "$SNAP" "$OUT" <<'PY'
import json, os, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
cfg = json.loads((src / "config.json").read_text())
cfg["_dsv4_tp_pad"] = {
    "recipe": "flash0731-tp3",
    "orig": {
        "num_attention_heads": cfg.get("num_attention_heads"),
        "o_groups": cfg.get("o_groups"),
        "index_n_heads": cfg.get("index_n_heads"),
        "n_routed_experts": cfg.get("n_routed_experts"),
        "moe_intermediate_size": cfg.get("moe_intermediate_size"),
    },
}
heads = int(os.environ.get("VLLM_DSV4_PAD_HEADS", "72"))
groups = int(os.environ.get("VLLM_DSV4_PAD_GROUPS", "9"))
cfg["num_attention_heads"] = heads
cfg["o_groups"] = groups
# indexer stays 64
# PAD_MODE=heads (TP=3+EP): keep stock experts/inter. full: 384/2304 B12X pad.
pad_mode = os.environ.get("PAD_MODE", "full")
if pad_mode == "heads":
    # leave n_routed_experts / moe_intermediate_size at snapshot values
    pass
else:
    cfg["n_routed_experts"] = 384
    cfg["moe_intermediate_size"] = 2304
# Keep logical vocab 129280. Overlay VocabParallelEmbedding uses
# padding_size=lcm(64,3)=192 so the *padded* table is 129408 and
# org_vocab_size stays 129280 (checkpoint shape).
(dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
for p in src.iterdir():
    if p.name == "config.json":
        continue
    link = dst / p.name
    if link.exists() or link.is_symlink():
        link.unlink()
    # Relative to HF_CACHE so the same links work inside the container
    # (HF_CACHE is bind-mounted at /cache/huggingface).
    rel = os.path.relpath(p, dst)
    os.symlink(rel, link)
print(
    f"wrote {dst} heads={heads} groups={groups} "
    f"experts={cfg.get('n_routed_experts')} "
    f"inter={cfg.get('moe_intermediate_size')} mode={pad_mode}"
)
PY

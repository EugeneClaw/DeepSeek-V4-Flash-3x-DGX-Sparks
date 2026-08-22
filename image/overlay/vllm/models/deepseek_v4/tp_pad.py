# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 Flash TP pad helpers for non-divisible world sizes (e.g. TP=3).

Pad recipe (see tp3-pad/notes/DESIGN.md):
  num_attention_heads  64 -> 72
  o_groups              8 -> 9
  index_n_heads        64 -> 72
  n_routed_experts    256 -> 258 via num_redundant_experts=2 (caller)

Enable with env VLLM_DSV4_TP_PAD=1, or auto when heads % tp_size != 0.
"""
from __future__ import annotations

import os
from typing import Any

import torch

# Stock Flash-0731 → TP=3-friendly
# SM120 get_padded_num_q_heads(24) → 32 (kernel buffer), so logical
# 72/3=24 is legal. 72*512/9=4096 matches stock 64*512/8.
# 96/12 remains a fallback if a DeepGEMM Q/O path still rejects 24.
PAD_HEADS_TO = int(os.environ.get("VLLM_DSV4_PAD_HEADS", "72"))
PAD_O_GROUPS_TO = int(os.environ.get("VLLM_DSV4_PAD_GROUPS", "9"))
# Indexer is replicated (not TP-sharded) and DeepGEMM fp8_fp4_mqa_logits
# requires num_heads ∈ {16,32,64}. Leave index_n_heads at stock 64.
PAD_INDEX_HEADS_TO = 64
# 256%3≠0. topkGatingSoftplusSqrt only instantiates specific expert counts
# (see _moe_C_stable_libtorch: 192/256/320/384/448/512/576…). 258 is rejected.
# 384 is the smallest supported count that is both ≥256 and %3==0.
PAD_ROUTED_EXPERTS_TO = 384
# B12X forbids use_ep; non-EP shards intermediate by TP and requires
# intermediate % (tp * 128) == 0 (B12X rounds inter/tp up to 128).
# 2048 -> 2304 = 6*384 (384=3*128).
PAD_MOE_INTERMEDIATE_TO = 2304

_APPLIED_FLAG = "_dsv4_tp_pad_applied"
_ORIG_HEADS = "_dsv4_tp_pad_orig_heads"
_ORIG_GROUPS = "_dsv4_tp_pad_orig_o_groups"
_ORIG_INDEX = "_dsv4_tp_pad_orig_index_n_heads"
_ORIG_EXPERTS = "_dsv4_tp_pad_orig_routed_experts"
_ORIG_MOE_INTER = "_dsv4_tp_pad_orig_moe_intermediate"


def _pad_mode() -> str:
    env = os.environ.get("VLLM_DSV4_TP_PAD", "").strip().lower()
    if env in ("heads", "attn"):
        return "heads"
    if env in ("1", "true", "yes", "on", "full"):
        return "full"
    if env in ("0", "false", "no", "off"):
        return "off"
    return "auto"


def tp_pad_enabled(tp_size: int, n_heads: int) -> bool:
    mode = _pad_mode()
    if mode == "off":
        return False
    if mode in ("full", "heads"):
        return True
    return tp_size > 1 and (n_heads % tp_size != 0)


def pad_experts_enabled() -> bool:
    """False when EP shards experts (no 256→384 tax)."""
    return _pad_mode() != "heads"


def apply_config_tp_pad(config: Any, tp_size: int) -> bool:
    """Mutate hf_config in-place to padded dims. Idempotent. Returns True if padded.

    Accepts either stock Flash-0731 (64/8) or a config.json that was already
    rewritten to 72/9 by ``pad_hf_config_json`` (needed so VllmConfig head-div
    checks pass before model __init__).
    """
    if getattr(config, _APPLIED_FLAG, False):
        return True
    n_heads = int(config.num_attention_heads)
    n_groups = int(getattr(config, "o_groups", 0) or 0)

    n_experts = int(getattr(config, "n_routed_experts", 0) or 0)
    n_inter = int(getattr(config, "moe_intermediate_size", 0) or 0)

    # Already padded on disk (early config.json rewrite).
    if n_heads == PAD_HEADS_TO and n_groups == PAD_O_GROUPS_TO:
        if not tp_pad_enabled(tp_size, 64):
            return False
        setattr(config, _ORIG_HEADS, 64)
        setattr(config, _ORIG_GROUPS, 8)
        setattr(config, _ORIG_INDEX, 64)
        do_experts = pad_experts_enabled()
        setattr(
            config,
            _ORIG_EXPERTS,
            256 if (do_experts and n_experts == PAD_ROUTED_EXPERTS_TO) else n_experts,
        )
        setattr(
            config,
            _ORIG_MOE_INTER,
            2048
            if (do_experts and n_inter == PAD_MOE_INTERMEDIATE_TO)
            else (n_inter or 2048),
        )
        if hasattr(config, "index_n_heads"):
            config.index_n_heads = PAD_INDEX_HEADS_TO
        if do_experts:
            if hasattr(config, "n_routed_experts") and n_experts != PAD_ROUTED_EXPERTS_TO:
                config.n_routed_experts = PAD_ROUTED_EXPERTS_TO
            if hasattr(config, "moe_intermediate_size") and n_inter != PAD_MOE_INTERMEDIATE_TO:
                config.moe_intermediate_size = PAD_MOE_INTERMEDIATE_TO
        setattr(config, _APPLIED_FLAG, True)
        print(
            f"[dsv4-tp-pad] config already padded (heads/groups "
            f"{PAD_HEADS_TO}/{PAD_O_GROUPS_TO}, index_n_heads="
            f"{PAD_INDEX_HEADS_TO}, "
            f"experts {getattr(config, _ORIG_EXPERTS)}->"
            f"{getattr(config, 'n_routed_experts', n_experts)}, "
            f"moe_inter {getattr(config, _ORIG_MOE_INTER)}->"
            f"{getattr(config, 'moe_intermediate_size', n_inter)}, "
            f"mode={_pad_mode()}); "
            f"orig stock for weight pad (tp_size={tp_size})",
            flush=True,
        )
        return True

    if not tp_pad_enabled(tp_size, n_heads):
        return False

    # Only implement the known Flash-0731 recipe for now.
    if n_heads != 64 or n_groups != 8:
        raise RuntimeError(
            f"VLLM_DSV4_TP_PAD: unsupported stock shape heads={n_heads} "
            f"o_groups={getattr(config, 'o_groups', None)}; "
            f"only Flash-0731 (64 heads / 8 groups) pad recipe is implemented."
        )
    if PAD_HEADS_TO % tp_size != 0 or PAD_O_GROUPS_TO % tp_size != 0:
        raise RuntimeError(
            f"VLLM_DSV4_TP_PAD: pad targets heads={PAD_HEADS_TO} groups={PAD_O_GROUPS_TO} "
            f"not divisible by tp_size={tp_size}"
        )
    if (PAD_HEADS_TO // tp_size) % (PAD_O_GROUPS_TO // tp_size) != 0:
        raise RuntimeError("VLLM_DSV4_TP_PAD: local heads not divisible by local groups")
    do_experts = pad_experts_enabled()
    if do_experts and PAD_ROUTED_EXPERTS_TO % tp_size != 0:
        raise RuntimeError(
            f"VLLM_DSV4_TP_PAD: padded experts {PAD_ROUTED_EXPERTS_TO} "
            f"not divisible by tp_size={tp_size}"
        )

    setattr(config, _ORIG_HEADS, n_heads)
    setattr(config, _ORIG_GROUPS, n_groups)
    setattr(config, _ORIG_INDEX, int(getattr(config, "index_n_heads", n_heads)))
    setattr(config, _ORIG_EXPERTS, n_experts if n_experts else 256)
    setattr(config, _ORIG_MOE_INTER, n_inter if n_inter else 2048)

    config.num_attention_heads = PAD_HEADS_TO
    config.o_groups = PAD_O_GROUPS_TO
    if hasattr(config, "index_n_heads"):
        config.index_n_heads = PAD_INDEX_HEADS_TO
    if do_experts:
        if hasattr(config, "n_routed_experts"):
            config.n_routed_experts = PAD_ROUTED_EXPERTS_TO
        if hasattr(config, "moe_intermediate_size"):
            config.moe_intermediate_size = PAD_MOE_INTERMEDIATE_TO

    setattr(config, _APPLIED_FLAG, True)
    print(
        f"[dsv4-tp-pad] applied: heads {n_heads}->{PAD_HEADS_TO}, "
        f"o_groups {getattr(config, _ORIG_GROUPS)}->{PAD_O_GROUPS_TO}, "
        f"index_n_heads {getattr(config, _ORIG_INDEX)}->{PAD_INDEX_HEADS_TO}, "
        f"n_routed_experts {getattr(config, _ORIG_EXPERTS)}->"
        f"{getattr(config, 'n_routed_experts', n_experts)}, "
        f"moe_intermediate {getattr(config, _ORIG_MOE_INTER)}->"
        f"{getattr(config, 'moe_intermediate_size', n_inter)} "
        f"mode={_pad_mode()} (tp_size={tp_size})",
        flush=True,
    )
    return True


def pad_hf_config_json(config_path: str) -> dict:
    """Rewrite HF config.json so early VllmConfig checks see TP-divisible dims.

    Must run after make_vision_model_dir (which regenerates config.json from the
    stock snapshot). Idempotent. Returns the written config dict.
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    cfg = json.loads(path.read_text())
    before = {
        "num_attention_heads": cfg.get("num_attention_heads"),
        "o_groups": cfg.get("o_groups"),
        "index_n_heads": cfg.get("index_n_heads"),
    }
    cfg["num_attention_heads"] = PAD_HEADS_TO
    cfg["o_groups"] = PAD_O_GROUPS_TO
    if "index_n_heads" in cfg:
        cfg["index_n_heads"] = PAD_INDEX_HEADS_TO
    # Stash provenance for humans / restore
    cfg["_dsv4_tp_pad"] = {
        "recipe": "flash0731-tp3",
        "orig": before,
        "padded": {
            "num_attention_heads": PAD_HEADS_TO,
            "o_groups": PAD_O_GROUPS_TO,
            "index_n_heads": PAD_INDEX_HEADS_TO,
        },
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(
        f"[dsv4-tp-pad] wrote {path}: heads {before['num_attention_heads']}->"
        f"{PAD_HEADS_TO}, o_groups {before['o_groups']}->{PAD_O_GROUPS_TO}",
        flush=True,
    )
    return cfg


def orig_heads(config: Any) -> int:
    return int(getattr(config, _ORIG_HEADS, config.num_attention_heads))


def orig_groups(config: Any) -> int:
    return int(getattr(config, _ORIG_GROUPS, config.o_groups))


def orig_routed_experts(config: Any) -> int:
    return int(getattr(config, _ORIG_EXPERTS, getattr(config, "n_routed_experts", 256)))


def orig_moe_intermediate(config: Any) -> int:
    return int(
        getattr(config, _ORIG_MOE_INTER, getattr(config, "moe_intermediate_size", 2048))
    )


def pad_loaded_weight_for_moe(
    name: str,
    loaded_weight: torch.Tensor,
    *,
    padded_experts: int,
    orig_experts: int,
    padded_intermediate: int = PAD_MOE_INTERMEDIATE_TO,
    orig_intermediate: int = 2048,
) -> torch.Tensor:
    """Zero-pad MoE gate tensors and expert intermediate dims for TP pad.

    Expert body weights for pad expert indices (orig_experts..padded-1) are
    missing from the checkpoint and stay at module init (zeros).
    Intermediate pad (2048->2304) is required because B12X rejects EP and
    non-EP MoE shards intermediate by TP (2048%3≠0).
    """
    if not getattr(loaded_weight, "shape", None):
        return loaded_weight

    # ffn.gate.weight: [n_experts, hidden]
    if "ffn.gate.weight" in name or (
        ".gate.weight" in name and "experts" not in name and "wgate" not in name
    ):
        if (
            orig_experts != padded_experts
            and loaded_weight.ndim >= 1
            and loaded_weight.shape[0] == orig_experts
        ):
            return _pad_dim0(loaded_weight, padded_experts)
        return loaded_weight

    # ffn.gate.bias / e_score_correction_bias: [n_experts]
    if (
        "ffn.gate.bias" in name
        or "e_score_correction_bias" in name
        or (name.endswith("gate.bias") and "experts" not in name)
    ):
        if (
            orig_experts != padded_experts
            and loaded_weight.ndim == 1
            and loaded_weight.shape[0] == orig_experts
        ):
            return _pad_dim0(loaded_weight, padded_experts)
        return loaded_weight

    if orig_intermediate == padded_intermediate:
        return loaded_weight
    if ".experts." not in name and "experts." not in name:
        return loaded_weight

    # MXFP4 expert layouts (stock Flash-0731):
    #   w1/w3.weight [I, H_packed?] with I=2048  -> pad dim0 to padded_I
    #   w1/w3.scale  [I, ...]                     -> pad dim0
    #   w2.weight    [H, I/2] with I/2=1024       -> pad dim1 to padded_I/2
    #   w2.scale     [H, I/(2*block)]             -> pad dim1
    is_w13 = any(x in name for x in (".w1.", ".w3.", ".w1", ".w3"))
    is_w2 = ".w2." in name or name.endswith(".w2.weight") or name.endswith(".w2.scale")
    # Prefer path components: experts.N.w1.weight
    if ".w1." in name or ".w3." in name or name.endswith("w1.weight") or name.endswith("w3.weight") or name.endswith("w1.scale") or name.endswith("w3.scale"):
        is_w13 = True
        is_w2 = False
    if ".w2." in name or name.endswith("w2.weight") or name.endswith("w2.scale"):
        is_w2 = True
        is_w13 = False

    if is_w13:
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_intermediate:
            return _pad_dim0(loaded_weight, padded_intermediate)
        return loaded_weight

    if is_w2:
        # packed intermediate on dim1
        orig_packed = orig_intermediate // 2
        pad_packed = padded_intermediate // 2
        if loaded_weight.ndim >= 2 and loaded_weight.shape[1] == orig_packed:
            return _pad_dim1(loaded_weight, pad_packed)
        # scale may use I/(2*sf_block); sf_block=16 -> orig 64, pad 72
        # ratio pad/orig should match
        if loaded_weight.ndim >= 2 and loaded_weight.shape[1] * 2 * 16 == orig_intermediate * 2:
            # shape[1] == orig_intermediate / 32? 2048/32=64. padded 2304/32=72
            pass
        if loaded_weight.ndim >= 2 and orig_intermediate > 0:
            # generic: if dim1 scales with intermediate (weight I/2 or scale I/k)
            for k in (2, 16, 32, 64):
                if loaded_weight.shape[1] * k == orig_intermediate:
                    new_d1 = padded_intermediate // k
                    return _pad_dim1(loaded_weight, new_d1)
        return loaded_weight

    return loaded_weight


def vocab_padding_size(tp_size: int, base: int = 64) -> int:
    """Pad quantum so pad_vocab_size(v, pad) is always divisible by tp_size.

    VocabParallelEmbedding pads org vocab to a multiple of ``padding_size``,
    then shards by TP. Stock DeepSeek-V4 vocab (129280) is %64==0 but %3==1,
    so DEFAULT_VOCAB_PADDING_SIZE=64 alone is not enough for TP=3.
    Using lcm(base, tp_size) (e.g. 192 for tp=3) fixes that without changing
    the logical vocab_size.
    """
    import math

    if tp_size <= 1:
        return base
    return math.lcm(base, tp_size)


def _pad_dim0(t: torch.Tensor, new_n0: int) -> torch.Tensor:
    if t.shape[0] == new_n0:
        return t
    if t.shape[0] > new_n0:
        return t[:new_n0]
    pad_shape = list(t.shape)
    pad_shape[0] = new_n0 - t.shape[0]
    return torch.cat([t, t.new_zeros(pad_shape)], dim=0)


def _pad_dim1(t: torch.Tensor, new_n1: int) -> torch.Tensor:
    if t.shape[1] == new_n1:
        return t
    if t.shape[1] > new_n1:
        return t[:, :new_n1]
    pad_shape = list(t.shape)
    pad_shape[1] = new_n1 - t.shape[1]
    return torch.cat([t, t.new_zeros(pad_shape)], dim=1)


def pad_loaded_weight_for_attn(
    name: str,
    loaded_weight: torch.Tensor,
    *,
    head_dim: int,
    o_lora_rank: int,
    padded_heads: int,
    padded_groups: int,
    orig_heads: int,
    orig_groups: int,
    index_head_dim: int = 128,
) -> torch.Tensor:
    """Zero-pad a single checkpoint tensor to the padded attention geometry.

    Handles wq_b / wo_a / wo_b / attn_sink / indexer.weights_proj / indexer.wq_b
    and their scales. Unknown names are returned unchanged.
    """
    if not getattr(loaded_weight, "shape", None):
        return loaded_weight

    # attn_sink: [n_heads]
    if "attn_sink" in name:
        if loaded_weight.ndim == 1 and loaded_weight.shape[0] == orig_heads:
            return _pad_dim0(loaded_weight, padded_heads)
        return loaded_weight

    # Indexer is replicated; DeepGEMM MQA requires index heads ∈ {16,32,64}.
    # When PAD_INDEX_HEADS_TO == stock (64), leave indexer weights alone even
    # if main attention heads are padded to 96.
    if "indexer" in name:
        idx_pad = PAD_INDEX_HEADS_TO
        if idx_pad == orig_heads:
            return loaded_weight
        if "weights_proj" in name:
            if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_heads:
                return _pad_dim0(loaded_weight, idx_pad)
            return loaded_weight
        if "wq_b" in name:
            out_orig = orig_heads * index_head_dim
            out_pad = idx_pad * index_head_dim
            if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == out_orig:
                return _pad_dim0(loaded_weight, out_pad)
            if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_heads:
                return _pad_dim0(loaded_weight, idx_pad)
            if loaded_weight.ndim >= 2 and loaded_weight.shape[1] == out_orig:
                return _pad_dim1(loaded_weight, out_pad)
            return loaded_weight
        return loaded_weight

    # main attn wq_b: ColumnParallel out = n_heads * head_dim
    # weight [out, in] = [n_heads*head_dim, q_lora] e.g. (32768, 1024)
    # scale often [out/block, ...] e.g. (256, 8) with block=128
    if (
        (".wq_b." in name or name.endswith("wq_b.weight") or name.endswith("wq_b.scale"))
        and "indexer" not in name
    ):
        out_orig = orig_heads * head_dim
        out_pad = padded_heads * head_dim
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == out_orig:
            return _pad_dim0(loaded_weight, out_pad)
        if loaded_weight.ndim >= 2 and loaded_weight.shape[1] == out_orig:
            return _pad_dim1(loaded_weight, out_pad)
        for block in (128, 64, 32, 16, 8):
            if out_orig % block == 0 and loaded_weight.ndim >= 1:
                if loaded_weight.shape[0] == out_orig // block:
                    return _pad_dim0(loaded_weight, out_pad // block)
                if (
                    loaded_weight.ndim >= 2
                    and loaded_weight.shape[1] == out_orig // block
                ):
                    return _pad_dim1(loaded_weight, out_pad // block)
        return loaded_weight

    # wo_a: weight out = n_groups * o_lora_rank (8192->9216); scale is often
    # per-head [n_heads, ...] e.g. (64, 32) — must pad heads, not groups.
    if ".wo_a." in name or "attn.wo_a" in name:
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_heads:
            return _pad_dim0(loaded_weight, padded_heads)
        out_orig = orig_groups * o_lora_rank
        out_pad = padded_groups * o_lora_rank
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == out_orig:
            return _pad_dim0(loaded_weight, out_pad)
        if loaded_weight.ndim >= 2 and loaded_weight.shape[1] == out_orig:
            return _pad_dim1(loaded_weight, out_pad)
        # BMM packed layouts: [groups, ...]
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_groups:
            return _pad_dim0(loaded_weight, padded_groups)
        return loaded_weight

    # wo_b: RowParallel in = n_groups * o_lora_rank; scale often (32, n_heads)
    if ".wo_b." in name or "attn.wo_b" in name:
        in_orig = orig_groups * o_lora_rank
        in_pad = padded_groups * o_lora_rank
        # weight [out=hidden, in]
        if loaded_weight.ndim >= 2 and loaded_weight.shape[1] == in_orig:
            return _pad_dim1(loaded_weight, in_pad)
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == in_orig:
            return _pad_dim0(loaded_weight, in_pad)
        # scale (..., n_heads) e.g. (32, 64)
        if loaded_weight.ndim >= 2 and loaded_weight.shape[-1] == orig_heads:
            pad_shape = list(loaded_weight.shape)
            pad_shape[-1] = padded_heads - orig_heads
            return torch.cat(
                [loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=-1
            )
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_heads:
            return _pad_dim0(loaded_weight, padded_heads)
        return loaded_weight

    # Generic: any remaining attn tensor whose leading dim is stock n_heads
    if "attn" in name and loaded_weight.ndim >= 1 and loaded_weight.shape[0] == orig_heads:
        return _pad_dim0(loaded_weight, padded_heads)

    return loaded_weight

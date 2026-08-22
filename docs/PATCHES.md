# Patches

Two layers: (1) files bind-mounted from `overlay/` and baked into
`image/`, which are the 3×-specific fixes; (2) Mia / Anemll hotfixes
copied from
[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
and applied at container start by `docker-compose.yml`.

None of these edit the host kernel. The Cutlass convert lives in the
image so every node has the same digest.

## 3× overlays (always mounted)

| File | Why |
|---|---|
| `overlay/vllm/models/deepseek_v4/tp_pad.py` | General `_pad_loaded_for_tp_narrow`. Skips vocab-named tensors. |
| `overlay/vllm/models/deepseek_v4/attention.py` | 64→72 head / 8→9 group pad (SM120 legal). |
| `overlay/vllm/models/deepseek_v4/nvidia/model.py` | Wires the pad into the NVIDIA DeepSeek-V4 path. |
| `overlay/vllm/models/deepseek_v4/nvidia/dspark.py` | Same pad on the DSpark target. |
| `overlay/vllm/model_executor/models/qwen3_dspark.py` | Vocab-parallel embedding `padding_size = lcm(64, 3) = 192`. |
| `overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | Identity dummy EPLB on the draft (`expert_load_view`). |

## Image (Cutlass EP)

| File | Why |
|---|---|
| `image/overlay/.../oracle/mxfp4.py` | `[w1;w3]→[w3;w1]` + `block_scale_interleave` for `FLASHINFER_CUTLASS_MXFP4_MXFP8`. |
| `image/overlay/.../flashinfer_cutlass_moe.py` | DeepSeek α=1.0 **omits** `swiglu_alpha/beta/limit` so FlashInfer stays on SiLU. |
| `image/patches/patch_flashinfer_moe_init.py` | FlashInfer 0.6.15 `MoERunner.init` 8-arg vs AOT 7-arg shim. |

## Applied at container start (Mia / Anemll)

| Patch | Issue | What it does |
|---|---|---|
| `hotfix-encoding-dsv4-issue21.py` | #21 | Installs the 0731 `encoding_dsv4.py` into the image tokenizer path. |
| `hotfix-nvfp4-ds-mla-issue22.sh` | #22 | `nvfp4_ds_mla` long-context decode. Do not skip. |
| `hotfix-dsv4-issue26-hybrid-swa-min.py` | #26 | Hybrid SWA min + prefix-cache spacing. |
| `hotfix-dsv4-issue27-partial-prefill-concurrency.py` | #27 | One in-flight long prefill so decode is not starved. Pairs with `LONG_PREFILL_TOKEN_THRESHOLD=1024`. |
| `hotfix-dsv4-issue31-v2-thinking-budget-gpu.py` | #31 | Opt-in `thinking_token_budget` on the GPU sampler. |
| `hotfix-dsv4-issue43-decode-fairness-and-diag.py` | #43 | Decode fairness when mixed prefill/decode is in flight. |
| `hotfix-dsv4-issue55-tool-truncation.py` | #55 | Truncated tool call → `finish_reason: length`, drop invalid JSON args. |
| `hotfix-dsv4-suppress-stops-in-reasoning.py` | — | Client `stop` strings wait for `</think>`. |
| `hotfix-dsv4-mtp-buffer-50312.sh` | v0.27 | MTP buffer. |
| `hotfix-dsv4-adaptive-topk-50004.sh` | v0.27 | Adaptive top-k. |
| `hotfix-dsv4-skip-topk-49486.sh` | v0.27 | Skip top-k. |
| `hotfix-dsv4-dense-prefill-indexer-48407.sh` | v0.27 | Dense prefill indexer. |
| `hotfix-dsv4-skip-empty-c128-48957.sh` | v0.27 | Skip empty c128. |
| `hotfix-dsv4-flashmla-workspace-50298.sh` | v0.27 | FlashMLA workspace. |
| `hotfix-dsv4-grammar-advance.sh` | v0.27 | Grammar advance. |

`DSPARK_SKIP_HOTFIX=1` skips only the v0.27 shell backports. #21 / #26 /
#27 / #31 / #43 / #55 still run. `DSPARK_SKIP_ISSUE22_HOTFIX=1` skips
#22 — don't, on this recipe.

## Not applied

`patches/compressor.py` is a PP=3 stride experiment. It unblocks serve
and then decode falls to ~0.2 tok/s. Left in the tree for provenance;
compose does not mount it.

## Security / auth (fail-closed)

| File | Why |
|---|---|
| `patches/hotfix-vllm-redact-api-key-log.sh` | vLLM logs the full `--api-key` list in `non-default args` at startup. This redacts to `<redacted:N value(s)>` (count preserved — the documented way to verify one flag carried N keys), with a behavioural `--status` check. Keyed starts apply + verify it and refuse to serve otherwise. Port of Mia PR #89. |

## Vendored-but-deferred

| File | Why deferred |
|---|---|
| `patches/hotfix-gb10-spin-wait.sh` | TP=2-specific busy-wait sleep. The 3× runs TP=3 over the mesh; enable only after a 3× A/B. Not invoked by any entrypoint today. |

## Changes vs the vendored 2026-08-15 baseline (this port)

- All vendored `.sh` hotfixes are now **transactional**: every hunk is validated against a staged in-memory view before anything touches the tree, publishes via same-directory atomic rename, preserves modes, verifies committed bytes, and restores originals on any failure (Mia PR #103). The entrypoint no longer swallows failures: a missing or failing hotfix aborts the container before `exec vllm` (was `|| true` + silent skip). Escape hatches unchanged (`DSPARK_SKIP_HOTFIX`, `_ISSUE22`, `_SUPPRESS_STOPS`).
- `hotfix-dsv4-issue27-partial-prefill-concurrency.py`: now honors `DSPARK_MAX_INFLIGHT_PREFILLS` (1–3, default 2). The 2026-08-15 version hard-coded the serialising gate. Upstream A/B: 32K×c4 per-stream decode 8.2 → 24.6 tok/s (Mia #90).
- `hotfix-dsv4-issue31-v2-thinking-budget-gpu.py`: now **opt-in** (`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`, default 0). Default-on reproduced the omit-`thinking_token_budget` decode cliff (Mia #66).
- All vendored `.py` hotfixes gained read-only `--status` modes.
- `hotfix-dsv4-issue55-tool-truncation.py` no longer runs under `|| true`.

# Results — 3× DGX Spark DeepSeek-V4-Flash-0731

Date: **2026-08-14**. Hardware: three DGX Spark (GB10 / `sm_121`),
pairwise ConnectX-7 triangle, shared LAN for Gloo.

Runtime: [Anemll `dspark-vllm-gx10:0.1.1`](https://github.com/Anemll/dspark-vllm-gx10)
plus the Cutlass-EP layer in `image/`.

Measured lane: [SuperDeepseek-V4-Flash-abliterated](https://huggingface.co/Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX)
@ `2a7dd6a12d46f752e290c6255140a42600897864` (same 0731 geometry:
64 heads, 256 experts, inter 2048, vocab 129280).

Method: `scripts/measure_tps.py` against `http://127.0.0.1:8888/v1`,
`temperature=0`, thinking off, 128 generated tokens, two trials per
prompt size. Decode tok/s excludes the first token. Prefill tok/s is
prompt tokens / TTFT. A unique nonce defeats prefix-cache between
trials; "warm" is the second trial.

2× baseline is the Mia Anemll 0.1.1 recipe on the same two of these
three boxes, RoCE, MTP=5, util 0.835.

## Headline

| | 2× RoCE MTP=5 | TP=3 pad only | EP Socket MTP=0 | **EP Mesh MTP=5 (shipped)** |
|---|---:|---:|---:|---:|
| Weight / GPU | ~75–80 GiB | 81.5 GiB | 51.57 GiB | **54.09 GiB** |
| KV pool | 2,532,977 (2.42×) | 2,263,960 (2.16×) | 6,344,497 (6.05×) | **4,908,491 (4.68×)** |
| KV memory | 17.05 GiB | — | 41.13 GiB | **33.41 GiB** |
| Decode ~300 | 81.2 | ~11 | 9.5 | **85.2** |
| Decode ~2.1k | 78–81 | ~11 | 8.8 | **85.5** |
| Decode ~8.2k | — | — | 5.8 | **82.9** |
| Prefill ~300 cold/warm | 656 / — | ~70 | 78 / 114 | **47 / 569** |
| Prefill ~2.1k cold/warm | 1029 / — | ~95 | 105 / 118 | **296 / 968** |
| Prefill ~8.2k | 973 | — | 118 | **977** |
| Output | coherent | coherent | coherent | **coherent** |

Shipped stack: TP=3 + EP + EPLB=2 + Cutlass SiLU + NCCL mesh + DSpark
MTP=5 + dummy draft EPLB + heads pad 72/9. Smoke:
`"Say hi in 5 words."` → `"Hello! I'm ready to help."`

That is the 2× decode rate with **1.94×** the KV. It does **not** hit
the aspirational pair ">80 tok/s **and** >6M KV" in one process: MTP
draft weights + mesh registrations shrink the pool from EP-only's 6.34M
down to 4.91M. `GPU_MEMORY_UTILIZATION=0.87` is the leftover cheap
lever; this repo ships 0.835 because that is the number that booted
clean on the first mesh+MTP serve.

## Why the third node is not +124 GB of KV

MLA KV under tensor parallel is a **per-rank full copy**. The third
Spark holds:

- an expert-parallel shard (86 local / 258 global with 2 EPLB), which
  is why weights drop from ~80 GiB to ~54 GiB and the KV pool grows;
- a 64→72 head pad so TP=3 is legal on SM120 (`get_padded_num_q_heads`).

It does not hold a third replica of the cache. Data-parallel would
split the pool, not grow it. Pipeline-parallel *would* grow it a lot
(~20M tokens in one boot) and is not shippable today — see negatives.

## What we tried and rejected

These are here so you do not re-spend a night on them.

| Attempt | Result |
|---|---|
| PP=3 + DSpark | Draft does not implement `SupportsPP`. No boot. |
| PP=3 MTP=0 + regular graphs | Weights split correctly (~48 GiB/rank), dies in compressor (`state_cache.strides[0]` not divisible by 16). |
| PP=3 + copy-back stride hotfix | Serves; decode **0.2–0.3 tok/s**. Do not ship a clone of the compressor page on every token. |
| TP=3, no EP, Socket, MTP=0 | Serves, coherent, **~11 tok/s / 2.08M KV**. Loses to 2× on every metric. |
| TP=3, no EP, Mesh, MTP=5 | Serves, **~86 tok/s / 1.61M KV**. Decode wins, context loses. |
| Heads pad 96/12 vs 72/9 | 72/9 is legal and recovers ~180k KV. Shipped. |
| `--moe-backend flashinfer_b12x` + EP | B12X rejects `use_ep`. |
| `--moe-backend triton` | Kernel does not support `cuda` on GB10. |
| Stock Anemll Cutlass + EP | Convert missing; then SwiGLU-bias tensors → CJK / English-ish loops. |
| Cutlass convert + α=1 tensors still passed | Oracle cosine **0.87**. 43 layers of that is loops. |
| Cutlass convert, **omit** swiglu tensors | Oracle cosine **0.999**. Shipped. |
| Stock `NCCL_NET=IB` on the triangle | Cross-ported hop pairs IPv4 RoCEv2 with `fe80::`; QP INIT→RTR timeout. |
| TCP on the QSFP Ethernet names | Hang. |
| Two-hop IP forward across the triangle | Adjacent ICMP works; opposite `/24` is 100% loss. |
| Socket on the shared LAN | Boots TP/EP. Decode ~9–35 tok/s. Debug Gloo only. |
| MTP=5 + EP without dummy EPLB | `EPLB requires expert_load_view != None` after the draft loads. |
| DP | Splits the KV pool. Not a context win. |

## Cutlass oracle (why the image looks like that)

Software MXFP4 dequant vs FlashInfer `cutlass_fused_moe` on one expert
(layer 0 expert 0, hidden 4096, inter 2048). Serving path quantizes
activations to MXFP8 first; a naive bf16 oracle is classified as NVFP4
and asks for six scales.

| Variant | cosine | notes |
|---|---:|---|
| **swap+interleave + plain SiLU** | **0.999** | shipped |
| swap+interleave + α=1 tensor | 0.873 | FlashInfer switches to SwigluBias |
| swap only (no interleave) | 0.756 | |
| identity | 0.442 | |
| GPT-OSS even/odd + swap | −0.018 | |

FlashInfer 0.6.15: if SwiGLU **and** any of `alpha` / `beta` / `limit`
is present → `ActivationType.SwigluBias`. DeepSeek α=1 must omit those
tensors, not pass ones.

The image also shims FlashInfer `MoERunner.init` (Python 8-arg vs AOT
`fused_moe_120` 7-arg).

## Reproduce

On node0, after `./start.sh` is READY:

```bash
python3 scripts/measure_tps.py \
  --url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --tag local --prompts 256,2048,8192 --gen 128 --trials 2
```

Compare the boot lines `GPU KV cache size` and
`Available KV cache memory` to the table above. Different util, seqs,
or MTP will move the pool.

## Leftover knobs (not in the default)

- `GPU_MEMORY_UTILIZATION=0.87` or `0.90` after a healthy first boot.
- `MAX_NUM_SEQS=6` (Mia 2× default). Capture size grows with
  `seqs × (mtp+1)`.
- Official 0731 (`ABLITERATED=0`) should load if `config.json` matches.
  We did not re-bench it on 3×.
- PP=3 with an aligned compressor allocation (not a copy-back) would be
  the context jackpot. It is not this recipe.

# `dspark-vllm-gx10:0.1.1-cutlass-ep`

Thin layer on [`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`](https://github.com/Anemll/dspark-vllm-gx10).
It does **not** rebuild vLLM, FlashInfer, or B12X.

```bash
# on every DGX Spark, from the repo root
./scripts/build-image.sh
```

Or build once and copy:

```bash
docker save dspark-vllm-gx10:0.1.1-cutlass-ep | ssh worker docker load
```

What the layer adds:

1. Cutlass MXFP4 `convert_weight`: DeepSeek packed `[w1;w3]` → FlashInfer
   `[w3;w1]`, then `block_scale_interleave`.
2. MoE runner: when DeepSeek `swiglu_alpha=1.0`, do **not** pass
   `swiglu_alpha/beta/limit` into FlashInfer (that switch is SwiGLU-bias /
   GPT-OSS and produces garbage on this checkpoint).
3. `MoERunner.init` arity shim for FlashInfer 0.6.15 on GB10.

`start.sh` refuses to launch if this tag is missing on any rank.

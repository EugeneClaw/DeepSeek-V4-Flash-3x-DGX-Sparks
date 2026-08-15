# Credits

This recipe stands on other people's work. Please keep this file if you fork.

## Runtime and 2× serving recipe

- **[Anemll / dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)** — GB10 vLLM image with DSpark, `nvfp4_ds_mla`, and FlashInfer.
- **[MiaAI-Lab / DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)** — worker-first compose, Anemll env surface, issue #21/#22/#26/#27/#31/#43/#55 hotfixes, thinking-budget and stop-in-reasoning behavior, KV-ceiling documentation. Several files under `patches/` are vendored from that repo.
- **[drowzeys ("Keys")](https://github.com/drowzeys/)** — DSpark concurrency / `nvfp4_ds_mla` long-context work that Mia packages.
- **Rafael Caricio, Tony Deangelo, Fraser Price** — earlier DSpark overlay and 2× Spark packaging.

## 3× fabric

- **[autoscriptlabs / nccl-mesh-plugin](https://github.com/autoscriptlabs/nccl-mesh-plugin)** — subnet-aware RoCE QPs. Required because stock NCCL IB cannot pair a pairwise QSFP triangle as dual-rail.

## Weights

- Official: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Validated 3× lane: [Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX](https://huggingface.co/Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX) (same 0731 geometry)

## This repo

3-node EP + Cutlass SiLU convert + DSpark EPLB dummy + mesh launch: FlyCockpit.

Repo scripts and docs: Apache-2.0 (`LICENSE`). Vendored vLLM overlay snippets keep their upstream SPDX headers. Weights and base images have their own terms. Do not publish prebuilt `.ko` or cluster-specific hostnames.

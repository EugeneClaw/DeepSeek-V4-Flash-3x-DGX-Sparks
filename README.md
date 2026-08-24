# DeepSeek-V4-Flash on 3× DGX Spark

Serve **DeepSeek-V4-Flash-0731** across three NVIDIA DGX Sparks with a
pairwise ConnectX-7 triangle, expert parallel, and DSpark speculative decoding.

> Forked from [FlyCockpit/DeepSeek-V4-Flash-3x-DGX-Sparks](https://github.com/FlyCockpit/DeepSeek-V4-Flash-3x-DGX-Sparks) — all original work is theirs; this fork adds testing, hardening, and documentation.
> **This fork** extends the FlyCockpit 3× recipe with:
> - API-key authentication (single + multi-key, with log redaction)
> - Fail-fast hotfix validation (exit 1 on missing patches, not silent continue)
> - Performance backports from upstream vLLM PRs (#50004, #50298, #50312, #48957, #49486, #48407)
> - `GPU_MEMORY_UTILIZATION=0.87` tested and recommended (+13% MTP acceptance vs 0.835)
> - First published prose benchmark for the 3× configuration
> - Honest comparison vs Mia 2× recipe (see "Which recipe should I use?" below)

See [CHANGELOG.md](CHANGELOG.md) and [results/RESULTS-2026-08-24.md](results/RESULTS-2026-08-24.md) for details.

This is a consumer runbook, not an official NVIDIA or DeepSeek product.
The git repo is the maintained surface — no hosted binaries.

| | 2× Spark (RoCE, Mia recipe) | **This 3× recipe** |
|---|---:|---:|
| Decode (single stream, 300–8k prompt) | ~81 tok/s | **~85 tok/s** (original) / **~99 tok/s** (tested, util=0.87) |
| KV pool | ~2.53M tokens | **4.91M tokens** (original) / **~5.0M** (tested) |
| Concurrency @ 1M | ~2.4× | **4.68×** (original) / **4.78×** (tested) |
| Warm prefill @ 2k / 8k | ~1.0k / ~1.0k tok/s | **~970 / ~980 tok/s** (original) / **~1,425 / ~1,579** (tested) |
| Per-request ceiling | 1,048,576 | 1,048,576 |

Original recipe numbers measured 2026-08-14 on GB10 / Anemll 0.1.1.
Tested configuration numbers from 2026-08-24: [results/RESULTS-2026-08-24.md](results/RESULTS-2026-08-24.md). Original recipe numbers: [results/RESULTS.md](results/RESULTS.md).

You do **not** get 3× the 2× KV. Tensor-parallel MLA keeps a full KV
copy on every rank; the third box holds an expert-parallel shard plus a
small attention pad, not a third replica of the cache. Expert parallel
is what almost doubles the pool. Plain TP=3 without EP is *slower and
smaller* than 2× (~11 tok/s / ~2.1M KV).

---

## What you need

- **3× DGX Spark** (GB10, `sm_121`)
- A **QSFP cable between every pair** (a triangle). This is almost never
  one RoCE `/24`. See [docs/FABRIC.md](docs/FABRIC.md).
- A **shared L2 Ethernet** that can reach all three boxes (typical Spark
  LAN name: `enP7s7`). Gloo, TCPStore, and `--master-addr` live here.
  NCCL data does **not**.
- Passwordless SSH from node0 (head) to the two workers
- Docker + NVIDIA Container Toolkit on every node
- `libibverbs-dev` on node0 (to build the mesh plugin)
- ~200 GB free disk per node (weights + image)
- **earlyoom disabled** on every host — deep-context load looks like a
  memory hog and the daemon will SIGKILL vLLM

Stock NCCL IB cannot treat that triangle as dual-rail. This recipe
builds [autoscriptlabs/nccl-mesh-plugin](https://github.com/autoscriptlabs/nccl-mesh-plugin)
locally. Do not pin `NCCL_IB_GID_INDEX`.

---

## Quick start

Run everything from **node0**.

### 1. Checkout and env

```bash
git clone https://github.com/FlyCockpit/DeepSeek-V4-Flash-3x-DGX-Sparks.git
cd DeepSeek-V4-Flash-3x-DGX-Sparks
cp .env.example .env
```

Set at least:

```env
NODE0_HOST=spark-0
NODE1_HOST=spark-1
NODE2_HOST=spark-2
NODE0_IP=10.0.0.10
NODE1_IP=10.0.0.11
NODE2_IP=10.0.0.12
GLOO_IFACE=enP7s7
SSH_USER=ubuntu
```

`NODE*_IP` is the **shared LAN**, not a pairwise QSFP address.
`start.sh` refuses to launch if any `NODE*_HOST` still contains
`CHANGE_ME`.

If the checkout path is not the same on the workers, set `NODE1_DIR` /
`NODE2_DIR`. Leave them empty to use this directory on every node.

### 2. Image on all three nodes

```bash
./scripts/build-image.sh
```

Repeat on each worker, or build once and `docker save | ssh docker load`.
The tag is `dspark-vllm-gx10:0.1.1-cutlass-ep` — a thin layer on
[Anemll `dspark-vllm-gx10:0.1.1`](https://github.com/Anemll/dspark-vllm-gx10)
that adds the Cutlass MXFP4 convert DeepSeek needs for expert parallel.
See [image/README.md](image/README.md).

### 3. Mesh plugin (node0)

```bash
./scripts/build-mesh-plugin.sh
```

`start.sh` rsyncs `nccl-mesh/libnccl-net-mesh.so` to the workers.

### 4. Weights on all three nodes

```bash
./scripts/prepare-model.sh
```

Default is the validated abliterated 0731 snapshot
([SuperDeepseek](https://huggingface.co/Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX)
@ `2a7dd6a1…`). Official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
@ `9e165c30…` is `ABLITERATED=0` after that cache is warm. Prepare
forces Hugging Face online; keep `HF_HUB_OFFLINE=1` in `.env` afterwards
so a hub retry cannot fill a worker disk.

### 5. Start (workers first, then head)

```bash
./start.sh
```

First boot is 5–15 minutes (weight load + CUDA-graph capture). The
script polls `http://127.0.0.1:8888/v1/models` until it returns 200.

Always `./stop.sh` before a relaunch. Recreating a live cluster races
Gloo.

### 6. Check it

```bash
./status.sh
./smoke.sh
```

Expect `"id": "deepseek-v4-flash-0731"` and `"max_model_len": 1048576`.
Smoke (`temperature=0`, thinking off) should look like
`Hello! I'm ready to help.`

Boot log (trust the live numbers on your cluster):

```text
Available KV cache memory: 33.41 GiB
GPU KV cache size: 4,908,491 tokens
Maximum concurrency for 1,048,576 tokens per request: 4.68x
```

API: `http://NODE0_IP:8888/v1`. For head-only tests set
`VLLM_HOST=127.0.0.1`.

> ⚠️ **Security warning:** The default `VLLM_HOST=0.0.0.0` with no API key
> exposes an unauthenticated inference server. **Do not expose port 8888
> to the Internet.** This recipe is designed for a trusted LAN. API
> authentication (when configured) does not protect all endpoints
> (`/metrics`, `/health`, `/tokenize` remain keyless). Use network-level
> access control (firewall, VLAN, or bind to `127.0.0.1`).

Day-to-day: `./status.sh`, `./logs.sh`, `./stop.sh`.

Validate **direct** `:8888` first, then any agent harness.

---

## Default profile

| Knob | Default |
|---|---|
| Image | `dspark-vllm-gx10:0.1.1-cutlass-ep` (Anemll 0.1.1 + Cutlass EP) |
| Checkpoint | SuperDeepseek abliterated 0731 (`ABLITERATED=1`) |
| Served name | `deepseek-v4-flash-0731` |
| Parallelism | TP=3 + EP + EPLB (2 redundant) · PP=1 · DP=1 |
| MoE | `flashinfer_cutlass` · SiLU (no SwiGLU-bias tensors) |
| Fabric | `NCCL_NET=Mesh` · `NCCL_ALGO=Ring` · Gloo on `GLOO_IFACE` |
| Context ceiling | `MAX_MODEL_LEN=1048576` |
| Concurrent seqs | `MAX_NUM_SEQS=4` |
| Batch tokens | `MAX_NUM_BATCHED_TOKENS=8192` |
| Long-prefill cap | `LONG_PREFILL_TOKEN_THRESHOLD=1024` |
| KV | `nvfp4_ds_mla` · block 256 · util **0.835** (~4.91M tokens) |
| Spec | `MTP_NUM_TOKENS=5` |
| Thinking | `DEFAULT_THINKING=off` |
| Graphs | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (keep this) |

`max_model_len` and `max_num_seqs` are **ceilings**, not reservations.
The limit is `sum(live tokens) ≤ KV pool`.

```
4 ×  50k  =  200k   easy
4 × 200k  =  800k   easy
4 × 500k  =  2.0M   fits
4 ×   1M  =  4.0M   near the 4.91M pool
6 ×   1M  =  6.0M   does not fit — extras queue
```

After a clean first serve you can try more KV / more slots:

```env
GPU_MEMORY_UTILIZATION=0.87
MAX_NUM_SEQS=6
```

Then `./stop.sh && ./start.sh`.

---

## Why this is not the 2× recipe on three boxes

The [Mia 2× runbook](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
is the right starting point for two Sparks on one RoCE `/24`. Three
Sparks change three axes:

1. **Fabric.** Two CX7 ports per box almost always wire as a pairwise
   triangle (`192.168.100/101/102`), not one mesh. Stock NCCL IB pairs
   the wrong GID on the cross-ported hop and `ibv_modify_qp` INIT→RTR
   times out. This recipe uses the subnet-aware mesh plugin and keeps
   Gloo on the shared LAN.
2. **Expert parallel.** TP=3 on 64 heads / 256 experts pads to 72/9
   (legal) but B12X rejects EP. Cutlass on stock Anemll 0.1.1 cannot
   convert DeepSeek MXFP4. The image in `image/` does the
   `[w1;w3]→[w3;w1]` + `block_scale_interleave` convert and **omits**
   `swiglu_alpha/beta/limit` so FlashInfer stays on plain SiLU (α=1).
   Passing those tensors switches the kernel to GPT-OSS SwiGLU-bias and
   the model emits garbage or English-ish loops.
3. **DSpark + EPLB.** The draft is not `MixtureOfExperts`, so vLLM's
   EPLB controller skips `add_model()` and then crashes with
   `expert_load_view != None`. The mounted `speculator.py` installs an
   identity dummy on the draft. MTP=5 without that overlay does not
   boot.

Everything else that is load-bearing on Anemll 0.1.1 is taken from the
2× recipe: worker-first start, `HF_HUB_OFFLINE` after cache,
`LONG_PREFILL=1024`, prefix-cache retention 4096, breakable graphs off,
FlashInfer sampler, `nvfp4_ds_mla` block 256, async scheduling + chunked
prefill, capture size `seqs × (mtp+1)`, generation-config `vllm`, and
hotfixes #21 / #22 / #26 / #27 / #31 / #43 / #55 plus suppress-stops.
See [docs/PATCHES.md](docs/PATCHES.md).

---

## `.env` switches

Copy [`.env.example`](.env.example) → `.env`. Restart all three ranks
after a flip (`./stop.sh` then `./start.sh`).

### Weights

| Variable | Default | What it does |
|---|---|---|
| **`ABLITERATED`** | `1` | `1` = SuperDeepseek (measured lane). `0` = official 0731. |
| `DSPARK_REVISION_*` | pinned SHAs in `.env.example` | Empty = tip of that repo. The pad script needs a snapshot SHA. |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-0731` | Name clients send as `model`. |
| `HF_HUB_OFFLINE` | `1` | Keep `1` after both caches are warm. |

```bash
# in .env
ABLITERATED=0
./scripts/prepare-model.sh
./stop.sh && ./start.sh
```

### Thinking, API

| Variable | Default | What it does |
|---|---|---|
| **`DEFAULT_THINKING`** | `off` | `off` / `low` / `high` / `max`. Request-level `chat_template_kwargs` still wins. |
| `VLLM_HOST` | `0.0.0.0` | `127.0.0.1` for head-only tests. |
| `VLLM_PORT` | `8888` | OpenAI-compatible API. |
| `VLLM_API_KEY` | *(empty)* | Single API key. Mutually exclusive with `DSPARK_API_KEYS`. |
| `DSPARK_API_KEYS` | *(empty)* | Space-separated list of API keys (one `--api-key` flag, N keys). Default empty = unauthenticated. |

`max_tokens` counts **think + answer**. With `DEFAULT_THINKING=max`, a
harness cap of 256/512 often returns empty `content` because reasoning
eats the whole budget. Raise `max_tokens` into the tens of thousands,
set thinking `off`/`low`, or send `thinking_token_budget` (hotfix #31):

```json
{
  "max_tokens": 8192,
  "thinking_token_budget": 1024,
  "temperature": 0.6,
  "top_p": 0.95,
  "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
}
```

Client `stop` strings wait for `</think>` (suppress-stops hotfix). A
tool call cut off by `max_tokens` reports `finish_reason: "length"`
instead of a poisoned `tool_calls` payload (hotfix #55).

#### API authentication (optional, default off)

Set **one** of these in `.env` — never both (start, smoke, status, and
all three container entrypoints exit 2 before any side effect):

- `VLLM_API_KEY=…` — single key.
- `DSPARK_API_KEYS="k1 k2 k3"` — multiple independently revocable keys
  (quoted; single line; space/tab separated; duplicates allowed).

What you get: `Authorization: Bearer <key>` required on `/v1`, `/v2`,
`/inference`; every key in the list is accepted; rotation = edit `.env`
+ `./stop.sh && ./start.sh`. What you don't get: everything else stays
**keyless** (`/invocations`, `/generative_scoring`, `/tokenize`,
`/health`, `/metrics`…), so a keyed deployment still needs network-level
access control on the port; and vLLM does not attribute requests to
keys — this is revocation and blast-radius control, not per-user logs.
Keys are visible to host `ps` / `docker inspect` (argv/env) by design;
they are kept out of the startup Docker log by a fail-closed redaction
hotfix (`'api_key': ['<redacted:N value(s)>']`). Probes (`smoke.sh`,
`status.sh`, and the start.sh readiness poll) send the first parsed key
automatically.

### Serve shape

| Variable | Default | What it does |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | Per-request ceiling (1M). |
| `MAX_NUM_SEQS` | `4` | Concurrent slots. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill tokens per step. |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Issue #27 chunk cap. `0` lets one prefill starve decode. |
| `GPU_MEMORY_UTILIZATION` | `0.835` | Larger = bigger KV pool. Try `0.87` after a healthy boot. |
| `MTP_NUM_TOKENS` | `5` | DSpark draft depth. Capture size = `seqs × (k+1)`. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | **Keep 0.** Unset enables Anemll's slower breakable graphs. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | Issue #26 SWA prefix-cache spacing. |

Do not change `TENSOR_PARALLEL_SIZE`, `ENABLE_EP`, `MOE_BACKEND`, or
`TRANSPORT` unless you are reproducing a negative from
[results/RESULTS.md](results/RESULTS.md).

---

## What speed to expect

Full tables: **[results/RESULTS.md](results/RESULTS.md)**.

| Workload | What you should see |
|---|---|
| One chat, 300–8k prompt | **~83–85 decode tok/s** after the first token |
| Warm prefill @ 300 / 2k / 8k | ~570 / ~970 / ~980 tok/s |
| Cold first request after boot | 6–9 s TTFT (graphs); later requests ~0.5–2 s at 2k |
| Four short chats | pool has room; we did not publish a c=4 aggregate |

Reproduce:

```bash
python3 scripts/measure_tps.py \
  --url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --tag local --prompts 256,2048,8192 --gen 128 --trials 2
```

---

## If it misbehaves

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) covers the cases we
hit while building this:

- API never 200 → rank-2-first, same image tag, `./stop.sh` before relaunch
- `ibv_modify_qp` + `fe80::` → you are on stock IB, not the mesh plugin
- CJK / English-ish loops → stock Anemll Cutlass (SwiGLU-bias). Use this image.
- `EPLB requires expert_load_view != None` → speculator overlay not mounted
- Empty `content` with thinking on → `max_tokens` too small

---

## Files

| Path | Purpose |
|---|---|
| [`.env.example`](.env.example) | Cluster template (no hostnames) |
| [`docker-compose.yml`](docker-compose.yml) | Anemll serve + overlays + hotfixes |
| `start.sh` / `stop.sh` / `status.sh` / `logs.sh` / `smoke.sh` | Three-node ops |
| [`scripts/prepare-model.sh`](scripts/prepare-model.sh) | HF snapshot on head **and** workers |
| [`scripts/build-image.sh`](scripts/build-image.sh) | Cutlass-EP image |
| [`scripts/build-mesh-plugin.sh`](scripts/build-mesh-plugin.sh) | NCCL mesh `.so` |
| [`scripts/measure_tps.py`](scripts/measure_tps.py) | Prefill / decode probe |
| [`overlay/`](overlay/) | Head pad, vocab pad, DSpark EPLB dummy |
| [`image/`](image/) | Cutlass MXFP4 convert + FlashInfer init shim |
| [`patches/`](patches/) | Mia / Anemll hotfixes applied at container start |
| [docs/FABRIC.md](docs/FABRIC.md) | Pairwise triangle + why mesh |
| [docs/PATCHES.md](docs/PATCHES.md) | What each hotfix is for |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Failure modes we actually hit |
| [results/RESULTS.md](results/RESULTS.md) | Dated benches and dead ends |

---

## Security / exposure model

This recipe is designed for a **trusted LAN** (private network behind a
firewall). The default configuration serves on `0.0.0.0:8888` without
authentication. Optional API-key support (`VLLM_API_KEY` /
`DSPARK_API_KEYS`) guards `/v1`, `/v2`, and `/inference` routes only —
other endpoints (`/metrics`, `/health`, `/tokenize`, `/detokenize`)
remain keyless.

**Do not expose this server to the public Internet** without
network-level access control (firewall rules, VLAN isolation, or
binding to `127.0.0.1`). The recipe builds no hosted binaries; all
images and plugins are compiled locally from source.


## License and credits

Repo scripts and docs: Apache-2.0 (`LICENSE`). Vendored vLLM overlay
snippets keep their upstream SPDX headers. Weights and the Anemll base
image have their own terms. No prebuilt `.ko` or cluster hostnames are
published.

## Which recipe should I use?

**Use the 3× recipe (this repo) if you want:**
- **Maximum context capacity:** ~5M KV tokens (2× the 2× recipe's 2.5M)
- **High concurrency:** 4.78× at 1M context (2× the 2× recipe's 2.38×)
- **Long-context performance:** 99 tok/s at 8k prompts (50% faster than 2× at 66)
- **Counting/benchmark speed:** 97–99 tok/s (exceeds 2× recipe's 83)
- **1M-token context ceiling** with room for multiple concurrent long chats

**Use [Mia's 2× recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) if you want:**
- **Maximum per-stream prose speed:** 68–75 tok/s on natural text (1.6–1.8× faster than 3× at ~42)
- **Better draft acceptance:** ~60–70% (vs 3× at ~35%) — the 2× TP topology gives the DSpark draft model denser expert routing, producing better predictions
- **Simpler fabric:** one RoCE /24, no mesh plugin needed
- **Lower power:** one fewer node (one Spark sits idle, ~20W)

**The trade-off is structural, not tunable.** 3-way expert parallelism gives you 2× the KV pool and 2× the concurrency, but halves the DSpark draft model's prediction quality on open text. The draft uses mean-pooled auxiliary hidden states, and with 3-way EP each GPU sees sparser expert routing than under 2-way TP. No in-recipe knob (GPU utilisation, max sequences, MTP depth, sample method, model variant) closes this gap. See [results/RESULTS-2026-08-24.md](results/RESULTS-2026-08-24.md) for the full experiment record.

This recipe stands on [Anemll](https://github.com/Anemll/dspark-vllm-gx10),
[MiaAI-Lab's 2× runbook](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
[drowzeys ("Keys")](https://github.com/drowzeys/), and
[autoscriptlabs/nccl-mesh-plugin](https://github.com/autoscriptlabs/nccl-mesh-plugin).
Full list: [`CREDITS.md`](CREDITS.md).

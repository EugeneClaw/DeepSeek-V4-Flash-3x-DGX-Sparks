# Troubleshooting

## API never becomes 200

- `docker compose -p dsv4-0731-3n logs --tail 120` on each rank.
- Rank 2 must start **before** rank 0. `start.sh` already does this.
  Recreating a live cluster without `./stop.sh` first can race Gloo
  (`Connection refused` to a worker that is about to die).
- Confirm the same image tag exists on all three nodes:
  `docker image inspect dspark-vllm-gx10:0.1.1-cutlass-ep`.

## NCCL / mesh

Look for `NET/Mesh/0` and `NET/Mesh/1` on every pair, including the
cross-ported hop. If you see `ibv_modify_qp` + IPv4 GID paired with
`fe80::`, you are on stock IB, not the mesh plugin.

- `libnccl-net-mesh.so` must be in `./nccl-mesh` on every node.
- `NCCL_SOCKET_IFNAME` must be the **shared** LAN (`==enP7s7` style),
  not a QSFP name.

## Garbage / CJK / English-ish loops

That was the unpatched Cutlass path (GPT-OSS SwiGLU-bias tensors).
This image sets DeepSeek α=1 and **omits** `swiglu_alpha/beta/limit`.
If you swapped the image back to stock Anemll without the Cutlass
layer, EP will load and then emit garbage.

Direct `:8888` first, then the agent harness (Mia's rule).

## `EPLB requires expert_load_view != None`

The overlay `speculator.py` installs an identity dummy on the DSpark
draft. If that file is not mounted, MTP=5 dies after the draft loads.
Check the log line:

```
DSpark: installed identity EPLB dummy on 3 draft modules
```

## KV smaller than ~4.9M

Draft weights + mesh registrations cost several GiB vs EP-only. First
boot at `GPU_MEMORY_UTILIZATION=0.835`. After a clean serve, try
`0.87` or `0.90` (E8 needed 0.87 on the mesh plugin for a 1M ceiling
check when weights were larger).

## earlyoom

Disable **earlyoom** on every host. Deep-context load looks like a
memory hog and the daemon will SIGKILL vLLM.

## Thinking returns empty `content`

`DEFAULT_THINKING=max` plus a small `max_tokens` (256/512) spends the
whole budget inside `<think>`. Raise `max_tokens` into the tens of
thousands, set thinking `off`/`low`, or send `thinking_token_budget`
(Mia issue #31 hotfix is applied at container start).

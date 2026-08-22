# Changelog

## Unreleased — 2× security & robustness port

Bringing the latest Mia 2× security & robustness work to the 3× recipe. Vendored at [MiaAI-Lab@a462a9e](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/commit/a462a9e) (2026-08-22). All new behavior is default-off or default-unchanged; keyless boot is byte-identical.

### Added
- **Multi-key API auth via `DSPARK_API_KEYS` (default empty = unauthenticated)**, with fail-closed startup-log redaction for keyed starts. vLLM receives all space-separated keys through exactly one `--api-key` flag; `VLLM_API_KEY` stays as the mutually exclusive single-key option. Both set → exit 2 before side effects (launcher, probes, and all three container entrypoints). Parser: trims/collapses whitespace, preserves order and duplicates, rejects CR/LF/VT/FF, backslashes, and dash-leading tokens without echoing token bytes. start/smoke/status send the first parsed key. Guarded routes: `/v1`, `/v2`, `/inference` — everything else stays keyless (`/invocations`, `/generative_scoring`, `/tokenize`, `/health`, `/metrics`…), so network-level access control is still required; this provides revocation, not per-key attribution.
- **Safe env handling** (Mia #98/#5fc02ce/#5410f88, adapted to 3 nodes): `.env` is normalised once (BOM/CRLF) into a private 0600 snapshot; the operator's file stays byte-identical; an ambient `DSPARK_API_KEYS` that disagrees with `.env` is rejected. `.env.3n` is published atomically (tmp+rename, 0600) on head **and both workers** — a failed transfer can no longer truncate or expose credentials. `sync_tree` no longer ships a stale node-0 `.env.3n` to workers.
- `DSPARK_MAX_INFLIGHT_PREFILLS` (1–3, default 2): two overlapping chunked prefills (Mia #90; 8.2 → 24.6 tok/s on 32K×c4 upstream).
- `DRAFT_SAMPLE_METHOD` (probabilistic|greedy, default unchanged) validated before the speculative JSON is built (Mia #84).
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` + `TILELANG_CACHE_DIR` on the HF volume: ride out mid-serve TileLang JIT instead of EngineCore death (Mia #65/#87).
- HF hub download timeouts in `scripts/prepare-model.sh` (Mia #97, adapted to the host-CLI flow).
- `start.sh` exits 3 when the head container already exists (reboot under `restart: unless-stopped`); supervisors treat 3 as already-up.

### Changed
- **Hotfix boot is fail-closed and transactional** (Mia PR #103): every vendored `.sh` hotfix validates all hunks against a staged view, publishes atomically, verifies bytes, rolls back on failure; the entrypoint FATALs on a missing file and aborts on nonzero exit (was: silent skip + `|| true`).
- `hotfix-dsv4-issue31-v2-thinking-budget-gpu.py` is opt-in (`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX`, default 0 — Mia #66 omit-field decode cliff).
- `smoke.sh` runs `CONCURRENCY` parallel requests (default 1) and exits nonzero on any failure.

### Deferred (intentionally)
- `hotfix-gb10-spin-wait.sh`: vendored but not invoked (TP=2-specific; needs a 3× mesh A/B first).
- NCCL passthrough knobs (`NCCL_IB_MERGE_NICS` etc.): 2× RoCE-oriented; the 3× mesh transport is configured separately.
- Responses-API live verifier, RULER-lite eval fixes, benchmark output cap, VL sidecar, assistant-final hotfix.


## 0.1.0 — 2026-08-15

Initial public runbook.

- Three-node launch: worker-first `start.sh`, Anemll-shaped compose,
  Cutlass-EP image, NCCL mesh plugin, DSpark MTP=5.
- Expert parallel (86/258 + 2 EPLB) with DeepSeek SiLU convert. Head
  pad 72/9; experts stay at 256.
- Identity dummy `expert_load_view` on the DSpark draft so EPLB + MTP
  can coexist.
- Mia 2× hotfixes vendored and applied at container start: #21, #22,
  #26, #27, #31, #43, #55, suppress-stops, plus the v0.27 perf
  backports (MTP buffer, adaptive top-k, skip top-k, dense prefill
  indexer, skip empty c128, FlashMLA workspace, grammar advance).
- Measured lane: ~85 decode tok/s, 4.91M KV, 1M ceiling. See
  [results/RESULTS.md](results/RESULTS.md).

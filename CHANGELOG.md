# Changelog

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

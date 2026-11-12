# Outline · track ⑥ multimodal cache v1 (Research-to-Product, team theme, v8 form)

Value thesis: ADS asks for ms-level access to clips in the Lance lake. Physics rules out moving whole clips from OBS; Lance has no cache for object bytes and nothing shared across jobs. A tiered UB-pool cache with prefetch and a value-per-byte policy makes ms time-to-first-batch reachable.

1. Title — Millisecond Data Serving from Lance
2. We speed up ADS stage (6) — pipeline, (6) highlighted.
3. Three key technologies in the Lance read path — consumers → reader → L1/L2/L3 → OBS map + status.
4. "ms" needs a close hit and a partial read — bandwidth model table + what Lance caches today.
5. KT1 tiered byte cache — read through four tiers, Lance hooks, prior art, gate.
6. KT2 prefetch — illustrative timeline, prefetch flow with (4) hints, policy, gate.
7. KT3 policy + placement — U = F·C·R/S, object classes, rendezvous placement on UB pool, baselines, gate.
8. Summary + roadmap — 3 cards, 4 gated phases.

Change log: v1 (11 Nov 2026) first version.

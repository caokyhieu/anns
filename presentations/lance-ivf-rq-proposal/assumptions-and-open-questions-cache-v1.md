# Assumptions and open questions · track ⑥ v1

Assumptions (from the user, 11 Nov 2026)
- "Serve" means serving everything in Lance format: original objects (video etc.), vectors, tags; ms latency needs more efficient caching.
- UB / Lingqu pooled memory is available.

Assumptions (ours, to confirm)
- "ms level" for a 4 GB clip means time to first batch, not whole-clip transfer (slide 4 model: whole clip ≥ 40 ms even at 100 GB/s).
- The training loader exposes its epoch order (seeded shuffle).
- Clips are stored as Lance blob columns reachable through take_blobs / read_range.

Open questions for phase 1
- Latency and bandwidth of UB pool, local NVMe and OBS on ADS nodes.
- The ms target value agreed with ADS, and the per-step window size.
- UB pool capacity per job / per pod; how pooled memory is exposed (API, allocation).
- Clip reuse across epochs, jobs and stages (drives KT3).
- Staffing and sizes (S/M/L are rough estimates).

# V6 assumptions and open questions

- Kunpeng: the user confirmed SVE/SVE2 support. The exact model, vector length, core/NUMA layout, RAM and BF16 support are not recorded. The deck does not assume BF16 or SME.
- Every Kunpeng number is a placeholder. No native ARM build or timing exists yet, partly because the pinned Rust toolchain download failed certificate validation on the x86 host.
- x86 results are single runs, except the GIST blocked-L2 table (2 repeats). Noise is about ±1 s on builds and ±0.005 on recall.
- Suite C recall covers nprobes 1–2 and k=10 only. Tail latency is excluded as a harness cache artefact; phase 4 fixes the cache and adds k=100 and nprobes ≥ 3.
- Suite B peak RSS is not compared, because those builds shared one process.
- The ±0.005 recall@10 tolerance, the T-shirt sizes, the 2-week phase-1 estimate and the kernel gates are proposals for leadership to confirm.
- The standalone PiPNN check uses different graph settings and uint8 data. It cross-checks the result and is not the replacement comparison.
- Previews were rendered with LibreOffice and a substitute font for Microsoft YaHei. Native PowerPoint rendering was not checked.

# Outline · track ④ hybrid retrieval v1 (Research-to-Product, team theme, v8 form)

Value thesis: ADS mining runs many filtered vector searches; Lance today rebuilds the tag allow-list each query and reads partitions with no passing row. Removing that work speeds mining without recall loss.

1. Title — Faster Vector + Tag Retrieval in Lance
2. We speed up ADS stage (4) — pipeline, (4) highlighted. Visual: copied ADS diagram.
3. Three key technologies in Lance filtered search — query-path map + 3 verified gaps + status. Visual: architecture map.
4. Filtered IVF loads partitions where no row passes — 4-step anatomy + model table. Visual: method figure + table.
5. KT1 predicate-result cache — lookup flow, design table, evidence/TBD, gate.
6. KT2 partition skip — partition × tag grid, prepare_partition change, model + reported table, gate.
7. KT3 batched mining — queries × partitions grid, load counts, prior art, gate.
8. Summary + roadmap — 3 cards, 4 gated phases.

Change log: v1 (11 Nov 2026) first version.

# Filtered-search methods from the literature, run on our indexes

FILTERED_SEARCH_LITERATURE.md read six papers and checked their claims against our
earlier measurements. This note **implements** their methods and runs them on the
indexes this repository builds: HNSW (hnswlib), our from-scratch **fastpipnn** graph
(native/), a plain k-NN graph, and IVF. Data and filters are the same as
FILTERED_SEARCH.md, plus AND / OR filters. Script: `filtered_methods_experiment.py`,
raw numbers: `filtered-methods-results.json`.

## Setup

200,000 rows, 64-d clustered embeddings (400 clusters), k = 100, 20 queries per filter,
beam (ef) 200 unless noted. Every cell is **recall@100 / distance computations per
query** against exact filtered search. Distance computations include entry points and
IVF centroids. Brute force over the passing rows costs exactly `passing rows`.

| Index | How built | Max / mean out-degree | Build |
| --- | --- | --- | --- |
| hnsw-d32 | hnswlib M = 16, base layer | 32 / 16.7 | 19 s for both HNSW, 4 threads |
| hnsw-d64 | hnswlib M = 32, base layer | 64 / 20.2 | |
| pipnn-d64 | fastpipnn defaults (local_k 2, α 1.2) | 64 / 13.8 | 3.2 s |
| pipnn-d64-dense | fastpipnn, local_k 8, α 2.0 | 64 / 47.1 | 4.8 s |
| knn-d32 / knn-d64 | plain nearest neighbours, no pruning (KGraph style) | 32 / 64 | 8.4 s |
| IVF | numpy k-means, 256 partitions, base nprobe 16 | – | 1.1 s |

`native/build/pipnn_graph` (new, `native/src/pipnn_graph.cpp`) builds the fastpipnn
graph over float32 vectors and writes its adjacency for the Python search code.

Methods, by paper:

| Method | From | What runs |
| --- | --- | --- |
| acorn | ACORN-1 / NaviX `blind` | Walk passing nodes only, plus all passing 2-hop neighbours behind failing ones |
| navix | NaviX | Per-candidate choice from local selectivity: `onehop-s` above 50%, `directed` 2-hop (1-hop sorted by distance, capped at the degree) while the expected passing 2-hop count is ≥ 3 × degree, `blind` below that |
| entries = 1…512 | Li et al. claim 5 | navix with that many random passing rows as entry points (default 64) |
| milvus | Amanbayev et al. | Dual-pool traversal from one entry, brute force when > 93% of rows fail, exact re-run when fewer than k come back |
| ivf-fixed / ivf-scaled | baseline / our planner | Probe 16, or 16 / s partitions; score passing rows |
| ivf-skip | FILTERED_SEARCH.md §4.2 | Partition × label counts: probe the 16 nearest partitions that hold a passing row |
| caps-h3 / caps-h8 (+skip) | CAPS | Attribute-frequency tree of height 3 or 8 per partition: read only the sub-partitions that can match |
| sieve | SIEVE | Greedy sub-index choice for the workload's 11 filters, budget 0.5 × base edges (used 0.46 ×); per query the cheaper of the best subsuming sub-index and brute force, by SIEVE's cost model |
| pathfinder | PathFinder | One HNSW per attribute value (5 weather values, 4 tags: 1.12 × the rows) plus the root graph; per DNF clause the covering graph with the highest utility U, union over OR |

## Results

### Single-label filters

| Method | rain 20% | pedestrian 10% | cut_in 1% | emergency 0.1% | tunnel 1%, inside | tunnel 1%, outside |
| --- | --- | --- | --- | --- | --- | --- |
| acorn / hnsw-d32 | 1.00 / 1,357 | 0.98 / 961 | 0.41 / 312 | 0.37 / 71 | 1.00 / 545 | 0.99 / 675 |
| acorn / hnsw-d64 | 1.00 / 2,336 | 0.99 / 1,646 | 0.63 / 413 | 0.40 / 75 | 1.00 / 559 | 0.99 / 691 |
| acorn / pipnn-d64 | 1.00 / 1,007 | 0.87 / 767 | 0.19 / 249 | 0.37 / 70 | 1.00 / 547 | 0.99 / 673 |
| acorn / pipnn-d64-dense | 1.00 / 3,618 | 1.00 / 2,648 | **0.78** / 497 | 0.46 / 88 | 1.00 / 551 | 1.00 / 763 |
| acorn / knn-d64 | 0.10 / 424 | 0.12 / 346 | 0.18 / 240 | 0.41 / 77 | 1.00 / 533 | 0.98 / 667 |
| navix / hnsw-d64 | 0.90 / 1,862 | 0.94 / 1,478 | 0.62 / 416 | 0.41 / 75 | 1.00 / 631 | 0.99 / 880 |
| navix / pipnn-d64-dense | 0.93 / 1,864 | 0.93 / 1,400 | 0.60 / 417 | 0.45 / 86 | 1.00 / 573 | 0.99 / 832 |
| milvus / hnsw-d32 | 1.00 / 3,170 | 1.00 / 6,628 | 1.00 / 1,969 | 1.00 / 193 | 1.00 / 1,989 | 1.00 / 1,989 |
| milvus / pipnn-d64 | 1.00 / 3,027 | 1.00 / 6,204 | 1.00 / 1,969 | 1.00 / 193 | 1.00 / 1,989 | 1.00 / 1,989 |
| ivf-fixed | 1.00 / 8,245 | 0.93 / 4,135 | 0.58 / 633 | 0.26 / 288 | 1.00 / 1,402 | 0.29 / 708 |
| ivf-scaled | 1.00 / 23,240 | 1.00 / 15,989 | 1.00 / 2,225 | 1.00 / 449 | 1.00 / 2,245 | 1.00 / 2,245 |
| ivf-skip | 1.00 / 8,289 | 0.93 / 4,158 | 0.58 / 637 | 0.28 / 292 | 1.00 / 2,245 | **1.00** / 2,245 |
| sieve | 1.00 / 1,329 | 1.00 / 1,732 | 1.00 / 1,969 | 1.00 / 193 | 1.00 / 1,989 | 1.00 / 1,989 |
| pathfinder | 1.00 / 1,324 | 1.00 / 1,722 | 1.00 / 1,367 | 1.00 / 193 | 1.00 / 566 | 0.99 / 713 |
| brute force | 1.00 / 40,013 | 1.00 / 20,007 | 1.00 / 1,969 | 1.00 / 193 | 1.00 / 1,989 | 1.00 / 1,989 |
| GLS ρ (mean) | +0.04 | −0.03 | −0.38 | −0.91 | +0.98 | −1.00 |

### AND / OR filters

| Method | rain AND cut_in 0.2% | night AND pedestrian 2% | pedestrian OR tunnel 11% | cut_in OR emergency 1.1% | (rain AND ped) OR tunnel 3%, query outside tunnel |
| --- | --- | --- | --- | --- | --- |
| acorn / hnsw-d64 | 0.27 / 107 | 0.83 / 625 | 0.99 / 1,766 | 0.69 / 444 | 0.83 / 883 |
| acorn / pipnn-d64-dense | 0.37 / 142 | **0.91** / 792 | 1.00 / 2,726 | **0.81** / 536 | **0.92** / 1,138 |
| navix / hnsw-d64 | 0.28 / 109 | 0.79 / 637 | 0.94 / 1,572 | 0.67 / 447 | 0.81 / 940 |
| milvus / hnsw-d32 | 1.00 / 399 | 1.00 / 3,961 | 1.00 / 6,462 | 1.00 / 2,161 | 1.00 / 5,906 |
| ivf-fixed | 0.35 / 327 | 0.68 / 1,011 | 0.92 / 4,762 | 0.59 / 665 | 0.68 / 1,483 |
| ivf-scaled | 1.00 / 655 | 1.00 / 4,217 | 1.00 / 17,017 | 1.00 / 2,417 | 1.00 / 6,162 |
| sieve | 1.00 / 399 | 1.00 / 1,754 | 1.00 / 1,702 | 1.00 / 2,161 | 1.00 / 1,718 |
| pathfinder | 1.00 / **1,926** | 1.00 / **6,325** | 1.00 / 2,495 | 1.00 / 1,558 | 1.00 / **6,820** |
| brute force | 1.00 / 399 | 1.00 / 3,961 | 1.00 / 21,789 | 1.00 / 2,161 | 1.00 / 5,906 |

PathFinder's plans: `cut_in` graph for rain AND cut_in, `pedestrian` graph for night AND
pedestrian, `pedestrian` + `tunnel` graphs for both OR filters with pedestrian, and
`cut_in` + `emergency_vehicle` for the last one. SIEVE built sub-indexes for
`pedestrian OR tunnel`, `(rain AND pedestrian) OR tunnel`, `night AND pedestrian`,
`rain` and `pedestrian`; it answers everything else by brute force.

### Recall against work: NaviX vs ACORN (beam 100 / 200 / 400)

| Graph, filter | acorn | navix |
| --- | --- | --- |
| hnsw-d64, rain 20% | 0.99 / 873 · 1.00 / 2,336 · 1.00 / 4,544 | 0.90 / 859 · 0.90 / 1,862 · 0.92 / 3,642 |
| hnsw-d64, cut_in 1% | 0.49 / 257 · 0.63 / 413 · 0.75 / 663 | 0.50 / 253 · 0.62 / 416 · 0.74 / 663 |
| pipnn-d64-dense, pedestrian 10% | 0.96 / 1,379 · 1.00 / 2,648 · 1.00 / 4,567 | 0.82 / 818 · 0.93 / 1,400 · 0.94 / 2,550 |
| pipnn-d64-dense, cut_in 1% | 0.60 / 304 · 0.78 / 497 · 0.89 / 804 | 0.45 / 257 · 0.60 / 417 · 0.71 / 678 |

### Entry points (navix on hnsw-d64; Li et al. claim 5)

| Entries | rain 20% | cut_in 1% | emergency 0.1% | rain AND cut_in 0.2% | tunnel outside |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.91 / 1,893 | 0.44 / 285 | 0.01 / 1 | 0.01 / 2 | 0.95 / 1,036 |
| 8 | 0.91 / 1,855 | 0.59 / 385 | 0.06 / 10 | 0.05 / 19 | 0.95 / 813 |
| 64 | 0.91 / 1,867 | 0.62 / 416 | 0.41 / 74 | 0.28 / 111 | 0.99 / 887 |
| 512 | 0.90 / 2,236 | 0.85 / 765 | 1.00 / 193 | 1.00 / 399 | 0.99 / 1,189 |

### Rows read by the IVF variants (I/O, not distance computations)

| Plan | rain 20% | cut_in 1% | tunnel outside | rain AND cut_in | night AND ped |
| --- | ---: | ---: | ---: | ---: | ---: |
| ivf-fixed | 39,643 | 39,643 | 40,552 | 39,643 | 39,643 |
| ivf-scaled | 114,417 | 200,000 | 200,000 | 200,000 | 200,000 |
| ivf-skip | 39,867 | 40,042 | 7,932 | 40,266 | 39,867 |
| caps-h3 | 25,323 | 39,643 | 40,164 | 25,323 | 31,146 |
| caps-h8 | 23,467 | 39,643 | 40,164 | 23,467 | 29,517 |
| caps-h8 + skip | 23,617 | 40,042 | **5,467** | 23,815 | 29,653 |

CAPS never changes recall (it reads fewer rows from the same partitions), so its recall
equals ivf-fixed, or ivf-skip with `+skip`.

## What the measurements say

1. **Only plans with an exact fallback hold recall everywhere.** Milvus's rule (brute
   force when > 93% of rows fail, safety net) and SIEVE are ≥ 0.996 on all 11 filters.
   Every pure graph or pure IVF plan drops below 0.5 somewhere. This confirms the
   literature note's top recommendation (§8.1).
2. **SIEVE is the best overall plan here.** With 0.46 × extra edges it does no worse
   than brute force or Milvus on every filter, and it is 2–4 × cheaper than Milvus on
   the broad ones (rain 1,329 against 3,170; pedestrian OR tunnel 1,702 against 6,462).
   Its cost model sent every filter under 2,200 passing rows to brute force and built
   sub-indexes only for broad filters. That is the reverse of FILTERED_SEARCH.md §4.3,
   which builds sub-tables for **rare** labels. At this scale rare labels are cheapest to
   brute-force. Sub-indexes pay off for the broad filters a graph walk handles badly.
   At 10^9 rows, "rare" means 10^6 rows and the balance moves back.
3. **PathFinder needs a brute-force switch.** It wins when a per-value graph is dense in
   passing rows: `tunnel` 566 against 1,989 for brute force, `cut_in` 1,367 against
   1,969. On conjunctions its best covering graph still has 20% passing rows, and the
   dual-pool walk inside it costs more than brute force: 1,926 against 399 for
   rain AND cut_in, and 6,325 against 3,961 for night AND pedestrian. Adding SIEVE's
   `γ · card(f)` check to its plan choice removes those cases.
4. **On fastpipnn, density decides 2-hop search, and the default build is too sparse.**
   With defaults (mean degree 13.8), acorn reaches 0.19 recall at 1%, the worst of the
   pruned graphs. With local_k 8 and α 2.0 (mean degree 47, built in 4.8 s), it is the
   **best** predicate-agnostic graph we tested: 0.78 at 1% (0.89 at beam 400), 0.91 on
   night AND pedestrian, 0.92 on the anti-correlated OR. hnsw-d64 reaches 0.63 / 0.83 /
   0.83, because its pruning heuristic keeps only 20 of 64 slots filled on average. So
   PiPNN's cheap build is a good fit for ACORN-style filtering, provided it is built
   for filtering (larger local_k, looser α) rather than tuned for unfiltered search. The
   memory cost is about 2.3 × the edges of hnsw-d64.
5. **NaviX's adaptive rule did not beat uncapped 2-hop at equal work.** At 20%, acorn
   on hnsw-d64 reaches 0.99 with 873 distance computations. navix sits at 0.90–0.92 up to
   3,642, because its `directed` step stops after a degree's worth of candidates and
   loses the long-range ones. At 1% the two are equal on HNSW, and navix is worse on
   dense PiPNN. NaviX's reported 2 × gain comes from a disk-based system, where 1-hop
   distance computations replace random page reads. Our cost here is distance
   computations in memory. Treat this as "not reproduced in memory", not as refuted.
6. **Unpruned k-NN graphs fail on clustered data (against Li et al. claim 4).** With
   no long edges, the k-NN graph cannot leave a cluster. It gets 0.04–0.20 recall at
   1–20%, and works only when the label sits in the query's cluster (tunnel inside,
   1.00). Li et al.'s finding (keep nearest neighbours below 5%) does not carry over to
   strongly clustered embeddings. On this data, pruning (diverse edges) is required,
   which agrees with NaviX's ACORN-1 observation.
7. **Entry points matter only at low selectivity (agrees with Li et al. claim 5).** From
   1 to 64 entries, rain stays at 0.91, while cut_in goes from 0.44 to 0.62 and
   emergency_vehicle from 0.01 to 0.41. At 512 entries, cut_in reaches 0.85 for 765
   distance computations (brute force costs 1,969). Entry points sampled from the
   passing set are a partial brute force: below about 1%, sampling more of them is a
   cheap way to buy recall between a pure walk and a full scan.
8. **Partition label counts (ivf-skip) fix anti-correlation for correlated labels
   only.** For `tunnel` with the query outside, recall goes from 0.29 to 1.00 while
   reading 7,932 rows, against 40,552 for ivf-fixed and 200,000 for ivf-scaled. For
   uncorrelated labels every partition holds passing rows, so nothing is skipped and
   nothing changes (cut_in 0.58 either way). There, nprobe ∝ 1/s or brute force is
   still needed.
9. **CAPS's frequency tree is captured by the most common attribute.** Every row has
   exactly one weather value (20% each), so the tree splits on weather values first.
   After five levels the remainder is empty, and tags never get a bucket: h = 3 and
   h = 8 read the same rows for tag filters. It helps weather filters and weather
   conjunctions (rain AND cut_in: 39,643 rows down to 23,467). Combined with skipping,
   it reads 5,467 rows for tunnel. For the AV table, the tree should be built per
   attribute group, or only over multi-valued tags, so that one dense categorical
   column does not fill it.
10. **GLS ρ ranks filter difficulty but is biased for rare filters.** Mean ρ matches
    the correlated cases (+0.98 tunnel inside, −1.00 outside). Over all queries, its
    rank correlation with per-query recall is 0.56 for ivf-fixed and 0.26 for navix on
    hnsw-d32: it predicts IVF recall better than graph recall. When s · k < 1
    (emergency_vehicle: 0.1 expected passing rows in the unfiltered top 100), ρ is −1
    for most queries even though the label is independent of the embedding. Compute it
    with k ≥ several / s, or only for filters above about 1%.

## What changes in the design

- Planner: add the SIEVE cost check (`γ · card(f)` against indexed cost) to **every** plan,
  including PathFinder-style AND / OR plans, and keep Milvus's safety net.
- Sub-indexes: choose them with SIEVE's benefit-per-byte rule from the query log. Do not
  assume rare labels come first: at shard sizes where rare filters stay under the
  brute-force threshold, broad filters and OR filters benefit more.
- Hot-tier filtered graph: build fastpipnn with a larger local_k and looser α (here
  local_k 8, α 2.0) and search it ACORN-style (uncapped 2-hop) with 64+ passing entry
  points. Fall back to brute force below the threshold.
- IVF tier: keep partition × label counts (they fix correlated labels), and scale nprobe
  or brute-force for uncorrelated ones. If CAPS ordering is used, keep dense categorical
  columns such as weather out of the frequency tree.

## Limits

Synthetic data, one seed, 20 queries per filter (±0.02–0.05 recall noise), 200K rows,
in-memory. Work is distance computations, not latency or I/O; the Python graph search
is not a speed benchmark. The fastpipnn and hnswlib graphs come from the native code,
but all filtered search runs in Python over their adjacency lists. NaviX, SIEVE,
PathFinder and CAPS are re-implemented from the papers, not the authors' code.

## Run

```bash
cmake -S native -B native/build && make -C native/build pipnn_graph
```

```bash
uv run --locked --extra lance --extra compare python filtered_methods_experiment.py
```

Options: `--n`, `--queries`, `--beam`, `--beam-sweep`, `--entries`, `--entry-sweep`,
`--caps-heights`, `--sieve-budget`. A full run takes a few minutes on 4 cores.

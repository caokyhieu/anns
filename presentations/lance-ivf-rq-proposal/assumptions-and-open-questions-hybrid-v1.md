# Assumptions and open questions · track ④ v1

Assumptions (from the user, 11 Nov 2026)
- Baseline index is IVF_RQ; filters are tags (bitmap-indexable equality / AND predicates).

Assumptions (ours, to confirm)
- ADS mining uses selective tag combinations (well below 1%) and reuses tag sets across many query vectors.
- Mining is batch work judged on throughput.
- Tags are roughly independent of the embedding for the model on slide 4; real correlation changes the numbers.

Open questions for phase 1
- Tag vocabulary size T and selectivity distribution on ADS data.
- Predicate repeat rate and share of query time spent building the allow-list.
- Partition reads per query and recall@10 at the current nprobes / search_effort.
- Whether ADS needs NOT / range predicates (KT2 falls back to today's path for those).
- Staffing and sizes (S/M/L are rough estimates).

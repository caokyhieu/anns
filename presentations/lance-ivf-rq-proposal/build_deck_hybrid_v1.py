#!/usr/bin/env python3
"""Track 4 v1: faster vector + tag (hybrid) retrieval in Lance, ADS stage (4). Same form as build_deck_v8.py.

Key technologies, in query-path order:
  KT1 predicate-result cache (allow-list per tag predicate and dataset version)
  KT2 filter-aware partition skipping (conservative per-partition tag summary, checked before load)
  KT3 batched mining (partition-major schedule over a window of queries)
Slides: 1 title · 2 ADS pipeline (4) · 3 Lance filtered-search map · 4 problem anatomy + model · 5 KT1 · 6 KT2 · 7 KT3 · 8 summary
No ADS measurements exist yet: every ADS number is TBD, [Model] or [Reported]; nothing is presented as measured.
Run: uv run --no-project --with python-pptx python build_deck_hybrid_v1.py
"""
import math
from ads_track_common import load, ads_slide, finish

LANCE = 'lance-format/lance main @ 04e619963'
EXTRA = {
    'PF': f'{LANCE}: rust/lance/src/index/prefilter.rs (DatasetPreFilter: deleted_ids, filtered_ids; selects_nothing = whole mask empty; '
          'create_deletion_mask cached under RowAddrMaskKey in the metadata cache) and rust/lance/src/session/caches.rs L167 (RowAddrMaskKey: version, e-tag, restrict hash).',
    'IVF': f'{LANCE}: rust/lance/src/index/vector/ivf/v2.rs L1331-1361 prepare_partition: tokio::try_join!(load_query_partition, pre_filter.wait_for_ready); '
           'L1241 prefilter_for_partition; L1399 selects_nothing check before scan; L2179 write_cache=true admits the loaded partition.',
    'MODEL': 'Our model: rows per partition n = 10,000,000 / K; tags independent of the embedding; P(no passing row in a probed partition) = (1 - s)^n ≈ exp(-s·n). '
             'K = 16,384: n ≈ 610; K = 4,096: n ≈ 2,441. Summary size K × T bits = 16,384 × 1,024 / 8 = 2 MiB.',
    'PC': 'Schmidt, Kipf, Horn, Saxena, Kraska, Predicate Caching: Query-Driven Secondary Indexing for Cloud Data Warehouses, SIGMOD 2024; '
          'slides https://www.db.in.tum.de/~schmidt/papers/predicate-caching-slides.pdf (up to 10× on selected queries, Redshift prototype).',
    'SIEVE': 'Hu et al., SieveIVF: Threshold-Aware IVF Execution for Large-Scale Training Data Deduplication, arXiv:2608.03199 (Aug 2026), implemented in Lance: '
             '4.1-7.6× (10M, recall loss 0.03-1.13 pp) and 6.1-8.4× (100M, 1.43-2.29 pp) vs fixed-probe IVF, W = 8.',
    'VADER': 'Chatzakis et al., VADER: Filtered Vector Search with Declarative Recall, arXiv:2609.38053, VLDB 2027 (up to 53% faster, +28% quality vs best baseline).',
    'LAKE': 'Jain, Griffin, Zawad, Filtered Vector Search in a Disaggregated Lakehouse, arXiv:2608.05441 (Aug 2026; 355 of 444 files pruned; ~94× with partitioned layout).',
    'GROUP': 'CaGR-RAG, arXiv:2505.01164 (p99 up to -51.55%, higher hit ratio); CALL, arXiv:2509.18670 (p99 up to -33%).',
}
g = load('ADS · Lance Hybrid Retrieval', EXTRA, 'ads-lance-hybrid-retrieval-v1.pptx')
globals().update({k: v for k, v in g.items() if not k.startswith('__')})


def empty_share(s_, K):
    return math.exp(-s_ * 10_000_000 / K)


def pct(v):
    return f'{v * 100:.1f}%' if v < .1 else f'{v * 100:.0f}%'


# ================================================================ 1 title
s = D.title_slide('Faster Vector + Tag Retrieval in Lance',
                  'Predicate-result cache · Filter-aware partition skipping · Batched mining',
                  presenters=['Research proposal v1 · 11 November 2026'], team='Vector Index Optimization')
notes(s, 'Proposal to make ADS multimodal mining faster: vector search combined with tag filters over the Lance data lake. '
      'Three key technologies on the IVF_RQ query path: KT1 caches the evaluated tag allow-list, KT2 skips IVF partitions in which no row can pass '
      'the filter before loading them, and KT3 runs mining queries in batches that share partitions. The gaps are verified in Lance source; '
      'ADS measurements are the first phase.', ['ADS', 'DOC'])

# ================================================================ 2 ADS pipeline
ads_slide(g, 'We speed up ADS stage (4): vector + tag retrieval',
          '[Company context] Redrawn from ADS-PLAN slide 1; target is (4) Multimodal retrieval: vector search joined with scalar tags.',
          '(4)', (4.98, 5.14, 2.12, .6, 6.59, 5.07, 5.14),
          ['OUR KEY TECHNOLOGIES', 'KT1 predicate cache', 'KT2 partition skip · KT3 batching'],
          'Mining runs many filtered vector searches over one lake: work repeats.', 'All three key technologies sit in this box.',
          'ADS pipeline from ADS-PLAN slide 1. Stage (4), multimodal retrieval, searches the Lance vector indexes built in stage (3) together with scalar '
          'tags such as weather, time of day or scene type. The ADS framework names the problem: the join of vector and scalar data is inefficient. '
          'All three key technologies change only what happens inside this box. The next slide opens it.')

# ================================================================ 3 project map
s = new('Three key technologies in Lance filtered search',
        'Lance IVF_RQ query with a tag prefilter, each key technology (KT) at its integration point · gaps [Source] · dashed = proposed')
s.section(.8, 1.42, 8.6, 'LANCE FILTERED VECTOR SEARCH (IVF_RQ + TAG PREFILTER)')
stages = [('Parse filter', 'night ∧ rain'), ('Build allow-list', 'bitmap index → mask'), ('Route + probe', 'nearest nprobes'),
          ('Load partition', 'RQ codes → cache'), ('Scan + top-k', 'mask per row')]
xs = [.8, 2.55, 4.3, 6.05, 7.8]
for i, ((a, b), x) in enumerate(zip(stages, xs)):
    node(s, f'st{i}', x, 1.78, 1.5, .62, a, b, 'neutral', size=9.5)
    if i < 4:
        edge(s, f'e{i}', [(x + 1.5, 2.09), (x + 1.75, 2.09)])
TOP, BOT = 2.62, 3.82
s.rect(.8, TOP, 3.25, BOT - TOP, 'blue_light', 'blue', .9, dashed=True)
txt(s, .95, TOP + .05, 3.0, .24, 'KT1 · Predicate-result cache', size=10, color='blue_dark', bold=True)
txt(s, .95, 2.94, 3.0, .85, ['Reuse the allow-list per (tags, version)', 'Append: evaluate new fragments only', 'Exact: same mask, same results'], size=8.5)
edge(s, 'kt1', [(3.3, TOP), (3.3, 2.4)], color='blue', dashed=True)
s.rect(4.3, TOP, 3.25, BOT - TOP, 'purple_light', 'purple', .9, dashed=True)
txt(s, 4.45, TOP + .05, 3.0, .24, 'KT2 · Filter-aware partition skip', size=10, color='purple_dark', bold=True)
txt(s, 4.45, 2.94, 3.0, .85, ['Tag summary per partition, checked first', 'Skip, and do not admit, empty partitions', 'Probe budget counts useful partitions'], size=8.5)
edge(s, 'kt2a', [(5.05, TOP), (5.05, 2.4)], color='purple', dashed=True)
edge(s, 'kt2b', [(6.8, TOP), (6.8, 2.4)], color='purple', dashed=True)
s.rect(7.8, TOP, 1.5, BOT - TOP, 'green_light', 'green', .9, dashed=True)
txt(s, 7.88, TOP + .05, 1.38, .24, 'KT3 · Batching', size=9.5, color='green_dark', bold=True)
txt(s, 7.88, 2.94, 1.38, .85, ['one load,', 'many queries', 'partition-major'], size=8.5)
edge(s, 'kt3', [(8.55, TOP), (8.55, 2.4)], color='green', dashed=True)
KY = 4.32
s.rect(.8, KY, 8.5, 1.68, 'pale_red', 'red', 1.0)
gaps = [('1 · Allow-list rebuilt per query', 'only scalar-index pages are cached', 2.4),
        ('2 · Partition loaded before mask', 'load ∥ prefilter wait (try_join)', 5.05),
        ('3 · Empty test is global only', 'selects_nothing() = whole mask', 6.8)]
for i, (a, b, ux) in enumerate(gaps):
    gx = .98 + i * 2.75
    node(s, f'gap{i}', gx, KY + .18, 2.6, .78, a, b, 'white', size=9)
    edge(s, f'up{i}', [(ux, KY + .18), (ux, BOT)], color='red', dashed=True)
txt(s, .98, KY + 1.06, 8.15, .28, 'Today in Lance [Source] · where filtered search repeats or wastes work', size=10, color='title', bold=True, align='ctr')
txt(s, .98, KY + 1.34, 8.15, .25, 'Gaps 2 and 3: a partition with no passing row still costs a read, cache space and a probe slot', size=8, color='muted', align='ctr')
s.rule(9.55, 1.45, 4.55, vertical=True)
s.section(9.8, 1.42, 2.8, 'STATUS · 11 NOV 2026')
st = [('KT1 predicate cache', ['Gap verified in Lance source', 'Design after SIGMOD 2024 paper'], 'blue', 1.25),
      ('KT2 partition skip', ['Gap verified: load before mask', 'Model: 54% of probes empty', 'at 0.1% tags, K = 16,384'], 'purple', 1.45),
      ('KT3 batched mining', ['Reported 4–8× (SieveIVF, on Lance)', 'ADS trace not yet collected'], 'green', 1.3)]
y = 1.78
for h, lines, col, dy in st:
    s.rect(9.8, y + .04, .06, .22, col, None)
    txt(s, 9.95, y, 2.6, .28, h, size=10, color='ink', bold=True, anchor='ctr')
    bullets(s, 9.95, y + .3, 2.6, 1.2, lines, size=8.5, line=1.25)
    y += dy
end(s, 'KT1 removes repeated filter work; KT2 and KT3 remove wasted partition loads.', 'Each ships separately.')
notes(s, 'The query path is Lance vector search on an IVF_RQ index with a tag prefilter. The filter is evaluated through the scalar (bitmap) index into a row '
      'address mask; IVF routing picks the nearest nprobes partitions by centroid distance; each partition is loaded through the index cache and scanned, '
      'checking the mask row by row. Three gaps are verified in the source. First, DatasetPreFilter computes filtered_ids for every query; only the bitmap '
      'index pages it touches are cached, and only the deletion mask is cached as a whole. Second, prepare_partition loads the partition and waits for the '
      'prefilter concurrently, so a partition is read and admitted before Lance knows whether any of its rows pass. Third, selects_nothing() is true only when '
      'the whole mask is empty, not per partition. KT1 targets the first gap, KT2 the second and third, KT3 batches queries so shared partitions load once.',
      ['PF', 'IVF', 'MODEL', 'SIEVE', 'PC'], limits='Status reflects source reading only; no ADS measurement yet.')

# ================================================================ 4 problem anatomy
s = new('Filtered IVF loads partitions where no row passes',
        'One filtered query in Lance today, redrawn by us from the source [Source] · red tags = the key technology that removes the cost')
panels = [('1 · Evaluate tag filter', 'Bitmaps are ANDed into an allow-list; pages are cached, the result is not', 'repeated per query → KT1', 'outline'),
          ('2 · Rank partitions', 'nprobes nearest centroids; the filter is not consulted', 'blind to filter → KT2', 'outline'),
          ('3 · Load + admit', 'Runs in parallel with the mask: read and cached even if no row passes', 'wasted I/O → KT2', 'accent'),
          ('4 · Scan with mask', 'Every row is scored, then rows outside the mask are dropped', 'wasted compute → KT2', 'outline')]
x = .8
for i, (h, b, k, v) in enumerate(panels):
    yy = s.panel(x, 1.45, 2.82, 3.05, h)
    if i == 0:
        s.grid(x + .45, yy + .15, 1, 10, cell=.17, fill='white', cells={(0, c): 'pink' for c in (1, 3, 4, 7, 8)})
        lab(s, x + .1, yy + .15, .33, 'night', size=7.5, align='l')
        s.grid(x + .45, yy + .45, 1, 10, cell=.17, fill='white', cells={(0, c): 'pink' for c in (0, 3, 6, 8)})
        lab(s, x + .1, yy + .45, .33, 'rain', size=7.5, align='l')
        s.rule(x + .45, yy + .72, 1.7)
        s.grid(x + .45, yy + .8, 1, 10, cell=.17, fill='white', cells={(0, c): 'red' for c in (3, 8)})
        lab(s, x + .1, yy + .8, .33, 'mask', size=7.5, align='l', bold=True)
    elif i == 1:
        cpts = [(x + .5, yy + .25), (x + .95, yy + .6), (x + 1.4, yy + .2), (x + 1.85, yy + .7), (x + 2.3, yy + .3),
                (x + .7, yy + 1.0), (x + 1.6, yy + 1.1), (x + 2.2, yy + 1.0)]
        for j, (px, py) in enumerate(cpts):
            dot(s, px, py, 'red' if j in (1, 2, 3, 6) else 'muted', .11)
        s.rect(x + .7, yy + .02, 1.45, 1.25, None, 'red', .9, dashed=True, shape=MSO_SHAPE.OVAL)
        s.rect(x + 1.38, yy + .6, .1, .1, 'ink', None)
        lab(s, x + 1.5, yy + .55, .4, 'q', size=8, color='ink', bold=True, align='l')
    elif i == 2:
        for j, (lbl, empty) in enumerate([('P7 · 3 pass', False), ('P2 · 0 pass', True), ('P9 · 1 pass', False), ('P4 · 0 pass', True)]):
            s.rect(x + .45, yy + .1 + j * .29, 1.9, .23, 'white' if empty else 'pale_red', 'red' if empty else 'red', .75, dashed=empty)
            lab(s, x + .45, yy + .1 + j * .29, 1.9, lbl + ('  loaded anyway' if empty else ''), size=7.5, color='title' if empty else 'ink')
    else:
        s.grid(x + .6, yy + .1, 5, 8, cell=.2, fill='surface', cells={(1, 2): 'red', (3, 6): 'red'})
        lab(s, x + .45, yy + 1.15, 1.95, 'score all · keep 2', size=7.5)
    txt(s, x + .15, 3.38, 2.55, .65, b, size=9, line=1.25)
    s.tag(x + .15, 4.1, k, v)
    if i < 3:
        s.arrow(x + 2.84, 2.75, .14, .25)
    x += 2.98
head(s, .8, 4.62, 5.9, 'Model · share of probed partitions with no passing row', 'purple')
sel = [1e-4, 1e-3, 5e-3, 1e-2]
rows = [['Tag selectivity s', '0.01%', '0.1%', '0.5%', '1%'],
        ['K = 16,384 (≈ 610 rows / partition)'] + [pct(empty_share(v, 16384)) for v in sel],
        ['K = 4,096 (≈ 2,441 rows / partition)'] + [pct(empty_share(v, 4096)) for v in sel]]
s.table(.8, 4.92, 5.9, rows, col_widths=[2.7, .8, .8, .8, .8], row_h=.27, size=8.5,
        highlight={(1, 2): 'pale', (1, 1): 'pale', (2, 1): 'pale'}, align=['l', 'r', 'r', 'r', 'r'])
txt(s, .8, 5.78, 5.9, .3, '10M rows, tags independent of the embedding: P(empty) ≈ e^(−s·n). Measure on ADS.', size=8, color='muted')
s.rect(6.95, 4.62, 5.6, 1.45, 'pale_red', 'red', .9)
txt(s, 7.1, 4.68, 5.3, .3, 'Why ADS mining is the bad case [Assumption]', size=10, color='title', bold=True)
bullets(s, 7.1, 5.0, 5.3, 1.05, ['rare scenarios combine several tags → selectivity well below 1%',
                                 'the same tag sets are reused with many query vectors',
                                 'batch jobs: throughput counts more than one query’s p99'], size=9, line=1.3)
end(s, 'At selective filters, most probed partitions return nothing but are still loaded;', 'the filter result is rebuilt every time.')
notes(s, 'Walk through one filtered query. Step 1: the bitmap index for each tag is read (pages cached) and ANDed into an allow-list; that result is not cached, '
      'so the next query with the same tags repeats it. Step 2: IVF picks the nprobes nearest partitions by centroid distance only. Step 3: each probed partition '
      'is loaded while the mask is still being computed; on the normal path it is admitted to the index cache even if none of its rows pass. Step 4: the scan '
      'scores rows and drops those outside the mask. The model shows how often step 3 is wasted: with 10M rows split into 16,384 partitions, about 610 rows each, '
      'and tags independent of the embedding, a 0.1% selective filter leaves about 54% of probed partitions with no passing row. With larger partitions (K=4,096) '
      'the effect is smaller. Real tags are correlated with the embedding, which can raise or lower these numbers, so phase 1 measures them on ADS data.',
      ['IVF', 'PF', 'MODEL'], limits='Model assumes independence between tags and embedding and uniform partition sizes; ADS selectivities are assumptions.')

# ================================================================ 5 KT1 predicate cache
s = new('KT1 · Cache the tag allow-list, not only pages',
        '[Proposal] after Predicate Caching (Schmidt et al., SIGMOD 2024) · key and invalidation follow Lance’s existing RowAddrMaskKey [Source]')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'Allow-list lookup per query')
node(s, 'k1_pred', 1.0, y0 + .1, 2.4, .55, 'Tag predicate', 'night ∧ rain ∧ ¬highway', 'white', size=9)
edge(s, 'k1_e1', [(2.2, y0 + .65), (2.2, y0 + .82)])
node(s, 'k1_norm', 1.0, y0 + .82, 2.4, .55, 'Normalize + hash', 'sorted terms, canonical form', 'white', size=9)
edge(s, 'k1_e2', [(2.2, y0 + 1.37), (2.2, y0 + 1.54)])
node(s, 'k1_look', 1.0, y0 + 1.54, 2.4, .6, 'Lookup, metadata cache', 'version · e-tag · predicate · fragments', 'blue', size=9)
edge(s, 'k1_hit', [(1.6, y0 + 2.14), (1.6, y0 + 2.45)], color='red')
lab(s, 1.65, y0 + 2.2, .5, 'hit', size=8, color='title', align='l', bold=True)
node(s, 'k1_reuse', .98, y0 + 2.45, 1.95, .6, 'Reuse mask', 'no bitmap work', 'accent', size=9)
edge(s, 'k1_miss', [(3.4, y0 + 1.84), (4.25, y0 + 1.84), (4.25, y0 + 2.45)])
lab(s, 3.5, y0 + 1.62, .6, 'miss', size=8, align='l', bold=True)
node(s, 'k1_eval', 3.2, y0 + 2.45, 2.0, .6, 'Bitmap → mask', 'then admit', 'white', size=9)
s.rect(.98, y0 + 3.2, 4.25, .82, 'blue_light', 'blue', .75, dashed=True)
txt(s, 1.1, y0 + 3.22, 4.0, .78, [[('New dataset version', {'bold': True, 'color': 'blue_dark'})],
                                  [('append: evaluate only the new fragments and OR them in', {})],
                                  [('delete: AND the deletion mask Lance already caches', {})]], size=8.5, line=1.25)
head(s, 6.6, 1.45, 5.95, 'Design [Proposal] on existing Lance pieces [Source]', 'blue')
rows = [['Aspect', 'Design', 'Lance piece'],
        ['Key', 'version · e-tag · predicate · fragments', 'RowAddrMaskKey pattern'],
        ['Value', 'RowAddrMask (roaring allow-list)', 'DatasetPreFilter.filtered_ids'],
        ['Append', 'new fragments only, OR into mask', 'fragments are immutable'],
        ['Delete', 'AND the cached deletion mask', 'create_deletion_mask'],
        ['Admission', 'eval time × reuse ÷ mask bytes', 'metadata cache (S3-FIFO)']]
s.table(6.6, 1.82, 5.95, rows, col_widths=[1.05, 2.6, 2.3], row_h=.34, size=8.5, align=['l', 'l', 'l'])
head(s, 6.6, 4.0, 5.95, 'Evidence and what we measure first', 'blue')
rows = [['Item', 'Value'],
        ['Predicate Caching in Redshift [Reported]', 'up to 10× on selected queries'],
        ['Share of query time spent building the mask, ADS', TBD],
        ['Predicate repeat rate in the ADS mining trace', TBD]]
s.table(6.6, 4.32, 5.95, rows, col_widths=[3.6, 2.35], row_h=.29, size=8.5, highlight={(r, 1): 'pale' for r in (2, 3)}, align=['l', 'r'])
s.strip(6.6, 5.55, 5.95, .52, 'First gate:', 'repeat rate and mask-time share on the ADS trace justify a cache; results stay identical.', size=9)
end(s, 'Same tags, same version, same mask:', 'compute it once and reuse it exactly.')
notes(s, 'KT1 caches the evaluated allow-list, not just the bitmap pages. The predicate is normalized (sorted terms, canonical boolean form) and hashed; the key adds '
      'the dataset version, the manifest e-tag and the fragment set, exactly as Lance already does for the deletion mask under RowAddrMaskKey, so a recreated '
      'dataset or a restricted fragment scan never reuses a wrong entry. On a hit the mask is reused with no bitmap work. Following Predicate Caching, entries '
      'are not thrown away when data is appended: Lance fragments are immutable, so a new version only needs the new fragments evaluated and ORed in; deletions '
      'are applied by ANDing the deletion mask Lance already caches. KT1 is exact: the mask is identical to the one computed today. The gate is empirical: '
      'if ADS mining filters rarely repeat, or mask building is a small share of query time, KT1 is dropped and effort moves to KT2.',
      ['PF', 'PC', 'LCACHE'], limits='The 10× figure is from a cloud data warehouse, not a vector search engine; ADS values are TBD.')

# ================================================================ 6 KT2 partition skip
s = new('KT2 · Skip partitions where no row can pass',
        '[Proposal] conservative tag summary per IVF partition, checked before load · adaptive stop as in SieveIVF (on Lance) and VADER (VLDB 2027)')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'Partition × tag summary (K × T bits)')
GX, GY, CELL = 1.6, y0 + .55, .24
present = {0: (1, 4, 6), 1: (0, 2), 2: (1, 3), 3: (1, 4, 7), 4: (1, 2, 4), 5: (5, 6), 6: (3, 4), 7: (1, 4)}
cells = {(r, c): ('red' if c in (1, 4) else 'pink') for r, cs in present.items() for c in cs}
s.grid(GX, GY, 8, 8, cell=CELL, fill='white', cells=cells)
for c in range(8):
    lab(s, GX + c * CELL, GY - .22, CELL, f'T{c + 1}', size=7.5, color='title' if c in (1, 4) else 'muted', bold=c in (1, 4))
for c in (1, 4):
    s.rect(GX + c * CELL - .01, GY - .01, CELL + .02, 8 * CELL + .02, None, 'red', 1.0)
lab(s, GX + .1, GY - .42, 1.8, 'query tags T2 ∧ T5', size=8, color='title', bold=True, align='l')
for r in range(8):
    lab(s, 1.05, GY + r * CELL, .5, f'P{r + 1}', size=7.5, align='r')
    ok = 1 in present[r] and 4 in present[r]
    s.tag(GX + 8 * CELL + .15, GY + r * CELL + .01, 'probe' if ok else 'skip', 'accent' if ok else 'outline', size=7.5)
lab(s, 1.0, GY + 8 * CELL + .04, .7, 'by distance', size=7.5, align='l')
bullets(s, 1.0, y0 + 2.75, 5.2, 1.4, ['Built at index build / optimize: 16,384 × 1,024 tags = 2 MiB [Model]',
                                      'Conservative: a partition is skipped only if a required tag is absent',
                                      'Never skips a partition with a passing row; stale after deletes is still safe',
                                      'NOT / range predicates the summary cannot decide → probe as today'], size=8.5, line=1.3)
head(s, 6.6, 1.45, 5.95, 'Change in prepare_partition', 'purple')
node(s, 'k2_today', 6.6, 1.8, 1.25, .5, 'Today', None, 'neutral', size=9)
node(s, 'k2_t1', 8.0, 1.8, 2.15, .5, 'load ∥ wait mask', 'read + admit always', 'white', size=8.5)
edge(s, 'k2_te', [(10.15, 2.05), (10.4, 2.05)])
node(s, 'k2_t2', 10.4, 1.8, 2.15, .5, 'scan with mask', '0-row: scored, dropped', 'white', size=8.5)
node(s, 'k2_new', 6.6, 2.42, 1.25, .5, 'KT2', None, 'purple', size=9)
node(s, 'k2_n1', 8.0, 2.42, 2.15, .5, 'summary check', 'skip: no read, no admit', 'accent', size=8.5)
edge(s, 'k2_ne', [(10.15, 2.67), (10.4, 2.67)], color='red')
node(s, 'k2_n2', 10.4, 2.42, 2.15, .5, 'load → scan', 'useful partitions only', 'white', size=8.5)
head(s, 6.6, 3.1, 5.95, 'Expected effect [Model] and prior art [Reported]', 'purple')
rows = [['Item', 'Value'],
        ['Probe reads avoidable, 0.1% tags, K = 16,384 [Model]', '≈ ' + pct(empty_share(1e-3, 16384))],
        ['SieveIVF on Lance, 10M / 100M rows [Reported]', '4.1–7.6× / 6.1–8.4×'],
        ['SieveIVF filtered recall loss [Reported]', '0.03–2.29 pp'],
        ['VADER vs best baseline [Reported]', 'up to 53% faster'],
        ['Lakehouse: files pruned before ANN [Reported]', '355 of 444']]
s.table(6.6, 3.42, 5.95, rows, col_widths=[4.25, 1.7], row_h=.29, size=8.5, highlight={(1, 1): 'pale'}, align=['l', 'r'])
s.strip(6.6, 5.3, 5.95, .75, 'First gate:', 'recall@10 equal to today at the same useful-probe count, and partition reads per query fall as the model predicts on ADS tags.', size=9)
end(s, 'Check the summary before the read:', 'empty partitions cost nothing and free their probe slot.')
notes(s, 'KT2 stores, per IVF partition, one presence bit per tag (T1..T8 in the figure; up to about 1,024 tags in practice, 2 MiB for 16,384 partitions). '
      'Before loading a probed partition the query ANDs the bits of its required tags; if any required tag is absent from the partition, no row can pass, '
      'so the partition is skipped without a read and without cache admission, and the next useful partition takes its probe slot. The check is conservative: '
      'presence of every tag in the partition is necessary but not sufficient for a passing row, so KT2 never skips a partition that holds one; NOT and range '
      'predicates fall back to today\'s path. Deletions make the summary a superset, which stays safe; unindexed appended rows are searched exactly by Lance '
      'already. In the code this moves a check ahead of the try_join in prepare_partition. Prior art: SieveIVF, built in Lance, stops probing after W empty '
      'partitions and reports 4-8x speedups with small recall loss; VADER stops at a declared recall target; the IBM lakehouse work prunes whole files first.',
      ['IVF', 'MODEL', 'SIEVE', 'VADER', 'LAKE'], limits='Speedups are reported on other workloads; the 54% figure is a model under independence.')

# ================================================================ 7 KT3 batched mining
s = new('KT3 · Batch mining queries that share partitions',
        '[Proposal] partition-major execution over a window of queries · after SieveIVF continuous batching, CaGR-RAG and CALL query grouping')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'Queries × probed partitions (after KT2 skip)')
probes = {0: (0, 2, 5), 1: (0, 2, 6), 2: (2, 5, 8), 3: (0, 5, 6), 4: (2, 6, 8), 5: (0, 5, 9)}
QX, QY, QC = 1.6, y0 + .45, .3
s.grid(QX, QY, 6, 10, cell=QC, fill='white', cells={(q, p): 'red' for q, ps in probes.items() for p in ps})
for p in range(10):
    lab(s, QX + p * QC, QY - .22, QC, f'P{p + 1}', size=7.5)
for q in range(6):
    lab(s, 1.05, QY + q * QC, .5, f'Q{q + 1}', size=7.5, align='r')
total = sum(len(v) for v in probes.values())
distinct = len({p for v in probes.values() for p in v})
s.stat(1.0, y0 + 2.45, 2.3, f'{total}', 'loads, query by query', size=22, label_size=8.5)
s.stat(3.3, y0 + 2.45, 2.3, f'{distinct}', 'loads, partition-major', size=22, label_size=8.5)
txt(s, 1.0, y0 + 3.35, 5.2, .8, f'Illustrative window of 6 queries: {total} partition reads become {distinct}; each loaded partition is scanned once '
    'for every query that probes it, with each query’s own mask (KT1).', size=8.5, line=1.3)
head(s, 6.6, 1.45, 5.95, 'How a batch runs [Proposal]', 'green')
bullets(s, 6.75, 1.8, 5.8, 1.5, ['collect a window of (vector, tags) mining queries',
                                 'route all queries, skip empty partitions (KT2), union the rest',
                                 'load each partition once, scan it for every query that needs it',
                                 'batch loads use their own admission class: no online-cache pollution'], size=9, line=1.35)
head(s, 6.6, 3.0, 5.95, 'Prior art [Reported]', 'green')
rows = [['System', 'Idea', 'Reported'],
        ['SieveIVF (on Lance)', 'group ready queries by partition', 'part of 4.1–8.4×'],
        ['CaGR-RAG', 'group queries by shared clusters', 'p99 up to −51.55%'],
        ['CALL', 'context-aware grouping', 'p99 up to −33%']]
s.table(6.6, 3.32, 5.95, rows, col_widths=[1.75, 2.45, 1.75], row_h=.3, size=8.5, align=['l', 'l', 'r'])
s.strip(6.6, 4.85, 5.95, .95, 'First gate:', 'mining throughput up at equal recall@10, and online query p99 unchanged while a batch runs.', size=9)
end(s, 'Mining is batch work:', 'read a shared partition once, not once per query.')
notes(s, 'ADS mining issues many filtered queries, for example one per seed clip, and is judged on throughput. KT3 collects a window of queries, routes all of them, '
      'drops partitions skipped by KT2 and loads each remaining partition once, scanning it for every query that probes it. In the illustrative window of six '
      'queries, ' + f'{total} query-by-query reads become {distinct}. ' + 'Loads made for a batch use a separate admission class so a large batch cannot evict the partitions online '
      'queries depend on. Prior art: SieveIVF uses continuous batching by partition inside Lance; CaGR-RAG and CALL group queries by shared clusters and report '
      'lower p99 and higher hit ratios. The counts on this slide are an illustration, not a measurement.',
      ['SIEVE', 'GROUP', 'LCACHE'], limits='Load counts are illustrative; real overlap depends on the ADS query mix.')

# ================================================================ 8 summary + roadmap
s = new('Summary · recall-safe by design, ADS trace next',
        'Three key technologies · gaps verified in Lance source · [Proposal] sizes S ≈ 1–2, M ≈ 3–5, L ≈ 6–10 engineer-weeks')
cols = [('KT1 · Predicate cache', 'blue', 'blue_light', TBD, 'mask time saved on ADS', ['reported up to 10× (Redshift)', 'exact: identical results']),
        ('KT2 · Partition skip', 'purple', 'purple_light', '≈ 54%', 'probe reads avoidable [Model]', ['at 0.1% tags, K = 16,384', 'reported 4–8× (SieveIVF)']),
        ('KT3 · Batched mining', 'green', 'green_light', TBD, 'mining throughput on ADS', ['reported p99 −33 to −52%', 'one read, many queries'])]
x = .8
for h, col, fill, val, lbl, lines in cols:
    s.rect(x, 1.45, 3.8, 1.5, 'white', col, 1.0, dashed=val == TBD)
    s.rect(x, 1.45, 3.8, .38, fill, None)
    txt(s, x + .15, 1.45, 3.5, .38, h, size=10.5, color='ink', bold=True, anchor='ctr')
    s.stat(x + .15, 1.92, 1.6, val, lbl, size=26, label_size=8.5)
    bullets(s, x + 1.8, 2.02, 1.95, 1.2, lines, size=8.5, line=1.3)
    x += 3.97
s.section(.8, 3.2, 11.7, 'ROADMAP · EACH PHASE HAS A GATE')
ph = [('1 · ADS trace + baseline', 'S', 'tag selectivity, repeat rate, reads per query', 'accent'),
      ('2 · KT2 partition skip', 'M', 'tag summary, check before load', 'outline'),
      ('3 · KT1 predicate cache', 'S', 'allow-list in metadata cache', 'outline'),
      ('4 · KT3 batching, upstream', 'M', 'batched search, Lance PRs', 'dashed')]
x = .8
for i, (h, sz, b, v) in enumerate(ph):
    w = 2.75
    fill = 'red' if v == 'accent' else 'white'
    s.rect(x, 3.55, w, .5, fill, 'red', 1.0, dashed=v == 'dashed')
    txt(s, x + .08, 3.55, w - .16, .5, f'{h} · {sz}', size=9, color='white' if v == 'accent' else 'title', bold=True, anchor='ctr')
    txt(s, x + .05, 4.12, w - .1, .5, b, size=8.5, line=1.2)
    if i < 3:
        s.arrow(x + w + .02, 3.69, .1, .22)
    x += 3.0
s.strip(.8, 4.75, 11.75, .62, 'Recall-safe by default:', 'KT1 is exact, KT2’s summary never skips a partition with a passing row, adaptive stopping is opt-in; '
        'a failed gate keeps today’s path.', size=9)
end(s, 'The gaps are in the code today; ADS data decides the size of the gain.', 'Phase 1 fills every TBD.')
notes(s, 'Summary. KT1 caches the tag allow-list per predicate and dataset version and is exact. KT2 checks a conservative per-partition tag summary before '
      'loading, so empty partitions cost no read, no cache space and no probe slot; under independence about half of probed partitions are empty at 0.1% '
      'selectivity with 16,384 partitions. KT3 batches mining queries so each shared partition is read once. Phase 1 collects an ADS mining trace (tags, '
      'selectivity, repeat rate, partition reads per query, recall) on IVF_RQ; KT2 comes first because it attacks I/O, then KT1, then KT3 and upstream pull '
      'requests. Sizes are rough estimates, not commitments.',
      ['DOC', 'MODEL', 'PC', 'SIEVE', 'GROUP'], limits='No ADS measurement yet; sizes and staffing unconfirmed.')

finish(g, 'hybrid-v1')

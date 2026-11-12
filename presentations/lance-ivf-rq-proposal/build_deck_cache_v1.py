#!/usr/bin/env python3
"""Track 6 v1: millisecond serving of multimodal Lance data (objects + vectors + tags) through a tiered cache, ADS stage (6).

Same form as build_deck_v8.py. Key technologies:
  KT1 tiered byte cache under the Lance reader: process RAM -> UB / Lingqu pooled memory -> local NVMe -> OBS
  KT2 prefetch driven by the training sampler (known order) plus retrieval results from stage (4) as hints
  KT3 cost-aware admission and shared placement across jobs and stages (U = F * C * R / S)
Slides: 1 title · 2 ADS pipeline (6) · 3 Lance read-path map · 4 latency model + what Lance caches · 5 KT1 · 6 KT2 · 7 KT3 · 8 summary
No ADS or UB measurements exist yet: UB numbers are TBD; the bandwidth table is a labeled model.
Run: uv run --no-project --with python-pptx python build_deck_cache_v1.py
"""
from ads_track_common import load, ads_slide, finish

LANCE = 'lance-format/lance main @ 04e619963'
EXTRA = {
    'BLOB': f'{LANCE}: rust/lance/src/dataset/blob/file.rs (BlobFile: read_range / read_ranges, DEFAULT_READ_BUFFER_SIZE = 512 KiB per handle); '
            'rust/lance/src/dataset.rs take_blobs / take_blobs_by_indices.',
    'PAGE': f'{LANCE}: rust/lance-encoding/src/encodings/logical/primitive.rs PageDataCacheKey / FieldDataCacheKey (page metadata, not data bytes); '
            'rust/lance-io/src/spill.rs LocalSpillStore (build-time scratch, not a cache); rust/lance-io/src/uring.rs (io_uring readers for NVMe).',
    'HOOKS': f'{LANCE}: rust/lance-io/src/object_store.rs L292 WrappingObjectStore; rust/lance-core/src/cache/backend.rs CacheTier::Local, peek_tier, '
             'get_without_promotion, get_rows; codec.rs (experimental envelope, decode error = miss); registry.rs register_backend; '
             'rust/lance-index/src/vector/storage.rs OriginLatencyClass + LANCE_RQ_ORIGIN_LATENCY (doc: a process whose store is wrapped in a local cache declares Low); '
             'rust/lance/src/session.rs with_index_origin_latency.',
    'BW': 'Our model: transfer time = bytes / aggregate bandwidth B; 4 GB = 4,000 MB, 64 MiB = 64/1024 GB. Excludes first-byte latency, decode, queueing. '
          'UB / Lingqu, NVMe and OBS latency and bandwidth for ADS nodes are TBD (phase 1).',
    'SEN': 'Desai et al., Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca, USENIX FAST 2026, '
           'https://www.usenix.org/conference/fast26/presentation/desai (-45.23% makespan vs PyTorch, up to 3.45× throughput vs next best loader).',
    'LAKE': 'Jain, Griffin, Zawad, Filtered Vector Search in a Disaggregated Lakehouse, arXiv:2608.05441 (rendezvous-hashed per-file cache over object storage).',
    'LRA': 'Pace et al., Lance: Efficient Random Access in Columnar Storage through Adaptive Structural Encodings, arXiv:2504.15247 (2025).',
    'DL': 'Mohan et al., Analyzing and Mitigating Data Stalls in DNN Training (CoorDL), PVLDB 14(5), 2021; Khan et al., SHADE: Enable Fundamental '
          'Cacheability for Distributed Deep Learning Training, USENIX FAST 2023; Pinto et al., Hoard: A Distributed Data Caching System to Accelerate '
          'Deep Learning Training on the Cloud, arXiv:1812.00669. Ideas only; no numbers quoted.',
    'ACC': 'Boosting Cache Performance by Access Time Measurements, ACM Transactions on Storage, https://doi.org/10.1145/3572778',
    'DHNSW': 'Efficient Vector Search on Disaggregated Memory with d-HNSW, arXiv:2505.11783 (compute nodes cache representatives, data in a shared memory pool).',
}
g = load('ADS · Lance Multimodal Cache', EXTRA, 'ads-lance-multimodal-cache-v1.pptx')
globals().update({k: v for k, v in g.items() if not k.startswith('__')})

# ================================================================ 1 title
s = D.title_slide('Millisecond Data Serving from Lance',
                  'UB-pool tiered cache · Sampler prefetch · Cost-aware placement',
                  presenters=['Research proposal v1 · 11 November 2026'], team='Vector Index Optimization')
notes(s, 'Proposal to serve everything stored in the Lance data lake (original objects such as video and point clouds, vectors, tags and index '
      'partitions) to training and mining at millisecond latency. The ADS framework sets the target: pulling a 4 GB clip from OBS must drop to ms level. '
      'Three key technologies: KT1 a tiered byte cache under the Lance reader with UB / Lingqu pooled memory as the shared tier, KT2 prefetch from the '
      'known training order, KT3 one cost-aware policy for admission and placement. Gaps are verified in Lance source; UB numbers are measured in phase 1.',
      ['ADS', 'DOC'])

# ================================================================ 2 ADS pipeline
ads_slide(g, 'We speed up ADS stage (6): serving the data lake',
          '[Company context] Redrawn from ADS-PLAN slide 1; target is (6) Data cache in Lance format: a 4 GB clip from OBS at ms level.',
          '(6)', (10.14, 2.02, 1.38, .8, 10.83, 2.82, 2.97),
          ['KEY TECHNOLOGIES', 'KT1 tiered cache', 'KT2 prefetch', 'KT3 placement'],
          'Training waits on the lake: a 4 GB clip from OBS must reach ms level.', 'All three key technologies sit in this box.',
          'ADS pipeline from ADS-PLAN slide 1. Stage (6) caches Lance-format data for the dataload step that feeds (7) video decode and model training. '
          'The ADS framework sets the target: pulling a single 4 GB clip from OBS must drop to millisecond level, with Lingqu (UB) communication acceleration. '
          'The same lake also serves (4) mining and (5) ground truth, so a cache here serves objects, vectors and tags together.')

# ================================================================ 3 project map
s = new('Three key technologies in the Lance read path',
        'Consumers read objects, vectors and tags through Lance; each key technology (KT) at its integration point · gaps [Source] · dashed = proposed')
s.section(.8, 1.42, 8.6, 'CONSUMERS → LANCE READER → CACHE TIERS → ORIGIN')
cons = [('(6)→(7) Training loader', 'clips, frames, labels'), ('(4) Mining', 'vectors + tags'), ('(5) GT / OCC', 'clips, 3D')]
for i, (a, b) in enumerate(cons):
    node(s, f'c{i}', .8 + i * 2.9, 1.78, 2.7, .55, a, b, 'neutral', size=9.5)
    edge(s, f'ce{i}', [(2.15 + i * 2.9, 2.33), (2.15 + i * 2.9, 2.5)])
node(s, 'reader', .8, 2.5, 8.5, .42, 'Lance reader · take / take_blobs / read_range / vector search', None, 'white', size=9.5)
edge(s, 'r_down', [(5.05, 2.92), (5.05, 3.12)])
s.rect(.8, 3.12, 1.35, 1.95, 'purple_light', 'purple', .9, dashed=True)
txt(s, .88, 3.16, 1.22, .45, 'KT2 · Prefetch', size=9.5, color='purple_dark', bold=True)
txt(s, .88, 3.5, 1.22, 1.5, ['sampler order', '(4) results as hints', 'first chunk first'], size=8.5)
edge(s, 'kt2', [(2.15, 4.0), (2.4, 4.0)], color='purple', dashed=True)
s.rect(2.4, 3.12, 4.6, 1.95, 'blue_light', 'blue', .9, dashed=True)
txt(s, 2.52, 3.15, 4.4, .26, 'KT1 · Tiered byte cache', size=10, color='blue_dark', bold=True)
tiers = [('L1 · process RAM', 'LanceCache: index + metadata [Source]', 'white'),
         ('L2 · UB / Lingqu pooled memory', 'shared by all jobs and stages [Proposal]', 'accent'),
         ('L3 · local NVMe', 'node-local, disposable [Proposal]', 'dashed')]
for i, (a, b, v) in enumerate(tiers):
    node(s, f't{i}', 2.55, 3.45 + i * .53, 4.3, .46, a, b, v, size=9, sub_size=7.5)
s.rect(7.25, 3.12, 2.05, 1.95, 'green_light', 'green', .9, dashed=True)
txt(s, 7.33, 3.16, 1.92, .45, 'KT3 · Policy', size=9.5, color='green_dark', bold=True)
txt(s, 7.33, 3.5, 1.92, 1.5, ['U = F·C·R / S', 'admit + evict', 'placement on pool', 'encoded or decoded'], size=8.5)
edge(s, 'kt3', [(7.25, 4.0), (7.0, 4.0)], color='green', dashed=True)
edge(s, 'origin', [(4.7, 5.07), (4.7, 5.27)])
node(s, 'obs', 2.4, 5.27, 4.6, .5, 'OBS · Lance files (origin, source of truth)', None, 'dark', size=9.5)
txt(s, .8, 5.8, 8.5, .28, 'Every tier is a disposable copy of immutable Lance files: a miss or a lost node falls back to OBS', size=8, color='muted', align='ctr')
s.rule(9.55, 1.45, 4.55, vertical=True)
s.section(9.8, 1.42, 2.8, 'STATUS · 11 NOV 2026')
st = [('KT1 tiered cache', ['Gap verified: no data-byte cache', 'Hooks exist: object-store wrapper,', 'CacheTier::Local (no backend)'], 'blue', 1.45),
      ('KT2 prefetch', ['Training order known ahead [Assumption]', 'Blob range reads exist [Source]'], 'purple', 1.2),
      ('KT3 policy', ['Baselines in Lance: S3-FIFO, TinyLFU', 'OriginLatencyClass: RQ switch only'], 'green', 1.3)]
y = 1.78
for h, lines, col, dy in st:
    s.rect(9.8, y + .04, .06, .22, col, None)
    txt(s, 9.95, y, 2.6, .28, h, size=10, color='ink', bold=True, anchor='ctr')
    bullets(s, 9.95, y + .3, 2.6, 1.2, lines, size=8.5, line=1.25)
    y += dy
end(s, 'KT1 adds the tiers; KT2 fills them ahead of need; KT3 decides what stays.', 'Each ships separately.')
notes(s, 'All consumers read through the Lance reader: the training loader takes clips and labels (blobs plus columns), mining reads vectors, tags and index '
      'partitions, ground truth reads clips and 3D data. Today Lance caches index partitions and metadata in process RAM (L1); column and blob bytes are not '
      'cached at all, and the CacheBackend trait has a Local tier with no implementation. KT1 adds two tiers below L1: UB / Lingqu pooled memory shared by all '
      'jobs and stages (L2), and node-local NVMe (L3). KT2 fills L2 before data is needed, using the training sampler order and the clips selected by mining. '
      'KT3 is one admission, eviction and placement policy for all object classes. OBS stays the only source of truth.',
      ['LCACHE', 'BLOB', 'PAGE', 'HOOKS'], limits='UB pool availability is an assumption from ADS; no UB measurement yet.')

# ================================================================ 4 latency model + what Lance caches
s = new('"ms" needs a close hit and a partial read',
        '[Model] transfer time at aggregate bandwidth B · [Source] what Lance caches today · UB / NVMe / OBS figures on ADS nodes are TBD')
head(s, .8, 1.45, 5.5, 'Model · transfer time = bytes ÷ aggregate bandwidth', 'blue')
rows = [['Bandwidth B', 'Whole 4 GB clip', '64 MiB window']]
for b in (1, 10, 50, 100):
    rows.append([f'{b} GB/s', f'{4000 / b:,.0f} ms', f'{64 / 1024 * 1000 / b:.1f} ms' if b > 1 else f'{64 / 1024 * 1000 / b:.0f} ms'])
s.table(.8, 1.8, 5.5, rows, col_widths=[1.6, 1.95, 1.95], row_h=.32, size=9,
        highlight={(r, 2): 'pale' for r in (2, 3, 4)}, align=['l', 'r', 'r'])
txt(s, .8, 3.45, 5.5, .45, 'Transfer only; first-byte latency, decode and queueing come on top. A whole clip stays above 40 ms even at 100 GB/s.',
    size=8, color='muted')
y0 = s.panel(.8, 4.0, 5.5, 1.95, 'Two levers')
node(s, 'lev1', .95, y0 + .1, 2.55, .95, 'Read less', 'the range a step needs: 64× fewer bytes than the clip', 'white', size=9, sub_size=8)
node(s, 'lev2', 3.6, y0 + .1, 2.55, .95, 'Read closer', 'from L2 UB pool, not OBS: first-byte latency TBD', 'accent', size=9, sub_size=8)
head(s, 6.6, 1.45, 5.95, 'What Lance caches today [Source]', 'red')
rows = [['Data', 'Cached?', 'Where / how'],
        ['IVF partitions (RQ codes)', 'yes', 'index cache 6 GiB, S3-FIFO'],
        ['Manifests, file + page metadata', 'yes', 'metadata cache 1 GiB'],
        ['Deletion masks', 'yes', 'metadata cache, per version'],
        ['Column + blob bytes (video, lidar)', 'no', '512 KiB read-ahead per BlobFile'],
        ['Shared or persistent tier', 'no', 'CacheTier::Local, no backend'],
        ['Spill store', 'scratch', 'index build only, not a cache']]
s.table(6.6, 1.82, 5.95, rows, col_widths=[2.45, .85, 2.65], row_h=.34, size=8.5,
        highlight={(r, c): 'pale' for r in (4, 5) for c in (1, 2)}, align=['l', 'ctr', 'l'])
s.strip(6.6, 4.4, 5.95, 1.55, 'The gap for ADS:', 'the biggest bytes, original objects, have no cache in Lance, and nothing is shared between processes, '
        'jobs or pipeline stages. Every read of a clip goes back to OBS.', size=9.5)
end(s, 'Milliseconds come from reading only the needed range from a close tier;', 'Lance has neither tier for object bytes today.')
notes(s, 'The ADS target is a 4 GB clip at millisecond level. Physics first: moving a whole 4 GB clip takes 4,000 ms at 1 GB/s, 400 ms at 10 GB/s and still '
      '40 ms at 100 GB/s, before first-byte latency, decode and queueing. Millisecond serving therefore means time to first useful batch: read only the window '
      'a training step needs (64 MiB here, 64 times less than the clip; Lance supports row and byte-range reads), and read it from a tier close to the GPU node. '
      'The right table is what Lance caches today, read from source: index partitions and metadata in process RAM, deletion masks; but no cache for column or '
      'blob bytes (a BlobFile only has a 512 KiB read-ahead buffer per handle), and no shared or persistent tier (the trait defines CacheTier::Local but no backend '
      'implements it). The spill store is build-time scratch space. UB / Lingqu, NVMe and OBS numbers on ADS nodes are measured in phase 1.',
      ['BW', 'LCACHE', 'BLOB', 'PAGE', 'HOOKS'], limits='Bandwidth rows are a model, not measurements of any ADS tier; the 64 MiB window is illustrative.')

# ================================================================ 5 KT1 tiered cache
s = new('KT1 · A tiered byte cache: RAM, UB pool, NVMe',
        '[Proposal] read-through cache of immutable Lance file blocks under the reader · integration points exist in Lance [Source]')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'One read, four tiers')
node(s, 'k1_req', 1.0, y0 + .1, 3.2, .5, 'read_range(clip.lance, 37–41 MiB)', None, 'white', size=8.5)
edge(s, 'k1_e0', [(2.6, y0 + .6), (2.6, y0 + .74)])
s.grid(1.0, y0 + .74, 1, 8, cell=.4, fill='white', cells={(0, 2): 'red', (0, 3): 'red'})
lab(s, 4.25, y0 + .82, 1.9, '4 MiB aligned blocks', size=8, align='l')
tiers = [('L1 · process RAM', 'ns–µs · per process', 'white'), ('L2 · UB / Lingqu pool', 'shared · latency TBD', 'accent'),
         ('L3 · local NVMe', 'node-local · latency TBD', 'dashed'), ('OBS origin', 'source of truth', 'dark')]
for i, (a, b, v) in enumerate(tiers):
    yy = y0 + 1.35 + i * .66
    node(s, f'k1_t{i}', 1.0, yy, 2.9, .5, a, b, v, size=9, sub_size=7.5)
    if i < 3:
        s.route([(1.7, yy + .5), (1.7, yy + .66)], color='muted')
        s.route([(3.2, yy + .66), (3.2, yy + .5)], color='red')
txt(s, 4.1, y0 + 1.4, 2.1, 2.5, [[('Key', {'bold': True, 'color': 'ink'})], [('store · path · e-tag · block', {})],
                                  [('No invalidation', {'bold': True, 'color': 'ink'})], [('Lance files are immutable; old versions age out', {})],
                                  [('Failure', {'bold': True, 'color': 'ink'})], [('lost node = refill from OBS', {})],
                                  [('Arrows', {'bold': True, 'color': 'ink'})], [('grey ↓ miss · red ↑ fill', {})]], size=8.5, line=1.25)
head(s, 6.6, 1.45, 5.95, 'Lance integration points [Source]', 'blue')
rows = [['Hook', 'Use in KT1'],
        ['WrappingObjectStore', 'byte cache for data + index files'],
        ['CacheTier::Local, get_without_promotion', 'index entries below RAM, scans skip promotion'],
        ['CacheCodec (experimental)', 'serialize entries; decode error = miss'],
        ['with_index_origin_latency(Low)', 'RQ reader stops paying for slow origin']]
s.table(6.6, 1.82, 5.95, rows, col_widths=[2.75, 3.2], row_h=.36, size=8.5, align=['l', 'l'])
head(s, 6.6, 3.75, 5.95, 'Prior art [Reported]', 'blue')
rows = [['System', 'Reported'],
        ['Seneca (FAST 2026), cache for ML data loading', '−45% makespan, 3.45× throughput'],
        ['Lakehouse ANN, per-file cache over object store', 'rendezvous-hashed placement']]
s.table(6.6, 4.07, 5.95, rows, col_widths=[3.45, 2.5], row_h=.3, size=8.5, align=['l', 'r'])
s.strip(6.6, 5.2, 5.95, .85, 'First gate:', 'p50 time-to-first-batch for a 4 GB clip served from L2 meets the ADS ms target (value set with ADS); OBS bytes per epoch fall.', size=9)
end(s, 'Immutable files make the cache simple:', 'no invalidation, and any tier can be lost safely.')
notes(s, 'KT1 caches fixed-size aligned blocks of Lance files (4 MiB in the figure) keyed by store, path, e-tag and block number. A read first checks process RAM, '
      'then the UB / Lingqu pooled memory shared by every process on the pod, then node-local NVMe, then OBS; a miss fills the closer tiers on the way back. '
      'Because Lance data and index files are immutable and each dataset version names its files, there is no invalidation: blocks of files no version uses '
      'simply age out. Losing a node only costs a refill. Lance already has the hooks: WrappingObjectStore wraps every object-store read, the CacheBackend trait '
      'has a Local tier and a no-promotion read for scans, CacheCodec serializes index entries (experimental, decode failure becomes a miss), and the source '
      'comment on OriginLatencyClass says a process with a locally cached store should declare Low so the RaBitQ reader drops its high-latency policies.',
      ['HOOKS', 'BLOB', 'SEN', 'LAKE', 'LRA'], limits='Tier latencies are placeholders; block size is a parameter to tune; Seneca numbers are on its own workloads.')

# ================================================================ 6 KT2 prefetch
s = new('KT2 · Prefetch from the known training order',
        '[Proposal] the sampler’s order is fixed per epoch, so the next clips are known · timeline illustrative, not measured')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'Training step timeline (illustrative)')
lab(s, .95, y0 + .2, .9, 'Today', size=8.5, color='ink', bold=True, align='l')
x0 = 1.8
for j in range(3):
    s.span(x0, y0 + .12, .75, .32, 'grey_bar', 'fetch', size=7.5, color='white')
    s.span(x0 + .75, y0 + .12, .45, .32, 'pink', 'train', size=7.5)
    x0 += 1.2
lab(s, .95, y0 + .72, .9, 'KT2', size=8.5, color='title', bold=True, align='l')
x0 = 1.8
s.span(x0, y0 + .64, .75, .32, 'grey_bar', 'fetch', size=7.5, color='white')
for j in range(4):
    s.span(x0 + .75 + j * .47, y0 + .64, .45, .32, 'pink', 'train', size=7.5)
    if j < 3:
        s.span(x0 + .75 + j * .47, y0 + 1.0, .45, .14, 'red', None)
lab(s, x0 + .75, y0 + 1.16, 1.9, 'next clips prefetched into L2', size=7.5, color='title', align='l')
fl = [('Sampler', 'seeded epoch order'), ('Next W clips', 'lookahead window'), ('Lance take plan', 'rows → byte ranges'), ('L2 prefetch queue', 'first chunk first')]
for i, (a, b) in enumerate(fl):
    yy = y0 + 1.6 + i * .62
    node(s, f'k2_f{i}', 1.0, yy, 2.6, .5, a, b, 'accent' if i == 3 else 'white', size=9, sub_size=7.5)
    if i < 3:
        edge(s, f'k2_fe{i}', [(2.3, yy + .5), (2.3, yy + .62)])
node(s, 'k2_hint', 3.85, y0 + 2.85, 2.3, .55, '(4) mining results', 'clips chosen for training', 'purple', size=9, sub_size=7.5)
edge(s, 'k2_he', [(5.0, y0 + 3.4), (5.0, y0 + 3.73), (3.6, y0 + 3.73)], color='purple', dashed=True)
head(s, 6.6, 1.45, 5.95, 'Prefetch policy [Proposal]', 'purple')
bullets(s, 6.75, 1.8, 5.8, 1.7, ['lookahead W clips, sized to the L2 budget per job',
                                 'first chunk of each clip first: time to first frame',
                                 'priority: training > mining hints > background warm-up',
                                 'cancel on epoch change or job stop',
                                 'one prefetch serves every job reading the same clip'], size=9, line=1.35)
head(s, 6.6, 3.55, 5.95, 'Prior art (idea, numbers not transferred)', 'purple')
rows = [['Work', 'Idea we reuse'],
        ['Seneca (FAST 2026)', 'serve cached samples first across jobs'],
        ['CoorDL (VLDB 2021)', 'coordinate data prep across jobs'],
        ['SHADE (FAST 2023)', 'cache what matters most to training']]
s.table(6.6, 3.87, 5.95, rows, col_widths=[2.2, 3.75], row_h=.3, size=8.5, align=['l', 'l'])
s.strip(6.6, 5.2, 5.95, .85, 'First gate:', 'GPU data-stall time per step near zero at the target batch rate, with prefetched bytes ≤ 1.2× bytes consumed.', size=9)
end(s, 'The order is known before the step runs:', 'fetch during compute, not after it.')
notes(s, 'The training loader shuffles with a seed per epoch, so which clips come next is known before they are needed. KT2 turns the next W clips into Lance '
      'take plans (row ids to byte ranges, including blob ranges) and queues them into L2, first chunk of each clip first, so a step starts as soon as its '
      'first window is close. Mining in stage (4) produces the clip lists that later become training data, so its results are prefetch hints of lower priority. '
      'Prefetch is cancelled when an epoch or job ends, and one prefetched copy serves every job that reads the clip. The timeline is an illustration of '
      'overlap, not a measurement. Seneca, CoorDL and SHADE are cited for their ideas only.',
      ['SEN', 'DL', 'BLOB'], limits='Assumes the loader exposes its order (seeded shuffle); timeline illustrative; 1.2× threshold is a proposal.')

# ================================================================ 7 KT3 policy
s = new('KT3 · One policy keeps valuable bytes close',
        '[Proposal] utility per byte decides admission, eviction and tier · baselines are the S3-FIFO and TinyLFU backends Lance already has [Source]')
y0 = s.panel(.8, 1.45, 5.55, 4.6, 'Value per byte, one score for every object class')
txt(s, 1.0, y0 + .1, 5.2, .45, 'U(x) = F̂(x) · Ĉ(x) · R̂(x) ÷ S(x)', size=16, color='title', bold=True, align='ctr', anchor='ctr')
txt(s, 1.0, y0 + .58, 5.2, .3, 'F̂ recent frequency · Ĉ load cost (tier, decode) · R̂ reuse ahead (KT2 hints) · S bytes', size=8, color='muted', align='ctr')
rows = [['Object', 'Size S', 'Load cost C', 'Reuse R'],
        ['Clip byte range', 'large', 'OBS read', 'epochs, jobs'],
        ['Decoded frame window', 'larger', 'decode (⑦ 1.5 s)', 'hot clips'],
        ['IVF_RQ partition', 'medium', 'read + decode', 'queries'],
        ['Tag allow-list', 'small', 'bitmap AND', 'repeat filters'],
        ['Vector / tag columns', 'small', 'OBS read', 'mining']]
s.table(1.0, y0 + 1.0, 5.2, rows, col_widths=[1.65, .95, 1.4, 1.2], row_h=.32, size=8.5, align=['l', 'l', 'l', 'l'])
txt(s, 1.0, y0 + 3.0, 5.2, 1.0, 'Encoded or decoded: cache the decoded window only when its reuse pays for its extra bytes; otherwise keep the clip '
    'encoded and decode again (links to ⑦ remote decoding).', size=8.5, line=1.3)
head(s, 6.6, 1.45, 5.95, 'Shared placement on the UB pool', 'green')
for i in range(4):
    node(s, f'k3_n{i}', 6.65 + i * 1.47, 1.82, 1.35, .5, f'pool node {i + 1}', None, 'white' if i != 1 else 'accent', size=8.5)
for i in range(3):
    node(s, f'k3_g{i}', 6.95 + i * 1.95, 2.75, 1.6, .45, f'GPU node {chr(65 + i)}', None, 'neutral', size=8.5)
    s.route([(7.75 + i * 1.95, 2.75), (7.75 + i * 1.95, 2.55), (8.68, 2.55), (8.68, 2.32)], color='red')
txt(s, 6.6, 3.28, 5.95, .3, 'clip 42 hashes to one owner (rendezvous); every job and stage reads that copy over UB', size=8, color='muted', align='ctr')
head(s, 6.6, 3.65, 5.95, 'Compared with what Lance has', 'green')
rows = [['Policy', 'Status', 'Role'],
        ['S3-FIFO (QuickCacheBackend)', 'default [Source]', 'baseline'],
        ['TinyLFU (MokaCacheBackend)', 'built-in [Source]', 'baseline'],
        ['OriginLatencyClass', 'RQ switch [Source]', 'becomes cost input Ĉ']]
s.table(6.6, 3.97, 5.95, rows, col_widths=[2.6, 1.75, 1.6], row_h=.29, size=8.5, align=['l', 'l', 'l'])
s.strip(6.6, 5.25, 5.95, .8, 'First gate:', 'OBS bytes and refills per epoch fall vs S3-FIFO at equal pool size; no job slows by more than 5%.', size=9)
end(s, 'One copy on the pool, chosen by value per byte,', 'serves every job and stage.')
notes(s, 'KT3 replaces per-process, frequency-only eviction with one score: value per byte, frequency times load cost times expected reuse, divided by size. '
      'Load cost comes from the tier a miss would hit and from decode work; expected reuse comes from KT2 lookahead and from mining hints. The same score ranks '
      'clip byte ranges, decoded frame windows, IVF_RQ partitions, tag allow-lists from the hybrid retrieval track and vector or tag columns, and decides whether '
      'a hot clip is worth keeping decoded. Placement uses rendezvous hashing, so each block has one owner in the UB pool and all jobs and stages read the same '
      'copy, which removes the repeated loading across pipelines named in the ADS framework. Baselines are the two policies Lance already ships, S3-FIFO and '
      'TinyLFU, at equal memory; OriginLatencyClass becomes the cost input instead of an on/off switch.',
      ['ACC', 'LCACHE', 'S3F', 'TLFU', 'HOOKS', 'DHNSW'], limits='Size and cost columns are qualitative; the 5% threshold is a proposal.')

# ================================================================ 8 summary + roadmap
s = new('Summary · hooks exist in Lance, UB budget next',
        'Three key technologies · gaps verified in Lance source · [Proposal] sizes S ≈ 1–2, M ≈ 3–5, L ≈ 6–10 engineer-weeks')
cols = [('KT1 · Tiered cache', 'blue', 'blue_light', '64×', 'fewer bytes: window vs clip [Model]', ['L2 UB pool + L3 NVMe', 'no format change']),
        ('KT2 · Prefetch', 'purple', 'purple_light', TBD, 'data-stall time per step', ['training order known ahead', '(4) results as hints']),
        ('KT3 · Policy', 'green', 'green_light', TBD, 'OBS bytes per epoch', ['vs S3-FIFO and TinyLFU', 'one copy for all jobs'])]
x = .8
for h, col, fill, val, lbl, lines in cols:
    s.rect(x, 1.45, 3.8, 1.5, 'white', col, 1.0, dashed=val == TBD)
    s.rect(x, 1.45, 3.8, .38, fill, None)
    txt(s, x + .15, 1.45, 3.5, .38, h, size=10.5, color='ink', bold=True, anchor='ctr')
    s.stat(x + .15, 1.92, 1.6, val, lbl, size=26, label_size=8.5)
    bullets(s, x + 1.8, 2.02, 1.95, 1.2, lines, size=8.5, line=1.3)
    x += 3.97
s.section(.8, 3.2, 11.7, 'ROADMAP · EACH PHASE HAS A GATE')
ph = [('1 · Trace + tier budget', 'S', 'UB / NVMe / OBS latency and bandwidth, ADS access trace', 'accent'),
      ('2 · KT1 tiers', 'M', 'L2 UB pool + L3 NVMe under Lance', 'outline'),
      ('3 · KT2 prefetch', 'M', 'sampler order + mining hints', 'outline'),
      ('4 · KT3 policy, upstream', 'L', 'cost-aware admission, Lance PRs', 'dashed')]
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
s.strip(.8, 4.75, 11.75, .62, 'The origin stays the truth:', 'every tier is a disposable copy of immutable files, so a miss, a lost node or a failed gate '
        'costs only speed, never correctness.', size=9)
end(s, 'Lance has the hooks but no tier for object bytes;', 'phase 1 measures UB and sets the ms target with ADS.')
notes(s, 'Summary. KT1 adds UB pooled memory and local NVMe tiers under the Lance reader, using hooks that already exist; with partial reads it moves 64 times fewer '
      'bytes than a whole clip in the illustrative window. KT2 prefetches from the known training order plus mining hints. KT3 is one cost-aware policy and '
      'shared placement, compared with the S3-FIFO and TinyLFU backends Lance ships. Phase 1 measures latency and bandwidth of UB / Lingqu, NVMe and OBS on ADS '
      'nodes and collects an access trace, and agrees the ms target (time to first batch) with ADS. Sizes are rough estimates, not commitments.',
      ['DOC', 'BW', 'HOOKS', 'SEN'], limits='No UB or ADS measurement yet; sizes and staffing unconfirmed.')

finish(g, 'cache-v1')

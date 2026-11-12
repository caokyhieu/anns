#!/usr/bin/env python3
"""V8: condensed 8-slide deck, organised by key technology (KT) with result tables.

Changes vs v7:
  * WP1/WP2/WP3 renamed to key technologies and renumbered in pipeline order:
      KT1 assignment (option A tree routing, option B PiPNN graph over centroids, both measured)
      KT2 PiPNN partition index (IVF_PIPNN_FLAT)
      KT3 ARM SVE/SVE2 kernels on Kunpeng (the layer under KT1 and KT2)
  * PiPNN algorithm slide moved before KT1, because KT1-B and KT2 both use it.
  * Each KT slide has a small diagram plus a results table (build, match, recall@10, p50/p90 latency).
  * No ask slide; slide 8 is a summary and roadmap.
Slides: 1 title · 2 ADS pipeline · 3 Lance build map · 4 PiPNN · 5 KT1 · 6 KT2 · 7 KT3 · 8 summary
Reuses the header/helpers of build_deck_v6_1.py (everything before slide 1) so styling is identical.
Run: uv run --no-project --with python-pptx python build_deck_v8.py
"""
from pathlib import Path
import copy, json

ROOT = Path(__file__).resolve().parent
src = (ROOT / 'build_deck_v6_1.py').read_text()
prefix = src.split('# ================================================================ 1 title')[0]
g = {'__file__': str(ROOT / 'build_deck_v6_1.py'), '__name__': 'v61_prefix'}
exec(compile(prefix, 'build_deck_v6_1.py[prefix]', 'exec'), g)
globals().update({k: v for k, v in g.items() if not k.startswith('__')})
OUT = ROOT / 'ads-lance-kunpeng-index-build-v8.pptx'


def head(s, x, y, w, t, col='ink', size=10.5):
    """Small bold heading with a coloured left tick."""
    s.rect(x, y + .05, .06, .2, col, None)
    txt(s, x + .14, y, w - .14, .3, t, size=size, color='ink', bold=True, anchor='ctr')


def dot(s, x, y, col='muted', d=.08):
    s.rect(x - d / 2, y - d / 2, d, d, col, None, shape=MSO_SHAPE.OVAL)


# ================================================================ 1 title
s = D.title_slide('Faster Lance Index Builds on Kunpeng',
                  'Fast assignment · PiPNN partition graphs · ARM SVE/SVE2 kernels',
                  presenters=['Project proposal v8 · 10 November 2026'], team='Vector Index Optimization')
notes(s, 'Proposal to make ADS vectors searchable sooner by cutting Lance index-build time on our Kunpeng servers. '
      'Three key technologies: KT1 faster row assignment (tree routing or a PiPNN graph over centroids), KT2 PiPNN graphs per partition, '
      'and KT3 ARM SVE/SVE2 kernels that both run on. KT1 and KT2 are built in a Lance branch and measured on x86; Kunpeng numbers are pending.',
      ['ADS', 'PLAN'])

# ================================================================ 2 ADS context + contribution
srcs = Presentation(str(V51)).slides[1]
s2 = D.prs.slides.add_slide(D._blank)
for el in list(s2.shapes._spTree):
    if el.tag.endswith('}sp') or el.tag.endswith('}pic'):
        s2.shapes._spTree.remove(el)
for el in srcs.shapes._spTree:
    if el.tag.split('}')[1] in ('sp', 'cxnSp', 'grpSp', 'pic', 'graphicFrame'):
        s2.shapes._spTree.append(copy.deepcopy(el))
for sh in s2.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text
    if t == 'ADS · Faster Lance Index Builds':
        sh.text_frame.paragraphs[0].runs[0].text = KICKER
    elif t.startswith('Target (3)'):
        sh.text_frame.paragraphs[0].runs[0].text = 'We speed up ADS stage (3): building vector indexes'
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r.text = ''
    elif t.startswith('Index construction is the intervention point.'):
        runs = sh.text_frame.paragraphs[0].runs
        runs[0].text = 'Index build gates retrieval: new vectors wait until it ends.'
        for r in runs[1:]:
            r.text = ''
        if len(runs) > 1:
            runs[1].text = ' All three key technologies sit in this box.'
s2c = Canvas(D, s2, 'We speed up ADS stage (3): building vector indexes')
D.canvases.append(s2c)
s2c.rect(5.0, 4.12, 1.02, .95, None, 'red', 1.5, dashed=True)
s2c.route([(5.51, 5.07), (5.51, 5.16)], color='red')
s2c.rect(4.98, 5.16, 2.12, .56, 'white', 'red', 1.0)
s2c.text(5.03, 5.17, 2.04, .54, [[('OUR KEY TECHNOLOGIES', {'bold': True, 'color': 'title', 'size': 7.5})],
                                  [('KT1 assignment · KT2 PiPNN index', {'size': 7.5, 'color': 'ink'})],
                                  [('KT3 ARM kernels', {'size': 7.5, 'color': 'ink'})]], line=1.1, anchor='ctr', align='ctr')
s2.notes_slide.notes_text_frame.text = (
    'ADS pipeline from ADS-PLAN slide 1. Stage (3), vector production, builds the Lance vector indexes that stage (4) multimodal retrieval '
    'searches. Until the build finishes, new vectors are not searchable, so build time is the delay we attack. All three key technologies '
    'change only what happens inside this box: how rows are assigned to partitions (KT1), how each partition graph is built (KT2), and how fast '
    'the distance kernels run on Kunpeng (KT3). The next slide opens the box.\n\nSources:\n' + SRC['ADS'])

# ================================================================ 3 project map
s = new('Three key technologies in the Lance build',
        'Lance IVF create_index with each key technology (KT) at its integration point · solid = built and measured in Lance branch · dashed = proposed')
s.section(.8, 1.42, 8.6, 'LANCE INDEX BUILD (create_index)')
stages = [('Sample + train', 'hierarchical k-means'), ('Assign rows', '10M rows → K partitions'),
          ('Shuffle + encode', 'RaBitQ codes, write'), ('Partition subindex', 'flat / HNSW graph'), ('Commit', 'index files')]
xs = [.8, 2.55, 4.3, 6.05, 7.8]
for i, ((a, b), x) in enumerate(zip(stages, xs)):
    node(s, f'st{i}', x, 1.78, 1.5, .62, a, b, 'neutral', size=9.5)
    if i < 4:
        edge(s, f'e{i}', [(x + 1.5, 2.09), (x + 1.75, 2.09)])
TOP, BOT = 2.62, 3.82
s.rect(.8, TOP, 3.25, BOT - TOP, 'blue_light', 'blue', .9)
txt(s, .95, TOP + .05, 3.0, .24, 'KT1 · Assignment: A or B', size=10, color='blue_dark', bold=True)
node(s, 'asg_tree', .95, 2.98, 1.43, .7, 'A · Tree routing', 'keep k-means tree, beam 4–16', 'white', size=9, sub_size=7.5)
node(s, 'asg_pipnn', 2.48, 2.98, 1.43, .7, 'B · PiPNN graph', 'over the K centroids', 'purple', size=9, sub_size=7.5)
edge(s, 'kt1a', [(1.66, TOP), (1.66, 2.4)], color='blue', dashed=True)
edge(s, 'kt1b', [(3.3, TOP), (3.3, 2.4)], color='blue')
s.rect(4.3, TOP, 1.5, BOT - TOP, 'green_light', 'green', .9)
txt(s, 4.38, TOP + .05, 1.38, .24, 'Existing patch', size=9.5, color='green_dark', bold=True)
txt(s, 4.38, 2.94, 1.38, .7, ['Constant RQ scale', 'portable, opt-in'], size=8.5)
edge(s, 'grn', [(5.05, TOP), (5.05, 2.4)], color='green')
s.rect(6.05, TOP, 3.25, BOT - TOP, 'purple_light', 'purple', .9)
txt(s, 6.2, TOP + .05, 3.0, .24, 'KT2 · PiPNN partition index', size=10, color='purple_dark', bold=True)
txt(s, 6.2, 2.94, 3.0, .85, ['IVF_PIPNN_FLAT: PiPNN graph per partition', 'Replaces per-partition HNSW insertion',
                              'Same builder as KT1 option B'], size=8.5)
edge(s, 'kt2', [(6.8, TOP), (6.8, 2.4)], color='purple')
KY = 4.32
for i, x in enumerate([1.66, 3.2, 5.05, 6.8]):
    edge(s, f'up{i}', [(x, KY), (x, BOT)], color='red', dashed=True)
s.rect(2.48, 3.9, 6.82, .03, 'purple', None)
s.rect(3.75, 3.99, 4.3, .19, 'white', None)
lab(s, 3.4, 3.98, 5.0, 'One PiPNN builder at both sites: KT1 option B and KT2', size=7.5, color='purple_dark', bold=True)
s.rect(.8, KY, 8.5, 1.68, 'pale_red', 'red', 1.0)
txt(s, .95, KY + .06, 8.2, .26, 'KT3 · ARM CPU kernel dispatch layer (lance-linalg + PiPNN builder), selected at runtime',
    size=10, color='title', bold=True)
node(s, 'k_neon', .98, KY + .4, 2.55, .85, 'ARM NEON · baseline', 'all ARMv8 · existing Lance kernels + gaps', 'white', size=9)
node(s, 'k_sve', 3.68, KY + .4, 2.75, .85, 'SVE / SVE2 · Kunpeng (main)', '6 kernels done · blocked L2 next', 'accent', size=9.5)
node(s, 'k_sme', 6.58, KY + .4, 2.55, .85, 'SME / SME2 · exploratory', 'outer-product tiles for batch L2', 'dashed', size=9)
txt(s, .98, KY + 1.35, 8.15, .25, 'Runtime probe: SME → SVE2 / SVE → NEON → portable scalar · vector-length agnostic · no global -march flags',
    size=8, color='muted', align='ctr')
s.rule(9.55, 1.45, 4.55, vertical=True)
s.section(9.8, 1.42, 2.8, 'STATUS · 10 NOV 2026')
st = [('KT1 assignment', ['A · tree:4: measured, 2.49× faster', 'B · PiPNN graph: measured, 2.04× faster'], 'blue', 1.25),
      ('KT2 PiPNN index', ['IVF_PIPNN_FLAT builds, persists, searches', 'Measured: 4.9× faster, recall ≥ HNSW'], 'purple', 1.25),
      ('KT3 ARM kernels', ['6 SVE/SVE2 kernels, QEMU-validated', 'Blocked L2 + argmin: in progress',
                           'Native Kunpeng run: next'], 'red', 1.6)]
y = 1.78
for h, lines, col, dy in st:
    s.rect(9.8, y + .04, .06, .22, col, None)
    txt(s, 9.95, y, 2.6, .28, h, size=10, color='ink', bold=True, anchor='ctr')
    bullets(s, 9.95, y + .3, 2.6, 1.2, lines, size=8.5, line=1.25)
    y += dy
end(s, 'KT1 and KT2 change what is computed; KT3 makes it fast on Kunpeng.', 'Each ships separately.')
notes(s, 'The pipeline is Lance create_index for IVF indexes. KT1, assignment, has two measured options. Option A reuses the hierarchical k-means tree '
      'that training already builds and Lance currently flattens, and routes each row with a small beam. Option B builds a PiPNN graph over the K '
      'centroids and assigns each row with a graph search, replacing the K-wide scan or Lance\'s centroid HNSW. KT2 adds IVF_PIPNN_FLAT, a PiPNN graph '
      'per partition, next to IVF_HNSW_FLAT; it uses the same PiPNN builder. The constant RaBitQ scale patch is portable and opt-in. KT3 sits under all '
      'of them and targets ARM only: NEON baseline, SVE/SVE2 on Kunpeng as the main target, SME/SME2 exploratory. Kernels are selected at runtime.',
      ['PLAN', 'SVE', 'RA', 'RC', 'P1', 'ACLE'], limits='SME is exploratory; availability depends on the chip. Kunpeng numbers are pending.')

# ================================================================ 4 PiPNN algorithm
s = new('PiPNN · Build the graph with matrix math, not search',
        'Algorithm behind KT1 option B and KT2 · redrawn by us after Rubel et al., KDD 2026 · red tags = where KT3 kernels plug in')
panels = [('1 · Partition into leaves', 'Random leaders; each point joins its few nearest → overlapping leaves ≤ 1,024', 'P0 kernel: routing'),
          ('2 · Leaf Gram → k-NN', 'All pairs in a leaf as one dense matrix; keep k nearest each way', 'P0 kernel: Gram'),
          ('3 · HashPrune', 'Bounded reservoir per point keeps near candidates in varied directions', 'P1: sketches'),
          ('4 · RobustPrune', 'Final degree-bounded graph; search starts at the medoid', 'P1: batched L2')]
x = .8
for i, (h, b, k) in enumerate(panels):
    yy = s.panel(x, 1.45, 2.82, 3.05, h)
    if i == 0:
        for (cx, cy, r) in [(1.55, 2.5, .5), (2.15, 2.7, .55), (1.85, 3.1, .45)]:
            s.rect(cx - r, cy - r * .7, 2 * r, 1.4 * r, None, 'muted', .9, shape=MSO_SHAPE.OVAL)
        for (px, py) in [(1.4, 2.45), (1.7, 2.65), (2.0, 2.8), (2.3, 2.55), (1.9, 3.15), (2.4, 2.85), (1.6, 2.95)]:
            s.rect(px, py, .07, .07, 'muted', None, shape=MSO_SHAPE.OVAL)
        s.rect(1.95, 2.83, .09, .09, 'red', None, shape=MSO_SHAPE.OVAL)
    elif i == 1:
        s.grid(x + .65, yy + .12, 6, 6, cell=.2, fill='white',
               cells={**{(r, r): 'surface' for r in range(6)}, (0, 2): 'red', (2, 0): 'red', (1, 4): 'red', (4, 1): 'red', (3, 5): 'pink', (5, 3): 'pink'})
    elif i == 2:
        for j, (lbl, keep) in enumerate([('c1 1.2', True), ('c2 2.0', True), ('c3 1.5', True), ('c4 3.1', False), ('c5 0.9', True)]):
            s.rect(x + .5, yy + .1 + j * .25, 1.8, .2, 'pale_red' if keep else 'white', 'red' if keep else 'border', .75)
            lab(s, x + .5, yy + .1 + j * .25, 1.8, lbl + ('' if keep else '  evicted'), size=8, color='ink' if keep else 'muted')
    else:
        pts = [(x + .7, yy + .45), (x + 1.4, yy + .15), (x + 2.1, yy + .5), (x + 1.0, yy + 1.15), (x + 1.8, yy + 1.2), (x + 1.4, yy + .7)]
        for a, b_ in [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (0, 3), (2, 4)]:
            s.connector(pts[a][0], pts[a][1], pts[b_][0], pts[b_][1], 'muted', .9, arrow_end=False)
        for j, (px, py) in enumerate(pts):
            s.rect(px - .05, py - .05, .1, .1, 'ink' if j == 5 else 'muted', None, shape=MSO_SHAPE.OVAL)
    txt(s, x + .15, 3.38, 2.55, .65, b, size=9, line=1.25)
    s.tag(x + .15, 4.1, k, 'accent' if 'P0' in k else 'outline')
    if i < 3:
        s.arrow(x + 2.84, 2.75, .14, .25)
    x += 2.98
s.rect(.8, 4.8, 5.75, 1.05, 'surface', 'border', .75)
txt(s, .95, 4.87, 5.5, .3, 'HNSW today · insert one vector at a time', size=10, color='ink', bold=True)
bullets(s, .95, 5.2, 5.5, .6, ['each insert runs a graph search, then links + prunes', 'pointer chasing, locks, serial dependency'], size=9, line=1.3)
s.rect(6.8, 4.8, 5.75, 1.05, 'pale_red', 'red', .9)
txt(s, 6.95, 4.87, 5.5, .3, 'PiPNN · no search during build', size=10, color='title', bold=True)
bullets(s, 6.95, 5.2, 5.5, .6, ['leaves are independent dense blocks: parallel, SIMD-friendly',
                                'used twice: graph over centroids (KT1-B) and per partition (KT2)'], size=9, line=1.3)
end(s, 'PiPNN turns graph building into kernel work,', 'which is why it depends on KT3 on Kunpeng.')
notes(s, 'PiPNN (Rubel et al., KDD 2026) builds a proximity graph without insertion searches. Points are routed to overlapping leaves via random '
      'leaders; inside each leaf all pairwise distances are computed as a dense matrix and the k nearest are kept as candidate edges in both '
      'directions; HashPrune keeps a bounded, direction-diverse reservoir per point; RobustPrune produces the final degree-bounded graph. '
      'We use one native C++ builder in two places: over the 16,384 centroids for assignment (KT1 option B) and inside each IVF partition (KT2). '
      'Steps 1 and 2 are where the P0 blocked kernels from KT3 plug in.',
      ['P1', 'PLAN', 'RA', 'RC'], 'Diagram is our redrawn interpretation; values illustrative.')

# ================================================================ 5 KT1 assignment
s = new('KT1 · Tree or PiPNN assignment: ~2× faster build',
        '[x86] BIGANN10M (10M × 128), IVF_RQ 5-bit, K = 16,384, same centroids (only assignment differs) · AMD EPYC 9124 · 1 run each (±1 s, ±0.005 noise)')
PH = 1.95
# option A: tree
y0 = s.panel(.8, 1.45, 3.75, PH, 'A · Route down the k-means tree')
rx = 2.4
node(s, 'a_root', rx, y0 + .02, .55, .26, 'root', None, 'dark', size=8)
l1 = [1.05, 1.6, 2.15, 2.7, 3.25, 3.8]
for i, x in enumerate(l1):
    keep = i in (1, 3)
    s.rect(x, y0 + .55, .4, .2, 'blue_light' if keep else 'white', 'blue' if keep else 'border', .9)
    s.connector(rx + .27, y0 + .28, x + .2, y0 + .55, 'blue' if keep else 'border', .9 if keep else .5, arrow_end=False)
for j, x in enumerate([1.55, 1.8, 2.05, 2.65, 2.9, 3.15]):
    keep = j in (1, 3)
    s.rect(x, y0 + .98, .2, .17, 'blue_light' if keep else 'white', 'blue' if keep else 'border', .8)
    px = 1.8 if j < 3 else 2.9
    s.connector(px, y0 + .75, x + .1, y0 + .98, 'blue' if keep else 'border', .8 if keep else .5, arrow_end=False)
s.rect(1.8, y0 + 1.3, .2, .17, 'red', None)
s.connector(1.9, y0 + 1.15, 1.9, y0 + 1.3, 'red', 1.0)
lab(s, 2.1, y0 + 1.28, 2.4, 'beam 4 ≈ 256 distances / row', size=8, color='title', align='l')
# option B: PiPNN graph over centroids
y0 = s.panel(4.7, 1.45, 3.75, PH, 'B · Search a PiPNN centroid graph')
cpts = [(5.1, y0 + .3), (5.6, y0 + .1), (6.1, y0 + .45), (5.4, y0 + .8), (6.6, y0 + .15), (7.1, y0 + .55), (6.4, y0 + 1.0),
        (7.6, y0 + .25), (7.9, y0 + .9), (5.9, y0 + 1.25), (7.3, y0 + 1.25)]
for a, b_ in [(0, 1), (0, 3), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (3, 9), (6, 9), (5, 7), (7, 8), (6, 10), (8, 10), (2, 6)]:
    s.connector(cpts[a][0], cpts[a][1], cpts[b_][0], cpts[b_][1], 'border', .8, arrow_end=False)
for (px, py) in cpts:
    dot(s, px, py, 'purple', .1)
path = [0, 2, 6, 10]
for a, b_ in zip(path, path[1:]):
    s.connector(cpts[a][0], cpts[a][1], cpts[b_][0], cpts[b_][1], 'red', 1.2, arrow_end=True)
dot(s, cpts[10][0], cpts[10][1], 'red', .14)
lab(s, 4.85, y0 + 1.28, 1.0, 'beam search', size=8, color='title', align='l')
# right: headline
RX = 8.65
head(s, RX, 1.45, 3.9, 'Headline', 'blue')
s.stat(RX + .14, 1.75, 1.9, '2.49×', 'A · tree:4 vs exact', size=22, label_size=8.5)
s.stat(RX + 2.05, 1.75, 1.9, '2.04×', 'B · PiPNN beam 15', size=22, label_size=8.5)
bullets(s, RX + .14, 2.65, 3.75, .8, ['recall and p50/p90 unchanged (±1 ms)', 'no gain at small K (GIST1M, K = 1,000)'], size=8.5, line=1.3)
# results table
s.section(.8, 3.55, 11.7, 'RESULTS · BIGANN10M · RECALL@10 AND LATENCY AT NPROBES 256, 1,970 QUERIES')
rows = [['Method', 'Build s', 'Speedup', 'Assign CPU s', 'Recall@10', 'p50 ms', 'p90 ms'],
        ['exact (baseline)', '37.67', '1.00×', '760.9', '0.9430', '9.63', '9.93'],
        ['Lance default (HNSW)', '27.07', '1.39×', '376.0', '0.9423', '9.06', '9.43'],
        ['A · tree:4', '15.11', '2.49×', '59.4', '0.9442', '9.30', '9.63'],
        ['A · tree:16', '18.97', '1.99×', '137.3', '0.9443', '9.78', '10.16'],
        ['B · PiPNN beam 15', '18.43', '2.04×', '174.2', '0.9417', '9.23', '9.50'],
        ['B · PiPNN beam 32', '19.98', '1.89×', '221.8', '0.9428', '9.06', '9.34']]
hl = {(r, c): 'pale' for r in (3, 5) for c in range(7)}
s.table(.8, 3.85, 11.75, rows, col_widths=[2.6, 1.45, 1.45, 1.65, 1.5, 1.5, 1.6], row_h=.3, size=9, highlight=hl,
        align=['l'] + ['r'] * 6)
end(s, 'Both options halve build time at the same recall and latency;', 'tree:4 is fastest, PiPNN beam 15 close behind.')
notes(s, 'Run pilotB-bigann10m-k16384, MODE=fixed: the centroid model is trained once (17.7 s, excluded) and shared, so only assignment differs. '
      'Option A routes each row down the hierarchical k-means tree with a beam; option B builds a PiPNN graph over the 16,384 centroids '
      '(degree 64, leaf 256, ~0.05 s) and assigns each row with a beam search. Match = share of a 100k-row sample assigned to the same centroid as exact. '
      'Recall@10 against exact top-100 ground truth; latency is per-query at k=10. Assignment does not change the query path, so latency differences '
      'are within noise. At equal match the tree is a little faster than the PiPNN graph (tree:16 0.997 at 18.97 s vs PiPNN beam 32 0.993 at 19.98 s). '
      'GIST1M K=16384 (retrained centroids): tree:4 1.37x; GIST1M K=1000: no gain.',
      ['RA', 'RA1K', 'LSRC'], 'One run per configuration. Peak RSS not compared (builds shared one process).')

# ================================================================ 6 KT2 PiPNN partition index
s = new('KT2 · PiPNN partition graphs build 4.9× faster',
        '[x86] IVF_PIPNN_FLAT vs IVF_HNSW_FLAT, BIGANN10M (10M × 128 f32), K = 5, same exact assignment · AMD EPYC 9124 · 1 build each, separate processes')
y0 = s.panel(.8, 1.45, 3.4, 4.6, 'New graph per partition')
node(s, 'ivf', .95, y0 + .1, 3.1, .4, 'IVF centroids (unchanged)', None, 'neutral', size=9)
for j in range(5):
    px = .95 + j * .64
    s.route([(px + .27, y0 + .5), (px + .27, y0 + .7)], color='muted')
    s.rect(px, y0 + .7, .54, .85, 'white', 'purple', .9)
    pts = [(px + .13, y0 + .88), (px + .4, y0 + .92), (px + .25, y0 + 1.12), (px + .42, y0 + 1.35), (px + .12, y0 + 1.4)]
    for a, b_ in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]:
        s.connector(pts[a][0], pts[a][1], pts[b_][0], pts[b_][1], 'purple', .7, arrow_end=False)
    for (qx, qy) in pts:
        dot(s, qx, qy, 'purple', .07)
lab(s, .95, y0 + 1.62, 3.1, 'one graph per partition', size=8)
s.rect(.95, y0 + 1.9, 1.4, .55, 'white', 'border', .75)
txt(s, .95, y0 + 1.9, 1.4, .55, [[('Today', {'bold': True, 'color': 'ink'})], [('HNSW insertion', {})]], size=8.5, align='ctr', anchor='ctr')
s.arrow(2.4, y0 + 2.05, .2, .25)
s.rect(2.65, y0 + 1.9, 1.4, .55, 'purple_light', 'purple', .9)
txt(s, 2.65, y0 + 1.9, 1.4, .55, [[('KT2', {'bold': True, 'color': 'purple_dark'})], [('PiPNN build', {})]], size=8.5, align='ctr', anchor='ctr')
bullets(s, .95, y0 + 2.55, 3.15, .6, ['same partitions, storage, query path', 'cost: +1.5 GB peak memory'], size=8.5, line=1.3)
s.rect(.95, y0 + 3.12, 3.1, 1.0, 'orange_light', 'orange', .75)
txt(s, 1.05, y0 + 3.14, 2.95, .96, [[('Why p90 ≈ 2 s for both', {'bold': True, 'color': 'ink'})],
                                     [('~1 in 5 queries probes a partition evicted from Lance\'s default 6 GiB index cache and reloads it (~2 s). '
                                       'Fast queries: p90 5–19 ms. Rerun with a larger cache.', {})]], size=8, line=1.2)
# build table, compact: metrics as columns
head(s, 4.45, 1.45, 8.1, 'Build cost', 'purple')
rows = [['Index', 'Build wall s', 'Build CPU s', 'Index size MB', 'Peak RSS GB'],
        ['HNSW', '537.7', '12,590', '6,994', '16.2'],
        ['PiPNN', '109.9', '1,830', '5,915', '17.7'],
        ['Change', '4.9× faster', '6.9× less', '−15%', '+1.5 GB']]
s.table(4.45, 1.8, 8.1, rows, col_widths=[1.3, 1.7, 1.7, 1.7, 1.7], row_h=.29, size=8.5,
        highlight={(3, c): 'pale' for c in (1, 2, 3)}, align=['l', 'r', 'r', 'r', 'r'])
# recall / latency table
head(s, 4.45, 3.1, 8.1, 'Recall@10 and latency (ms) · 1,970 queries', 'purple')
rows = [['nprobes · ef', 'R@10 HNSW', 'R@10 PiPNN', 'p50 HNSW', 'p50 PiPNN', 'p90 HNSW', 'p90 PiPNN'],
        ['1 · 32', '0.744', '0.767', '4.4', '3.3', '2,004', '2,144'], ['1 · 64', '0.802', '0.821', '4.2', '4.1', '1,972', '2,181'],
        ['1 · 128', '0.839', '0.848', '7.0', '6.3', '1,982', '1,997'], ['1 · 256', '0.855', '0.857', '5.3', '7.5', '1,995', '2,014'],
        ['1 · 512', '0.860', '0.859', '5.9', '7.0', '1,997', '2,163'], ['2 · 32', '0.845', '0.874', '6.1', '5.5', '2,138', '3,028'],
        ['2 · 64', '0.909', '0.935', '10.0', '10.5', '2,145', '3,066'], ['2 · 128', '0.949', '0.964', '12.2', '11.9', '2,136', '3,209'],
        ['2 · 256', '0.966', '0.973', '12.9', '9.9', '2,319', '2,580'], ['2 · 512', '0.971', '0.976', '10.2', '13.8', '2,097', '2,591']]
s.table(4.45, 3.42, 8.1, rows, col_widths=[1.2, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15], row_h=.245, size=8,
        highlight={(r, 2): 'pale' for r in range(1, 11) if r != 5}, align=['l'] + ['r'] * 6)
end(s, 'Drop-in for IVF_HNSW_FLAT: faster, smaller, as accurate.', 'Memory is the trade-off.')
notes(s, 'Suite C pilot. Both indexes use the same exact assignment and FLAT f32 storage; only the per-partition graph differs. HNSW: M 16, '
      'ef_construction 100. PiPNN: degree 32, leaf 1024, local_k 2, reservoir 160, alpha 1.2. Recall@10 over 1,970 tuning queries at nprobes 1 and 2 '
      'and ef 32 to 512 (the planned grid stopped early; k=100 and nprobes >= 3 are not measured). Tail latency (p90, p95, p99) is a harness artefact: '
      'the dataset was opened with Lance\'s default 6 GiB index cache, smaller than either index (7.0 and 5.9 GB), so one partition is reloaded on every query '
      'that probes it; the same query IDs are slow at every ef, for both indexes. Build, CPU, memory, size, recall and p50 are unaffected. Standalone C++ '
      'cross-check: FastPiPNN 222 s vs HNSW 1,557 s (7.0x).',
      ['RC', 'NAT', 'LSRC', 'MEM'], 'Single build per index; k=10 only; nprobes 1-2 measured.')

# ================================================================ 7 KT3 ARM kernels
s = new('KT3 · One blocked ARM kernel under KT1 and KT2',
        '[Proposal] diagram drawn by us, sizes illustrative · SVE is vector-length agnostic · correctness checked in QEMU, Kunpeng timings pending')
y0 = s.panel(.8, 1.45, 5.45, 4.6, 'Blocked L2 + argmin: rows × centroids')
cons = [('k-means', 'blue', 1.45), ('KT1 tree / graph', 'blue', 1.65), ('KT2 PiPNN leaves', 'purple', 1.85)]
cx = .95
for a, col, w in cons:
    s.rect(cx, y0 + .1, w, .38, 'white', col, .9)
    txt(s, cx, y0 + .1, w, .38, a, size=8.5, color='ink', bold=True, align='ctr', anchor='ctr')
    s.route([(cx + w / 2, y0 + .48), (cx + w / 2, y0 + .72)], color=col)
    cx += w + .1
gy = y0 + .88
s.grid(.98, gy, 4, 9, cell=.145, fill='pale_red', cells={(r, c): 'red' for r in range(4) for c in range(2)})
lab(s, .98, gy + .63, 1.3, 'X: rows × D', size=8)
txt(s, 2.33, gy + .12, .3, .3, '×', size=13, color='ink', bold=True, align='ctr')
s.grid(2.65, gy, 4, 9, cell=.145, fill='white', cells={(r, c): 'pink' for r in range(4) for c in range(2)})
lab(s, 2.65, gy + .63, 1.3, 'C: centroids × D', size=8)
s.arrow(4.03, gy + .17, .17, .22)
s.grid(4.27, gy, 4, 4, cell=.145, fill='white', cells={(0, 2): 'red', (1, 0): 'red', (2, 3): 'red', (3, 1): 'red'})
lab(s, 4.1, gy + .63, .95, 'd² block', size=8)
s.arrow(4.9, gy + .17, .17, .22)
s.rect(5.12, gy, .98, .58, 'red', None)
txt(s, 5.12, gy, .98, .58, ['argmin', '+ bias'], size=8.5, color='white', bold=True, align='ctr', anchor='ctr')
bullets(s, .98, gy + 1.0, 5.15, 1.0, ['Today: pairwise NEON, one vector × all centroids', 'Blocked: each loaded chunk feeds 4×4 dot products in registers',
                                      'Predicated tails: same code at any vector length'], size=8.5, line=1.3)
ty = y0 + 3.15
lab(s, .95, ty - .25, 4.0, 'Selected at runtime on ARM', size=8, color='muted', align='l', bold=True)
node(s, 'k3_neon', .95, ty, 1.6, .65, 'NEON', 'baseline', 'white', size=9, sub_size=8)
node(s, 'k3_sve', 2.65, ty, 1.75, .65, 'SVE / SVE2', 'Kunpeng · main', 'accent', size=9, sub_size=8)
node(s, 'k3_sme', 4.5, ty, 1.6, .65, 'SME / SME2', 'exploratory', 'dashed', size=9, sub_size=8)
head(s, 6.5, 1.45, 6.0, 'Kernel status', 'red')
rows = [['ARM kernel', 'Serves', 'Status', 'Kunpeng'],
        ['f32 dot + L2, pair and batch', 'flat / HNSW search, k-means', 'done · QEMU ✓', TBD],
        ['u8 dot (UDOT) + u8 L2', 'uint8 vectors (BIGANN)', 'done · QEMU ✓', TBD],
        ['4-bit LUT accumulate', 'RQ / PQ fast scan', 'done · QEMU ✓', TBD],
        ['P0 · blocked L2 + argmin / top-k', 'k-means, KT1, PiPNN routing', 'in progress', TBD],
        ['P0 · leaf Gram + top-k', 'KT2 PiPNN leaf k-NN', 'planned', TBD]]
s.table(6.5, 1.82, 6.05, rows, col_widths=[2.2, 1.9, 1.15, .8], row_h=.4, size=8.5,
        highlight={(r, 3): 'pale' for r in range(1, 6)}, align=['l', 'l', 'l', 'ctr'])
txt(s, 6.5, 4.3, 6.05, .5, 'QEMU ✓ = clang-18 -Werror, validated at 5 vector lengths (128–2048-bit), VL changed mid-process, guard pages for tail over-reads.',
    size=8, color='muted')
s.strip(6.5, 5.0, 6.05, 1.0, 'First gate:', 'blocked beats pairwise at equal vector width on the same Kunpeng host, with recall within ±0.005. '
        'If not, the dispatcher keeps the old path.', size=9)
end(s, 'Kernel speed on Kunpeng is the open question;', 'KT3 answers it for both KT1 and KT2.')
notes(s, 'Lance scores one vector against all centroids (vector x matrix). The proposed kernel scores a block of rows against a block of centroids '
      '(matrix x matrix): accumulators stay in registers, so each loaded chunk feeds several dot products. The epilogue adds the k-means balance bias '
      'and keeps a running argmin or small top-k; the selected distance is recomputed exactly. The same block kernel computes the PiPNN leaf Gram matrix. '
      'Six SVE/SVE2 kernels already exist and are validated under QEMU; their Kunpeng speed and the gain of blocking over pairwise at equal width '
      'are not yet measured. Earlier x86 tile numbers are not used because vector width and formulation were confounded.',
      ['PLAN', 'SVE', 'ACLE'], 'Diagram sizes are illustrative. No native ARM timing yet.')

# ================================================================ 8 summary + roadmap
s = new('Summary · measured on x86, Kunpeng next',
        'Three key technologies · x86 results measured by us · [Proposal] sizes S ≈ 1–2, M ≈ 3–5, L ≈ 6–10 engineer-weeks')
cols = [('KT1 · Assignment', 'blue', 'blue_light', '2.5×', 'faster build [x86]', ['A tree:4 · B PiPNN 2.0×', 'recall within 0.005']),
        ('KT2 · PiPNN index', 'purple', 'purple_light', '4.9×', 'faster build [x86]', ['6.9× less CPU, −15% size', 'recall ≥ HNSW at 9/10']),
        ('KT3 · ARM kernels', 'red', 'pale_red', 'TBD', 'Kunpeng speedup', ['6 SVE kernels done', 'blocked L2 + Gram: next'])]
x = .8
for h, col, fill, val, lbl, lines in cols:
    s.rect(x, 1.45, 3.8, 1.5, 'white', col, 1.0, dashed=val == 'TBD')
    s.rect(x, 1.45, 3.8, .38, fill, None)
    txt(s, x + .15, 1.45, 3.5, .38, h, size=10.5, color='ink', bold=True, anchor='ctr')
    s.stat(x + .15, 1.92, 1.6, val, lbl, size=26, label_size=8.5)
    bullets(s, x + 1.8, 2.02, 1.95, 1.2, lines, size=8.5, line=1.3)
    x += 3.97
s.section(.8, 3.2, 11.7, 'ROADMAP · EACH PHASE HAS A GATE')
ph = [('1 · Kunpeng bring-up', 'S', 'native build, SVE vs NEON baseline', 'accent'),
      ('2 · P0 SVE kernels (KT3)', 'M', 'blocked L2 + argmin, leaf Gram', 'outline'),
      ('3 · Integrate on ARM', 'M', 'KT1 + KT2 with new kernels', 'outline'),
      ('4 · Harden, upstream', 'L', 'repeats, full grids, Lance PRs', 'dashed')]
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
s.strip(.8, 4.75, 11.75, .62, 'Nothing is forced on:', 'every kernel sits behind runtime dispatch and every algorithm is an opt-in build option, '
        'so a failed gate keeps the current path.', size=9)
end(s, 'Two key technologies measured on x86; kernel speed on Kunpeng is next.', 'Phase 1 fills every TBD.')
notes(s, 'Summary: KT1 (assignment) and KT2 (PiPNN partition index) are built and measured on x86: 2.5x and 4.9x faster builds at equal recall. '
      'KT3 kernel speed on Kunpeng is the open item. Phase 1 brings up the native toolchain, runs the full Lance tests on ARM and measures existing '
      'SVE vs NEON kernels plus the frozen build suites. Phase 2 adds the P0 blocked kernels; phase 3 integrates KT1 and KT2 on Kunpeng; phase 4 adds '
      'repeats, k=100, full query grids with a fixed index cache, per-process memory and upstream pull requests. Sizes are rough estimates, not commitments.',
      ['PLAN', 'SVE', 'RA', 'RC'], 'Sizes and staffing are unconfirmed; Kunpeng values are placeholders.')

# ================================================================ save
warnings = D.save(str(OUT))
(ROOT / 'layout-warnings-v8.json').write_text(json.dumps(warnings, indent=1))
(ROOT / 'diagram-sources-v8.json').write_text(json.dumps(DIAGRAMS, indent=1, ensure_ascii=False))
manifest = []
for i, sl in enumerate(D.prs.slides, 1):
    texts = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    manifest.append({'slide': i, 'title': texts[1] if i > 1 and len(texts) > 1 else texts[0] if texts else '',
                     'notes_chars': len(sl.notes_slide.notes_text_frame.text)})
(ROOT / 'slide-manifest-v8.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
print('wrote', OUT, 'warnings:', len(warnings))

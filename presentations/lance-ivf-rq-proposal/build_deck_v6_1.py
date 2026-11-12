#!/usr/bin/env python3
"""V6.1: Faster Lance index builds on Kunpeng, a complete project proposal.

v6.1 removes the 4x4 blocked-L2 tile claims: the x86 run compared an AVX-512 tile with
Lance's 256-bit AVX+FMA batch kernel, so tiling and vector width are confounded.

WP1 ARM SVE/SVE2 kernels (lead) · WP2 tree assignment · WP3 PiPNN partition graphs.
Run: python build_deck_v6.py   (python-pptx, lxml)
Inputs beside this file: team_deck.py, team-template.pptx, style.json and
ads-lance-ivf-rq-proposal-v5.1.pptx (slide 2 = ADS context, copied verbatim).
Outputs: ads-lance-kunpeng-index-build-v6.1.pptx, diagram-sources-v6.1.json,
slide-manifest-v6.1.json, layout-warnings-v6.1.json.
All numbers are copied from the run records named in SRC; Kunpeng cells are
placeholders ("TBD") until native runs exist.
"""
from pathlib import Path
import copy, json
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE
from team_deck import TeamDeck, Canvas, C, VARIANTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ads-lance-kunpeng-index-build-v6.1.pptx'
V51 = ROOT / 'ads-lance-ivf-rq-proposal-v5.1.pptx'
STYLE = json.loads((ROOT / 'style.json').read_text())
C.update(STYLE['palette'])
for name in ['blue', 'green', 'purple', 'orange']:
    dark = C.get(name + '_dark', C['ink'])
    VARIANTS[name] = (name + '_light', name, 0.8, False, dark, 'body')
    VARIANTS[name + '_future'] = ('white', name, 1.0, True, dark, 'body')

GH = 'https://github.com/caokyhieu/anns/blob/arm-kernels/'
SRC = {
    'ADS': 'ADS-PLAN.pptx slide 1 (internal company context; diagram retained from v5.1).',
    'RA': 'lance_index_build/results/ivf_rq_assignment_results.md (runs pilotB-bigann10m-k16384, pilotA-gist-k16384, probeA-gist-assign; '
          '2x AMD EPYC 9124, 32 cores, AVX-512; Lance fcd3787076 + benchmark patch 3853233ca7e7; REPEATS=1).',
    'RA1K': 'lance_index_build/results/pilotA-gist-k1000/summary.md (GIST1M K=1000, same host).',
    'RC': 'lance_index_build/results/pilotC_hnsw_vs_pipnn_results.md (run pilotC-bigann10m-k5; same EPYC host; Lance fcd3787076 + patch c86a06454cfd; '
          'single build per index; query grid nprobes 1-2, k=10).',
    'MEM': 'lance_index_build/results/pilotC-mem-1m/builds.jsonl (1M BIGANN prefix, K=5 memory gate).',
    'RG': 'lance_ivf_rq/README.md and results/results.jsonl (GIST1M, K=1000, 4-core Intel Xeon 2.8 GHz AVX-512, 2 repeats, means; Lance 04e6199).',
    'SVE': 'lance_arm_sve/README.md and lance-arm-sve.patch (Lance 6e18b21; six C kernels, eight dispatch candidates; QEMU 7.2 multi-VL validation; '
           'no native ARM timing yet).',
    'PLAN': 'LANCE_INDEX_BUILD_PLAN.md (7 Nov 2026): suites A/B/C, kernel priorities §4, ablations §5.5, memory model §6, phases §8.',
    'NAT': 'native/results/my-run/summary.json (standalone C++ PiPNN/FastPiPNN vs HNSW, BIGANN10M uint8, 4 build threads, 200 queries, 1 repeat).',
    'LSRC': 'Lance source: rust/lance/src/dataset.rs L242 DEFAULT_INDEX_CACHE_SIZE = 6 GiB; lance-index/src/vector/kmeans.rs hierarchical k-means '
            '(HIER_K=16); vector/utils.rs centroid-HNSW threshold K x D >= 1,000,000 with AMX exception.',
    'P1': 'Rubel et al., PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing, KDD 2026, arXiv:2602.21247v3: https://arxiv.org/html/2602.21247v3',
    'ACLE': 'Linux arm64 SVE ABI https://docs.kernel.org/arch/arm64/sve.html ; Arm ACLE https://arm-software.github.io/acle/main/acle.html',
}

KICKER = 'ADS · Lance Index Builds on Kunpeng'
D = TeamDeck(kicker=KICKER, template=str(ROOT / 'team-template.pptx'))
for master in D.prs.slide_masters:  # same header-width fix as v2-v5 for substitute fonts
    for sh in master.shapes:
        if sh.has_text_frame and 'IRELAND RESEARCH CENTER, SNI Lab' in sh.text:
            sh.left, sh.width = Inches(8.25), Inches(4.70)

DIAGRAMS = []
TBD = 'TBD'


def notes(s, spoken, keys, limits=''):
    text = spoken + '\n\nEvidence labels: [x86] = measured by us on x86 (host named in the source); [Kunpeng TBD] = placeholder, ' \
        'to be filled from native Kunpeng runs; [Source] = Lance code; [Reported] = paper; [Proposal] = planned work.\n'
    if limits:
        text += '\nLimits: ' + limits + '\n'
    text += '\nSources:\n' + '\n'.join(SRC[k] for k in keys)
    if isinstance(s, Canvas):
        s.notes(text)
    else:
        s.notes_slide.notes_text_frame.text = text


def new(title, tag):
    return D.slide(title, tagline=tag)


def end(s, lead, rest=''):
    s.takeaway(lead, rest, y=6.35, h=.40, size=11)


def txt(s, x, y, w, h, t, size=9.5, color='body', bold=False, align='l', line=1.3, anchor='t'):
    return s.text(x, y, w, h, t, size=size, color=color, bold=bold, align=align, line=line, anchor=anchor)


def bullets(s, x, y, w, h, items, size=9.5, color='body', line=1.35):
    return s.text(x, y, w, h, ['· ' + i if isinstance(i, str) else i for i in items], size=size, color=color, line=line)


def lab(s, x, y, w, t, size=8, color='muted', bold=False, align='ctr'):
    return s.text(x, y, w, .2, t, size=size, color=color, bold=bold, align=align, anchor='ctr', wrap=False, line=1.1)


def node(s, id, x, y, w, h, label, sub=None, variant='white', size=9, sub_size=8):
    s.node(x, y, w, h, label, sub, variant, size=size, sub_size=sub_size)
    DIAGRAMS.append({'slide': len(D.prs.slides), 'type': 'node', 'id': id, 'box': [x, y, w, h],
                     'label': label, 'sub': sub, 'variant': variant})


def edge(s, id, pts, color='muted', dashed=False, label=None):
    s.route(pts, color=color, dashed=dashed)
    DIAGRAMS.append({'slide': len(D.prs.slides), 'type': 'edge', 'id': id, 'points': pts,
                     'color': color, 'dashed': dashed, 'label': label})


def hbar(s, x, y, wmax, h, value, vmax, color, label, size=8.5, lcolor='black'):
    w = max(.03, wmax * value / vmax)
    s.rect(x, y, w, h, color, None)
    s.text(x + w + .06, y, 1.6, h, label, size=size, color=lcolor, bold=True, anchor='ctr', wrap=False)


def tbd_box(s, x, y, w, h, label='Kunpeng TBD', size=9):
    s.rect(x, y, w, h, 'white', 'red', 1.0, dashed=True)
    s.text(x, y, w, h, label, size=size, color='title', bold=True, align='ctr', anchor='ctr', wrap=False)


# ================================================================ 1 title
s = D.title_slide('Faster Lance Index Builds on Kunpeng',
                  'ARM SVE/SVE2 kernels · tree assignment · PiPNN partition graphs',
                  presenters=['Project proposal v6.1 · 9 November 2026'], team='Vector Index Optimization')
notes(s, 'This is the project proposal for making ADS vectors searchable sooner by cutting Lance index-build time on our Kunpeng servers. '
      'The project has three work packages that share one kernel layer. WP1, new ARM SVE/SVE2 kernels for Kunpeng, leads. WP2 reuses the '
      'hierarchical k-means tree for row assignment. WP3 replaces per-partition HNSW graphs with PiPNN graphs. WP2 and WP3 are implemented in a '
      'Lance branch and measured on x86. Kunpeng numbers are placeholders until the native runs are done.',
      ['ADS', 'PLAN'])

# ================================================================ 2 executive summary
s = new('Three build wins measured; Kunpeng kernels next',
        '[x86] measured by us, equal recall · [Kunpeng TBD] filled after native runs · the wins are dense-distance loops WP1 ports to SVE')
cols = [
    ('2.5×', 'WP2 · TREE', 'IVF_RQ · BIGANN10M · K 16,384',
     ['37.7 s → 15.1 s', 'assignment CPU 12.8× lower', 'recall@10 within 0.005 of exact'], 'blue'),
    ('4.9×', 'WP3 · PiPNN GRAPHS', 'vs HNSW · BIGANN10M · K 5',
     ['537.7 s → 109.9 s, 6.9× less CPU', 'index 15% smaller', 'recall higher at 9 of 10 settings'], 'purple'),
    ('4.9×', 'CONSTANT RQ SCALE', 'IVF_RQ RQ9 · GIST1M · K 1,000',
     ['190.3 s → 38.5 s', 'per-vector scale search removed', 'R@10 within 0.002 at RQ9'], 'green'),
]
x = .8
for val, head, scope, lines, col in cols:
    s.rect(x, 1.45, 2.78, 3.05, 'white', 'border', .75)
    s.rect(x, 1.45, 2.78, .06, col, None)
    s.section(x + .18, 1.62, 2.5, head, size=8.5)
    s.text(x + .18, 1.9, 2.5, .75, val, size=36, color='title', bold=True, anchor='ctr')
    txt(s, x + .18, 2.68, 2.5, .25, scope, size=9, color='ink', bold=True)
    bullets(s, x + .18, 3.02, 2.5, 1.2, lines, size=9.5)
    s.tag(x + .18, 4.15, 'x86 measured', 'outline')
    x += 2.95
# Kunpeng column
s.rect(x, 1.45, 2.88, 3.05, 'pale_red', 'red', 1.0, dashed=True)
s.section(x + .18, 1.62, 2.6, 'WP1 · KUNPENG (LEAD)', size=8.5)
s.text(x + .18, 1.9, 2.6, .75, TBD, size=36, color='title', bold=True, anchor='ctr')
txt(s, x + .18, 2.68, 2.6, .25, 'Same builds, Kunpeng SVE/SVE2', size=9, color='ink', bold=True)
bullets(s, x + .18, 3.02, 2.6, 1.2, ['6 SVE/SVE2 kernels ready (QEMU)', 'blocked L2 + argmin, leaf Gram next',
                                     'gain vs pairwise SVE: to measure'], size=9.5)
s.tag(x + .18, 4.15, 'placeholder', 'accent')
s.strip(.8, 4.75, 11.75, .62, 'Why kernels lead:',
        'all three loops are dense distance math. On x86 they ran on Lance\'s AVX2 / AVX-512 kernels; on Kunpeng only NEON pairwise kernels '
        'and our SVE ports exist, and no ARM timing has been taken. Kernel speed is the first unknown to resolve.')
s.strip(.8, 5.48, 11.75, .52, 'Ask:', 'Kunpeng server time (SVE/SVE2), owners for WP1–WP3, and an agreed recall tolerance (proposed ±0.005 recall@10).')
end(s, 'Measured on x86, ready for Kunpeng.', 'The remaining unknown is ARM kernel speed, which is what WP1 resolves.')
notes(s, 'Three measured results, each from our own runs on x86 at equal recall. Tree assignment: BIGANN10M IVF_RQ K=16384, fixed centroids, '
      'build 37.67 s exact vs 15.11 s tree:4 (2.49x), assignment thread-seconds 760.9 vs 59.4 (12.8x), recall@10 0.7680 vs 0.7724 at nprobes 16 and '
      'equal at nprobes 64 and above. PiPNN: IVF_PIPNN_FLAT 109.90 s vs IVF_HNSW_FLAT 537.72 s, CPU 1830 vs 12590 s, index 5915 vs 6994 MB, peak RSS '
      '17.67 vs 16.16 GB; recall@10 higher at 9 of 10 settings measured on both, 0.0008 lower at nprobes 1, ef 512. Constant RaBitQ scale: GIST1M RQ9 190.3 s to 38.5 s on a 4-core '
      'Xeon; both arms use the same distance kernel, so the gain is the encoding change alone. The Kunpeng column is intentionally empty. Do not compare x86 and Kunpeng seconds directly; '
      'speedups will be computed within each machine.',
      ['RA', 'RC', 'RG', 'SVE'], 'Single runs except the GIST constant-scale table (2 repeats). Recall tolerance is a proposal to agree on.')

# ================================================================ 3 ADS context (copied from v5.1 slide 2)
src = Presentation(str(V51)).slides[1]
s3 = D.prs.slides.add_slide(D._blank)
for el in list(s3.shapes._spTree):
    if el.tag.endswith('}sp') or el.tag.endswith('}pic'):
        s3.shapes._spTree.remove(el)
for el in src.shapes._spTree:
    if el.tag.split('}')[1] in ('sp', 'cxnSp', 'grpSp', 'pic', 'graphicFrame'):
        s3.shapes._spTree.append(copy.deepcopy(el))
for sh in s3.shapes:
    if sh.has_text_frame and sh.text_frame.text == 'ADS · Faster Lance Index Builds':
        sh.text_frame.paragraphs[0].runs[0].text = KICKER
s3.notes_slide.notes_text_frame.text = src.notes_slide.notes_text_frame.text
D.canvases.append(Canvas(D, s3, 'Target (3): faster index construction'))

# ================================================================ 4 where build time goes
s = new('Build time sits in three dense-distance loops',
        '[x86] stage spans from our runs · bars are each build\'s own total (100%) · labels in seconds')
s.section(.8, 1.42, 7.5, 'WHERE THE SECONDS GO')
rows = [
    ('IVF_RQ · BIGANN10M · K = 16,384', 'fixed centroids (training 17.7 s excluded)', 37.67,
     [(31.29, 'shuffle incl. assignment 31.3', 'blue'), (6.38, 'merge 6.4', 'grey_bar')],
     'Assignment: 761 CPU-s = 16,384 distances × 10M rows'),
    ('IVF_RQ RQ9 · GIST1M · K = 1,000', '4-core Xeon, upstream Lance', 221.0,
     [(18.8, 'train', 'blue'), (197.3, 'shuffle: assign + 9-bit RQ encode + write 197.3', 'green'), (4.9, '', 'grey_bar')],
     'Encoding: per-vector scale search for multi-bit codes'),
    ('IVF_HNSW_FLAT · BIGANN10M · K = 5', 'per-partition graphs of 1.1–3.2M rows', 537.72,
     [(537.72, 'training + per-partition HNSW insertion 537.7 (assignment only 5.6 CPU-s)', 'purple')],
     'Graph build: one beam search per inserted vector'),
]
y = 1.78
for title, sub, total, segs, why in rows:
    txt(s, .8, y, 5.0, .22, title, size=10, color='ink', bold=True)
    txt(s, 5.0, y + .01, 3.3, .22, sub, size=8, color='muted', align='r')
    x0, W = .8, 7.5
    cx = x0
    for v, label, col in segs:
        w = W * v / total
        s.rect(cx, y + .28, w, .42, col, 'white', .75)
        if label:
            s.text(cx + .06, y + .28, w - .1, .42, label, size=8.5, color='white', bold=True, anchor='ctr', wrap=False)
        cx += w
    txt(s, .8, y + .74, 7.5, .22, [[('→ ', {'color': 'title', 'bold': True}), (why, {})]], size=8.5, color='body')
    y += 1.17
s.rule(8.6, 1.45, 3.75, vertical=True)
s.section(8.85, 1.42, 3.7, 'LOOP → WHAT FIXES IT')
fix = [('1 · Nearest-centroid search', 'k-means + full-data assignment', 'WP2 tree routing · WP1 SVE kernels', 'blue'),
       ('2 · Multi-bit RQ encoding', 'scale search per vector', 'constant scale (existing patch)', 'green'),
       ('3 · Partition graph build', 'HNSW insertion search', 'WP3 PiPNN · WP1 leaf Gram kernel', 'purple')]
y = 1.8
for h, sub, f, col in fix:
    s.rect(8.85, y, 3.7, 1.0, 'white', 'border', .75)
    s.rect(8.85, y, .06, 1.0, col, None)
    txt(s, 9.02, y + .08, 3.45, .24, h, size=10, color='ink', bold=True)
    txt(s, 9.02, y + .34, 3.45, .22, sub, size=8.5, color='muted')
    txt(s, 9.02, y + .6, 3.45, .3, [[('Fix: ', {'bold': True, 'color': 'title'}), (f, {})]], size=8.5)
    y += 1.12
s.strip(.8, 5.4, 11.75, .62, 'All three are dense L2 / dot-product math over many vectors.',
        'On x86 they run on Lance\'s AVX2 / AVX-512 kernels. On ARM, Lance has pairwise NEON kernels plus our six SVE ports.')
end(s, 'The bottleneck is the same arithmetic everywhere.', 'One kernel layer speeds up all three loops.')
notes(s, 'Each bar is one build normalized to its own total, so the shares are comparable even though the totals differ. BIGANN10M K=16384: '
      'build 37.67 s, shuffle 31.29 s, which contains assignment (760.9 thread-seconds of CPU), merge about 6.1 to 7.1 s; training 17.7 s was done once '
      'and excluded because the run used fixed centroids. GIST1M RQ9 on the 4-core Xeon: build 221.0 s, IVF training 18.8 s, shuffle 197.3 s, which '
      'covers assignment, RaBitQ encoding and writing. IVF_HNSW_FLAT: 537.72 s total, assignment only 5.59 thread-seconds; the stage split between '
      'k-means training and graph insertion was not recorded, so the bar is one segment.',
      ['RA', 'RG', 'RC', 'SVE'], 'Different hosts for row 2 (4-core Xeon) and rows 1 and 3 (32-core EPYC); only within-row shares are compared.')

# ================================================================ 5 project map
s = new('One project: three work packages, one kernel layer',
        'Lance IVF build pipeline with each work package at its integration point · solid = implemented on x86 · dashed red = to build for Kunpeng')
s.section(.8, 1.42, 8.6, 'LANCE INDEX BUILD (create_index)')
stages = [('Sample + train', 'hierarchical k-means'), ('Assign rows', '10M rows → K partitions'),
          ('Shuffle + encode', 'RaBitQ codes, write'), ('Partition subindex', 'flat / HNSW graph'), ('Commit', 'index files')]
xs = [.8, 2.55, 4.3, 6.05, 7.8]
for i, ((a, b), x) in enumerate(zip(stages, xs)):
    node(s, f'st{i}', x, 1.78, 1.5, .62, a, b, 'neutral', size=9.5)
    if i < 4:
        edge(s, f'e{i}', [(x + 1.5, 2.09), (x + 1.75, 2.09)])
# WP boxes above / below
s.rect(.8, 2.62, 3.25, 1.05, 'blue_light', 'blue', .9)
txt(s, .95, 2.68, 3.0, .24, 'WP2 · Tree assignment', size=10, color='blue_dark', bold=True)
txt(s, .95, 2.94, 3.0, .7, ['Keep the k-means tree; route each row with beam 4–16', 'Replaces the K-wide scan / centroid HNSW'], size=8.5)
edge(s, 'wp2a', [(2.42, 2.62), (2.42, 2.4)], color='blue')
edge(s, 'wp2b', [(3.3, 2.62), (3.3, 2.4)], color='blue')
s.rect(4.3, 2.62, 1.5, 1.05, 'green_light', 'green', .9)
txt(s, 4.38, 2.68, 1.38, .24, 'Existing patch', size=9.5, color='green_dark', bold=True)
txt(s, 4.38, 2.94, 1.38, .7, ['Constant RQ scale', 'portable, opt-in'], size=8.5)
edge(s, 'grn', [(5.05, 2.62), (5.05, 2.4)], color='green')
s.rect(6.05, 2.62, 3.25, 1.05, 'purple_light', 'purple', .9)
txt(s, 6.2, 2.68, 3.0, .24, 'WP3 · IVF_PIPNN_FLAT', size=10, color='purple_dark', bold=True)
txt(s, 6.2, 2.94, 3.0, .7, ['PiPNN graph per partition (v3 subindex)', 'Replaces per-partition HNSW insertion'], size=8.5)
edge(s, 'wp3', [(6.8, 2.62), (6.8, 2.4)], color='purple')
# kernel layer
s.rect(.8, 4.0, 8.5, 1.55, 'pale_red', 'red', 1.0)
txt(s, .95, 4.06, 8.2, .26, 'WP1 · CPU kernel dispatch layer (lance-linalg + PiPNN builder), selected at runtime', size=10, color='title', bold=True)
node(s, 'k_x86', .98, 4.42, 2.55, .95, 'x86 AVX2 / AVX-512', 'existing Lance kernels', 'white', size=9)
node(s, 'k_neon', 3.68, 4.42, 2.55, .95, 'ARM NEON', 'existing Lance pairwise kernels', 'white', size=9)
node(s, 'k_sve', 6.38, 4.42, 2.75, .95, 'ARM SVE / SVE2 · Kunpeng', '6 kernels done · blocked L2 next', 'accent', size=9.5)
for i, x in enumerate([1.55, 3.3, 5.05, 6.8]):
    edge(s, f'up{i}', [(x, 4.0), (x, 3.67 if i != 2 else 3.67)], color='red', dashed=True)
# status column
s.rule(9.55, 1.45, 4.55, vertical=True)
s.section(9.8, 1.42, 2.8, 'STATUS · 8 NOV 2026')
st = [('WP1 kernels', ['6 SVE/SVE2 kernels: clang-18 -Werror, QEMU 128–2048-bit VL', 'Blocked L2 + argmin (SVE): in progress',
                       'Native Kunpeng run: next'], 'red'),
      ('WP2 tree', ['Retained tree + beam routing in Lance branch', 'x86 suites A/B measured'], 'blue'),
      ('WP3 PiPNN', ['IVF_PIPNN_FLAT builds, persists, searches', 'x86 suite C measured (build + recall)'], 'purple')]
y = 1.78
for h, lines, col in st:
    s.rect(9.8, y + .04, .06, .22, col, None)
    txt(s, 9.95, y, 2.6, .28, h, size=10, color='ink', bold=True, anchor='ctr')
    bullets(s, 9.95, y + .3, 2.6, 1.0, lines, size=8.5, line=1.25)
    y += 1.3 if h != 'WP1 kernels' else 1.45
end(s, 'WP2/WP3 change what is computed; WP1 makes it fast on Kunpeng.', 'Each ships separately.')
notes(s, 'The pipeline is Lance create_index for IVF indexes. WP2 changes how rows find their partition: it reuses the hierarchical k-means tree '
      'that training already builds and Lance currently flattens. WP3 adds a new partition subindex type, IVF_PIPNN_FLAT, next to IVF_HNSW_FLAT. '
      'The existing patch (constant RaBitQ scale) is portable and opt-in. WP1 sits under all of them: the distance kernels are '
      'chosen at runtime by Lance\'s CPU dispatcher, and the PiPNN builder must share the same kernels. Kernel work on SVE needs no global '
      '-march flags; objects are probed at build time and selected at runtime.',
      ['PLAN', 'SVE', 'RA', 'RC'])

# ================================================================ 7 kernel mechanism (WP1)
s = new('Proposed: one blocked L2 + argmin for every loop',
        'Proposal 1 · WP1 · [Proposal] method figure drawn by us · gain over pairwise not yet measured · SVE is vector-length agnostic · sizes illustrative')
y0 = s.panel(.8, 1.4, 5.05, 3.25, '① Register block over D, in VL-sized steps')  # h set below
s.grid(1.0, y0 + .15, 4, 10, cell=.17, fill='pale_red', cells={(r, c): 'red' for r in range(4) for c in range(2)})
lab(s, 1.0, y0 + .88, 1.7, 'X block: rows × D', size=8)
s.grid(3.2, y0 + .15, 4, 10, cell=.17, fill='white', cells={(r, c): 'pink' for r in range(4) for c in range(2)})
lab(s, 3.2, y0 + .88, 1.7, 'C block: centroids × D', size=8)
txt(s, 1.0, y0 + 1.15, 4.7, .5, [[('acc', {'italic': True, 'bold': True}), ('ᵢⱼ += Σ (x', {'italic': True}), ('ᵢ', {'italic': True}),
                                   (' · c', {'italic': True}), ('ⱼ', {'italic': True}), (')   over one VL chunk (red columns)', {})]], size=9.5, color='ink')
bullets(s, 1.0, y0 + 1.5, 4.7, 1.4, ['row × centroid accumulators stay in SVE Z registers; each loaded chunk is reused across the block',
                                     'svwhilelt predicate covers the tail: no scalar remainder, no over-read',
                                     'd² = ‖x‖² + ‖c‖² − 2 x·c, norms precomputed once per block'], size=8.5)
y1 = s.panel(6.0, 1.4, 2.95, 3.25, '② Fused epilogue')
s.grid(6.35, y1 + .15, 4, 4, cell=.22, fill='white', cells={(0, 2): 'red', (1, 0): 'red', (2, 3): 'red', (3, 1): 'red'})
lab(s, 7.3, y1 + .45, 1.5, 'distance block', size=8)
bullets(s, 6.2, y1 + 1.1, 2.6, 1.9, ['+ balance bias (k-means)', 'running argmin or small top-k', 'first strictly smaller wins; NaN rows stay unassigned',
                                     'chosen distance recomputed exactly for RaBitQ residuals'], size=8.5)
y2 = s.panel(9.1, 1.4, 3.45, 3.25, '③ Consumers')
cons = [('k-means training', 'rows × K centroids', 'blue'), ('Tree beam step (WP2)', 'rows × 16·beam children', 'blue'),
        ('PiPNN leader routing (WP3)', 'rows × leaders', 'purple'), ('PiPNN leaf k-NN (WP3)', 'leaf Gram L×L, L = 256–1024', 'purple')]
yy = y2 + .1
for a, b, col in cons:
    s.rect(9.3, yy, 3.05, .6, 'white', col, .9)
    txt(s, 9.4, yy + .04, 2.9, .26, a, size=9, color='ink', bold=True)
    txt(s, 9.4, yy + .3, 2.9, .24, b, size=8, color='muted')
    yy += .7
s.strip(.8, 4.95, 11.75, .78, 'Contracts:',
        'same winner rule as upstream, exact selected distance, scalar fallback when SVE is absent, objects built with -msve-vector-bits=scalable '
        'and no global -march, vector length queried per call. Tests: ties, NaN, tails, irregular strides, 5 vector lengths, guard pages.')
end(s, 'First gate: does blocking beat pairwise at equal width?', 'If yes, k-means, tree routing and both PiPNN loops share it.')
notes(s, 'Mechanism of the proposed P0 kernel. Lance today scores one vector against all centroids (vector x matrix). This kernel scores a '
      'block of rows against a block of centroids (matrix x matrix), loaded in vector-length chunks along the dimension; the accumulators stay in '
      'registers, so each loaded chunk feeds several dot products instead of one. Its benefit over the pairwise SVE kernel at the same vector length '
      'is not yet measured on any machine; that is the first WP1 gate. The tail is handled with an SVE predicate, which is why the same code is '
      'correct at any vector length. The epilogue adds the k-means balance bias, keeps a running argmin or a small top-k, and recomputes the selected '
      'distance exactly. The same block kernel, with a leaf-by-leaf shape, computes the PiPNN leaf Gram matrix. This is the plan section 4 P0 row; contracts '
      'and test list come from the plan and from the existing SVE patch validation.',
      ['PLAN', 'SVE', 'ACLE'], 'Diagram sizes are illustrative. Blocked-kernel performance at equal vector width is not yet measured.')

# ================================================================ 8 kernel inventory (WP1)
s = new('Six SVE kernels ready; blocked kernels are next',
        'Proposal 1 · WP1 · status of every ARM kernel in scope · Done = QEMU-validated at 128–2048-bit VL, native Kunpeng timing pending')
s.text(.8, 1.4, 11.7, .24, [[('Done', {'bold': True, 'color': 'title'}), (' = implemented, emulator-validated    ', {}),
                             ('P0', {'bold': True, 'color': 'title'}), (' = needed by WP2/WP3 hot loops    ', {}),
                             ('P1 / P2', {'bold': True, 'color': 'title'}), (' = after profiling on Kunpeng', {})]], size=9.5)
rows = [['Pri', 'Kernel (ARM)', 'Serves', 'Status', 'Kunpeng result'],
        ['Done', 'f32 dot + squared L2, pair and batch (SVE FMA)', 'HNSW / flat search, k-means today', 'QEMU ✓ · dispatcher ✓', TBD],
        ['Done', 'u8 dot (SVE UDOT) + u8 squared L2', 'uint8 vectors (BIGANN)', 'QEMU ✓ · overflow tests', TBD],
        ['Done', '4-bit LUT accumulate (SVE gathers; SVE2 TBL high-accuracy)', 'PQ / RQ fast scan', 'QEMU ✓ · exact order', TBD],
        ['P0', 'Blocked f32 L2 + fused argmin / top-k + bias', 'k-means, tree beam, PiPNN routing', 'SVE in progress; gain unmeasured', TBD],
        ['P0', 'Leaf Gram / L2 blocks + top-k, bounded scratch', 'PiPNN leaf k-NN', 'x86 AVX-512 in PiPNN; SVE planned', TBD],
        ['P1', 'Batched candidate L2 + row gather', 'RobustPrune, graph search', 'planned', '—'],
        ['P1', 'Projection sketches, sign compare, u16 hash', 'PiPNN HashPrune', 'planned', '—'],
        ['P2', 'Small-k selection; u8 integer distance blocks', 'byte-input leaf builds', 'planned', '—']]
hl = {(r, 4): 'pale' for r in range(1, 6)}
hl.update({(4, 0): 'accent', (5, 0): 'accent'})
s.table(.8, 1.75, 11.75, rows, col_widths=[.6, 4.0, 2.75, 3.0, 1.4], row_h=.36, size=8.5, highlight=hl)
s.section(.8, 5.1, 5.5, 'ALREADY VERIFIED')
bullets(s, .8, 5.36, 5.6, .7, ['clang-18 -Wall -Wextra -Werror · QEMU at 5 vector lengths, VL changed mid-process',
                               'guard pages expose tail over-reads · Rust dispatcher selects SVE only when the OS reports it'], size=8.5, line=1.25)
s.section(6.7, 5.1, 5.8, 'NOT YET')
bullets(s, 6.7, 5.36, 5.85, .7, ['native Rust build + full Lance tests on ARM (toolchain certificate issue on our host)',
                                 'any Kunpeng timing · SME (deferred until profiles justify it)'], size=8.5, line=1.25)
end(s, 'The pairwise base is done; blocked kernels are the bet.', 'P0 rows decide whether WP2/WP3 gains survive on Kunpeng.')
notes(s, 'Done rows come from lance_arm_sve: six C entry points and eight Rust dispatch candidates, selected by Lance\'s cached CPU dispatcher when '
      'Linux reports SVE or SVE2. They retain upstream\'s per-vector algorithm, so they speed up existing paths but do not add blocking. P0 rows are '
      'the kernels the new algorithms need, from plan section 4. The SVE blocked kernel is in progress; its gain over the pairwise SVE '
      'path has not been measured at equal vector width. The PiPNN builder has AVX-512 and BF16 Gram paths on x86; the ARM leaf-Gram kernel is planned in f32 first, because BF16 on the target '
      'Kunpeng is not confirmed. The Kunpeng result column stays empty until native runs.',
      ['SVE', 'PLAN'], 'Emulator runs establish correctness only, not speed.')

# ================================================================ 9 Kunpeng evaluation (WP1)
s = new('Kunpeng runs reuse frozen configs, five ablations',
        'Proposal 1 · WP1 · [Proposal] evaluation design · placeholders fill in as runs complete · speedups computed within one machine only')
s.section(.8, 1.42, 4.3, 'ABLATION LADDER (SAME KUNPENG HOST)')
steps = [('1', 'Stock Lance', 'reference', 'neutral'), ('2', '+ constant RQ scale', 'portable encoding change', 'neutral'),
         ('3', '+ existing SVE kernels', 'pairwise SVE vs NEON', 'neutral'), ('4', '+ WP2 tree / WP3 PiPNN', 'algorithm gain, old kernels', 'neutral'),
         ('5', '+ new P0 SVE blocked kernels', 'kernel gain on new loops', 'accent')]
y = 1.78
for n, a, b, v in steps:
    s.badge(.8, y + .1, n)
    node(s, f'ab{n}', 1.25, y, 3.8, .52, a, b, v if v == 'accent' else 'white', size=9.5)
    if n != '5':
        s.arrow(3.05, y + .55, .15, .1, direction='down')
    y += .67
s.rule(5.3, 1.45, 4.45, vertical=True)
s.section(5.55, 1.42, 7.0, 'RESULTS TABLE · TO BE FILLED')
rows = [['Measurement', 'x86 EPYC (ours)', 'Kunpeng 1–3', 'Kunpeng 5'],
        ['IVF_RQ build, BIGANN10M K = 16,384, exact', '37.7 s', TBD, TBD],
        ['… with tree:4 assignment (WP2)', '15.1 s', TBD, TBD],
        ['IVF_HNSW_FLAT build, BIGANN10M K = 5', '537.7 s', TBD, '—'],
        ['IVF_PIPNN_FLAT build, same (WP3)', '109.9 s', TBD, TBD],
        ['Blocked vs pairwise L2 + argmin, same ISA', 'to run', TBD, TBD],
        ['Pair kernels: SVE vs NEON (Criterion)', '—', TBD, '—'],
        ['Recall@10 change vs exact / HNSW', '≤ 0.005', TBD, TBD]]
hl = {(r, c): 'pale' for r in range(1, 8) for c in (2, 3) if rows[r][c] == TBD}
s.table(5.55, 1.75, 7.0, rows, col_widths=[3.15, 1.35, 1.25, 1.25], row_h=.37, size=8.5, highlight=hl)
txt(s, 5.55, 4.82, 7.0, .3, 'Recorded per run: Kunpeng model, SVE vector length, cores / NUMA, affinity, frequency policy, compiler, thread counts.',
    size=8.5, color='muted')
s.strip(.8, 5.3, 11.75, .7, 'Gate for each new kernel [Proposal]:',
        'faster than the existing NEON/SVE path on the same Kunpeng host in microbenchmark and in full build, with recall@10 within ±0.005. '
        'Otherwise the dispatcher keeps the old path. x86 vs Kunpeng seconds are context, never a speedup.')
end(s, 'Every Kunpeng number will have a same-machine baseline.', 'The harness and configs are frozen and run unchanged.')
notes(s, 'Ablation ladder from plan section 5.5. Step 2 adds the portable constant RaBitQ scale. Step 3 isolates the existing SVE ports with benchmark_native.py, which builds twice (LANCE_DISABLE_SVE_BUILD=1 vs default) '
      'and compares with Criterion. Step 4 runs WP2 and WP3 on the old kernels; step 5 adds the P0 blocked kernels on the same host and vector length, which isolates the effect of blocking. x86 column values are from our EPYC runs. '
      'The recall gate of ±0.005 is a proposal; the leadership may set a different tolerance.',
      ['PLAN', 'SVE', 'RA', 'RC'], 'All Kunpeng cells are placeholders.')

# ================================================================ 10 tree mechanism (WP2)
s = new('Reuse the trained tree to assign rows',
        'Proposal 2 · WP2 tree assignment · hierarchical k-means, 16 children per node, K = 16,384 → 3–4 levels · tree drawn smaller')
s.section(.8, 1.42, 3.6, 'TODAY · TREE THROWN AWAY')
s.card(.8, 1.72, 3.55, 2.0, None, ['k-means builds a 16-way tree, then keeps only the K leaf centroids',
                                   'Assignment: scan all K (16,384 distances / row), or a centroid HNSW when K·D ≥ 1M (Lance default)',
                                   'HNSW must be built and loses ~4–5% of exact winners at default ef'], 'neutral', size=8.5)
s.section(4.6, 1.42, 4.6, 'PROPOSED · ROUTE DOWN THE KEPT TREE')
# tree drawing
rx, ry = 6.7, 1.8
node(s, 'root', rx, ry, .7, .3, 'root', None, 'dark', size=8)
lvl1 = [4.75, 5.4, 6.05, 6.7, 7.35, 8.0, 8.65]
for i, x in enumerate(lvl1):
    keep = i in (2, 4)
    s.rect(x, 2.4, .5, .25, 'blue_light' if keep else 'white', 'blue' if keep else 'border', 1.0 if keep else .75)
    s.connector(rx + .35, ry + .3, x + .25, 2.4, 'blue' if keep else 'border', .9 if keep else .6, arrow_end=False)
lab(s, 8.9, 2.42, .6, '… 16', size=8)
for j, x in enumerate([5.75, 6.05, 6.35, 7.05, 7.35, 7.65]):
    keep = j in (1, 3)
    s.rect(x, 3.0, .24, .22, 'blue_light' if keep else 'white', 'blue' if keep else 'border', .9)
    px = 6.3 if j < 3 else 7.6
    s.connector(px, 2.65, x + .12, 3.0, 'blue' if keep else 'border', .8 if keep else .5, arrow_end=False)
s.rect(6.0, 3.5, .34, .22, 'red', None)
s.connector(6.17, 3.22, 6.17, 3.5, 'red', 1.0)
lab(s, 6.4, 3.5, 2.0, 'exact pick among final leaves', size=8, color='title', align='l')
lab(s, 4.6, 3.05, 1.1, 'beam = 2', size=8, color='blue_dark', bold=True)
s.text(4.6, 3.85, 4.6, .25, 'Keep the beam closest nodes per level → exact distance to the chosen centroid', size=8.5, color='body', align='ctr')
# cost table right
s.rule(9.45, 1.45, 2.5, vertical=True)
s.section(9.7, 1.42, 2.9, 'SAMPLE OF 100K ROWS [x86]')
s.table(9.7, 1.75, 2.85, [['Method', 'Time', 'Match'], ['exact', '0.248 s', '1.000'], ['tree:4', '0.023 s', '0.917'],
                          ['tree:16', '0.055 s', '0.997'], ['HNSW (default)', '1.28 s*', '0.952']],
        col_widths=[1.25, .8, .8], row_h=.33, size=8.5, highlight={(2, 0): 'pale', (2, 1): 'pale', (2, 2): 'pale'})
txt(s, 9.7, 3.5, 2.85, .5, '* + 0.11 s graph build. Tree prep 0.002 s: training already made it. Match = same centroid as exact.', size=8, color='muted')
s.strip(.8, 4.35, 11.75, .62, 'Distance work per row ≈ beam × 16 per level instead of K.',
        'With K = 16,384 and 3–4 levels, beam 4 touches a few hundred centroids instead of 16,384, and every touch is a small dense block (WP1 kernel).')
s.card(.8, 5.1, 5.75, .95, 'Why the occasional wrong leaf is cheap', 'Queries still probe nprobes partitions chosen exactly; a near-boundary row in a '
       'neighbouring partition is still found. Measured below: recall unchanged.', 'white', size=8.5, title_size=9.5)
s.card(6.8, 5.1, 5.75, .95, 'What stays unchanged', 'Query routing, index format and centroid layout. The tree is build-only state, '
       'not persisted. Exact / auto remain selectable.', 'white', size=8.5, title_size=9.5)
end(s, 'The tree is free and already trained.', 'Using it removes most assignment work without new index state.')
notes(s, 'Lance hierarchical k-means (enabled for K > 256, branching 16) produces a tree but returns only the flat leaf centroids. Full-data assignment then '
      'compares each row with all K centroids, or, when K x D >= 1,000,000 and the CPU has no AMX, builds an HNSW graph over the centroids (M 12, '
      'ef 15). Our branch keeps the tree arrays and routes each row down with a beam; the chosen leaf distance is recomputed exactly. Sample-based '
      'quality from pilotB: 100,000 rows, exact 0.248 s; tree:4 0.023 s with 0.9173 agreement; tree:16 0.055 s with 0.9968; HNSW default 1.277 s plus '
      '0.113 s build with 0.9520. Internal-node distances are a heuristic, so finite beams can miss the true centroid.',
      ['RA', 'LSRC', 'PLAN'], 'Tree drawing is schematic; counts per row are approximate (beam x 16 x depth).')

# ================================================================ 11 tree evidence (WP2)
s = new('Tree assignment cuts BIGANN10M build time 2.5×',
        'Proposal 2 · WP2 · [x86] IVF_RQ 5-bit, K = 16,384, fixed centroids (only assignment differs), 1,970 tuning queries, 1 run each')
s.section(.8, 1.42, 5.6, 'BUILD SECONDS · BIGANN10M (LOWER IS BETTER)')
cats = [('exact (baseline)', 37.67, 'grey_bar'), ('HNSW helper (Lance default)', 27.07, 'grey_bar'), ('PiPNN centroid graph, beam 32', 19.98, 'grey_bar'),
        ('tree:16', 18.97, 'blue'), ('tree:4 (proposed default)', 15.11, 'red')]
y = 1.8
for name, v, col in cats:
    txt(s, .8, y, 2.4, .3, name, size=8.5, color='ink', bold=col == 'red', anchor='ctr')
    hbar(s, 3.2, y + .04, 2.7, .24, v, 37.67, col, f'{v:.1f} s', size=8.5)
    y += .4
s.rule(6.65, 1.45, 3.0, vertical=True)
s.section(6.9, 1.42, 5.6, 'QUALITY AT THE SAME SETTINGS')
s.table(6.9, 1.75, 5.65, [['Method', 'vs exact', 'vs default', 'Match', 'R@10 np=64'],
                          ['exact', '1.00×', '0.72×', '1.000', '0.9066'],
                          ['HNSW default', '1.39×', '1.00×', '0.952', '0.9026'],
                          ['PiPNN graph', '1.89×', '1.35×', '0.993', '0.9061'],
                          ['tree:16', '1.99×', '1.43×', '0.997', '0.9083'],
                          ['tree:4', '2.49×', '1.79×', '0.917', '0.9062']],
        col_widths=[1.45, .95, 1.0, .85, 1.4], row_h=.34, size=8.5, highlight={(5, c): 'pale' for c in range(5)})
txt(s, 6.9, 3.85, 5.65, .25, 'Assignment CPU: 760.9 → 59.4 thread-s (12.8× less). Query latency unchanged (p50 within ±1 ms).', size=8.5, color='muted')
s.section(.8, 4.15, 11.7, 'OTHER DATA · WHERE IT HELPS LESS')
s.card(.8, 4.45, 3.8, 1.5, 'GIST1M, K = 16,384', ['complete builds (k-means retrained)', 'exact 31.4 s · Lance auto 24.3 s',
                                                  'tree:4 23.0 s (1.37×) · tree:1 19.9 s', 'training (~9.5 s) now dominates'], 'neutral', size=8.5, title_size=9.5)
s.card(4.78, 4.45, 3.8, 1.5, 'GIST1M, K = 1,000', ['K·D below the helper threshold', 'exact 11.7 s vs tree:4 11.6 s',
                                                   'no gain: assignment already cheap', 'auto keeps exact at small K'], 'neutral', size=8.5, title_size=9.5)
s.card(8.76, 4.45, 3.8, 1.5, 'Centroid graphs lose', ['HNSW and PiPNN over centroids are slower than the tree at equal match',
                                                      'graph route not pursued for assignment', '(PiPNN stays for partition graphs, WP3)'], 'white', size=8.5, title_size=9.5)
end(s, 'Biggest gain at large K with many rows: the ADS case.', 'At small K, Lance\'s exact path stays.')
notes(s, 'Table 1 of the assignment report. All builds share one trained centroid model, so only assignment differs. Build seconds: exact 37.67, '
      'HNSW default 27.07, HNSW ef 32 21.77, PiPNN beam 32 19.98, tree:16 18.97, tree:4 15.11, tree:1 14.96 (but 0.578 match and lower recall at '
      'nprobes 16). Recall@10 at nprobes 16: exact 0.7724, tree:4 0.7680, tree:16 0.7726; at nprobes 256: 0.9430, 0.9442, 0.9443. On GIST1M with '
      'K=16384 and retrained centroids: exact 31.38 s, auto (effectively HNSW) 24.25 s, tree:4 22.97 s, tree:1 19.89 s; recall@10 at nprobes 256 '
      '0.8984, 0.9057, 0.8979. GIST at K=1000: exact 11.73 s, tree:4 11.56 s, auto 11.56 s. Peak RSS is not compared because builds ran in one process.',
      ['RA', 'RA1K'], 'One run per configuration (about ±1 s, ±0.005 recall noise). vs default = speedup against Lance auto (HNSW helper).')

# ================================================================ 12 PiPNN mechanism (WP3)
s = new('PiPNN swaps insertion search for leaf matrix math',
        'Proposal 3 · WP3 · drawn by us after Rubel et al., KDD 2026 §3–4 · red tags = where WP1 kernels plug in · sizes illustrative')
panels = [('1 · Overlapping leaves', 'Random leaders; each point joins a few nearest → leaves ≤ 1,024', 'P0 kernel: routing'),
          ('2 · Leaf Gram → local k-NN', 'All pairs inside a leaf as one dense matrix; keep k nearest each way', 'P0 kernel: Gram'),
          ('3 · HashPrune', 'Bounded reservoir per point keeps near candidates in varied directions', 'P1: sketches'),
          ('4 · RobustPrune', 'Final degree-bounded graph; search starts at the medoid', 'P1: batched L2')]
x = .8
for i, (h, b, k) in enumerate(panels):
    yy = s.panel(x, 1.4, 2.82, 2.95, h)
    if i == 0:
        for (cx, cy, r) in [(1.55, 2.55, .5), (2.15, 2.75, .55), (1.85, 3.15, .45)]:
            s.rect(cx - r, cy - r * .7, 2 * r, 1.4 * r, None, 'muted', .9, shape=MSO_SHAPE.OVAL)
        for (px, py) in [(1.4, 2.5), (1.7, 2.7), (2.0, 2.85), (2.3, 2.6), (1.9, 3.2), (2.4, 2.9), (1.6, 3.0)]:
            s.rect(px, py, .07, .07, 'muted', None, shape=MSO_SHAPE.OVAL)
        s.rect(1.95, 2.88, .09, .09, 'red', None, shape=MSO_SHAPE.OVAL)
    elif i == 1:
        s.grid(x + .55, yy + .1, 7, 7, cell=.2, fill='white',
               cells={**{(r, r): 'surface' for r in range(7)}, (0, 2): 'red', (2, 0): 'red', (1, 4): 'red', (4, 1): 'red', (3, 5): 'pink', (5, 3): 'pink'})
    elif i == 2:
        for j, (lbl, keep) in enumerate([('c1 1.2', True), ('c2 2.0', True), ('c3 1.5', True), ('c4 3.1', False), ('c5 0.9', True)]):
            s.rect(x + .5, yy + .1 + j * .27, 1.8, .22, 'pale_red' if keep else 'white', 'red' if keep else 'border', .75)
            lab(s, x + .5, yy + .1 + j * .27, 1.8, lbl + ('' if keep else '  evicted'), size=8, color='ink' if keep else 'muted')
    else:
        pts = [(x + .7, yy + .5), (x + 1.4, yy + .2), (x + 2.1, yy + .55), (x + 1.0, yy + 1.2), (x + 1.8, yy + 1.25), (x + 1.4, yy + .75)]
        for a, b_ in [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (0, 3), (2, 4)]:
            s.connector(pts[a][0], pts[a][1], pts[b_][0], pts[b_][1], 'muted', .9, arrow_end=False)
        for j, (px, py) in enumerate(pts):
            s.rect(px - .05, py - .05, .1, .1, 'ink' if j == 5 else 'muted', None, shape=MSO_SHAPE.OVAL)
    txt(s, x + .15, 3.45, 2.55, .5, b, size=8.5)
    s.tag(x + .15, 4.02, k, 'accent' if 'P0' in k else 'outline')
    if i < 3:
        s.arrow(x + 2.84, 2.6, .14, .25)
    x += 2.98
s.section(.8, 4.55, 5.6, 'HNSW (TODAY) · PER-VECTOR INSERTION')
txt(s, .8, 4.82, 5.6, .9, ['Each new vector runs a beam search over the graph built so far, then links and prunes neighbours.',
                           'Irregular memory access, locks, and a serial dependency between insertions.'], size=9)
s.section(6.75, 4.55, 5.8, 'PiPNN (WP3) · NO SEARCH DURING BUILD')
txt(s, 6.75, 4.82, 5.8, .9, ['Leaves are independent: dense GEMM-like blocks, embarrassingly parallel.',
                             'Search is used only at query time; the graph format is our new Lance v3 subindex.'], size=9)
s.strip(.8, 5.65, 11.75, .45, 'Build cost moves from pointer chasing to matrix math,', 'the shape SIMD kernels run best.', size=9)
end(s, 'PiPNN turns graph building into kernel work.', 'That is why WP3 depends on WP1 on Kunpeng.')
notes(s, 'PiPNN (Rubel et al., KDD 2026) builds a proximity graph without insertion searches. Points are routed to overlapping leaves via random leaders '
      'with multi-level fanout; inside each leaf, all pairwise distances are computed as a dense matrix and the k nearest are kept as candidate edges in '
      'both directions; HashPrune keeps a bounded reservoir per point that favours near candidates in different directions; RobustPrune produces the '
      'final degree-bounded graph. Our Lance integration builds this per IVF partition through a C++ builder behind a narrow C ABI, with Rust owning '
      'storage and search. Parameters in our runs: degree 32, leaf 1024, local_k 2, reservoir 160, alpha 1.2.',
      ['P1', 'PLAN', 'RC'], 'Diagram is our redrawn interpretation; values illustrative.')

# ================================================================ 13 PiPNN evidence (WP3)
s = new('PiPNN builds partition graphs 4.9× faster',
        'Proposal 3 · WP3 · [x86] IVF_PIPNN_FLAT vs IVF_HNSW_FLAT, BIGANN10M (10M × 128 f32), K = 5, exact assignment, separate processes, 1 build each')
s.section(.8, 1.42, 4.3, 'BUILD COST (HNSW GREY · PiPNN RED)')
mets = [('Build wall time', 537.72, 109.90, 's', '4.9× faster'), ('Build CPU time', 12590.1, 1830.2, 's', '6.9× less'),
        ('Index size', 6993.5, 5915.4, 'MB', '−15%'), ('Peak RSS', 16.16, 17.67, 'GB', '+1.5 GB')]
y = 1.75
for name, h_, p_, u, verdict in mets:
    txt(s, .8, y, 2.4, .22, name, size=9.5, color='ink', bold=True)
    txt(s, 3.3, y, 1.8, .22, verdict, size=10, color='title' if '+' not in verdict else 'body', bold=True, align='r')
    vmax = max(h_, p_)
    hbar(s, .8, y + .26, 3.2, .2, h_, vmax, 'grey_bar', f'{h_:,.1f} {u}' if h_ < 100 else f'{h_:,.0f} {u}', size=8)
    hbar(s, .8, y + .5, 3.2, .2, p_, vmax, 'red', f'{p_:,.1f} {u}' if p_ < 100 else f'{p_:,.0f} {u}', size=8)
    y += .8
s.rule(5.3, 1.45, 4.45, vertical=True)
s.section(5.55, 1.42, 4.0, 'RECALL@10, SAME SETTINGS')
s.table(5.55, 1.75, 3.95, [['nprobes · ef', 'HNSW', 'PiPNN', 'Δ'],
                           ['1 · 32', '0.744', '0.767', '+0.023'], ['1 · 128', '0.839', '0.848', '+0.009'], ['1 · 512', '0.860', '0.859', '−0.001'],
                           ['2 · 32', '0.845', '0.874', '+0.029'], ['2 · 64', '0.909', '0.935', '+0.026'], ['2 · 128', '0.949', '0.964', '+0.014'],
                           ['2 · 512', '0.971', '0.976', '+0.005']],
        col_widths=[1.2, .9, .9, .95], row_h=.33, size=8.5, highlight={(r, 3): 'pale' for r in (1, 4, 5, 6)})
txt(s, 5.55, 4.45, 3.95, .55, 'Higher at 9 of 10 settings, −0.001 at the 10th (full grid in A2). p50 within a few ms.', size=8.5, color='muted')
s.rule(9.75, 1.45, 4.45, vertical=True)
s.section(10.0, 1.42, 2.6, 'STANDALONE CHECK')
txt(s, 10.0, 1.72, 2.55, .45, 'C++ builders, 10M uint8, 4 threads, no Lance', size=8.5, color='muted')
s.stat(10.0, 2.15, 2.55, '7.0×', 'FastPiPNN 222 s vs HNSW 1,557 s', size=24, label_size=8.5)
bullets(s, 10.0, 3.15, 2.55, 1.6, ['recall@10 at beam 64: 0.955 vs 0.958', 'peak memory 18.8 vs 6.1 GB', 'QPS at that point 3,787 vs 5,025'], size=8.5)
s.strip(.8, 5.15, 11.75, .8, 'A drop-in replacement for IVF_HNSW_FLAT:',
        'same partitions and storage, 4.9× faster and 6.9× cheaper to build, 15% smaller, equal or better recall. The cost is memory: +1.5 GB inside Lance '
        '(sequential partitions), up to 3× standalone. WP3 bounds concurrent partitions to a memory budget.')
end(s, 'Faster, smaller, as accurate; memory is the trade-off.', 'Tail latency is not claimed (see A2).')
notes(s, 'Suite C pilot. Both indexes use the same exact assignment and FLAT f32 storage; only the per-partition graph differs. HNSW: M 16, ef_construction '
      '100, 537.72 s, CPU 12,590 s, index 6,993.5 MB, peak RSS 16.16 GB. PiPNN: degree 32, leaf 1024, local_k 2, reservoir 160, 109.90 s, CPU 1,830 s, '
      '5,915.4 MB, peak 17.67 GB. Recall@10 over 1,970 tuning queries at nprobes 1 and 2 and ef 32 to 512. Tail latency was a harness artefact: the '
      'benchmark opened the dataset with Lance\'s default 6 GiB index cache, smaller than either index, so one partition was reloaded on every query '
      'that probed it, identically for both indexes; the same query IDs were slow at every ef. Build, CPU, memory, size and recall are unaffected. '
      'Standalone cross-check from native/results: HNSW M 32 ef_c 200 1,556.5 s and 6.07 GB; FastPiPNN degree 64 222.3 s and 18.8 GB; original PiPNN '
      '1,128 s and 20.2 GB.',
      ['RC', 'NAT', 'LSRC', 'MEM'], 'Single build per index; k=10 only; nprobes 1-2 measured. Standalone run uses different graph settings and uint8 data.')

# ================================================================ 14 roadmap
s = new('Kunpeng baseline first; every step has a gate',
        '[Proposal] sizes are rough (S ≈ 1–2 weeks, M ≈ 3–5, L ≈ 6–10 engineer-weeks) · owners to be named · x86 work already done is shown first')
s.section(.8, 1.42, 11.7, 'PHASES')
ph = [('Done on x86', 'Harness, suites A/B/C pilots, tree assignment, IVF_PIPNN_FLAT, 6 SVE kernels', 'tests pass · numbers in this deck', 'dark', ''),
      ('Phase 1 · Kunpeng bring-up', 'Native toolchain, full Lance tests on ARM, SVE vs NEON Criterion, ablation steps 1–3', 'baseline table filled', 'outline', 'S'),
      ('Phase 2 · P0 SVE kernels', 'Blocked L2 + argmin, leaf Gram + top-k; shared by Rust and the PiPNN builder', 'beats old path, scalar parity', 'accent', 'M'),
      ('Phase 3 · Integrate on ARM', 'WP2 + WP3 on Kunpeng with new kernels; frozen suites A/C, ablation steps 4–5', 'speedup at ±0.005 recall', 'outline', 'M'),
      ('Phase 4 · Harden + upstream', '3 repeats, k = 100, full nprobes, per-process RSS, remap / reload tests, Lance PRs', 'both architectures pass', 'dashed', 'L')]
x = .8
for i, (h, b, gate, v, size) in enumerate(ph):
    w = 2.25
    fill = {'dark': 'ink', 'accent': 'red'}.get(v, 'white')
    tcol = 'white' if v in ('dark', 'accent') else 'title'
    s.rect(x, 1.75, w, .6, fill, 'red' if v in ('outline', 'dashed') else fill, 1.0, dashed=v == 'dashed')
    txt(s, x + .1, 1.75, w - .2, .6, h, size=9.5, color=tcol, bold=True, anchor='ctr')
    if size:
        s.tag(x + w - .55, 1.55, size, 'accent')
    s.rect(x, 2.4, w, 1.05, 'surface', 'border', .75)
    txt(s, x + .1, 2.47, w - .2, .95, b, size=8.5, line=1.3)
    s.rect(x, 3.5, w, .5, 'white', 'border', .75)
    txt(s, x + .1, 3.5, w - .2, .5, [[('Gate: ', {'bold': True, 'color': 'title'}), (gate, {})]], size=8.5, anchor='ctr')
    if i < 4:
        s.arrow(x + w + .02, 1.93, .12, .22)
    x += 2.37
s.section(.8, 4.3, 5.6, 'EXIT / ROLLBACK')
bullets(s, .8, 4.57, 5.6, 1.1, ['Each kernel sits behind runtime dispatch: if it fails its gate, the old path stays',
                                'Tree and PiPNN are opt-in build options; exact / auto / HNSW remain the defaults until gated'], size=8.5, line=1.3)
s.section(6.75, 4.3, 5.8, 'WHAT THE DECISION UNBLOCKS NOW')
bullets(s, 6.75, 4.57, 5.8, 1.1, ['Phase 1 needs only Kunpeng server access and one engineer',
                                  'Phases 2–3 can run in parallel: kernel owner + integration owner'], size=8.5, line=1.3)
end(s, 'Small first step, measurable gates, nothing forced on by default.', 'Phase 1 fills every TBD in this deck.')
notes(s, 'Phases follow the plan section 8, re-ordered so Kunpeng bring-up comes first. T-shirt sizes are rough estimates for discussion, not commitments. '
      'Gates: phase 1 produces the same-machine baseline; phase 2 kernels must beat the existing path on Kunpeng with scalar parity under documented '
      'tolerances; phase 3 must show build speedup with recall within the agreed tolerance; phase 4 completes repeats, k=100, full query grids with an '
      'adequate index cache, per-process memory, serialization reload and remap tests, and prepares upstream pull requests.',
      ['PLAN', 'SVE'], 'Sizes and staffing are unconfirmed.')

# ================================================================ 15 ask
s = new('Approve WP1 on Kunpeng, then WP2 and WP3',
        'Decision requested from technical leadership · risks are the measured trade-offs, each with a mitigation')
s.text(.8, 1.42, 11.7, .4, 'Recommendation: fund a gated 3-package project, starting with a Kunpeng baseline.',
       size=13, color='ink', bold=True)
s.section(.8, 1.95, 3.7, 'WHAT WE KNOW')
for i, (a, b) in enumerate([('2.5× IVF_RQ build', 'tree assignment, equal recall (x86)'), ('4.9× partition graphs', 'PiPNN vs HNSW, recall on par or better (x86)'),
                            ('Kunpeng gap', 'no ARM timing yet; ARM has pairwise kernels')]):
    s.badge(.8, 2.3 + i * .82, i + 1, a, '', size=9.5)
    txt(s, 1.2, 2.58 + i * .82, 3.4, .4, b, size=8.5)
s.rule(4.75, 1.98, 4.0, vertical=True)
s.section(5.0, 1.95, 4.0, 'RISKS → MITIGATION')
s.table(5.0, 2.25, 4.1, [['Risk', 'Mitigation'], ['New kernel no faster', 'gate + fallback'],
                         ['PiPNN memory', 'bound partitions'],
                         ['Approx. assignment', 'tree:16 / exact'], ['New index format', 'v3 subindex'],
                         ['Single x86 runs', '3 repeats'],],
        col_widths=[2.1, 2.0], row_h=.36, size=8.5)
s.rule(9.3, 1.98, 4.0, vertical=True)
s.section(9.55, 1.95, 3.0, 'THE ASK')
asks = ['Kunpeng server time (SVE/SVE2), fixed affinity and NUMA', 'Owners: WP1 kernels, WP2/WP3 integration',
        'Recall tolerance (proposed ±0.005 recall@10)', 'Agreement to upstream to Lance']
for i, a in enumerate(asks):
    s.rect(9.55, 2.3 + i * .66, 3.0, .56, 'pale_red' if i == 0 else 'white', 'red', .9)
    txt(s, 9.65, 2.3 + i * .66, 2.85, .56, a, size=8.5, color='ink', bold=i == 0, anchor='ctr')
s.strip(.8, 5.15, 11.75, .8, 'Next 2 weeks after approval:',
        'run phase 1 on Kunpeng and return with the filled baseline table (slide 8): SVE vs NEON, and stock vs tree vs PiPNN builds on the same host.')
end(s, 'Measured on x86; Kunpeng speed is the open question.', 'One bounded step answers it.')
notes(s, 'The recommendation is a gated project, not a production rollout. What we know comes from our x86 measurements. The open question is ARM kernel '
      'speed on Kunpeng, which phase 1 and 2 answer. Risks are the trade-offs we measured: PiPNN memory, approximate assignment, a new subindex format, '
      'and single-run data. The ask is server time, two owners, a recall tolerance and agreement to upstream.',
      ['RA', 'RC', 'SVE', 'PLAN'])

# ================================================================ A1 full assignment table
s = new('Appendix A1 · All assignment methods, both datasets',
        '[x86] AMD EPYC 9124 ×2 · IVF_RQ 5-bit · K = 16,384 · 1 run each · match = same centroid as exact on a 100k sample')
s.table(.8, 1.4, 11.75, [['BIGANN10M (fixed centroids)', 'build s', 'vs exact', 'assign CPU-s', 'match', 'R@10 np16', 'np64', 'np256'],
                         ['exact', '37.67', '1.00×', '760.9', '1.0000', '0.7724', '0.9066', '0.9430'],
                         ['tree:1', '14.96', '2.52×', '49.5', '0.5778', '0.7128', '0.8808', '0.9371'],
                         ['tree:4', '15.11', '2.49×', '59.4', '0.9173', '0.7680', '0.9062', '0.9442'],
                         ['tree:16', '18.97', '1.99×', '137.3', '0.9968', '0.7726', '0.9083', '0.9443'],
                         ['pipnn 15:64:256', '18.43', '2.04×', '174.2', '0.9624', '0.7651', '0.9013', '0.9417'],
                         ['pipnn 32:64:256', '19.98', '1.89×', '221.8', '0.9931', '0.7702', '0.9061', '0.9428'],
                         ['hnsw 12:15:32', '21.77', '1.73×', '271.4', '0.9638', '0.7701', '0.9060', '0.9435'],
                         ['pipnn 64:64:256', '23.35', '1.61×', '296.8', '0.9990', '0.7710', '0.9062', '0.9425'],
                         ['hnsw (Lance default)', '27.07', '1.39×', '376.0', '0.9520', '0.7652', '0.9026', '0.9423'],
                         ['hnsw 12:15:64', '27.38', '1.38×', '416.2', '0.9953', '0.7723', '0.9068', '0.9436'],
                         ['pipnn 128:64:256', '27.91', '1.35×', '493.4', '0.9999', '0.7726', '0.9078', '0.9437']],
        col_widths=[2.6, 1.1, 1.1, 1.4, 1.1, 1.2, 1.1, 1.1], row_h=.27, size=8, highlight={(3, c): 'pale' for c in range(8)})
s.table(.8, 4.85, 11.75, [['GIST1M (centroids retrained)', 'build s', 'vs exact', 'assign CPU-s', 'match', 'train s', 'R@10 np64', 'np256'],
                          ['exact', '31.38', '1.00×', '384.7', '1.0000', '9.33', '0.7259', '0.8984'],
                          ['tree:4', '22.97', '1.37×', '18.9', '0.9635', '11.79', '0.7062', '0.8979'],
                          ['auto (Lance default = HNSW)', '24.25', '1.29×', '125.6', '—', '9.72', '0.7192', '0.9057']],
        col_widths=[2.6, 1.1, 1.1, 1.4, 1.1, 1.2, 1.1, 1.1], row_h=.27, size=8)
end(s, 'tree:4 is the best speed-quality point; tree:16 is the near-exact option.', 'Sources in notes.')
notes(s, 'Full tables from the assignment report. GIST rows: tree:1 19.89 s (1.58x, match 0.6712) and tree:16 22.82 s (match 0.9995) omitted for space. '
      'GIST percentiles rest on 193 tuning queries.', ['RA'], 'REPEATS=1: about ±1 s build and ±0.005 recall noise.')

# ================================================================ A2 suite C grid
s = new('Appendix A2 · Recall grid and tail latency',
        '[x86] BIGANN10M, K = 5, recall@10 over 1,970 tuning queries · every completed setting shown')
rows = [['nprobes', 'ef', 'HNSW R@10', 'PiPNN R@10', 'HNSW p50 ms', 'PiPNN p50 ms']]
grid = [(1, 32, .7444, .7669, 4.4, 3.3), (1, 64, .8023, .8214, 4.2, 4.1), (1, 128, .8391, .8478, 7.0, 6.3), (1, 256, .8545, .8565, 5.3, 7.5),
        (1, 512, .8601, .8593, 5.9, 7.0), (2, 32, .8452, .8742, 6.1, 5.5), (2, 64, .9089, .9346, 10.0, 10.5), (2, 128, .9492, .9636, 12.2, 11.9),
        (2, 256, .9655, .9732, 12.9, 9.9), (2, 512, .9713, .9762, 10.2, 13.8)]
for g in grid:
    rows.append([str(g[0]), str(g[1]), f'{g[2]:.4f}', f'{g[3]:.4f}', f'{g[4]}', f'{g[5]}'])
s.table(.8, 1.4, 6.6, rows, col_widths=[.9, .7, 1.25, 1.25, 1.25, 1.25], row_h=.3, size=8.5,
        highlight={(r, 3): 'pale' for r in range(1, 11) if grid[r - 1][3] > grid[r - 1][2]})
s.rule(7.65, 1.45, 4.6, vertical=True)
s.section(7.9, 1.42, 4.6, 'WHY P90–P99 ARE NOT REPORTED')
bullets(s, 7.9, 1.75, 4.65, 3.2, ['Both indexes showed ~2 s queries: 18% at nprobes 1, 34–37% at 2',
                                  'Exactly the same query IDs were slow at every ef → it depends on the partition probed, not on graph search',
                                  'Cause: the harness opened the dataset with Lance\'s default 6 GiB index cache; the indexes are 7.0 GB (HNSW) and 5.9 GB (PiPNN), '
                                  'so a partition is reloaded on every query that probes it',
                                  'Affects both indexes alike; build time, memory, size and recall are untouched',
                                  'Fix for phase 4: set index cache size explicitly, pre-warm, flag runs with > 1% queries over 100 ms'], size=8.5, line=1.3)
end(s, 'Recall comparison is complete for nprobes 1–2; tails need a cache fix, not a new index.', '')
notes(s, 'The query sweep stopped after nprobes 2 (PiPNN) and nprobes 3, ef 32 (HNSW). Slow-query analysis: per build, the slow sets are identical across '
      'the five ef values (HNSW 364 and 676 queries at nprobes 1 and 2; PiPNN 330 and 720). Lance DEFAULT_INDEX_CACHE_SIZE is 6 GiB. The host swap file '
      'being full was investigated and ruled out: it is filled by other tenants; this container used about 4 MB of swap.',
      ['RC', 'LSRC'])

# ================================================================ A3 constant scale
s = new('Appendix A3 · Constant RQ scale on GIST1M',
        '[x86] GIST1M, 1M × 960, K = 1,000 · 4-core Intel Xeon 2.8 GHz · 2 repeats, means · both arms of each pair use the same distance kernel')
s.table(.8, 1.4, 11.75, [['Code width', 'RaBitQ scale', 'Build s', 'IVF train s', 'Shuffle s', 'Speedup', 'R@10 np20', 'np50', 'np100'],
                         ['RQ5', 'per-vector search', '56.8', '13.0', '41.1', '1.00×', '0.800', '0.912', '0.949'],
                         ['RQ5', 'constant', '35.0', '12.7', '19.4', '1.62×', '0.792', '0.910', '0.946'],
                         ['RQ9', 'per-vector search', '190.3', '13.1', '172.6', '1.00×', '0.799', '0.917', '0.956'],
                         ['RQ9', 'constant', '38.5', '12.4', '21.9', '4.94×', '0.800', '0.919', '0.956']],
        col_widths=[1.1, 1.8, 1.0, 1.2, 1.1, 1.1, 1.2, 1.0, 1.0], row_h=.36, size=8.5,
        highlight={(2, 5): 'pale', (4, 5): 'pale'})
s.strip(.8, 3.55, 11.75, .9, 'How this relates to the project:',
        'constant scale changes only the encoding, so it is portable and enters the Kunpeng ablation as step 2. It costs up to 0.008 R@10 '
        '(RQ5, nprobes 20) and at most 0.003 at nprobes 50 or more; check more datasets before making it a default.')
end(s, 'Constant scale removes most of the multi-bit encoding cost.', '')
notes(s, 'From lance_ivf_rq results.jsonl. Constant scale is opt-in (LANCE_RQ_CONST_EX_SCALE=1): one scaling factor per dimension and bit width, '
      'averaged over 100 Gaussian vectors; stored correction factors are computed from the written codes. Each pair differs only in the scale '
      'setting. Comparisons against unpatched Lance are left out because those runs also changed the distance kernel and ISA width.',
      ['RG'], 'One machine, one dataset, 2 repeats.')

# ================================================================ A4 memory
s = new('Appendix A4 · PiPNN memory: model and measurements',
        'Plan §6 first-order model (decimal GB) vs measured peak RSS · PiPNN is the memory trade-off of WP3')
s.section(.8, 1.42, 5.5, 'FIRST-ORDER MODEL [PROPOSAL]')
s.table(.8, 1.72, 5.6, [['Term', 'Formula', 'N = 2M'], ['input f32 vectors', '4·N·D', '1.02 GB'], ['HashPrune slots', '8·N·S (S = 160)', '2.56 GB'],
                        ['final adjacency', '4·N·R (R = 64)', '0.51 GB'], ['sketches', '4·N·16', '0.13 GB'], ['subtotal', '', '≈ 4.22 GB'],
                        ['+ leaf scratch', '≈ 4·W·L²', 'per worker']],
        col_widths=[1.8, 2.2, 1.6], row_h=.33, size=8.5)
txt(s, .8, 4.15, 5.6, .5, 'At 10M rows across concurrently built partitions the same terms reach ≈ 21 GB. Capacity estimate, not RSS.', size=8.5, color='muted')
s.rule(6.65, 1.45, 3.6, vertical=True)
s.section(6.9, 1.42, 5.6, 'MEASURED PEAK RSS [x86]')
s.table(6.9, 1.72, 5.65, [['Run', 'HNSW', 'PiPNN'], ['Lance, 10M, K = 5, sequential partitions', '16.2 GB', '17.7 GB'],
                          ['Lance, 1M prefix, K = 5', '3.2 GB', '5.1 GB'], ['Standalone C++, 10M uint8 (FastPiPNN)', '6.1 GB', '18.8 GB']],
        col_widths=[3.45, 1.1, 1.1], row_h=.38, size=8.5)
txt(s, 6.9, 3.4, 5.65, .7, '1M prefix build time: HNSW 42.3 s vs PiPNN 8.6 s (4.9×), consistent with 10M.', size=8.5, color='muted')
s.strip(.8, 4.85, 11.75, .8, 'Mitigation in WP3:', 'one memory/CPU budget for active partitions, worker scratch and C++ threads; '
        'fail with a recorded capacity result rather than silently shrinking parameters. Kunpeng RAM per node must be recorded.')
end(s, 'Memory is bounded by how many partitions build at once.', '')
notes(s, 'Model from plan section 6 with N = 2M, D = 128, S = 160, R = 64. Measured: Suite C 10M peak 16.16 vs 17.67 GB; 1M memory gate 3.20 vs 5.08 GB; '
      'standalone FastPiPNN 18.8 GB vs HNSW 6.07 GB (original PiPNN 20.2 GB).', ['PLAN', 'RC', 'MEM', 'NAT'])

# ================================================================ A5 sources
s = new('Appendix A5 · Runs, sources and references',
        'All run folders are in the repository with manifest.json (revision, patch hash, machine, resolved options)')
s.table(.8, 1.4, 11.75, [['ID', 'Source', 'Used on'],
                         ['RA', 'lance_index_build/results/ivf_rq_assignment_results.md (pilotB, pilotA, probeA)', '2, 9, 10, A1'],
                         ['RC', 'lance_index_build/results/pilotC_hnsw_vs_pipnn_results.md (pilotC-bigann10m-k5)', '2, 12, A2'],
                         ['MEM', 'lance_index_build/results/pilotC-mem-1m', 'A4'],
                         ['RG', 'lance_ivf_rq/README.md, results/results.jsonl', '2, A3'],
                         ['SVE', 'lance_arm_sve/README.md, lance-arm-sve.patch, scripts/validate_emulated.py, benchmark_native.py', '5, 6, 7, 8'],
                         ['NAT', 'native/results/my-run/summary.json', '12, A4'],
                         ['PLAN', 'LANCE_INDEX_BUILD_PLAN.md (7 Nov 2026)', '5, 6–8, 13, A4'],
                         ['P1', 'Rubel et al., PiPNN, KDD 2026, arXiv:2602.21247v3', '11'],
                         ['ADS', 'ADS-PLAN.pptx slide 1 (internal)', '3']],
        col_widths=[.8, 8.9, 2.05], row_h=.36, size=8.5)
s.strip(.8, 5.15, 11.75, .75, 'Not established yet:',
        'any Kunpeng timing · native ARM Rust build · repeats ≥ 3 · k = 100 and nprobes ≥ 3 for suite C · per-process RSS for suite B. '
        'Previews may use a substitute font for Microsoft YaHei.')
end(s, 'Every number traces to a run folder.', '')
notes(s, 'Reference list with the slides that use each source.', list(SRC))

# ================================================================ save
warnings = D.save(str(OUT))
(ROOT / 'layout-warnings-v6.1.json').write_text(json.dumps(warnings, indent=1))
(ROOT / 'diagram-sources-v6.1.json').write_text(json.dumps(DIAGRAMS, indent=1, ensure_ascii=False))
manifest = []
for i, sl in enumerate(D.prs.slides, 1):
    texts = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    manifest.append({'slide': i, 'title': texts[1] if i > 1 and len(texts) > 1 else texts[0] if texts else '',
                     'notes_chars': len(sl.notes_slide.notes_text_frame.text)})
(ROOT / 'slide-manifest-v6.1.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

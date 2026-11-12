#!/usr/bin/env python3
"""Shared setup for the ADS research-track decks (track 4 hybrid retrieval, track 6 multimodal cache).

Reuses the header/helpers of build_deck_v6_1.py exactly as build_deck_v8.py does, so styling is identical
to v8; only the kicker, sources and evidence-label wording differ. The ADS pipeline slide is copied from
ads-lance-ivf-rq-proposal-v5.1.pptx slide 2 and re-targeted at stage (4) or (6).
"""
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parent
MARK = '# ================================================================ 1 title'
V61_KICKER = "KICKER = 'ADS · Lance Index Builds on Kunpeng'"

LANCE = 'lance-format/lance main @ 04e619963 (11 Nov 2026), verified by reading the source'
COMMON_SRC = {
    'ADS': 'ADS framework, main-framework.pptx (internal company context; pipeline diagram redrawn in ADS-PLAN slide 1 / v5.1 deck).',
    'DOC': 'lance_indexing_cache_research.md v2 (11 Nov 2026), sections 3, 3.9, 5.5, 5.6 and 11.',
    'LCACHE': f'{LANCE}: rust/lance-core/src/cache/quick.rs (quick_cache 0.6.24, S3-FIFO), moka.rs (moka 0.12, TinyLFU), '
              'backend.rs (CacheTier Resident/Local/Absent, get_without_promotion), rust/lance/src/dataset.rs L242/L246 (6 GiB index, 1 GiB metadata).',
    'S3F': 'Yang et al., FIFO Queues Are All You Need for Cache Eviction (S3-FIFO), SOSP 2023; quick_cache README.',
    'TLFU': 'Einziger, Friedman, Manes, TinyLFU: A Highly Efficient Cache Admission Policy, ACM TOS 2017, https://doi.org/10.1145/3149371',
}


def load(kicker, extra_src, out_name):
    src = (ROOT / 'build_deck_v6_1.py').read_text()
    prefix = src.split(MARK)[0]
    assert V61_KICKER in prefix, 'v6.1 kicker line moved; update ads_track_common.py'
    prefix = prefix.replace(V61_KICKER, f'KICKER = {kicker!r}')
    g = {'__file__': str(ROOT / 'build_deck_v6_1.py'), '__name__': 'v61_prefix'}
    exec(compile(prefix, 'build_deck_v6_1.py[prefix]', 'exec'), g)
    g['SRC'].update(COMMON_SRC)
    g['SRC'].update(extra_src)
    g['OUT'] = ROOT / out_name
    MSO_SHAPE, txt = g['MSO_SHAPE'], g['txt']
    SRC = g['SRC']
    Canvas = g['Canvas']

    def notes(s, spoken, keys, limits=''):
        text = spoken + '\n\nEvidence labels: [Source] = read in Lance code (' + LANCE + '); [Reported] = paper, author-reported, ' \
            'not reproduced on ADS data; [Model] = our analytic estimate, assumptions stated; [Assumption] = to be confirmed with ADS; ' \
            '[Proposal] = planned work; TBD = to be measured in phase 1.\n'
        if limits:
            text += '\nLimits: ' + limits + '\n'
        text += '\nSources:\n' + '\n'.join(SRC[k] for k in keys)
        if isinstance(s, Canvas):
            s.notes(text)
        else:
            s.notes_slide.notes_text_frame.text = text

    def head(s, x, y, w, t, col='ink', size=10.5):
        s.rect(x, y + .05, .06, .2, col, None)
        txt(s, x + .14, y, w - .14, .3, t, size=size, color='ink', bold=True, anchor='ctr')

    def dot(s, x, y, col='muted', d=.08):
        s.rect(x - d / 2, y - d / 2, d, d, col, None, shape=MSO_SHAPE.OVAL)

    g.update(notes=notes, head=head, dot=dot)
    return g


def ads_slide(g, title, tagline, target, callout, callout_lines, takeaway, rest, spoken):
    """Copy the v5.1 ADS pipeline slide and move the red emphasis from stage (3) to `target` ('(4)' or '(6)').

    callout = (x, y, w, h, anchor_x, anchor_y_from, anchor_y_to) in inches.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    D, Canvas, C = g['D'], g['Canvas'], g['C']
    srcs = Presentation(str(g['V51'])).slides[1]
    s2 = D.prs.slides.add_slide(D._blank)
    for el in list(s2.shapes._spTree):
        if el.tag.endswith('}sp') or el.tag.endswith('}pic'):
            s2.shapes._spTree.remove(el)
    for el in srcs.shapes._spTree:
        if el.tag.split('}')[1] in ('sp', 'cxnSp', 'grpSp', 'pic', 'graphicFrame'):
            s2.shapes._spTree.append(copy.deepcopy(el))

    def col(key):
        return RGBColor.from_string(C[key])

    shapes = list(s2.shapes)
    boxes = {}
    for sh in shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t == 'ADS · Faster Lance Index Builds':
            sh.text_frame.paragraphs[0].runs[0].text = g['KICKER']
        elif t.startswith('Target (3)'):
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = title
            for r in runs[1:]:
                r.text = ''
        elif t.startswith('[Company context]'):
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = tagline
            for r in runs[1:]:
                r.text = ''
        elif t.startswith('Index construction is the intervention point.'):
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = takeaway
            for r in runs[1:]:
                r.text = ''
            if len(runs) > 1:
                runs[1].text = ' ' + rest
        for key in ('(3)', '(4)', '(6)'):
            if t.startswith(key):
                boxes[key] = sh
    E = 914400

    def rect_behind(tb):
        for sh in shapes:
            if (not sh.has_text_frame or not sh.text_frame.text.strip()) and abs(sh.top - tb.top) < E * .02 \
                    and abs(sh.left - (tb.left - int(E * .06))) < E * .02:
                return sh
        raise RuntimeError('no box behind ' + tb.text_frame.text)

    def paint(tb, fill, line, lead, sub):
        r = rect_behind(tb)
        r.fill.solid()
        r.fill.fore_color.rgb = col(fill)
        if line:
            r.line.color.rgb = col(line)
            r.line.width = int(12700 * .75)
        else:
            r.line.fill.background()
        for p in tb.text_frame.paragraphs:
            for run in p.runs:
                run.font.color.rgb = col(lead if run.font.bold else sub)

    paint(boxes['(3)'], 'white', 'border', 'black', 'body')
    paint(boxes[target], 'red', None, 'white', 'on_red_sub')
    for sh in shapes:  # the red route into stage (3) goes back to neutral grey
        if sh.shape_type == 9:
            try:
                if str(sh.line.color.rgb) == C['red']:
                    sh.line.color.rgb = col('muted')
            except Exception:
                pass
    c = Canvas(D, s2, title)
    D.canvases.append(c)
    tb = boxes[target]
    bx, by, bw, bh = tb.left / E - .06, tb.top / E, tb.width / E + .12, tb.height / E
    c.rect(bx - .03, by - .06, bw + .06, bh + .12, None, 'red', 1.5, dashed=True)
    x, y, w, h, ax, ay0, ay1 = callout
    c.route([(ax, ay0), (ax, ay1)], color='red')
    c.rect(x, y, w, h, 'white', 'red', 1.0)
    paras = [[(callout_lines[0], {'bold': True, 'color': 'title', 'size': 7.5})]] + \
        [[(ln, {'size': 7.5, 'color': 'ink'})] for ln in callout_lines[1:]]
    c.text(x + .05, y + .01, w - .1, h - .02, paras, line=1.1, anchor='ctr', align='ctr')
    s2.notes_slide.notes_text_frame.text = spoken + '\n\nSources:\n' + g['SRC']['ADS']
    return s2


def finish(g, tag):
    """Save deck plus layout warnings, diagram sources and manifest, like build_deck_v8.py."""
    import json
    D, OUT = g['D'], g['OUT']
    warnings = D.save(str(OUT))
    (ROOT / f'layout-warnings-{tag}.json').write_text(json.dumps(warnings, indent=1))
    (ROOT / f'diagram-sources-{tag}.json').write_text(json.dumps(g['DIAGRAMS'], indent=1, ensure_ascii=False))
    manifest = []
    for i, sl in enumerate(D.prs.slides, 1):
        texts = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        manifest.append({'slide': i, 'title': texts[1] if i > 1 and len(texts) > 1 else texts[0] if texts else '',
                         'notes_chars': len(sl.notes_slide.notes_text_frame.text)})
    (ROOT / f'slide-manifest-{tag}.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print('wrote', OUT, 'warnings:', len(warnings))
    for w_ in warnings:
        print('  ', w_)

#!/usr/bin/env python3
"""Summarize the equal-width blocked vs per-vector ablation (scripts/tile_width_ablation.sh).

Usage:
  tile_width_report.py micro <results/.../tilewidth-micro-STAMP>
  tile_width_report.py e2e   <results/<arch>-<host>> <STAMP>

Speedups are ratios of medians (micro: median over rounds of each run's median
call; e2e: median over rounds). min-max ranges show run-to-run spread.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ['P256', 'B256', 'P512', 'B512', 'REF']


def fmt_range(xs):
    return f'{min(xs):.3f}–{max(xs):.3f}'


def micro(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / 'micro.jsonl').read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    meta = {}
    for r in rows:
        by[(r['case'], r['k'], r['arm'])].append(r['median_s'])
        meta[(r['case'], r['k'])] = r
        meta[(r['case'], r['k'], r['arm'])] = r['kernels']
    out = ['# Blocked vs per-vector exact assignment at equal SIMD width (microbenchmark)', '',
           f'Run: `{run_dir.name}`. Lance `compute_partitions_arrow_array`, real rows, centroids = base rows at a fixed stride.',
           'Times: median over rounds of the per-run median of 3 timed calls (1 warm-up). Ranges are min–max over rounds.', '',
           '| case | D | K | table | rows | thr | P256 s | B256 s | **B256 vs P256** | P512 s | B512 s | **B512 vs P512** | '
           'REF s | REF vs B512 (old, confounded) | P256 vs P512 (width, per-vector) | B256 vs B512 (width, blocked) |',
           '|' + '---|' * 16]
    cases = sorted({(c, k) for c, k, _ in by}, key=lambda x: (x[0].split('-')[1], x[0].split('-')[0], x[1]))
    for c, k in cases:
        m = {a: st.median(by[(c, k, a)]) for a in ARMS if (c, k, a) in by}
        r = meta[(c, k)]
        tb = r['table_bytes'] / 2**20
        sp = lambda a, b: f'{m[a] / m[b]:.2f}×' if a in m and b in m else '—'
        out.append(f"| {c} | {r['dim']} | {k} | {tb:.2f} MiB | {r['rows']:,} | {r['threads']} | "
                   f"{m['P256']:.3f} | {m['B256']:.3f} | **{sp('P256', 'B256')}** | {m['P512']:.3f} | {m['B512']:.3f} | "
                   f"**{sp('P512', 'B512')}** | {m.get('REF', float('nan')):.3f} | {sp('REF', 'B512')} | "
                   f"{sp('P256', 'P512')} | {sp('B256', 'B512')} |")
    out += ['', 'Speedup columns are slower ÷ faster as labelled: "B256 vs P256" = P256 time ÷ B256 time (> 1 means blocked is faster).',
            'Width columns: "P256 vs P512" = P256 ÷ P512 (> 1 means 512-bit is faster).', '',
            '## Spread (min–max seconds over rounds)', '', '| case | K | ' + ' | '.join(ARMS) + ' |', '|' + '---|' * (2 + len(ARMS))]
    for c, k in cases:
        out.append(f'| {c} | {k} | ' + ' | '.join(fmt_range(by[(c, k, a)]) if (c, k, a) in by else '—' for a in ARMS) + ' |')
    # Assignment agreement: every arm vs P256, same case/K/round.
    out += ['', '## Kernels selected', '', '| arm | blocked_l2 | dot_tile | l2_f32_batch | l2_f32_pair |', '|---|---|---|---|---|']
    c0, k0 = cases[0]
    for a in ARMS:
        kn = meta.get((c0, k0, a))
        if kn:
            out.append(f"| {a} | {kn['blocked_l2']} | {kn['dot_tile']} | {kn['l2_f32_batch']} | {kn['l2_f32_pair']} |")
    out += ['', '## Assignment agreement with P256 (fraction of rows with the same centroid)', '',
            '| case | K | ' + ' | '.join(ARMS[1:]) + ' |', '|' + '---|' * (1 + len(ARMS))]
    for c, k in cases:
        ref = run_dir / 'assign' / f'{c}-P256-r0-k{k}.u32'
        base = ref.read_bytes()
        n = len(base) // 4
        cells = []
        for a in ARMS[1:]:
            p = run_dir / 'assign' / f'{c}-{a}-r0-k{k}.u32'
            if not p.exists():
                cells.append('—'); continue
            other = p.read_bytes()
            same = sum(base[i:i + 4] == other[i:i + 4] for i in range(0, len(base), 4))
            cells.append(f'{same / n:.6f}')
        out.append(f'| {c} | {k} | ' + ' | '.join(cells) + ' |')
    loads = [float(r['load']) for r in rows]
    out += ['', f'1-minute load average at run start: {min(loads):.1f}–{max(loads):.1f} (shared 32-core host).']
    text = '\n'.join(out) + '\n'
    (run_dir / 'summary.md').write_text(text)
    print(text)


def e2e(results: Path, stamp: str):
    rec = defaultdict(list)
    kern = {}
    for d in sorted(results.glob(f'tw-{stamp}-*')):
        parts = d.name[len(f'tw-{stamp}-'):].rsplit('-', 2)  # workload, arm, rN
        wl, arm = parts[0], parts[1]
        bl = d / 'builds.jsonl'
        if not bl.exists():
            continue
        for l in bl.read_text().splitlines():
            b = json.loads(l)
            rec[(wl, arm)].append(b)
        kern[arm] = json.loads((d / 'manifest.json').read_text())['kernels']
    out = ['# Blocked vs per-vector exact assignment at equal SIMD width (end-to-end IVF_RQ builds)', '',
           f'Runs `tw-{stamp}-*`. Each cell: median over rounds (min–max). Exact assignment, same configs per arm.', '']
    for wl in sorted({w for w, _ in rec}):
        out += [f'## {wl}', '', '| arm | rounds | build s | assign thread-s | fixed train s | cpu s | recall@10 |', '|---|---|---|---|---|---|---|']
        med = {}
        for arm in ARMS:
            bs = rec.get((wl, arm))
            if not bs:
                continue
            g = lambda key: [b[key] for b in bs if b.get(key) is not None]
            build, asg, cpu = g('build_secs'), g('assign_thread_secs'), g('cpu_secs')
            tr = g('fixed_train_secs')
            med[arm] = (st.median(build), st.median(asg) if asg else None)
            rc = defaultdict(list)
            for b in bs:
                for s in b['search']:
                    if s['k'] == 10:
                        rc[s['nprobes']].append(s['recall'])
            rtxt = ', '.join(f'np{n} {st.median(v):.4f}' for n, v in sorted(rc.items()))
            cell = lambda xs: f'{st.median(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})' if xs else '—'
            out.append(f'| {arm} | {len(bs)} | {cell(build)} | {cell(asg)} | {cell(tr)} | {cell(cpu)} | {rtxt} |')
        out.append('')
        for a, b, label in [('P256', 'B256', 'blocked vs per-vector, 256-bit'), ('P512', 'B512', 'blocked vs per-vector, 512-bit'),
                            ('P256', 'P512', 'width effect, per-vector'), ('B256', 'B512', 'width effect, blocked')]:
            if a in med and b in med:
                asg = f', assignment {med[a][1] / med[b][1]:.2f}×' if med[a][1] and med[b][1] else ''
                out.append(f'- {label}: build {med[a][0] / med[b][0]:.2f}×{asg} ({a} ÷ {b})')
        out.append('')
    out += ['## Kernels selected', '', '| arm | blocked_l2 | dot_tile | l2_f32_batch | l2_f32_pair |', '|---|---|---|---|---|']
    for a in ARMS:
        if a in kern:
            k = kern[a]
            out.append(f"| {a} | {k.get('blocked_l2')} | {k.get('dot_tile')} | {k.get('l2_f32_batch')} | {k.get('l2_f32_pair')} |")
    text = '\n'.join(out) + '\n'
    (results / f'tw-{stamp}-summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    if sys.argv[1] == 'micro':
        micro(Path(sys.argv[2]))
    else:
        e2e(Path(sys.argv[2]), sys.argv[3])

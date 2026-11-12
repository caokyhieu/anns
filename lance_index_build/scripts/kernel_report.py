#!/usr/bin/env python3
"""ARM kernel session report: legacy Lance vs the new kernels at three levels.

Usage: scripts/kernel_report.py results/<host>/armk-<TAG> [-o OUT_DIR]

Reads what kernels_all.sh leaves in the session directory:
  criterion-target/, criterion-target-nosve/   kernel microbenchmarks (SVE vs no-SVE build)
  stages/<case>/<arm>/stages.jsonl             standalone stage bench, one dir per arm
  e2e/<case>/<arm>/builds.jsonl                end-to-end Lance build + search
Writes report.md (tables) and report.json (every aggregated number, to pick from).
Medians over repeats; speedup = legacy time / arm time (> 1 is faster). Standard library only.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ["legacy", "blocked", "sve", "new"]
# Standalone stages that make up each stage Lance's build reports (the projection).
E2E_FROM_STAGES = {
    "train_ivf": [("train", "kmeans")],
    "train_quantizer": [("quantizer", None)],
    "shuffle": [("assign", "auto"), ("encode", "residual"), ("encode", "rq")],
}


def jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def med(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def f(v, digits=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{digits}g}" if abs(v) < 1 else f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def speedup(base, other):
    return base / other if base and other else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------- level 1: criterion ----------

def criterion(tree, baseline):
    """{benchmark key: mean ns} for one criterion directory and baseline name."""
    out = {}
    if not tree.exists():
        return out
    for est in tree.rglob(f"{baseline}/estimates.json"):
        parts = list(est.relative_to(tree).parts[:-2])
        # The tile group is named after the dispatched kernel; compare across builds.
        parts[0] = re.sub(r"^index_tiles.*", "index_tiles", parts[0])
        out["/".join(parts)] = json.loads(est.read_text())["mean"]["point_estimate"]
    return out


def micro_section(session):
    new = criterion(session / "criterion-target", "default")
    legacy = criterion(session / "criterion-target-nosve", "nosve")
    rows = []
    for key in sorted(set(new) | set(legacy)):
        rows.append([key, legacy.get(key), new.get(key), speedup(legacy.get(key), new.get(key))])
    data = [dict(zip(["benchmark", "legacy_ns", "new_ns", "speedup"], r)) for r in rows]
    if not rows:
        return "_No criterion results (step `micro`)._", data
    md = ("Mean ns per call. legacy = build without SVE objects (Lance's NEON / scalar kernels "
          "and the NEON tile); new = SVE build.\n\n"
          + table(["benchmark", "legacy ns", "new ns", "speedup"], rows))
    return md, data


# ---------- level 2: standalone stages ----------

def stage_rows(session, case):
    by = defaultdict(lambda: defaultdict(list))  # (stage, variant) -> arm -> rows
    kernels = {}
    for d in sorted((session / "stages" / case).glob("*")):
        if not (d / "stages.jsonl").exists():
            continue
        for r in jsonl(d / "stages.jsonl"):
            if r.get("status") == "error":
                continue
            by[(r["stage"], r["variant"])][r["arm"]].append(r)
            kernels[r["arm"]] = r.get("kernels")
    return by, kernels


def metric_time(stage, rows):
    if stage == "probe":
        return med([r.get("latency_us_p50") for r in rows])
    return med([r.get("secs") for r in rows])


def quality(stage, variant, arm_rows, legacy_rows):
    if stage == "train":
        loss, ref = med([r.get("mean_loss") for r in arm_rows]), med([r.get("mean_loss") for r in legacy_rows])
        return f"loss {f(loss)} ({(loss / ref - 1) * 100:+.3f}% vs legacy)" if loss and ref else "–"
    if stage == "assign":
        ref = [r["vs_ref_exact"] for r in arm_rows if r.get("vs_ref_exact")]
        if ref:
            agree = med([v["id_agreement"] for v in ref])
            drift = med([v["rel_dist_diff_mean"] for v in ref])
            return f"ids = legacy {agree * 100:.4f}%, dist {drift:+.2e}"
        ex = [r["vs_exact"] for r in arm_rows if r.get("vs_exact")]
        if ex:
            return f"ids = exact {med([v['id_agreement'] for v in ex]) * 100:.3f}%"
        return f"objective {f(med([r.get('objective') for r in arm_rows]))}"
    if stage == "probe":
        rec = med([r.get("probe_recall") for r in arm_rows])
        return f"probe recall@{arm_rows[0].get('probe_recall_k')} {f(rec)}" if rec is not None else "–"
    return "–"


def stage_section(session, case):
    by, kernels = stage_rows(session, case)
    if not by:
        return None, {}
    arms = [a for a in ARM_ORDER if any(a in v for v in by.values())]
    order = {"train": 0, "quantizer": 1, "assign": 2, "encode": 3, "probe": 4}

    def sort_key(sv):
        m = re.search(r"\d+", sv[1]) if sv[0] == "probe" else None
        return (order.get(sv[0], 9), int(m.group()) if m else 0, sv[1])

    def in_build(stage, variant):
        # What Lance's build runs: the optional tree / pipnn / hnsw specs are extra.
        return stage in ("train", "quantizer", "encode") or (stage, variant) == ("assign", "auto")
    rows, data = [], {}
    legacy_build = 0.0
    for (stage, variant) in sorted(by, key=sort_key):
        per = by[(stage, variant)]
        times = {a: metric_time(stage, per.get(a, [])) for a in arms}
        base = times.get("legacy")
        cells = [f"{stage} / {variant}"] + [times[a] for a in arms]
        cells += [speedup(base, times.get(a)) for a in arms if a != "legacy"]
        q = {a: quality(stage, variant, per.get(a, []), per.get("legacy", [])) for a in arms}
        cells.append(q.get("new") or q.get(arms[-1]))
        rows.append(cells)
        data[f"{stage}/{variant}"] = {"time": times, "quality": q,
                                     "unit": "us p50 per query" if stage == "probe" else "s"}
        if in_build(stage, variant) and base:
            legacy_build += base
    header = (["stage / variant"] + [f"{a} (s)" for a in arms]
              + [f"×{a}" for a in arms if a != "legacy"] + ["accuracy (new arm)"])
    rows_md = table(header, rows).replace("probe / nprobes", "probe (µs p50) / nprobes")
    kern = table(["arm", "l2_f32_pair", "l2_f32_batch", "dot_tile", "blocked_l2"],
                 [[a] + [str((kernels.get(a) or {}).get(k)) for k in
                         ["l2_f32_pair", "l2_f32_batch", "dot_tile", "blocked_l2"]] for a in arms])
    shares = []
    for (stage, variant) in sorted(by, key=sort_key):
        t = metric_time(stage, by[(stage, variant)].get("legacy", []))
        if in_build(stage, variant) and t and legacy_build:
            shares.append(f"{stage}/{variant} {t / legacy_build * 100:.0f}%")
    md = (f"Kernels each arm dispatched:\n\n{kern}\n\n{rows_md}\n\n"
          f"Legacy build compute, standalone ({f(legacy_build)} s, Lance's `auto` assignment): "
          + ", ".join(shares))
    return md, {"stages": data, "kernels": kernels, "legacy_build_secs": legacy_build}


# ---------- level 3: end to end ----------

def e2e_builds(session, case):
    out = {}
    for d in sorted((session / "e2e" / case).glob("*")):
        rows = jsonl(d / "builds.jsonl") or jsonl(d / "built.jsonl")
        rows = [r for r in rows if r.get("status") != "error"]
        if rows:
            out[d.name] = rows
    return out


def e2e_section(session, case, stage_data):
    builds = e2e_builds(session, case)
    if not builds:
        return None, {}
    arms = [a for a in ARM_ORDER if a in builds]
    stage_names = []
    for rows in builds.values():
        for r in rows:
            for s in (r.get("stages") or {}):
                if s not in stage_names:
                    stage_names.append(s)
    agg = {a: {"build_secs": med([r["build_secs"] for r in builds[a]]),
               "peak_rss_gb": med([(r.get("peak_rss_bytes") or 0) / 1e9 for r in builds[a]]),
               "stages": {s: med([(r.get("stages") or {}).get(s) for r in builds[a]]) for s in stage_names}}
           for a in arms}
    rows = []
    for s in stage_names + ["build_secs"]:
        vals = [agg[a]["build_secs"] if s == "build_secs" else agg[a]["stages"][s] for a in arms]
        rows.append([s if s != "build_secs" else "**total build**"] + vals
                    + [speedup(vals[0], v) for v in vals[1:]])
    md = table(["Lance stage"] + [f"{a} (s)" for a in arms] + [f"×{a}" for a in arms[1:]], rows)

    # Search: per (k, nprobes), recall and latency per arm.
    search = defaultdict(dict)
    for a in arms:
        acc = defaultdict(list)
        for r in builds[a]:
            for s in r.get("search") or []:
                acc[(s["k"], s["nprobes"])].append(s)
        for key, ss in acc.items():
            search[key][a] = {"recall": med([s["recall"] for s in ss]),
                              "p50_us": med([s["p50_us"] for s in ss]),
                              "p99_us": med([s["p99_us"] for s in ss])}
    srows = []
    for (k, np_) in sorted(search):
        cell = search[(k, np_)]
        row = [f"k={k} nprobes={np_}"]
        for a in arms:
            c = cell.get(a, {})
            row += [c.get("recall"), c.get("p50_us")]
        lg, nw = cell.get(arms[0], {}).get("p50_us"), cell.get(arms[-1], {}).get("p50_us")
        row.append(speedup(lg, nw))
        srows.append(row)
    if srows:
        hdr = ["search"] + sum([[f"{a} recall", f"{a} p50 µs"] for a in arms], []) + ["×p50"]
        md += "\n\nSearch (warm, per query):\n\n" + table(hdr, srows)

    # Projection: legacy breakdown minus the standalone stage savings of the new arm.
    proj_md, proj = "", {}
    st = (stage_data or {}).get("stages", {})
    if "legacy" in agg and st:
        prow, total_saved = [], 0.0
        for e2e_stage, parts in E2E_FROM_STAGES.items():
            legacy_t = agg["legacy"]["stages"].get(e2e_stage)
            saved = 0.0
            for stage, variant in parts:
                for key, v in st.items():
                    s, var = key.split("/", 1)
                    if s == stage and (variant is None or var == variant):
                        lt, nt = v["time"].get("legacy"), v["time"].get("new")
                        if lt and nt:
                            saved += lt - nt
            if legacy_t is None:
                continue
            projected = max(legacy_t - saved, 0.0)
            total_saved += saved
            measured = agg.get("new", {}).get("stages", {}).get(e2e_stage)
            prow.append([e2e_stage, legacy_t, saved, projected, measured])
            proj[e2e_stage] = {"legacy": legacy_t, "saved": saved, "projected": projected, "measured_new": measured}
        lt = agg["legacy"]["build_secs"]
        prow.append(["**total build**", lt, total_saved, lt - total_saved, agg.get("new", {}).get("build_secs")])
        proj["total"] = {"legacy": lt, "saved": total_saved, "projected": lt - total_saved,
                         "measured_new": agg.get("new", {}).get("build_secs")}
        proj_md = ("\n\nProjection: the legacy build's own breakdown minus the time the standalone stages "
                   "saved (legacy − new). The rest of each stage (I/O, shuffle spill, writing) is assumed "
                   "unchanged; *measured new* is the actual end-to-end run with the kernels.\n\n"
                   + table(["Lance stage", "legacy (s)", "saved (s)", "projected (s)", "measured new (s)"], prow))
    return md + proj_md, {"builds": agg, "search": {f"k={k},nprobes={n}": v for (k, n), v in search.items()},
                          "projection": proj}


RECALL_TOLERANCE = 0.005


def sve_width(session, case):
    for d in sorted((session / "stages" / case).glob("*/logs/sve_vector_bytes.txt")):
        v = d.read_text().strip()
        if v.isdigit():
            return int(v) * 8
    return None


def gate_section(session, case, e2e_data):
    """The deck's KT3 first gate: blocked beats pairwise at equal vector width, with
    recall within ±RECALL_TOLERANCE of legacy."""
    by, _ = stage_rows(session, case)
    if not by:
        return None, {}
    bits = sve_width(session, case)
    pairs = [("legacy", "blocked", "NEON (128-bit)"),
             ("sve", "new", f"SVE ({bits}-bit)" if bits else "SVE (width not recorded)")]
    rows, data = [], {}
    for stage, variant in [("train", "kmeans"), ("assign", "exact"), ("assign", "auto")]:
        per = by.get((stage, variant))
        if not per:
            continue
        for pairwise, blocked, width in pairs:
            tp, tb = metric_time(stage, per.get(pairwise, [])), metric_time(stage, per.get(blocked, []))
            if tp is None or tb is None:
                continue
            agree = [r["vs_ref_exact"]["id_agreement"] for r in per.get(blocked, []) if r.get("vs_ref_exact")]
            sp = speedup(tp, tb)
            rows.append([f"{stage} / {variant}", width, f"{pairwise} → {blocked}", tp, tb, sp,
                         f"{med(agree) * 100:.4f}%" if agree else "–",
                         "pass" if sp and sp > 1 else "FAIL"])
            data[f"{stage}/{variant}/{width}"] = {"pairwise_s": tp, "blocked_s": tb, "speedup": sp,
                                                  "id_agreement_vs_legacy": med(agree)}
    md = ("Blocked vs pairwise at equal vector width (the deck's first gate). ids = share of rows "
          "assigned to the same centroid as legacy, with legacy's centroids.\n\n"
          + table(["stage", "width", "pairwise → blocked", "pairwise (s)", "blocked (s)", "×", "ids = legacy", "faster?"], rows))

    # Recall. The gate is the probe recall: every arm searches legacy's centroids, so any
    # difference comes from the kernels alone. End-to-end search trains its own k-means per
    # arm, and float rounding moves the centroids; there only a drop beyond the tolerance
    # counts against the arm (a gain is reported, not failed).
    rrows, worst, worst_drop = [], 0.0, 0.0
    probe = sorted(((k, v) for k, v in by.items() if k[0] == "probe"),
                   key=lambda kv: int(re.search(r"\d+", kv[0][1]).group()))
    for (stage, variant), per in probe:
        ref = med([r.get("probe_recall") for r in per.get("legacy", [])])
        for arm in ARM_ORDER[1:]:
            got = med([r.get("probe_recall") for r in per.get(arm, [])])
            if ref is not None and got is not None:
                worst = max(worst, abs(got - ref))
                rrows.append([f"probe {variant}", arm, ref, got, f"{got - ref:+.4f}"])

    def search_key(kv):
        nums = [int(n) for n in re.findall(r"\d+", kv[0])]
        return nums
    for key, cell in sorted((e2e_data or {}).get("search", {}).items(), key=search_key):
        ref = cell.get("legacy", {}).get("recall")
        for arm in ARM_ORDER[1:]:
            got = cell.get(arm, {}).get("recall")
            if ref is not None and got is not None:
                worst_drop = max(worst_drop, ref - got)
                rrows.append([f"search {key}", arm, ref, got, f"{got - ref:+.4f}"])
    if rrows:
        ok = worst <= RECALL_TOLERANCE and worst_drop <= RECALL_TOLERANCE
        md += (f"\n\nRecall vs legacy (tolerance ±{RECALL_TOLERANCE}): same centroids (probe) worst |Δ| = {worst:.4f}; "
               f"end-to-end search (own k-means) worst drop = {max(worst_drop, 0.0):.4f} → **{'pass' if ok else 'FAIL'}**\n\n"
               + table(["where", "arm", "legacy recall", "arm recall", "Δ"], rrows))
        data["recall"] = {"probe_worst_abs_delta": worst, "e2e_worst_drop": worst_drop,
                          "tolerance": RECALL_TOLERANCE, "pass": ok}
    data["sve_bits"] = bits
    return md, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()
    s, out = args.session, args.out or args.session
    cases = sorted({p.name for p in (s / "stages").glob("*")} | {p.name for p in (s / "e2e").glob("*")})
    md = [f"# ARM kernels: legacy Lance vs new kernels ({s.name})", ""]
    report = {"session": s.name, "cases": {}}
    micro_md, micro = micro_section(s)
    md += ["## 1. Kernels (criterion)", "", micro_md, ""]
    report["micro"] = micro
    for c in cases:
        sm, sd = stage_section(s, c)
        em, ed = e2e_section(s, c, sd)
        gm, gd = gate_section(s, c, ed)
        md += [f"## {c}", "",
               "### KT3 first gate (deck slide 7)", "", gm or "_No stage runs._", "",
               "### 2. Standalone IVF_RQ stages (Lance's stage functions, in memory, no index written)", "",
               sm or "_No stage runs._", "",
               "### 3. End-to-end Lance IVF_RQ build + search", "", em or "_No end-to-end runs._", ""]
        report["cases"][c] = {"stages": sd, "e2e": ed, "gate": gd}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out / 'report.md'} and report.json")


if __name__ == "__main__":
    main()

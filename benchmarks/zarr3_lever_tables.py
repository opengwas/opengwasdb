"""The markdown tables #244, #240, #246 and #253 quote, generated from #244's outputs.

Nothing in these tables is typed by hand. Each subcommand reads the committed
outputs (`--outputs`, by default `docs/benchmark-output/opengwasdb_zarr3_read_levers`)
and the #242 zarr 2.18 baseline, and prints one table:

  attribution  per lever, per shape (#244's Stage A re-run comment; ADR 0056)
  pairs        the #242 harness, 2.18 against zarr 3 + levers, two back-to-back pairs
  memory       peak RSS per query, 2.18 against zarr 3 after step 1
  time         set-L time per query, measured and expected (#246's body)
  eaf          the duplicate EAF read's share of each query (#253's body)
  budget       set L against zarr 3 after step 1 and the expected 0.2.0 shapes
  proposals    the three budget sets the user chose between (#240's decision comment)
  stage-b      #244's Stage B: each back-to-back pair of #242 harness runs (`--pair BASE
               HEAD`, repeatable) against set L and the whole-Analysis guard, with a verdict
  shapes       #246: the zarr 2.18 run (`--base`) against every zarr 3 configuration
               (`--head`, one store label per shape) against set L, the whole-Analysis
               guard and the 30%-of-a-limit confirm rule

The budget values are the user's decision (set L) or the proposals it was
chosen from; every other number is read from a file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks import _zarr3_levers as levers

#: The attribution run's configurations, in the order the table shows them.
CONFIG_TITLES = [
    ("base_218", "zarr 2.18 (745796c)"),
    ("i_none", "(i) zarr 3, no fix (708d179)"),
    ("ii_bt", "(ii) + Blosc threads"),
    ("iii_bt_fused", "(iii) + fused, 1 worker"),
    ("iv_head", "(iv) + arrays opened once (3339776)"),
]
SHORT = {
    "rand_10x100": "random 10 variants x 100 Analyses",
    "rand_100x10": "random 100 variants x 10 Analyses",
}
#: The order and names the budget tables use.
BUDGET_ORDER = [
    "tophits",
    "phewas",
    "regional_one_analysis",
    "rand_10x100",
    "rand_100x10",
    "regional",
    "bulk",
]
NAME = {
    "tophits": "Top hits for one Analysis",
    "phewas": "One variant, all Analyses",
    "regional_one_analysis": "One window, one Analysis",
    "rand_10x100": "10 variants × 100 Analyses",
    "rand_100x10": "100 variants × 10 Analyses",
    "regional": "One window, all Analyses",
    "bulk": "One whole Analysis",
}
PAIR_LABEL = {
    "tophits": "tophits",
    "phewas": "phewas",
    "regional_one_analysis": "regional, one Analysis",
    "rand_10x100": "random 10 variants × 100 Analyses",
    "rand_100x10": "random 100 variants × 10 Analyses",
    "bulk": "bulk (one Analysis, genome-wide)",
    "regional": "regional (1 Mb × all Analyses)",
}
#: The eaf_split outputs name the two lookups differently.
SPLIT = {"rand_10x100": "random_10x100", "rand_100x10": "random_100x10"}

#: Set L, as decided: time (typical ms, slow ms) and peak-memory cap (MB).
BUDGET_L = {
    "tophits": (50, 100, 300),
    "phewas": (100, 250, 300),
    "regional_one_analysis": (250, 500, 300),
    "rand_10x100": (500, 1000, 300),
    "rand_100x10": (3000, 5000, 500),
    "regional": (5000, 10_000, 1500),
    "bulk": (60_000, 90_000, 2000),
}
#: The three sets proposed: (typical ms, slow ms) per shape. L is the one chosen.
PROPOSALS = {
    "T": {
        "tophits": (10, 20),
        "phewas": (25, 50),
        "regional_one_analysis": (50, 100),
        "rand_10x100": (150, 250),
        "rand_100x10": (1000, 1500),
        "bulk": (30_000, 40_000),
        "regional": (2000, 3000),
    },
    "M": {
        "tophits": (25, 50),
        "phewas": (50, 100),
        "regional_one_analysis": (100, 200),
        "rand_10x100": (250, 500),
        "rand_100x10": (1500, 2500),
        "bulk": (30_000, 45_000),
        "regional": (3000, 5000),
    },
    "L": {shape: (budget[0], budget[1]) for shape, budget in BUDGET_L.items()},
}
PROPOSAL_ORDER = [
    "tophits",
    "phewas",
    "regional_one_analysis",
    "rand_10x100",
    "rand_100x10",
    "bulk",
    "regional",
]
USE = {
    "tophits": "interactive",
    "phewas": "interactive",
    "regional_one_analysis": "interactive",
    "rand_10x100": "interactive",
    "rand_100x10": "interactive",
    "bulk": "batch",
    "regional": "batch",
}


def fmt_attribution(ms: float) -> str:
    return f"{ms:,.0f}" if ms >= 100 else f"{ms:,.1f}" if ms >= 10 else f"{ms:,.2f}"


def fmt_pairs(ms: float) -> str:
    if ms >= 10_000:
        return f"{ms / 1000:.1f} s"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 100:
        return f"{ms:.0f} ms"
    if ms >= 10:
        return f"{ms:.1f} ms"
    return f"{ms:.2f} ms"


def fmt_budget(ms: float) -> str:
    if ms >= 10_000:
        return f"{ms / 1000:.0f} s"
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def fmt_mb(x: float) -> str:
    return f"{x / 1000:.2f} GB" if x >= 1000 else f"{x:.0f} MB"


def _attribution_path(outputs: Path) -> Path:
    return outputs / "attribution" / "attribution.jsonl"


def attribution(args: argparse.Namespace) -> None:
    path = _attribution_path(args.outputs)
    by_cfg = {cfg: levers.attribution_runs(path, cfg) for cfg, _ in CONFIG_TITLES}
    loads: dict[str, list[float]] = {}
    for line in levers.read_jsonl(path):
        loads.setdefault(line["config"], []).extend(
            [float(line["load_1m_before"]), float(line["load_1m_after"])]
        )
    print("| config | processes | effective settings (from the run) | 1-min load range |")
    print("|---|---:|---|---|")
    for cfg, title in CONFIG_TITLES:
        runs = by_cfg[cfg]
        eff = runs[0].get("effective", {})
        span = f"{min(loads[cfg]):.1f}-{max(loads[cfg]):.1f}"
        print(f"| {title} | {len(runs)} | `{json.dumps(eff)}` | {span} |")
    print()
    print("| shape | " + " | ".join(t for _, t in CONFIG_TITLES) + " |")
    print("|---|" + "---:|" * len(CONFIG_TITLES))
    for shape in levers.HARNESS_ORDER:
        cells = []
        base = statistics.median(r["ms"][shape] for r in by_cfg["base_218"])
        for cfg, _ in CONFIG_TITLES:
            vals = [float(r["ms"][shape]) for r in by_cfg[cfg]]
            med = statistics.median(vals)
            ratio = f" ({med / base:.2f}x)" if cfg != "base_218" else ""
            spread = f"{fmt_attribution(min(vals))}-{fmt_attribution(max(vals))}"
            cells.append(f"{fmt_attribution(med)}{ratio}<br><sub>{spread}</sub>")
        print(f"| {SHORT.get(shape, shape)} | " + " | ".join(cells) + " |")


def _time(path: Path) -> dict[str, dict[str, Any]]:
    return levers.harness_rows(path)["time"]


def pairs(args: argparse.Namespace) -> None:
    out = args.outputs
    p1b, p1h = _time(out / "stage-a-step1-base.json"), _time(out / "stage-a-step1-head.json")
    p2b = _time(out / "stage-a-step1-pair2-base.json")
    p2h = _time(out / "stage-a-step1-pair2-head.json")
    old_b = _time(out / "stage-a3-base-nousersite.json")
    old_h = _time(out / "stage-a3-head-nousersite.json")
    print(
        "| shape | zarr 2.18 median | zarr 3 + levers median | ratio | zarr 2.18 p95 "
        "| zarr 3 + levers p95 | rows | Stage A ratio (708d179) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in levers.HARNESS_ORDER:
        r1 = p1h[s]["median_ms"] / p1b[s]["median_ms"]
        r2 = p2h[s]["median_ms"] / p2b[s]["median_ms"]
        ro = old_h[s]["median_ms"] / old_b[s]["median_ms"]
        print(
            f"| {PAIR_LABEL[s]} | {fmt_pairs(p1b[s]['median_ms'])} / "
            f"{fmt_pairs(p2b[s]['median_ms'])} "
            f"| {fmt_pairs(p1h[s]['median_ms'])} / {fmt_pairs(p2h[s]['median_ms'])} "
            f"| **{r1:.2f}× / {r2:.2f}×** "
            f"| {fmt_pairs(p1b[s]['p95_ms'])} / {fmt_pairs(p2b[s]['p95_ms'])} "
            f"| {fmt_pairs(p1h[s]['p95_ms'])} / {fmt_pairs(p2h[s]['p95_ms'])} "
            f"| {p1b[s]['result_count']:,} | {ro:.2f}× |"
        )


def _head_time(outputs: Path) -> dict[str, tuple[float, float]]:
    """(median of the per-process medians, worst per-process p95) for `iv_head`."""
    runs = levers.attribution_runs(_attribution_path(outputs), "iv_head")
    return {
        s: (statistics.median(r["ms"][s] for r in runs), max(r["p95_ms"][s] for r in runs))
        for s in runs[0]["ms"]
    }


def _eaf_runs(outputs: Path) -> list[dict[str, Any]]:
    return [json.loads((outputs / "eaf" / f"eaf_split_{i}.json").read_text()) for i in (1, 2)]


def _eaf_share(outputs: Path) -> dict[str, float]:
    runs = _eaf_runs(outputs)
    share = {"tophits": 0.0}
    for s in BUDGET_ORDER[1:]:
        share[s] = statistics.median(r[SPLIT.get(s, s)]["result_eaf_share"] for r in runs)
    return share


def _expected(outputs: Path) -> dict[str, Any]:
    shapes: dict[str, Any] = json.loads((outputs / "slice" / "expected_020.json").read_text())[
        "shapes"
    ]
    return shapes


def _budget_rows(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    base = _time(args.baseline)
    head = _head_time(args.outputs)
    share = _eaf_share(args.outputs)
    exp = _expected(args.outputs)
    out = {}
    for s in BUDGET_ORDER:
        k = 1 - share[s]
        row: dict[str, Any] = {
            "218": (base[s]["median_ms"], base[s]["p95_ms"]),
            "z3": head[s],
            "z3fix": head[s][0] * k,
        }
        for n, key in (("v3_c1000_s", "c1000"), ("v3_c128_s", "c128"), ("v3_c64_s", "c64")):
            if s == "tophits":
                v = exp[s]["expected_if_tophit_arrays_sharded_ms"]
                row[key] = (v, exp[s]["expected_if_tophit_arrays_unsharded_ms"], v)
            else:
                e = exp[s][n]
                row[key] = (e["expected_ms"] * k, e["expected_lo_ms"] * k, e["expected_hi_ms"] * k)
        out[s] = row
    return out


def time_table(args: argparse.Namespace) -> None:
    print(
        "| query | zarr 2.18 (typical / slow) | zarr 3 after step 1 (typical / slow) "
        "| + EAF fix (expected) | 0.2.0, 1000 per chunk (expected) "
        "| 0.2.0, 128 per chunk (expected) | 0.2.0, 64 per chunk (expected) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s, r in _budget_rows(args).items():
        b, z = r["218"], r["z3"]
        cells = [
            f"{fmt_budget(b[0])} / {fmt_budget(b[1])}",
            f"{fmt_budget(z[0])} / {fmt_budget(z[1])}",
            fmt_budget(r["z3fix"]),
        ]
        for key in ("c1000", "c128", "c64"):
            v, lo, hi = r[key]
            cells.append(f"{fmt_budget(v)} ({fmt_budget(lo)}–{fmt_budget(hi)})")
        print(f"| {NAME[s]} | " + " | ".join(cells) + " |")


def memory_table(args: argparse.Namespace) -> None:
    out = args.outputs
    base = levers.harness_rows(args.baseline)["mem"]
    pb = levers.harness_rows(out / "rss-pair-base.json")["mem"]
    ph = levers.harness_rows(out / "rss-pair-head.json")["mem"]
    h1 = levers.harness_rows(out / "rss-step1-head.json")["mem"]
    print(
        "| query | zarr 2.18, #242 baseline | zarr 2.18, pair run "
        "| zarr 3 after step 1, pair run | zarr 3 after step 1, second run |"
    )
    print("|---|---:|---:|---:|---:|")
    for s in BUDGET_ORDER:
        print(
            f"| {NAME[s]} | {fmt_mb(base[s]['peak_mb'])} | {fmt_mb(pb[s]['peak_mb'])} "
            f"| {fmt_mb(ph[s]['peak_mb'])} | {fmt_mb(h1[s]['peak_mb'])} |"
        )
    pb_max = max(m["baseline_mb"] for m in pb.values())
    ph_min = min(m["baseline_mb"] for m in ph.values())
    ph_max = max(m["baseline_mb"] for m in ph.values())
    print(
        f"\nProcess baseline before each query: zarr 2.18 {pb['bulk']['baseline_mb']:.0f}–"
        f"{pb_max:.0f} MB; zarr 3 {ph_min:.0f}–{ph_max:.0f} MB."
    )


def budget_table(args: argparse.Namespace) -> None:
    ph = levers.harness_rows(args.outputs / "rss-pair-head.json")["mem"]
    h1 = levers.harness_rows(args.outputs / "rss-step1-head.json")["mem"]
    rows = _budget_rows(args)
    print(
        "| query | time budget (typical / slow) | memory cap | zarr 3 after step 1: time "
        "| memory | 0.2.0 at 128 or 64 per chunk (expected): time | memory |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s in BUDGET_ORDER:
        tm, tp, cap = BUDGET_L[s]
        z = rows[s]["z3"]
        peak = max(ph[s]["peak_mb"], h1[s]["peak_mb"])
        worst = max(rows[s]["c128"][2], rows[s]["c64"][2])

        def mark(ok: bool) -> str:
            return "✓" if ok else "✗"

        print(
            f"| {NAME[s]} | {fmt_budget(tm)} / {fmt_budget(tp)} | {fmt_mb(cap)} "
            f"| {mark(z[0] <= tm and z[1] <= tp)} {fmt_budget(z[0])} / {fmt_budget(z[1])} "
            f"| {mark(peak <= cap)} {fmt_mb(peak)} "
            f"| {mark(worst <= tm)} up to {fmt_budget(worst)} | not yet measured |"
        )


def eaf_table(args: argparse.Namespace) -> None:
    runs = _eaf_runs(args.outputs)
    print(
        "| query | total | EAF read inside SE decoding "
        "| EAF read for the result's `eaf` column | share of total |"
    )
    print("|---|---:|---:|---:|---:|")

    def spread(values: list[float]) -> str:
        lo, hi = fmt_budget(min(values)), fmt_budget(max(values))
        return f"{lo}–{hi}" if lo != hi else fmt_budget(values[0])

    for s in BUDGET_ORDER[1:]:
        k = SPLIT.get(s, s)
        share = [r[k]["result_eaf_share"] for r in runs]
        lo, hi = f"{100 * min(share):.0f}", f"{100 * max(share):.0f}"
        pct = f"{lo}%" if lo == hi else f"{lo}–{hi}%"
        print(
            f"| {NAME[s]} | {spread([r[k]['total_ms'] for r in runs])} "
            f"| {spread([r[k]['eaf_in_se_decode_ms'] for r in runs])} "
            f"| {spread([r[k]['eaf_for_result_ms'] for r in runs])} | {pct} |"
        )


def _positions(args: argparse.Namespace) -> dict[str, dict[str, tuple[float, float | None]]]:
    """Per shape: (median, p95 or None) for 2.18, zarr 3 after step 1, and 0.2.0 expected."""
    out = args.outputs
    base = _time(args.baseline)
    p1h = _time(out / "stage-a-step1-head.json")
    p2h = _time(out / "stage-a-step1-pair2-head.json")
    head = levers.attribution_medians(_attribution_path(out), "iv_head")
    exp = _expected(out)
    positions: dict[str, dict[str, tuple[float, float | None]]] = {}
    for s in PROPOSAL_ORDER:
        row: dict[str, tuple[float, float | None]] = {
            "2.18": (base[s]["median_ms"], base[s]["p95_ms"]),
            "head": (head[s], max(p1h[s]["p95_ms"], p2h[s]["p95_ms"])),
        }
        for k, n in (("c1000", "v3_c1000_s"), ("c128", "v3_c128_s"), ("c64", "v3_c64_s")):
            if s == "tophits":
                row[k] = (exp[s]["expected_if_tophit_arrays_sharded_ms"], None)
            else:
                row[k] = (exp[s][n]["expected_ms"], None)
        positions[s] = row
    return positions


def proposals(args: argparse.Namespace) -> None:
    pos = _positions(args)
    columns = ("2.18", "head", "c1000", "c128", "c64")
    for name, budget in PROPOSALS.items():
        print(
            f"\n**Set {name}** (median / p95 budget; ✓ meets the median budget, ✗ misses it; "
            "p95 checked where measured)\n"
        )
        print(
            "| shape | use | budget | zarr 2.18 | head (step 1) | 0.2.0 c1000 (exp.) "
            "| 0.2.0 c128 (exp.) | 0.2.0 c64 (exp.) |"
        )
        print("|---|---|---|---|---|---|---|---|")
        fails: dict[str, list[str]] = {k: [] for k in columns}
        for s in PROPOSAL_ORDER:
            bm, bp = budget[s]
            cells = []
            for k in columns:
                med, p95 = pos[s][k]
                ok = med <= bm and (p95 is None or p95 <= bp)
                if not ok:
                    fails[k].append(PAIR_LABEL[s].split(" (")[0])
                tail = "" if p95 is None else f" / {fmt_pairs(p95)}"
                cells.append(f"{'✓' if ok else '✗'} {fmt_pairs(med)}{tail}")
            print(
                f"| {PAIR_LABEL[s]} | {USE[s]} | {fmt_pairs(bm)} / {fmt_pairs(bp)} | "
                + " | ".join(cells)
                + " |"
            )
        print(
            "\nMisses: "
            + "; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in fails.items())
        )


#: The whole-Analysis guard in #244's criterion: in the same back-to-back pair,
#: zarr 3's median for one whole Analysis is at most this multiple of 2.18's.
WHOLE_ANALYSIS_GUARD = 1.25
#: A result this close to a limit, either side, as a fraction of the limit, needs
#: a second back-to-back pair before it counts.
CONFIRM_MARGIN = 0.30


def _near(value: float, limit: float) -> bool:
    return abs(value - limit) <= CONFIRM_MARGIN * limit


def _mark(value: float, limit: float) -> str:
    return ("✓" if value <= limit else "✗") + (" ≈" if _near(value, limit) else "")


def _run_line(name: str, path: Path) -> str:
    data = json.loads(path.read_text())
    env, store = data["environment"], data["stores"][0]
    return (
        f"- {name}: `{path.name}`, commit `{env['opengwasdb_commit']}`, zarr {env['zarr']}, "
        f"numcodecs {env['numcodecs']}, 1-minute load {store['load_average_1m_before']} → "
        f"{store['load_average_1m_after']}"
    )


def _judge_pair(base_path: Path, head_path: Path) -> tuple[bool, list[str]]:
    """Print one pair's table against set L; return whether it holds and what is near."""
    base, head = levers.harness_rows(base_path), levers.harness_rows(head_path)
    print(_run_line("zarr 2.18", base_path))
    print(_run_line("zarr 3", head_path))
    print(
        "\n| query | budget: typical / slow / memory | zarr 3 typical | zarr 3 slow "
        "| zarr 3 peak memory | zarr 2.18, same pair: typical / slow / peak memory |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    holds, near = True, []
    for s in BUDGET_ORDER:
        typical, slow, cap = BUDGET_L[s]
        t, m = head["time"][s], head["mem"][s]
        b, bm = base["time"][s], base["mem"][s]
        checks = [
            ("typical", t["median_ms"], typical),
            ("slow", t["p95_ms"], slow),
            ("memory", m["peak_mb"], cap),
        ]
        holds &= all(value <= limit for _, value, limit in checks)
        near += [f"{NAME[s]} {what}" for what, value, limit in checks if _near(value, limit)]
        print(
            f"| {NAME[s]} | {fmt_pairs(typical)} / {fmt_pairs(slow)} / {fmt_mb(cap)} "
            f"| {_mark(t['median_ms'], typical)} {fmt_pairs(t['median_ms'])} "
            f"| {_mark(t['p95_ms'], slow)} {fmt_pairs(t['p95_ms'])} "
            f"| {_mark(m['peak_mb'], cap)} {fmt_mb(m['peak_mb'])} "
            f"| {fmt_pairs(b['median_ms'])} / {fmt_pairs(b['p95_ms'])} / {fmt_mb(bm['peak_mb'])} |"
        )
    ratio = head["time"]["bulk"]["median_ms"] / base["time"]["bulk"]["median_ms"]
    holds &= ratio <= WHOLE_ANALYSIS_GUARD
    if _near(ratio, WHOLE_ANALYSIS_GUARD):
        near.append("the whole-Analysis guard")
    print(
        f"\nWhole-Analysis guard: zarr 3 median / zarr 2.18 median = {ratio:.2f}× "
        f"(limit {WHOLE_ANALYSIS_GUARD}×) {_mark(ratio, WHOLE_ANALYSIS_GUARD)}"
    )
    return holds, near


def stage_b(args: argparse.Namespace) -> None:
    """#244's Stage B: each back-to-back pair against set L and the whole-Analysis guard.

    ✓ meets the limit, ✗ misses it; ≈ marks a result within 30% of the limit, either
    side, which needs a second back-to-back pair before it counts.
    """
    if not args.pair:
        raise SystemExit("stage-b needs --pair BASE HEAD (zarr 2.18 run, then zarr 3 run)")
    pairs_given = " ".join(f"--pair {base} {head}" for base, head in args.pair)
    print("# OGS-00009 on zarr-python 3 against set L (#244 Stage B)\n")
    print(
        f"Generated by `python benchmarks/zarr3_lever_tables.py stage-b {pairs_given}`; "
        "no number is typed by hand. Each pair is the #242 harness run back to back, "
        "zarr 2.18 first, `--reps 5` with the peak-memory probes. ✓ meets the limit, "
        "✗ misses it; ≈ marks a result within 30% of a limit, either side, which needs "
        "a second pair before it counts."
    )
    every_pair_holds, near_any = True, False
    for number, (base_path, head_path) in enumerate(args.pair, start=1):
        print(f"\n**Pair {number}**\n")
        holds, near = _judge_pair(Path(base_path), Path(head_path))
        every_pair_holds &= holds
        near_any |= bool(near)
        print(f"Within 30% of a limit: {', '.join(near) if near else 'none'}.")
    if not every_pair_holds:
        verdict = "FAIL: a limit is missed"
    elif near_any and len(args.pair) < 2:
        verdict = "NOT YET: a result is within 30% of a limit; run a second back-to-back pair"
    else:
        verdict = "PASS: every query meets set L, and the whole-Analysis guard holds"
    print(f"\n**Verdict: {verdict}.**")


# ── #246: every configuration against set L ─────────────────────────────────

#: The store label #246's zarr 2.18 run uses, and the one its zarr 3 run uses for
#: the same physical shape.  The pair is the format 0.1.0 -> zarr 3 before/after.
SHAPES_218 = "v2-c1000"


def _shape_labels(head_path: Path) -> list[str]:
    return [str(store["label"]) for store in json.loads(head_path.read_text())["stores"]]


def _time_cell(rows: dict[str, Any], shape: str) -> str:
    typical, slow, _ = BUDGET_L[shape]
    t = rows["time"][shape]
    return (
        f"{_mark(t['median_ms'], typical)} {fmt_pairs(t['median_ms'])}"
        f" / {_mark(t['p95_ms'], slow)} {fmt_pairs(t['p95_ms'])}"
    )


def _memory_cell(rows: dict[str, Any], shape: str) -> str:
    _, _, cap = BUDGET_L[shape]
    peak = rows["mem"][shape]["peak_mb"]
    return f"{_mark(peak, cap)} {fmt_mb(peak)}"


def _shape_run_line(label: str, name: str, path: Path) -> str:
    store = levers.harness_store(path, label)
    env = json.loads(path.read_text())["environment"]
    effective = store.get("effective_reader") or env.get("effective_reader")
    return (
        f"- {name}: `{path.name}` store `{label}`, commit `{env['opengwasdb_commit']}`, "
        f"zarr {env['zarr']}, numcodecs {env['numcodecs']}, "
        f"effective reader `{json.dumps(effective, sort_keys=True)}`, "
        f"1-minute load {store['load_average_1m_before']} → {store['load_average_1m_after']}"
    )


def shapes_table(args: argparse.Namespace) -> None:
    """#246: every configuration against set L, and the whole-Analysis guard.

    ✓ meets the limit, ✗ misses it, ≈ marks a result within 30% of a limit,
    either side, which needs a second back-to-back pair before it counts.  The
    engine is the #242 harness run back to back -- zarr 2.18 first, then one
    zarr 3 process holding every configuration -- so the guard compares two runs
    of the same quiet window rather than one run against a committed artifact.
    """
    if args.base is None or args.head is None:
        raise SystemExit("shapes needs --base <zarr 2.18 run> and --head <zarr 3 run>")
    base_rows = levers.harness_rows(args.base, SHAPES_218)
    labels = _shape_labels(args.head)
    head_rows = {label: levers.harness_rows(args.head, label) for label in labels}

    print("# OGS-00009 across chunk and shard shapes (#246)\n")
    print(
        "Generated by `python benchmarks/zarr3_lever_tables.py shapes --base "
        f"{args.base} --head {args.head}`; no number is typed by hand. Each run is the "
        "#242 harness, `--reps 5` with the peak-memory probes, started at a 1-minute "
        "load below 3. ✓ meets the limit, ✗ misses it; ≈ marks a result within 30% of "
        "a limit, either side, which needs a second back-to-back pair before it counts.\n"
    )
    print("## Runs\n")
    print(_shape_run_line(SHAPES_218, "zarr 2.18", args.base))
    for label in labels:
        print(_shape_run_line(label, label, args.head))

    print("\n## Time per query against set L\n")
    print("| query | budget: typical / slow | " + " | ".join(labels) + " |")
    print("|---|---:|" + "---:|" * len(labels))
    for shape in BUDGET_ORDER:
        typical, slow, _ = BUDGET_L[shape]
        cells = [_time_cell(head_rows[label], shape) for label in labels]
        print(
            f"| {NAME[shape]} | {fmt_pairs(typical)} / {fmt_pairs(slow)} | "
            + " | ".join(cells)
            + " |"
        )

    print("\n## Peak memory per query against set L\n")
    print("| query | memory cap | " + " | ".join(labels) + " |")
    print("|---|---:|" + "---:|" * len(labels))
    for shape in BUDGET_ORDER:
        _, _, cap = BUDGET_L[shape]
        cells = [_memory_cell(head_rows[label], shape) for label in labels]
        print(f"| {NAME[shape]} | {fmt_mb(cap)} | " + " | ".join(cells) + " |")

    print("\n## Whole-Analysis guard\n")
    print(
        "| configuration | zarr 3, one whole Analysis | zarr 2.18, same pair "
        "| ratio | limit | verdict |"
    )
    print("|---|---:|---:|---:|---:|---|")
    guard_holds = True
    near_any = []
    for label in labels:
        head_bulk = head_rows[label]["time"]["bulk"]["median_ms"]
        ratio = head_bulk / base_rows["time"]["bulk"]["median_ms"]
        holds = ratio <= WHOLE_ANALYSIS_GUARD
        guard_holds &= holds
        if _near(ratio, WHOLE_ANALYSIS_GUARD):
            near_any.append(f"{label}: the whole-Analysis guard")
        print(
            f"| {label} | {fmt_pairs(head_rows[label]['time']['bulk']['median_ms'])} "
            f"| {fmt_pairs(base_rows['time']['bulk']['median_ms'])} | {ratio:.2f}× "
            f"| {WHOLE_ANALYSIS_GUARD:.2f}× | {_mark(ratio, WHOLE_ANALYSIS_GUARD)} |"
        )

    misses = [
        f"{label} {NAME[shape]}"
        for label in labels
        for shape in BUDGET_ORDER
        if head_rows[label]["time"][shape]["median_ms"] > BUDGET_L[shape][0]
        or head_rows[label]["time"][shape]["p95_ms"] > BUDGET_L[shape][1]
        or head_rows[label]["mem"][shape]["peak_mb"] > BUDGET_L[shape][2]
    ]
    near_any += [
        f"{label} {NAME[shape]} {what}"
        for label in labels
        for shape in BUDGET_ORDER
        for what, value, limit in (
            ("typical", head_rows[label]["time"][shape]["median_ms"], BUDGET_L[shape][0]),
            ("slow", head_rows[label]["time"][shape]["p95_ms"], BUDGET_L[shape][1]),
            ("memory", head_rows[label]["mem"][shape]["peak_mb"], BUDGET_L[shape][2]),
        )
        if _near(value, limit)
    ]
    print(f"\n**Misses:** {', '.join(misses) if misses else 'none'}.")
    print(f"**Within 30% of a limit:** {', '.join(near_any) if near_any else 'none'}.")
    print(
        "\n**Verdict: "
        + (
            "FAIL: a configuration misses a limit"
            if misses or not guard_holds
            else "PASS: every configuration meets set L and the whole-Analysis guard"
        )
        + ".**"
    )


TABLES = {
    "attribution": attribution,
    "pairs": pairs,
    "memory": memory_table,
    "time": time_table,
    "eaf": eaf_table,
    "budget": budget_table,
    "proposals": proposals,
    "stage-b": stage_b,
    "shapes": shapes_table,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("table", choices=sorted(TABLES))
    ap.add_argument("--outputs", type=Path, default=levers.OUTPUTS, help="#244's output directory")
    ap.add_argument(
        "--baseline", type=Path, default=levers.ZARR2_BASELINE, help="the #242 zarr 2.18 artifact"
    )
    ap.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BASE", "HEAD"),
        help="stage-b: one back-to-back pair of harness runs, zarr 2.18 then zarr 3; repeatable",
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=None,
        help="shapes: the #246 zarr 2.18 harness run (one store, `v2-c1000`)",
    )
    ap.add_argument(
        "--head",
        type=Path,
        default=None,
        help="shapes: the #246 zarr 3 harness run, holding every configuration",
    )
    args = ap.parse_args()
    TABLES[args.table](args)


if __name__ == "__main__":
    main()

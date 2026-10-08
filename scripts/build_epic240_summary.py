#!/usr/bin/env python3
"""Generate the epic #240 summary: Zarr v3 with sharding, before and after.

Writes `docs/benchmark-output/opengwasdb_zarr_v3_sharding_summary.md` from
committed artifacts only, so no number in it is typed. For each of OGS-00009,
OGS-00016 and OGS-00011 it tabulates files, size, the seven #242 query shapes
(and OGS-00011's two #252 Overflow shapes) with median time and peak RSS, in
three columns:

* **before**: `745796c` under Zarr 2.18 on the 0.1.0 store (the status quo);
* **this code on 0.1.0**: Zarr 3 reading the same store, so the format's own
  effect can be separated from the code's;
* **after**: Zarr 3 on the converted 0.2.0 store at ADR 0058's shapes.

Each table gives two ratios, as #250's report does: after / before (what a user
sees) and after / this code on 0.1.0 (the format alone). The sources table
names every artifact read, with its commit and time.

Usage, from the repository root::

    python3 scripts/build_epic240_summary.py            # print the document
    python3 scripts/build_epic240_summary.py --write    # write it
    python3 scripts/build_epic240_summary.py --check    # fail if the committed copy drifted
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

OUT = Path("docs/benchmark-output")
SUMMARY = OUT / "opengwasdb_zarr_v3_sharding_summary.md"
ADRS = ("0056", "0057", "0058", "0059", "0060")
#: Each Store's three columns: (artifact, store label). #246's first back-to-back
#: pair is the one ADR 0058 and #250's generators read; its second pair confirms it.
STORES = {
    "OGS-00009": {
        "before": ("opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_zarr2.json",
                   "v2-c1000"),
        "code_010": ("opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes.json",
                     "v2-c1000"),
        "after": ("opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes.json",
                  "v3-c64"),
    },
    "OGS-00016": {
        "before": ("opengwasdb_ogs00016_store_comparison_2_18.json", "v2-18-ogs00016"),
        "code_010": ("opengwasdb_ogs00016_store_comparison_zarr3.json", "v2-ogs00016"),
        "after": ("opengwasdb_ogs00016_store_comparison_zarr3.json", "v3-ogs00016"),
    },
    "OGS-00011": {
        "before": ("opengwasdb_ogs00011_store_comparison_2_18.json", "v2-18-ogs00011"),
        "code_010": ("opengwasdb_ogs00011_store_comparison_zarr3.json", "v2-ogs00011"),
        "after": ("opengwasdb_ogs00011_store_comparison_zarr3.json", "v3-ogs00011"),
    },
}
EXTRAS = "opengwasdb_ogs00011_extra_shapes_0_2_0.json"
AB252 = "opengwasdb_ogs00011_hybrid_252_ab.json"
CONVERSIONS = {
    "OGS-00016": "opengwasdb_ogs00016_conversion_0_2_0.json",
    "OGS-00011": "opengwasdb_ogs00011_conversion_0_2_0.json",
}
REPORTS = {
    "#246's OGS-00009 shape report": "opengwasdb_246_shapes/opengwasdb_ogs00009_shapes.html",
    "#250's OGS-00016 report (with OGS-00011)": "opengwasdb_ogs00016_0_2_0.html",
}
SHAPES = {
    "bulk": "One whole Analysis",
    "regional": "One window, all Analyses",
    "regional_one_analysis": "One window, one Analysis",
    "random_lookup_10_variants_100_analyses": "Lookup, 10 variants × 100 Analyses",
    "random_lookup_100_variants_10_analyses": "Lookup, 100 variants × 10 Analyses",
    "tophits": "Top hits for one Analysis",
    "phewas": "PheWAS, one variant",
}
EXTRA_SHAPES = {
    "bulk_overflow_heavy": "Overflow: bulk, largest Overflow Analysis",
    "phewas_off_axis": "Overflow: PheWAS of an off-panel variant",
}
EXTRA_COLUMNS = {"before": "a-2.18-0.1.0", "code_010": "b-code-0.1.0", "after": "c-code-0.2.0"}
COLUMNS = ("before", "code_010", "after")
COLUMN_NAMES = {"before": "before", "code_010": "this code on 0.1.0", "after": "after"}
#: A before column written this long before its Zarr 3 columns is not their pair.
UNPAIRED_AFTER_S = 3600


def _json(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def _store(column: tuple[str, str]) -> dict:
    name, label = column
    return next(s for s in _json(name)["stores"] if s["label"] == label)


def _fmt_time(ms: float) -> str:
    return f"{ms / 1000:,.2f} s" if ms >= 1000 else f"{ms:,.1f} ms"


def _fmt_rss(mib: float) -> str:
    return f"{mib / 1024:,.2f} GiB"


def _ratio(new: float, old: float) -> str:
    return f"{new / old:.2f}×"


def _measure(store: dict, shape: str) -> dict:
    """One shape's median, peak RSS and repetition count, or its limit hit."""
    timing = next(r for r in store["timings"] if r["query"] == shape)
    memory = next(r for r in store["memory"] if r["query"] == shape)
    return {"ms": timing.get("median_ms"), "mib": memory.get("peak_mb"),
            "reps": timing.get("repetitions"), "limit": timing.get("timed_out", False)}


def _cell(m: dict) -> str:
    if m["limit"] or m["ms"] is None:
        return "**limit hit**"
    return (f"{_fmt_time(m['ms'])} · {_fmt_rss(m['mib'])}"
            + (" (n=1)" if m["reps"] == 1 else ""))


def _ratios(after: dict, base: dict) -> str:
    if after["limit"] or base["limit"] or None in (after["ms"], base["ms"]):
        return "—"
    return f"{_ratio(after['ms'], base['ms'])} · {_ratio(after['mib'], base['mib'])}"


def _row(label: str, m: dict) -> str:
    return (f"| {label} | {_cell(m['before'])} | {_cell(m['code_010'])} | {_cell(m['after'])} "
            f"| {_ratios(m['after'], m['before'])} | {_ratios(m['after'], m['code_010'])} |")


def _size_rows(spec: dict) -> list[str]:
    old, new = _store(spec["before"])["footprint"], _store(spec["after"])["footprint"]
    return [
        f"| Files | {old['n_files']:,} | (same store) | {new['n_files']:,} "
        f"| {old['n_files'] / new['n_files']:,.0f}× fewer | — |",
        f"| Apparent size | {old['apparent_bytes'] / 1e9:,.2f} GB | (same store) "
        f"| {new['apparent_bytes'] / 1e9:,.2f} GB "
        f"| {(new['apparent_bytes'] / old['apparent_bytes'] - 1) * 100:+.2f} % | — |",
    ]


def _extra_rows() -> list[str]:
    extras = _json(EXTRAS)
    rows = []
    for shape, label in EXTRA_SHAPES.items():
        cells = {col: extras["shapes"][shape][EXTRA_COLUMNS[col]] for col in COLUMNS}
        rows.append(_row(label, {col: {"ms": r["median_ms"], "mib": r["peak_mib"],
                                       "reps": r["repetitions"], "limit": False}
                                 for col, r in cells.items()}))
    return rows


def _differing(digests: list[dict]) -> tuple[int, list[str]]:
    """(shapes every column measured, those whose per-array digests differ)."""
    common = set.intersection(*(set(d) for d in digests))
    keyed = {sh: {json.dumps(d[sh], sort_keys=True) for d in digests} for sh in common}
    return len(common), sorted(sh for sh, values in keyed.items() if len(values) != 1)


def _identity(spec: dict, name: str) -> str:
    """Whether every column returned the same result digest on every shape."""
    count, differing = _differing([_store(spec[col])["result_digests"] for col in COLUMNS])
    extra = ""
    if name == "OGS-00011":
        identity = _json(EXTRAS)["identity"]
        differing += sorted(sh for sh, entry in identity.items() if not entry["identical"])
        extra = f" and on the {len(identity)} Overflow shapes"
    if differing:
        return f"**Results differ between the columns on: {', '.join(differing)}.**"
    return (f"Every column returned identical results (sha256 per shape) on all {count} "
            f"shapes{extra}.")


def _store_table(name: str) -> list[str]:
    spec = STORES[name]
    measures = {
        shape: {col: _measure(_store(spec[col]), shape) for col in COLUMNS} for shape in SHAPES
    }
    lines = [
        f"### {name}",
        "",
        "| | before: 2.18 · 0.1.0 | this code · 0.1.0 | after: this code · 0.2.0 "
        "| after ÷ before (time · RSS) | after ÷ this code on 0.1.0 (time · RSS) |",
        "|---|---|---|---|---|---|",
        *_size_rows(spec),
        *(_row(label, measures[shape]) for shape, label in SHAPES.items()),
    ]
    if name == "OGS-00011":
        lines += _extra_rows()
    return [*lines, "", _identity(spec, name) + _pairing(spec), ""]


def _pairing(spec: dict) -> str:
    """A note when the before column ran in a different window from the Zarr 3 columns."""
    before, after = (_json(spec[col][0])["measured_at"] for col in ("before", "after"))
    gap = datetime.fromisoformat(after) - datetime.fromisoformat(before)
    if abs(gap.total_seconds()) < UNPAIRED_AFTER_S:
        return ""
    return (f" **The before column is not paired with the Zarr 3 columns**: it was written at "
            f"{before[11:16]} UTC and they at {after[11:16]} UTC.")


def _sources() -> list[str]:
    lines = ["| artifact | read for | commit | written (UTC) |", "|---|---|---|---|"]
    seen: dict[str, list[str]] = {}
    for name, spec in STORES.items():
        for col in COLUMNS:
            seen.setdefault(spec[col][0], []).append(f"{name} {COLUMN_NAMES[col]}")
    seen.setdefault(EXTRAS, []).append("OGS-00011 Overflow shapes")
    seen.setdefault(AB252, []).append("the caveats")
    for name, store in CONVERSIONS.items():
        seen.setdefault(store, []).append(f"{name} conversion peak RSS")
    for artifact, roles in seen.items():
        commit, written = _provenance(_json(artifact))
        lines.append(f"| `{artifact}` | {', '.join(dict.fromkeys(roles))} | {commit} | "
                     f"{written} |")
    return lines


def _stamp(iso: str) -> str:
    return iso[:16].replace("T", " ")


def _run_provenance(doc: dict) -> tuple[str, str]:
    """An extras aggregate: its columns' commits, and when its last run started."""
    commits = dict.fromkeys(env["commit"] for env in doc["environments"].values())
    runs = [row["run"]["measured_at"] for rows in doc["shapes"].values() for row in rows.values()]
    return ", ".join(f"`{c}`" for c in commits), f"{_stamp(max(runs))} (last run)"


def _provenance(doc: dict) -> tuple[str, str]:
    """(commit, when written) as each kind of artifact records them, or say they are not."""
    if "environments" in doc:
        return _run_provenance(doc)
    if "monitor" in doc:
        window = f"{doc['monitor']['first_sample_utc']}–{doc['monitor']['last_sample_utc']}"
        return f"not recorded (from `{doc['log']}`)", f"{window}, date not recorded"
    if "provenance" in doc:
        sides = doc["provenance"]
        return (", ".join(f"{side} `{sides[side]['commit']}`" for side in ("before", "after")),
                _stamp(sides["after"]["measured_at"]))
    return f"`{doc['commit']}`", _stamp(doc["measured_at"])


def _environments() -> str:
    extras = _json(EXTRAS)["environments"]
    return (f"The Overflow shapes ran under Zarr {extras['a-2.18-0.1.0']['zarr']} at "
            f"`{extras['a-2.18-0.1.0']['commit']}` (before) and Zarr "
            f"{extras['c-code-0.2.0']['zarr']} at `{extras['c-code-0.2.0']['commit']}` (this "
            "code), with every repetition gated at a 1-minute load below 3.")


def _slower_by_seconds() -> list[str]:
    """Every shape whose median on 0.2.0 is at least a second slower than on 2.18."""
    out = []
    for name, spec in STORES.items():
        before, after = _store(spec["before"]), _store(spec["after"])
        for shape, label in SHAPES.items():
            a, c = _measure(before, shape), _measure(after, shape)
            if None not in (a["ms"], c["ms"]) and c["ms"] - a["ms"] >= 1000:
                out.append(f"{name} {label}")
    for shape, label in EXTRA_SHAPES.items():
        row = _json(EXTRAS)["shapes"][shape]
        if row["c-code-0.2.0"]["median_ms"] - row["a-2.18-0.1.0"]["median_ms"] >= 1000:
            out.append(f"OGS-00011 {label}")
    return out


def _phewas_caveat() -> str:
    row = _json(EXTRAS)["shapes"]["phewas_off_axis"]
    ab = _json(AB252)
    before, after = (ab["shapes"]["phewas_off_axis"][side] for side in ("before", "after"))
    a, b, c = (row[EXTRA_COLUMNS[col]] for col in COLUMNS)
    overlap = min(c["elapsed_ms"]) <= max(b["elapsed_ms"])
    slower = _slower_by_seconds()
    return (
        f"The off-panel Overflow PheWAS on OGS-00011 is **{_ratio(c['median_ms'], a['median_ms'])} "
        f"the 2.18 time** on 0.2.0 ({_fmt_time(c['median_ms'])} against "
        f"{_fmt_time(a['median_ms'])}). "
        + ("It is the only measured shape at least a second slower than the status quo. "
           if slower == ["OGS-00011 " + EXTRA_SHAPES["phewas_off_axis"]] else
           f"Shapes at least a second slower than the status quo: {', '.join(slower)}. ")
        + ("The cost is this code's, not the format's: 0.2.0 is "
           f"{_ratio(c['median_ms'], b['median_ms'])} this code on 0.1.0, with overlapping "
           "repetitions. " if overlap else
           f"**0.2.0 is {_ratio(c['median_ms'], b['median_ms'])} this code on 0.1.0, with every "
           "repetition slower, so the format adds to the cost.** ")
        + "Which part of the code causes it is unestablished. #252's A/B, with the reader held "
        f"fixed, recorded the pre-#252 decode at {_fmt_time(before['elapsed_ms'])} and the "
        f"windowed scan at {_fmt_time(after['elapsed_ms'])} (one run each), and an earlier "
        "version of that artifact had the opposite sign. ADR 0060's by-variant index removes "
        "the scan either way."
    )


def _caveats() -> str:
    peaks = {name: _json(path)["peak_rss_gib"] for name, path in CONVERSIONS.items()}
    return (
        "**Caveats.** " + _phewas_caveat() + " Validation memory scales with the store (#254). "
        "The 0.2.0 conversions peaked at "
        + " and ".join(f"{gib:,.2f} GiB on {name}" for name, gib in peaks.items())
        + "; the phase times attribute those peaks to validation, which is inferred, not "
        "sampled. Follow-ups: #252 step 1 (restore a Ragged baseline), step 2 (the scaling "
        "experiment) and step 5 (build ADR 0060's by-variant index); #257 (single-pass and "
        "two-pass Hybrid builds disagree on association-less off-reference variants); #258 (a "
        "`--variant-reference` that names some rsids leaves the rest blank)."
    )


def _links() -> list[str]:
    lines = [f"- {title}: [`{path}`]({path})" for title, path in REPORTS.items()]
    for number in ADRS:
        adr = next(Path("docs/adr").glob(f"{number}-*.md"))
        title = adr.read_text().splitlines()[0].lstrip("# ").strip()
        lines.append(f"- ADR {number}: [{title}](../adr/{adr.name})")
    return lines


def build() -> str:
    lines = [
        "# Zarr v3 with sharding: before and after (epic #240)",
        "",
        "Generated by `python3 scripts/build_epic240_summary.py` from the committed artifacts "
        "named under *Sources*; regenerate rather than edit.",
        "",
        "**Before** is `745796c` under Zarr 2.18 on each store's 0.1.0 release, the status "
        "quo. **After** is this code under Zarr 3 on the store converted to 0.2.0 at ADR "
        "0058's shapes. The middle column is this code reading the 0.1.0 store, so the two "
        "ratios separate what a user sees (after ÷ before) from the format's own effect (after "
        "÷ this code on 0.1.0). Cells are median time · peak RSS (MiB / 1024 as GiB); "
        "\"this code\" is the commit each source names.",
        "",
        "## Before and after",
        "",
    ]
    for name in STORES:
        lines += _store_table(name)
    lines += [_environments(), "", _caveats(), "", "## Reports and decisions", "", *_links(),
              "", "## Sources", "", *_sources(), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {SUMMARY}")
    ap.add_argument("--check", action="store_true", help=f"fail if {SUMMARY} has drifted")
    args = ap.parse_args()
    text = build()
    if args.write:
        SUMMARY.write_text(text)
        print(f"{SUMMARY}: wrote {len(text)} characters")
    elif args.check:
        if not SUMMARY.is_file() or SUMMARY.read_text() != text:
            sys.exit(f"{SUMMARY}: differs from the artifacts; rerun with --write")
        print(f"{SUMMARY}: matches the artifacts")
    else:
        print(text)


if __name__ == "__main__":
    main()

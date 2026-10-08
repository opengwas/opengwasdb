#!/usr/bin/env python3
"""Generate (or splice) ADR 0058's "Checked on FinnGen (#250)" block.

Every number, and every verdict, is computed from a committed artifact, so the
ADR section is never retyped:

* `docs/benchmark-output/opengwasdb_ogs00016_store_comparison_zarr3.json` and
  `..._2_18.json` (the 0.1.0, 0.2.0 and 2.18 timings and the per-plane footprint),
* `docs/benchmark-output/opengwasdb_ogs00016_conversion_0_2_0.json`
  (the total size change, file counts and the bit-exact flag),
* `docs/benchmark-output/opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes.json`
  (this ADR's own OGS-00009 time and per-plane size ratios, `v3-c64` over
  `v2-c1000`, computed here rather than quoted).

The whole-Analysis guard is read against the **2.18** time, as this ADR defines
it, not against this code on 0.1.0. The question the check answers is the one
#250 asked: does FinnGen show a material size or time cost that OGS-00009 did
not? `MATERIAL` is the margin it calls material, and the block prints it.

Usage, from the repository root::

    python3 scripts/build_finngen_shape_check.py            # print the block
    python3 scripts/build_finngen_shape_check.py --write    # splice it into ADR 0058
    python3 scripts/build_finngen_shape_check.py --check    # fail if ADR 0058 has drifted

`--write` replaces everything between the markers

    <!-- BEGIN GENERATED: finngen-shape-check (#250) -->
    <!-- END GENERATED: finngen-shape-check (#250) -->

so the committed ADR text is exactly this output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path("docs/benchmark-output")
ADR = Path("docs/adr/0058-dense-chunk-and-shard-shapes.md")
BEGIN = "<!-- BEGIN GENERATED: finngen-shape-check (#250) -->"
END = "<!-- END GENERATED: finngen-shape-check (#250) -->"
CMP_Z3 = OUT / "opengwasdb_ogs00016_store_comparison_zarr3.json"
CMP_218 = OUT / "opengwasdb_ogs00016_store_comparison_2_18.json"
CONVERSION = OUT / "opengwasdb_ogs00016_conversion_0_2_0.json"
OGS00009 = OUT / "opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes.json"
#: ADR 0058's test 2: one whole Analysis within this multiple of the Zarr 2.18 time.
GUARD_LIMIT = 1.25
#: A FinnGen ratio more than this fraction above OGS-00009's counterpart is a
#: cost the decision's own data did not show.
MATERIAL = 0.10
PLANES = (
    "data.zarr/z",
    "data.zarr/se",
    "data.zarr/eaf",
    "data.zarr/top_hits/p_5e_04/z",
)
SHAPES = (
    "phewas",
    "random_lookup_10_variants_100_analyses",
    "random_lookup_100_variants_10_analyses",
    "regional",
    "bulk",
)


def _stores(path: Path) -> dict[str, dict]:
    return {store["label"]: store for store in json.loads(path.read_text())["stores"]}


def _medians(store: dict) -> dict[str, float]:
    return {row["query"]: row["median_ms"] for row in store["timings"]}


def _bytes(store: dict) -> dict[str, int]:
    return {array["node"]: array["apparent_bytes"] for array in store["footprint"]["arrays"]}


def _ratios(new: dict, old: dict, keys: tuple[str, ...]) -> dict[str, float]:
    return {key: new[key] / old[key] for key in keys}


def _inputs() -> dict:
    """Everything the block reads, straight from the committed artifacts."""
    finngen, ref = _stores(CMP_Z3), _stores(OGS00009)
    old, new = finngen["v2-ogs00016"], finngen["v3-ogs00016"]
    ref_old, ref_new = ref["v2-c1000"], ref["v3-c64"]
    base = next(iter(_stores(CMP_218).values()))
    conv = json.loads(CONVERSION.read_text())
    return {
        "dataset": old["dataset"],
        "conv": conv,
        "total": conv["after"]["apparent_bytes"] / conv["before"]["apparent_bytes"],
        "ref_total": ref_new["footprint"]["apparent_bytes"]
        / ref_old["footprint"]["apparent_bytes"],
        "bytes": (_bytes(old), _bytes(new)),
        "planes": _ratios(_bytes(new), _bytes(old), PLANES),
        "ref_planes": _ratios(_bytes(ref_new), _bytes(ref_old), PLANES),
        "guard": _medians(new)["bulk"] / _medians(base)["bulk"],
        "guard_010": _medians(new)["bulk"] / _medians(old)["bulk"],
        "shape_ratios": _ratios(_medians(new), _medians(old), SHAPES),
        "ref_ratios": _ratios(_medians(ref_new), _medians(ref_old), SHAPES),
    }


def _plane_table(data: dict) -> list[str]:
    old, new = data["bytes"]
    lines = [
        "| array | 0.1.0 | 0.2.0 | FinnGen ratio | OGS-00009 ratio (this ADR) |",
        "|---|---:|---:|---:|---:|",
    ]
    for node in PLANES:
        lines.append(
            f"| `{node.removeprefix('data.zarr/')}` | {old[node] / 1e9:.2f} GB | "
            f"{new[node] / 1e9:.2f} GB | {data['planes'][node]:.3f}x | "
            f"{data['ref_planes'][node]:.3f}x |"
        )
    return lines


def _ratio_table(data: dict) -> list[str]:
    lines = ["| shape | FinnGen | OGS-00009 (this ADR) |", "|---|---:|---:|"]
    for shape in SHAPES:
        lines.append(
            f"| {shape.replace('_', ' ')} | {data['shape_ratios'][shape]:.2f}x | "
            f"{data['ref_ratios'][shape]:.2f}x |"
        )
    return lines


def _label(key: str) -> str:
    """A shape as the tables print it, or a plane as a code span."""
    return key.replace("_", " ") if key in SHAPES else f"`{key.removeprefix('data.zarr/')}`"


def _above(ratios: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """{key: how far FinnGen's ratio sits above OGS-00009's, as a fraction} where it does."""
    return {
        key: ratios[key] / reference[key] - 1
        for key in ratios
        if ratios[key] > reference[key]
    }


def _excess_sentence(what: str, above: dict[str, float], of: int) -> str:
    if not above:
        return f"No FinnGen {what} ratio is above its OGS-00009 counterpart."
    names = ", ".join(_label(key) for key in above)
    return (
        f"FinnGen's {what} ratio is above OGS-00009's on {len(above)} of {of} ({names}), "
        f"by at most {max(above.values()) * 100:.1f} %."
    )


def _failures(data: dict) -> list[str]:
    """Each way FinnGen contradicts the decision; empty when it does not."""
    above = {
        **_above(data["shape_ratios"], data["ref_ratios"]),
        **_above(data["planes"], data["ref_planes"]),
        **_above({"total size": data["total"]}, {"total size": data["ref_total"]}),
    }
    failures = [
        f"{_label(key)} is {value * 100:.1f} % above OGS-00009"
        for key, value in above.items()
        if value > MATERIAL
    ]
    if data["guard"] > GUARD_LIMIT:
        failures.append(f"the whole-Analysis guard is {data['guard']:.2f}x, over {GUARD_LIMIT}x")
    if not data["conv"]["bit_exact"]:
        failures.append("the conversion was not verified bit-exact")
    return failures


def _verdict(data: dict) -> list[str]:
    grew = [shape for shape in SHAPES if data["shape_ratios"][shape] > 1]
    failures = _failures(data)
    conclusion = (
        f"**FinnGen contradicts the decision: {'; '.join(failures)}. This needs a correction "
        "in this ADR, not a footnote.**" if failures else
        "**The decision is not contradicted and is not corrected.**"
    )
    return [
        f"The whole-Analysis guard (0.2.0's median against the Zarr 2.18 median, the way this "
        f"ADR defines it) is **{data['guard']:.2f}x**, against the {GUARD_LIMIT}x limit; the same "
        f"ratio against this code on 0.1.0 is {data['guard_010']:.2f}x. The format costs time on "
        f"{len(grew)} of these {len(SHAPES)} shapes ({', '.join(map(_label, grew))}) and gains it "
        "on the rest. "
        + _excess_sentence("time", _above(data["shape_ratios"], data["ref_ratios"]), len(SHAPES))
        + " "
        + _excess_sentence("per-plane size", _above(data["planes"], data["ref_planes"]),
                           len(PLANES))
        + f" Total apparent size grows {(data['total'] - 1) * 100:+.2f} % against OGS-00009's "
        f"{(data['ref_total'] - 1) * 100:+.2f} %. A FinnGen ratio more than "
        f"{MATERIAL * 100:.0f} % above OGS-00009's would be a cost the decision's data did not "
        f"show. {conclusion} Full report: `docs/benchmark-output/opengwasdb_ogs00016_0_2_0.html`.",
        "",
        f"Converted {'bit-exactly' if data['conv']['bit_exact'] else '**not** bit-exactly'}.",
    ]


def build() -> str:
    data = _inputs()
    dataset = data["dataset"]
    sources = (
        f"`docs/benchmark-output/{CMP_Z3.name}`, `.../{CMP_218.name}`, "
        f"`.../{CONVERSION.name}` and `.../{OGS00009.relative_to(OUT).as_posix()}`"
    )
    intro = [
        "## Checked on FinnGen (#250)",
        "",
        f"Generated by `python3 scripts/build_finngen_shape_check.py` from {sources}. Run on the "
        f"full FinnGen R13 release ({dataset['n_analyses']:,} Analyses x "
        f"{dataset['n_variants']:,} variants). Total apparent size grows "
        f"**{(data['total'] - 1) * 100:+.2f} %** ({data['conv']['before']['n_files']:,} files -> "
        f"**{data['conv']['after']['n_files']:,}**). Per plane, 0.2.0 over 0.1.0, beside this "
        "ADR's OGS-00009 `v3-c64` over `v2-c1000`:",
        "",
    ]
    middle = [
        "",
        "The 0.1.0 -> 0.2.0 time ratios under Zarr 3, FinnGen beside OGS-00009:",
        "",
    ]
    return "\n".join(
        intro + _plane_table(data) + middle + _ratio_table(data) + [""] + _verdict(data) + [""]
    )


def splice(block: str, *, write: bool) -> int:
    text = ADR.read_text()
    before, _, rest = text.partition(BEGIN)
    _, _, after = rest.partition(END)
    if not (before and rest and after):
        raise SystemExit(f"{ADR}: missing the generated-block markers")
    updated = f"{before}{BEGIN}\n{block}{END}{after}"
    if write:
        ADR.write_text(updated)
        print(f"{ADR}: spliced {len(block)} characters")
        return 0
    if updated != text:
        print(f"{ADR}: the committed block differs from the artifacts", file=sys.stderr)
        return 1
    print(f"{ADR}: matches the artifacts")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="splice the block into ADR 0058")
    ap.add_argument("--check", action="store_true", help="fail if ADR 0058 is out of date")
    args = ap.parse_args()
    block = build()
    if args.write or args.check:
        raise SystemExit(splice(block, write=args.write))
    print(block)


if __name__ == "__main__":
    main()

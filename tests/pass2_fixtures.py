"""Two Dense VCF builds whose Pass 2 meets every case a Routing Key can, and their spills.

`tests/golden/pass2_spills.json` holds the spills these builds wrote at
`39c74f2`, when Pass 2 found each association's row by a binary search over its
Routing Key's bytes. Running this module writes the file again:

    pixi run -e dev python tests/pass2_fixtures.py

so a test comparing against it checks the integer key join against the search
it replaced, never the join against itself.

Both builds read the same two GWAS-VCF Analyses, declared hg19 and lifted by a
chain that maps chr2 onto its reverse strand and chr3 onto itself; chr5 has no
chain. `two_pass` routes by the sources' own union. `single_pass` routes by a
Variant Reference extracted from the first Analysis alone, so the second
Analysis's own sites are on no axis.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from extraction_fixtures import extract_from_manifest
from liftover_fixtures import write_two_strand_chain

from opengwasdb.layouts.dense import build_vcf
from opengwasdb.variants.axis import VariantAxis

GOLDEN = Path(__file__).parent / "golden" / "pass2_spills.json"

#: 300 inserted bases: a Routing Key of more than 300 bytes.
LONG = "ACGT" * 75
INSERTED = "A" + LONG
#: The long deletion's reference allele, as one Analysis spells it.
DELETED = "A" + LONG

_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FILTER=<ID=PASS,Description="All filters passed">\n'
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
    "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
)

#: ``(chrom, hg19 position, REF, ALT, ES, SE, AF)``, in file order.
Row = tuple[str, int, str, str, float, float, float]

ALPHA: list[Row] = [
    # Reverse strand: the lift complements both alleles and re-orients the row (#284).
    ("2", 1_000, "G", "A", 0.2, 0.1, 0.2),
    ("2", 2_000, "T", "A", 0.3, 0.1, 0.3),
    ("3", 1_000, "G", "A", 0.2, 0.1, 0.2),
    # A long insertion and a long deletion at one position: two variants.
    ("3", 5_000, "A", INSERTED, -0.1, 0.05, 0.1),
    ("3", 5_000, DELETED, "A", 0.3, 0.06, 0.02),
    # BMI_IRN's insertion and deletion of one repeat: two variants (ADR 0079).
    ("3", 6_000, "G", "GGGGCC", -0.00115727, 0.0135908, 0.00828552),
    ("3", 6_000, "GGGGCC", "G", 0.0226233, 0.05058, 0.000487289),
    # One site in lower and upper case, agreeing: one association, collapsed.
    ("3", 7_000, "c", "t", 0.5, 0.1, 0.4),
    ("3", 7_000, "C", "T", 0.5, 0.1, 0.4),
    # One site in two mixed-case spellings that disagree: both dropped.
    ("3", 8_000, "a", "G", 0.4, 0.1, 0.3),
    ("3", 8_000, "A", "g", 0.6, 0.1, 0.3),
    # An exact copy: collapsed.
    ("3", 9_000, "A", "G", 0.1, 0.1, 0.5),
    ("3", 9_000, "A", "G", 0.1, 0.1, 0.5),
    # A mixed-case deletion, which the other Analysis spells in upper case.
    ("3", 10_000, "acgT", "a", -0.2, 0.1, 0.25),
    # No chain covers chr5: the lift places it nowhere.
    ("5", 1_000, "A", "G", 0.7, 0.1, 0.5),
]

BETA: list[Row] = [
    ("2", 1_000, "g", "a", 0.25, 0.1, 0.2),
    ("3", 1_000, "G", "A", 0.25, 0.1, 0.2),
    ("3", 5_000, "a", INSERTED.lower(), -0.15, 0.05, 0.1),
    ("3", 6_000, "GGGGCC", "G", 0.02, 0.05, 0.0005),
    ("3", 10_000, "ACGT", "A", -0.3, 0.1, 0.25),
    # Sites only this Analysis names: on no axis of the single-pass build.
    ("3", 12_000, "T", "C", 0.9, 0.1, 0.6),
    ("3", 13_000, "A", "AT", 0.1, 0.1, 0.1),
    ("3", 13_000, "A", "AT", 0.2, 0.1, 0.1),
    ("3", 13_000, "AT", "A", 0.3, 0.1, 0.1),
    ("5", 2_000, "C", "T", 0.8, 0.1, 0.5),
]

ANALYSES = {"alpha": ALPHA, "beta": BETA}
BUILDS = ("two_pass", "single_pass")

#: The spill arrays `_spill_column` writes, in its order.
SPILL_ARRAYS = ("rows", "z", "se", "eaf")

Spills = dict[int, dict[str, np.ndarray]]


def _vcf(path: Path, rows: list[Row]) -> Path:
    body = "".join(
        f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t{es}:{se}:{af}\n"
        for chrom, pos, ref, alt, es, se, af in rows
    )
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def _manifest(directory: Path, names: list[str]) -> Path:
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_assembly"
    ]
    for name in names:
        source = _vcf(directory / f"{name}.vcf", ANALYSES[name])
        lines.append(f"{name}\t{source}\t{name}\t1000\tsd\tdeclared_standardised\t\thg19")
    manifest = directory / f"manifest-{'-'.join(names)}.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


@contextmanager
def captured_spills(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spills]:
    """Every column `_spill_column` writes while the block runs, by column.

    The build runs in this process (one worker), so the spill is seen exactly
    as it is handed to the writer, before the writer's own refusal.
    """
    spills: Spills = {}
    write = build_vcf._spill_column

    def capture(spill_dir: Path, col_idx: int, *arrays: np.ndarray) -> None:
        spills[col_idx] = {
            name: np.array(array, copy=True)
            for name, array in zip(SPILL_ARRAYS, arrays, strict=True)
        }
        write(spill_dir, col_idx, *arrays)

    monkeypatch.setattr(build_vcf, "_spill_column", capture)
    yield spills


def build(directory: Path, which: str) -> Path:
    """Build ``which`` of `BUILDS` under ``directory``; returns the Store."""
    directory.mkdir(parents=True, exist_ok=True)
    chain = write_two_strand_chain(directory / "strands.chain")
    manifest = _manifest(directory, list(ANALYSES))
    reference = None
    if which == "single_pass":
        reference = directory / "alpha.variant-ref.tsv.gz"
        extract_from_manifest(
            _manifest(directory, ["alpha"]), reference, chain_file=chain,
            liftover_failure_threshold=0.5,
        )
    store = directory / f"{which}.opengwasdb"
    build_vcf.build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", chain_file=chain,
        liftover_failure_threshold=0.5, variant_reference=reference,
    )
    return store


def axis(store: Path) -> list[str]:
    return [variant.alid for variant in VariantAxis(store).all()]


def pinned(spills: Spills, alids: list[str]) -> dict[str, object]:
    """What the golden file records of one build: its axis, and each column's
    spill arrays with their dtypes."""
    return {
        "alids": alids,
        "columns": {
            str(col): {
                name: {"dtype": str(array.dtype), "values": array.tolist()}
                for name, array in arrays.items()
            }
            for col, arrays in sorted(spills.items())
        },
    }


def _write_golden() -> None:
    golden = {}
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as patch:
        for which in BUILDS:
            with captured_spills(patch) as spills:
                store = build(Path(tmp) / which, which)
            golden[which] = pinned(spills, axis(store))
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    _write_golden()
    sys.exit(0)

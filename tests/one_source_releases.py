"""Releases of each layout built from one small source, for the 0.3.0 query tests.

`rows` are ``(chrom, pos, ref, alt, beta, se, af_alt)``, effect for `alt`, as
`test_indel_pairs` writes them. The Dense release is a FinnGen build, the Ragged one a
signals-only GWAS-SSF build of Analysis `trait_a`, and the Hybrid a GWAS-VCF build whose
Dense Component panel is `panel`; the rest of its rows are Ragged Overflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import test_indel_pairs as pair
from admission_helpers import ragged_ssf_store

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

Rows = Sequence[tuple[str, str, str, str, str, str, str]]


def dense_release(root: Path, rows: Rows) -> Path:
    store = root / "dense.opengwasdb"
    source = pair._finngen(root / "bmi_irn.tsv", list(rows))
    build_dense_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.finngen-r13"), store,
        store_id="s", release_id="r",
    )
    return store


def ragged_release(root: Path, rows: Rows) -> Path:
    store, _ = ragged_ssf_store(root, pair._ssf_rows(list(rows), "11"))
    return store


def hybrid_release(root: Path, rows: Rows, panel: Sequence[str]) -> Path:
    source = pair._gwas_vcf(root / "bmi_irn.vcf", list(rows))
    panel_file = root / "panel.txt"
    panel_file.write_text(
        "#opengwasdb-variant-reference\t1\n" + "".join(f"{a}\n" for a in panel),
        encoding="utf-8",
    )
    store = root / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.gwas-vcf"), store, reference_panel=panel_file,
        store_id="s", release_id="r",
    )
    return store


#: ``(chrom, pos, ref, alt, beta, se, af_alt)`` over three chromosomes, with a position
#: three variants share and a chromosome-2 variant at a position below chromosome 1's
#: last, so a range that leaks over a boundary shows. Every z is 6 or more, so a
#: signals-only Ragged build keeps every row.
ROWS = [
    ("1", "1000", "A", "G", "0.06", "0.01", "0.2"),
    ("1", "5000", "C", "A", "0.07", "0.01", "0.3"),
    ("1", "5000", "C", "G", "-0.08", "0.01", "0.25"),
    ("1", "5000", "C", "T", "0.09", "0.01", "0.35"),
    ("1", "9000", "G", "A", "-0.06", "0.01", "0.4"),
    ("2", "300", "T", "C", "0.07", "0.01", "0.2"),
    ("2", "7000", "A", "C", "-0.07", "0.01", "0.3"),
    ("X", "2500", "G", "T", "0.08", "0.01", "0.45"),
]
#: The Hybrid's Dense Component panel: the rest are Ragged Overflow.
PANEL = ["1:1000:A:G", "1:5000:C:A", "1:5000:C:T", "2:300:T:C"]

#: Every range question the rewrite's proof can pose (`query_digests.range_questions`);
#: a release of `ROWS` poses them all.
QUESTIONS = {
    "first", "to_first", "last", "from_last", "boundary/before", "boundary/after",
    "empty/between", "empty/chromosome", "shared", "shared/from_before", "shared/to_after",
}

#: Each layout's 0.3.0 release of `ROWS`, built under a root directory.
LAYOUTS: dict[str, Callable[[Path], Path]] = {
    "dense": lambda root: dense_release(root, ROWS),
    "ragged": lambda root: ragged_release(root, ROWS),
    "hybrid": lambda root: hybrid_release(root, ROWS, PANEL),
}

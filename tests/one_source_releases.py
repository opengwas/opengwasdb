"""Releases of each layout built from one small source, for the 0.3.0 query tests.

`rows` are ``(chrom, pos, ref, alt, beta, se, af_alt)``, effect for `alt`, as
`test_indel_pairs` writes them. The Dense release is a FinnGen build, the Ragged one a
signals-only GWAS-SSF build of Analysis `trait_a`, and the Hybrid a GWAS-VCF build whose
Dense Component panel is `panel`; the rest of its rows are Ragged Overflow.
"""

from __future__ import annotations

from collections.abc import Sequence
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

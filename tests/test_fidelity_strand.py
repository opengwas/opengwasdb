"""`fidelity.source` on a lift with reverse-strand blocks (issue #486).

A reverse-strand chain block complements a variant's alleles, which reverses
their order in its ALID, so the stored z is the negation of the source's
(`statistics_swap`, issue #284; `liftover_fixtures.STRAND_ROWS`). The check
compared the source's z with the stored one without that projection, so it
failed a correctly lifted Store on every reverse-strand row.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import test_extract_variant_reference as tev
from liftover_fixtures import (
    STRAND_ROWS,
    assert_strand_statistics,
    strand_vcf_rows,
    write_two_strand_chain,
)

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.query import query_store
from opengwasdb.validation import validate as validate_module
from opengwasdb.validation import validate_store

RULE = "fidelity.source"


@pytest.fixture
def lifted(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A Dense VCF build of `STRAND_ROWS` lifted by the two-strand chain:
    ``(store, manifest, chain)``."""
    chain = write_two_strand_chain(tmp_path / "two-strand.chain")
    vcf = tev._make_vcf(tmp_path, "strand_rows", strand_vcf_rows())
    manifest = tev._make_manifest(tmp_path, [("strand", vcf, "", "hg19")])
    store = tmp_path / "lifted.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", chain_file=chain,
        liftover_failure_threshold=0.5,
    )
    return store, manifest, chain


def test_the_fixture_stores_reverse_strand_rows_re_oriented(lifted) -> None:
    """Without reverse-strand rows stored with negated z, a pass below proves nothing."""
    store, _manifest, _chain = lifted
    assert any(z < 0 for *_, z, _eaf in STRAND_ROWS), "no reverse-strand row in the fixture"
    with query_store(store) as query:
        assert_strand_statistics(query, "strand")


def test_a_reverse_strand_lift_passes_source_fidelity(lifted) -> None:
    """Resolved through the stored `source_alid`, oriented by the chain the build used."""
    store, manifest, chain = lifted
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=chain)
    assert RULE not in result.failed(), result.errors


def test_a_reverse_strand_lift_passes_through_the_liftover_fallback(lifted, monkeypatch) -> None:
    """A Store predating `source_alid` resolves its rows by lifting the sample instead."""
    store, manifest, chain = lifted
    monkeypatch.setattr(validate_module, "_resolve_via_origin", lambda *_args: None)
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=chain)
    assert RULE not in result.failed(), result.errors


def test_a_chain_that_does_not_place_a_held_association_fails_loudly(
    lifted, tmp_path: Path
) -> None:
    """A Store holds chr2 rows by `source_alid`, but the chain given here places only chr3:
    the orientation of those rows cannot be known, so the check fails naming the chain,
    rather than comparing them under a guessed strand or dropping them unsaid."""
    store, manifest, _chain = lifted
    size = 1_000_000
    chr3_only = tmp_path / "chr3-only.chain"
    chr3_only.write_text(
        f"chain 1000 chr3 {size} + 0 {size} chr3 {size} + 0 {size} 2\n{size}\n\n",
        encoding="utf-8",
    )
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=chr3_only)
    assert RULE in result.failed()
    assert any("chain" in error and "2 " in error for error in result.errors), result.errors


def test_a_reverse_strand_row_that_disagrees_with_its_source_still_fails(
    lifted, tmp_path: Path
) -> None:
    """The projection is applied, not the comparison skipped: against a source whose effects
    are the negation of the built one's, every reverse-strand row disagrees."""
    store, _manifest, chain = lifted
    negated = [row.replace("0.2:0.1:0.2", "-0.2:0.1:0.2") for row in strand_vcf_rows()]
    assert negated != strand_vcf_rows(), "the negated source is the built one"
    other = tmp_path / "negated"
    other.mkdir()
    vcf = tev._make_vcf(other, "strand_rows", negated)
    manifest = tev._make_manifest(other, [("strand", vcf, "", "hg19")])
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=chain)
    assert RULE in result.failed()
    assert any("3 of 3 compared" in error for error in result.errors), result.errors

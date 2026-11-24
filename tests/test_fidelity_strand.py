"""`fidelity.source` on a lift with reverse-strand blocks (issue #486).

A reverse-strand chain block complements a variant's alleles, which reverses
their order in its ALID, so the stored z is the negation of the source's
(`statistics_swap`, issue #284; `liftover_fixtures.STRAND_ROWS`). The check
compared the source's z with the stored one without that projection, so it
failed a correctly lifted Store on every reverse-strand row.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import test_extract_variant_reference as tev
from liftover_fixtures import (
    STRAND_ROWS,
    assert_strand_statistics,
    strand_vcf_rows,
    write_two_strand_chain,
)
from region_tables import edit_manifest

from opengwasdb.build.liftover import BUILTIN_CHAIN_LABEL
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.query import query_store
from opengwasdb.validation import validate as validate_module
from opengwasdb.validation import validate_store
from opengwasdb.validation.fidelity_chain import (
    BUILTIN_CHAIN_NAME,
    ChainRefused,
    FidelityChain,
    fidelity_chain,
)

RULE = "fidelity.source"


def _provenance(store: Path, **entries: Any) -> None:
    """Set (a value) or remove (``None``) `provenance` entries of ``store``'s manifest."""

    def edit(manifest: dict[str, Any]) -> dict[str, Any]:
        provenance = {**manifest["provenance"], **entries}
        return {**manifest, "provenance": {k: v for k, v in provenance.items() if v is not None}}

    edit_manifest(store, edit)


def _chr3_only_chain(path: Path) -> Path:
    """A real chain placing chr3 alone: the fixture's chr2 rows land nowhere."""
    size = 1_000_000
    path.write_text(
        f"chain 1000 chr3 {size} + 0 {size} chr3 {size} + 0 {size} 2\n{size}\n\n",
        encoding="utf-8",
    )
    return path


def _fidelity_errors(store: Path, manifest: Path, chain: Path | None) -> list[str]:
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=chain)
    assert RULE in result.failed(), result.errors
    return result.errors


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
    rather than comparing them under a guessed strand or dropping them unsaid. The release
    records no chain here, so nothing refuses this one before the samples are lifted."""
    store, manifest, _chain = lifted
    _provenance(store, chain_file=None)
    chr3_only = _chr3_only_chain(tmp_path / "chr3-only.chain")
    errors = _fidelity_errors(store, manifest, chr3_only)
    assert any(
        f"{chr3_only} does not place 2 sampled" in error for error in errors
    ), errors


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


# ── The chain is the build's (#486 fix round 1) ──────────────────────────────


def test_a_copy_of_the_recorded_chain_is_the_recorded_chain(lifted, tmp_path: Path) -> None:
    """Matching: the same bytes at another path are the chain the build lifted with."""
    store, manifest, chain = lifted
    copy = shutil.copy(chain, tmp_path / "copy.chain")
    result = validate_store(store, source=manifest, source_assembly="hg19", chain_file=copy)
    assert RULE not in result.failed(), result.errors


def test_a_chain_that_contradicts_the_recorded_one_is_refused(lifted, tmp_path: Path) -> None:
    """Contradicting: other bytes are refused before a sample is read, naming both chains."""
    store, manifest, chain = lifted
    other = _chr3_only_chain(tmp_path / "other.chain")
    errors = _fidelity_errors(store, manifest, other)
    assert any(f"the chain {other} is not the one the release records" in e for e in errors)
    assert any(str(chain) in e for e in errors), errors


def test_a_chain_given_where_the_release_records_the_built_in_one_is_refused(
    lifted, tmp_path: Path
) -> None:
    """Contradicting: a release that records pyliftover's built-in chain names no file."""
    store, manifest, chain = lifted
    _provenance(store, chain_file=BUILTIN_CHAIN_LABEL)
    errors = _fidelity_errors(store, manifest, chain)
    assert any(BUILTIN_CHAIN_NAME in e and str(chain) in e for e in errors), errors


def test_a_chain_whose_bytes_differ_from_the_recorded_sha256_is_refused(lifted) -> None:
    """Contradicting: a recorded sha256 is the chain's identity, whatever its path holds."""
    store, manifest, chain = lifted
    _provenance(store, chain_sha256="0" * 64)
    errors = _fidelity_errors(store, manifest, chain)
    assert any(f"the chain {chain} is not the one the release records" in e for e in errors)
    errors = _fidelity_errors(store, manifest, None)
    assert any("now differs" in e and str(chain) in e for e in errors), errors


def test_with_no_chain_given_the_recorded_one_is_used(lifted) -> None:
    """Absent and resolvable: the recorded path is on disk, so the check lifts through it."""
    store, manifest, _chain = lifted
    result = validate_store(store, source=manifest, source_assembly="hg19")
    assert RULE not in result.failed(), result.errors


def test_with_no_chain_given_the_built_in_label_resolves_to_pyliftover() -> None:
    """Absent and resolvable: the built-in label lifts with pyliftover's own chain."""
    assert fidelity_chain({"chain_file": BUILTIN_CHAIN_LABEL}, None) == FidelityChain(
        None, BUILTIN_CHAIN_NAME
    )


def test_with_no_chain_given_a_recorded_chain_not_on_disk_is_refused(lifted) -> None:
    """Absent and unresolvable: the check names the recorded chain and asks for one."""
    store, manifest, chain = lifted
    chain.rename(chain.with_suffix(".moved"))
    errors = _fidelity_errors(store, manifest, None)
    assert any(str(chain) in e and "--chain-file" in e for e in errors), errors


def test_with_no_chain_given_or_recorded_the_check_is_refused(lifted) -> None:
    """Absent and unresolvable: nothing says which chain the build lifted with."""
    store, manifest, _chain = lifted
    _provenance(store, chain_file=None)
    errors = _fidelity_errors(store, manifest, None)
    assert any("records no liftover chain" in e for e in errors), errors
    with pytest.raises(ChainRefused, match="records no liftover chain"):
        fidelity_chain({}, None)

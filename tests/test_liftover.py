"""Tests for opengwasdb.build.liftover."""

from __future__ import annotations

import pytest
from liftover_fixtures import STRAND_CHAIN_SIZE, write_two_strand_chain

from opengwasdb.build.liftover import (
    LiftedAlleleMismatchError,
    LiftedRow,
    LiftoverFailureError,
    build_liftover_lookup,
    require_lifted_alleles_match_reference,
)

# Known good positions: hg19 1:100000 → hg38 1:100000 (identity in this region)
# hg19 1:1000000 → hg38 1:1064620
KNOWN_HG19_POSITIONS = [
    ("1", 100_000, "A", "G"),
    ("1", 1_000_000, "C", "T"),
]


def test_build_liftover_lookup_returns_dict_for_successful_variants():
    result = build_liftover_lookup(KNOWN_HG19_POSITIONS)

    assert len(result) == 2
    for key in KNOWN_HG19_POSITIONS:
        assert key in result


def test_build_liftover_lookup_output_alids_have_canonical_ordering():
    variants = [("1", 100_000, "G", "A")]  # G > A, so A is A1
    result = build_liftover_lookup(variants)

    assert len(result) == 1
    alid = list(result.values())[0]
    parts = alid.split(":")
    a1, a2 = parts[2], parts[3]
    assert a1 <= a2  # A1 is alphabetically first


def test_build_liftover_lookup_known_coordinate_maps_correctly():
    # hg19 1:100000 → hg38 1:100000 (identity at this known position)
    result = build_liftover_lookup([("1", 100_000, "A", "G")])
    alid = result[("1", 100_000, "A", "G")]
    chrom, pos, a1, a2 = alid.split(":")
    assert chrom == "1"
    assert int(pos) == 100_000  # this position maps identically


def test_build_liftover_lookup_accepts_chr_prefixed_input():
    variants = [("chr1", 100_000, "A", "G")]
    result = build_liftover_lookup(variants)
    assert len(result) == 1
    # Key in dict preserves original form
    assert ("chr1", 100_000, "A", "G") in result


def test_build_liftover_lookup_omits_failed_variants_below_threshold():
    # Positions that don't lift (e.g., in unsequenced gaps) are omitted.
    # hg19 1:200000 fails; 1:100000 succeeds.
    variants = [("1", 200_000, "A", "G"), ("1", 100_000, "C", "T")]
    result = build_liftover_lookup(variants, failure_threshold=0.6)

    assert ("1", 100_000, "C", "T") in result
    assert ("1", 200_000, "A", "G") not in result


def test_build_liftover_lookup_raises_when_failure_rate_exceeds_threshold():
    # Both positions in a known gap so both should fail.
    # hg19 1:200000 and 1:300000 both fail liftover.
    variants = [
        ("1", 200_000, "A", "G"),
        ("1", 300_000, "C", "T"),
    ]
    with pytest.raises(LiftoverFailureError, match="failure rate"):
        build_liftover_lookup(variants, failure_threshold=0.01)


def test_build_liftover_lookup_empty_input_returns_empty_dict():
    result = build_liftover_lookup([])
    assert result == {}


def test_build_liftover_lookup_output_chrom_is_bare():
    result = build_liftover_lookup([("1", 100_000, "A", "G")])
    alid = list(result.values())[0]
    chrom = alid.split(":")[0]
    assert not chrom.startswith("chr")


# ── The mapping strand (issue #284) ───────────────────────────────────────────
#
# A chain block has an orientation, and on a reverse-strand block the source's
# alleles are read on the other strand of the target. The expected values below
# are the chain fixture's own arithmetic -- `write_two_strand_chain` mirrors
# chr2 (1-based p -> STRAND_CHAIN_SIZE - p + 1) and leaves chr3 alone -- not a
# reading of the code under test.


def test_build_liftover_lookup_complements_alleles_on_a_reverse_strand_mapping(tmp_path):
    chain = write_two_strand_chain(tmp_path / "hg19ToHg38.chain")
    variants = [("2", 1_000, "A", "G"), ("3", 1_000, "A", "G")]

    result = build_liftover_lookup(
        variants, from_build="hg19", to_build="hg38", chain_file=chain
    )

    assert len(result) == 2, "both fixture rows must lift for the pair to mean anything"
    # Chr3 is a forward-strand chain block: the same variant, unchanged.
    assert result[("3", 1_000, "A", "G")] == "3:1000:A:G"
    # Chr2 is a reverse-strand block: the mirror position, A->T and G->C, so the
    # canonical pair is C:T. The bug this test pins kept A:G here.
    reversed_alid = result[("2", 1_000, "A", "G")]
    assert reversed_alid == f"2:{STRAND_CHAIN_SIZE - 1_000 + 1}:C:T"


# ── Checking the lift against reference alleles (issue #284) ─────────────────
#
# The rows below are the chain fixture's arithmetic done by hand: hg19
# 2:1000:A:G maps through the reverse-strand block to hg38 2:999001, where the
# reference holds C:T -- the complement of the source's pair.

_REVERSE_SOURCE = ("2", 1_000, "A", "G")
_REVERSE_LIFTED = LiftedRow(
    "2", STRAND_CHAIN_SIZE - 1_000 + 1, "T", "C", strand="-"
)


def _reference(tmp_path, alids):
    path = tmp_path / "reference.txt"
    path.write_text("".join(f"{alid}\n" for alid in alids), encoding="utf-8")
    return path


def test_lifted_alleles_are_verified_against_a_reference_that_holds_them(tmp_path):
    reference = _reference(tmp_path, ["2:999001:C:T"])

    check = require_lifted_alleles_match_reference(
        {_REVERSE_SOURCE: _REVERSE_LIFTED}, reference
    )

    assert check.n_lifted == 1
    assert check.n_verified == 1
    assert check.n_other_variant == 0
    assert check.n_absent == 0


def test_a_lift_that_kept_the_source_alleles_fails_on_its_own_strand(tmp_path):
    """The regression guard for #284 itself: with the projection dropped the
    stored pair is the source's A:G while the reference holds the projected
    C:T, and only the recorded strand distinguishes that from an ordinary
    second variant at the site."""
    reference = _reference(tmp_path, ["2:999001:C:T"])
    unprojected = LiftedRow(
        _REVERSE_LIFTED.chromosome, _REVERSE_LIFTED.position, "A", "G", strand="-"
    )
    assert frozenset({unprojected.effect_allele, unprojected.other_allele}) != frozenset(
        {_REVERSE_LIFTED.effect_allele, _REVERSE_LIFTED.other_allele}
    ), "the fixture rows must differ in allele, or this proves nothing"

    with pytest.raises(LiftedAlleleMismatchError, match="2:1000:A:G"):
        require_lifted_alleles_match_reference({_REVERSE_SOURCE: unprojected}, reference)


def test_a_reference_naming_another_variant_at_the_site_is_counted_not_failed(tmp_path):
    """A Variant Reference is a variant list, not a genome: a site it names with
    an unrelated pair cannot say whether this lift's variant exists, so it is
    counted rather than failed -- the class measured at 2,856 eQTLGen variants
    against the EUR alid panel with and without the strand fix."""
    reference = _reference(tmp_path, ["2:999001:G:T", "2:999002:A:C"])

    check = require_lifted_alleles_match_reference(
        {
            _REVERSE_SOURCE: _REVERSE_LIFTED,
            ("2", 1_001, "C", "T"): LiftedRow("2", 999_002, "C", "T"),
        },
        reference,
    )

    assert check.n_verified == 0
    assert check.n_other_variant == 2
    assert check.n_absent == 0

"""Canonical chromosome identity across source spelling conventions (issue #216)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import GwasSsfReader
from opengwasdb.variants import (
    VariantNormalisationError,
    normalise_chromosome,
    orient_to_canonical,
)


@pytest.mark.parametrize(
    ("spellings", "canonical"),
    [
        (("23", "X", "x", "chrX", "chrx", "chr23"), "X"),
        (("24", "Y", "y", "chrY", "chry", "chr24"), "Y"),
        (("25", "26", "M", "m", "MT", "mt", "chrM", "chrMT", "chr25", "chr26"), "MT"),
    ],
)
def test_normalise_chromosome_aliases_closed_non_autosomal_sets(
    spellings: tuple[str, ...], canonical: str
) -> None:
    assert {normalise_chromosome(spelling) for spelling in spellings} == {canonical}


def test_normalise_chromosome_leaves_autosomes_unchanged() -> None:
    assert [normalise_chromosome(str(number)) for number in range(1, 23)] == [
        str(number) for number in range(1, 23)
    ]


@pytest.mark.parametrize("invalid", ["", "chr", "0", "27", "chr27"])
def test_normalise_chromosome_rejects_labels_outside_the_closed_numeric_set(
    invalid: str,
) -> None:
    with pytest.raises(VariantNormalisationError, match="chromosome"):
        normalise_chromosome(invalid)


def _write_ssf(path: Path, chromosome: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
            "beta\tstandard_error\teffect_allele_frequency\n"
            f"{chromosome}\t100\tG\tA\t1.0\t0.5\t0.25\n"
        )


@pytest.mark.parametrize(
    ("letter_spelling", "numeric_spelling", "expected_alid"),
    [
        ("X", "23", "X:100:A:G"),
        ("Y", "24", "Y:100:A:G"),
        ("MT", "26", "MT:100:A:G"),
        ("M", "25", "MT:100:A:G"),
    ],
)
def test_gwas_ssf_sources_with_alias_spellings_produce_the_same_alid(
    tmp_path: Path,
    letter_spelling: str,
    numeric_spelling: str,
    expected_alid: str,
) -> None:
    letter_path = tmp_path / "letter.tsv.gz"
    numeric_path = tmp_path / "numeric.tsv.gz"
    _write_ssf(letter_path, letter_spelling)
    _write_ssf(numeric_path, numeric_spelling)

    letter_reader = GwasSsfReader(letter_path, StoredEffectScale.SD)
    numeric_reader = GwasSsfReader(numeric_path, StoredEffectScale.SD)
    [letter] = letter_reader.stream_associations()
    [numeric] = numeric_reader.stream_associations()
    [letter_variant] = letter_reader.stream_variants()
    [numeric_variant] = numeric_reader.stream_variants()

    assert letter_spelling != numeric_spelling, "fixture must exercise distinct source spellings"
    letter_alid = orient_to_canonical(
        letter.chromosome, letter.position, letter.alt, letter.ref
    ).variant.alid
    numeric_alid = orient_to_canonical(
        numeric.chromosome, numeric.position, numeric.alt, numeric.ref
    ).variant.alid
    assert letter_alid == numeric_alid == expected_alid
    assert letter_variant.chromosome == numeric_variant.chromosome == expected_alid.split(":")[0]

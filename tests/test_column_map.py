"""Loading a Column Map (issue #281, stores ADR 0036).

A Column Map is the data file that tells the generic tabular reader how one
Source Collection spells its columns and which conventions its values follow.
Every way it can be wrong is a refusal at load time, before any source is read:
a map that loads is a map whose every column and convention was declared, never
one whose gaps were filled in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from opengwasdb.readers.column_map import (
    BUILTIN_COLUMN_MAPS,
    ColumnMapError,
    load_column_map,
)

_VALID = """\
column_map_version = 1
name = "mvp-raw"
description = "dbGaP phs002453 raw deposit"

[columns]
chromosome = "chrom"
position = "pos"
ref = "ref"
alt = "alt"
effect_allele = "ea"
frequency = "af"
p_value = "pval"
rsid = "SNP_ID"

[optional_columns]
beta = "beta"
standard_error = "sebeta"
odds_ratio = "or"
confidence_interval = "ci"

[conventions]
missing_values = ["", "NA"]
unmapped_columns = "ignore"
alleles = "ref_alt_and_effect_allele"
frequency_allele = "effect_allele"
p_value_form = "p"
position_notation = "integer_or_exact_scientific"
chromosome_spelling = "adr-0052"
genome_build = "GRCh38"
confidence_interval_separator = ","
rsid_separator = "none"
"""


def _write(tmp_path: Path, text: str, name: str = "map.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_valid_map_loads_every_declared_column_and_convention(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID)

    column_map = load_column_map(path)

    assert column_map.name == "mvp-raw"
    assert column_map.columns["effect_allele"] == ("ea",)
    assert column_map.optional_columns["odds_ratio"] == ("or",)
    assert column_map.conventions.missing_values == ("", "NA")
    assert column_map.conventions.genome_build == "hg38"
    assert column_map.conventions.rsid_separator is None
    assert column_map.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_string_naming_a_file_loads_that_file(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID)

    assert load_column_map(str(path)).name == "mvp-raw"


@pytest.mark.parametrize("name", sorted(BUILTIN_COLUMN_MAPS))
def test_built_in_maps_load_by_name(name: str) -> None:
    column_map = load_column_map(name)

    assert column_map.name == name
    assert column_map.origin == f"builtin:{name}"
    assert len(column_map.sha256) == 64


def test_the_built_ins_are_the_two_tabular_formats_with_dedicated_readers() -> None:
    assert set(BUILTIN_COLUMN_MAPS) == {"gwas-ssf", "finngen-r13"}


def _replace(old: str, new: str) -> str:
    assert old in _VALID, f"fixture edit {old!r} must apply to the valid map"
    return _VALID.replace(old, new)


_REFUSALS = {
    "unknown top-level key": (_VALID + 'colour = "blue"\n', "colour"),
    "unknown field": (_replace('rsid = "SNP_ID"', 'rsid = "SNP_ID"\nlogp = "mlogp"'), "logp"),
    "unknown convention": (
        _replace('rsid_separator = "none"', 'rsid_separator = "none"\nstrand = "forward"'),
        "strand",
    ),
    "unknown convention value": (
        _replace('alleles = "ref_alt_and_effect_allele"', 'alleles = "sideways"'),
        "sideways",
    ),
    "undeclared required field": (_replace('chromosome = "chrom"\n', ""), "chromosome"),
    "required field declared optional": (
        _replace('chromosome = "chrom"\n', "").replace(
            "[optional_columns]\n", '[optional_columns]\nchromosome = "chrom"\n'
        ),
        "chromosome",
    ),
    "undeclared convention": (_replace('genome_build = "GRCh38"\n', ""), "genome_build"),
    "convention about an undeclared column": (
        _replace('frequency = "af"\n', ""),
        "frequency_allele",
    ),
    "column without its convention": (
        _replace('frequency_allele = "effect_allele"\n', ""),
        "frequency_allele",
    ),
    "no effect column": (
        _replace('beta = "beta"\n', "")
        .replace('odds_ratio = "or"\n', ""),
        "effect",
    ),
    "one provider column read as two fields": (
        _replace('odds_ratio = "or"', 'odds_ratio = "beta"'),
        "'beta'",
    ),
    "an interval with no separator": (
        _replace('confidence_interval_separator = ","', 'confidence_interval_separator = "none"'),
        "confidence_interval_separator",
    ),
    "unsupported version": (_replace("column_map_version = 1", "column_map_version = 2"), "2"),
    "not toml": ("column_map_version = \n", "TOML"),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_a_malformed_map_is_refused_naming_what_is_wrong(tmp_path: Path, case: str) -> None:
    text, expected = _REFUSALS[case]

    with pytest.raises(ColumnMapError, match=expected) as raised:
        load_column_map(_write(tmp_path, text))

    assert str(tmp_path / "map.toml") in str(raised.value), "the refusal must name the map"


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ('chromosome_spelling = "adr-0052"', 'chromosome_spelling = "plink"', "XY"),
        ('p_value_form = "p"', 'p_value_form = "neg_log10_p"', "-log10"),
        (
            'frequency_allele = "effect_allele"',
            'frequency_allele = "other_allele"',
            "other allele",
        ),
    ],
)
def test_a_known_convention_the_reader_cannot_honour_is_refused(
    tmp_path: Path, old: str, new: str, reason: str
) -> None:
    """Distinct from an unknown value: the reader knows what it means and says why
    it will not read it, rather than reading it as something else."""
    with pytest.raises(ColumnMapError, match="cannot honour") as raised:
        load_column_map(_write(tmp_path, _replace(old, new)))

    assert reason in str(raised.value)


def test_a_name_that_is_neither_built_in_nor_a_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ColumnMapError, match="gwas-ssf"):
        load_column_map(str(tmp_path / "no-such-map.toml"))


def test_a_built_in_name_shadowed_by_a_file_is_refused_as_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, _VALID, name="gwas-ssf")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ColumnMapError, match="ambiguous"):
        load_column_map("gwas-ssf")

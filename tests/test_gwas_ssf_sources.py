"""Every source the suite writes reads as batches as it reads per row (#319).

`test_gwas_ssf_batches` holds the adversarial cases; this holds the fixtures the
other test modules and `tests/fixtures/` already carry, so that no GWAS-SSF or
Column Map source in the suite escapes the identity check, and the declared
imputation score, whose rules the batch path reads in columns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import batch_fixtures as shared
import pytest
import ssf_cases
from test_allele_case import _ssf as _allele_case_ssf
from test_chromosome_normalisation import _write_ssf as _write_chromosome_ssf
from test_gwas_ssf_batches import _batched, _mapped, _per_row, _ssf
from test_malformed_identity import _ssf as _malformed_ssf
from test_malformed_identity import _truncated_ssf
from test_readers import _gwas_ssf_reader

import opengwasdb.readers.columnar as columnar
import opengwasdb.readers.gwas_ssf as gwas_ssf
import opengwasdb.readers.interface as interface
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.interface import StreamTally

FIXTURES = Path(__file__).parent / "fixtures"
SD, LOG_OR = StoredEffectScale.SD, StoredEffectScale.LOG_OR

#: A column map of the provider names `plain.toml` declares, over the fixture below.
_PLAIN = (
    'column_map_version = 1\nname = "plain"\n\n[columns]\nchromosome = "chr"\nposition = "bp"\n'
    'effect_allele = "a1"\nother_allele = "a2"\nbeta = "b"\nstandard_error = "se"\n\n'
    '[optional_columns]\nfrequency = "freq"\n\n[conventions]\nmissing_values = "any_non_numeric"\n'
    'unmapped_columns = "ignore"\nalleles = "effect_and_other"\nfrequency_allele = '
    '"effect_allele"\nposition_notation = "integer"\nchromosome_spelling = "adr-0052"\n'
    'genome_build = "GRCh37"\n'
)

Make = Callable[[Path, StoredEffectScale], Any]


def _readers_module(directory: Path) -> Path:
    return _path_of(_gwas_ssf_reader(directory))


def _path_of(reader: object) -> Path:
    assert isinstance(reader, GwasSsfReader)
    return Path(reader.path)


def _malformed(directory: Path) -> Path:
    return _path_of(_malformed_ssf(directory))


def _truncated(directory: Path) -> Path:
    return _path_of(_truncated_ssf(directory))


def _allele_cases(directory: Path) -> Path:
    path = directory / "allele.tsv.gz"
    _allele_case_ssf(path, "g", "A", 0.2, 0.1, rsid="rs1")
    return path


def _chromosomes(directory: Path) -> Path:
    path = directory / "chromosome.tsv.gz"
    _write_chromosome_ssf(path, "chrX")
    return path


def _plain_map_source(directory: Path) -> Path:
    path = directory / "plain.tsv"
    path.write_text(
        "chr\tbp\ta1\ta2\tb\tse\tfreq\n1\t100\tA\tG\t0.2\t0.1\t0.3\n2\t200\tC\tT\t-0.1\t0.05\tNA\n"
        "X\t300\tG\tA\t0.3\tNA\t0.4\n",
        encoding="utf-8",
    )
    return path


def _plain_mapped(path: Path, scale: StoredEffectScale) -> ColumnMapReader:
    map_path = path.parent / "plain.toml"
    map_path.write_text(_PLAIN, encoding="utf-8")
    return ColumnMapReader(path, load_column_map(map_path), scale)


def _mvp(path: Path, scale: StoredEffectScale) -> ColumnMapReader:
    return ColumnMapReader(path, load_column_map(FIXTURES / "dbgap_mvp" / "mvp-gia.toml"), scale)


#: Each source the suite keeps, as `(path, reader factory, scale)` written into a
#: directory, by name; the files under `tests/fixtures/` are keyed by path too.
def _written() -> dict[str, Callable[[Path], tuple[Path, Make, StoredEffectScale]]]:
    def written(write: Callable[[Path], Path], make: Make = _ssf) -> Any:
        return lambda directory: (write(directory), make, SD)

    return {
        "test_readers": written(_readers_module),
        "test_allele_case": written(_allele_cases),
        "test_chromosome_normalisation": written(_chromosomes),
        "test_malformed_identity": written(_malformed),
        "test_malformed_identity truncated": written(_truncated),
        "plain column map": written(_plain_map_source, _plain_mapped),
    }


#: The fixture files, by path under `tests/fixtures/`, with how each is read.
FIXTURE_SOURCES: dict[Path, tuple[Make, StoredEffectScale]] = {
    **{
        FIXTURES / "dbgap_mvp" / "gwas_ssf" / f"{shape}.tsv": (_ssf, scale)
        for shape, scale in (("binary", LOG_OR), ("continuous", SD), ("meta", SD))
    },
    **{
        FIXTURES / "dbgap_mvp" / "raw" / f"{shape}.tsv": (_mvp, scale)
        for shape, scale in (("binary", LOG_OR), ("continuous", SD), ("meta", SD))
    },
    FIXTURES / "finngen_r13.tsv": (
        lambda path, scale: ColumnMapReader(path, load_column_map("finngen-r13"), scale),
        LOG_OR,
    ),
}


def _identical(make: Make, path: Path, scale: StoredEffectScale) -> None:
    per_row = _per_row(make, path, scale)
    assert per_row.columns is not None and per_row.columns["z"], path
    assert per_row == _batched(make, path, scale)


@pytest.fixture(params=[None, 64, 1], ids=["default", "64B", "1B"])
def block(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int | None:
    if request.param is not None:
        monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", request.param)
    return cast(int | None, request.param)


@pytest.mark.parametrize("name", sorted(_written()))
def test_a_source_another_module_writes_reads_as_it_reads_per_row(
    tmp_path: Path, block: int | None, name: str
) -> None:
    path, make, scale = _written()[name](tmp_path)
    _identical(make, path, scale)
    if make is _ssf:
        _identical(_mapped, path, scale)


@pytest.mark.parametrize(
    "path", sorted(FIXTURE_SOURCES), ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_fixture_file_reads_as_it_reads_per_row(block: int | None, path: Path) -> None:
    make, scale = FIXTURE_SOURCES[path]
    _identical(make, path, scale)


#: Files under `tests/fixtures/` that are not sources, each with why. Nothing else
#: is exempt but a `.toml` Column Map, which maps a source and holds none.
EXCUSED: dict[Path, str] = {}


def unlisted_sources(root: Path, listed: Iterable[Path], excused: Iterable[Path]) -> list[Path]:
    """Every file under ``root`` that is neither a `.toml` map, nor read above, nor
    excused: whatever its name, since a reader gunzips a `.bgz` and reads a file
    with no extension all the same."""
    known = {*listed, *excused}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix != ".toml" and path not in known
    )


def test_every_source_file_under_the_fixtures_directory_is_read_above() -> None:
    """A source added to `tests/fixtures/` that this module neither lists nor
    excuses (with a reason) fails here, so the identity check cannot miss it."""
    assert unlisted_sources(FIXTURES, FIXTURE_SOURCES, EXCUSED) == []
    assert all(EXCUSED.values()), "an excused fixture says why"
    assert all(path.is_file() for path in [*FIXTURE_SOURCES, *EXCUSED])


def test_the_guard_names_a_source_whatever_its_name(tmp_path: Path) -> None:
    kept = tmp_path / "listed.tsv"
    names = ["new.bgz", "new.tab", "new", "new.tsv.gz", "map.toml"]
    for name in [kept.name, *names]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert unlisted_sources(tmp_path, [kept], []) == [
        tmp_path / name for name in sorted(names) if name != "map.toml"
    ]
    assert unlisted_sources(tmp_path, [kept], [tmp_path / "new.bgz"]) == [
        tmp_path / name for name in sorted(names) if name not in ("map.toml", "new.bgz")
    ]


def test_the_shadow_refuses_batches_that_differ_from_the_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shadow_gwas_ssf` re-reads every GWAS-SSF read of `test_readers` and
    `test_effect_recovery` as batches; it must be able to fail."""
    path = _readers_module(tmp_path)
    batches = GwasSsfReader.stream_association_batches

    def negated(self: GwasSsfReader, tally: StreamTally | None = None) -> Any:
        return (replace(batch, z=-batch.z) for batch in batches(self, tally))

    shared.shadow_gwas_ssf(monkeypatch)
    assert len(list(GwasSsfReader(path).stream_associations())) == 2
    monkeypatch.setattr(GwasSsfReader, "stream_association_batches", negated)
    with pytest.raises(AssertionError, match="batches differ"):
        list(GwasSsfReader(path).stream_associations())


# ── a declared imputation score ──────────────────────────────────────────────

_DECLARATION = interface.ImputationScoreDeclaration(
    "info", interface.ImputationScoreKind.IMPUTATION_INFO, "provider documentation: INFO"
)


def _declared(path: Path, scale: StoredEffectScale = SD) -> GwasSsfReader:
    return GwasSsfReader(path, scale, imputation_score_declaration=_DECLARATION)


def _scored_file(directory: Path, cells: list[str], newline: str = "\n") -> Path:
    path = directory / "scored.tsv"
    path.write_text(ssf_cases.scored(cells, newline=newline), encoding="utf-8")
    return path


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(
    ("cells", "statuses"),
    [
        (ssf_cases.SCORES, 4),
        (ssf_cases.MALFORMED_SCORES, 3),
        ([*ssf_cases.SCORES, *ssf_cases.MALFORMED_SCORES], 5),
    ],
    ids=["decidable", "malformed", "mixed"],
)
def test_a_declared_score_reads_in_columns_as_it_reads_per_row(
    tmp_path: Path, block: int | None, cells: list[str], statuses: int, newline: str
) -> None:
    path = _scored_file(tmp_path, cells * 2, newline)

    def make(path: Path, scale: StoredEffectScale) -> GwasSsfReader:
        return _declared(path, scale)

    per_row = _per_row(make, path, SD)
    assert per_row.columns is not None
    assert len(set(per_row.columns["imputation_status"])) == statuses
    assert per_row == _batched(make, path, SD)


def test_a_score_the_columns_decide_is_read_without_the_per_row_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every status a score takes is read from the columns: no row reaches the
    per-row parser and nothing is logged, so the identity above is the columnar
    parse's, not the per-row reader's twice. (An Analysis that declares a score
    was read row by row throughout until #319.)"""
    path = _scored_file(tmp_path, ssf_cases.SCORES * 3)
    expected = _per_row(lambda p, s: _declared(p, s), path, SD)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a row the columns decide was read row by row")

    monkeypatch.setattr(gwas_ssf, "_tabular_row", refuse)
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        assert _batched(lambda p, s: _declared(p, s), path, SD) == expected
    assert [r for r in caplog.records if r.name == gwas_ssf.__name__] == []


def test_a_block_with_a_malformed_score_alone_is_read_by_the_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A score pandas cannot read as `float()` reads it makes that block's rows
    be read from their own lines; the source is not read row by row throughout."""
    path = _scored_file(tmp_path, [*ssf_cases.SCORES, "bogus"])
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        list(_declared(path).stream_association_batches(StreamTally()))
    messages = [r.getMessage() for r in caplog.records if r.name == gwas_ssf.__name__]
    assert messages == [
        f"GWAS-SSF source {path}: 0 of 1 blocks read row by row, and 19 rows of the others "
        "read from their own line"
    ]


def test_the_score_tokens_are_the_ones_the_per_row_parser_reads_as_missing() -> None:
    """Text pandas reads as NaN is exactly an unpadded token the per-row parser
    reads as MISSING; padded or differently spelt text makes the column text."""
    candidates = ["", ".", "NA", "#NA", "NaN", "nan", "None", "NAN", "#na", "N/A", "null", "na",
                  "NULL", "-", "--", "?", "Nan", "none", "NONE"]
    missing = {
        token for token in candidates
        if interface.parse_imputation_score(token, declared=True).status
        is interface.ImputationScoreStatus.MISSING
    }
    assert missing == set(columnar.MISSING_SCORES)

"""GWAS-SSF's association batches are exactly its per-row associations (#319).

The per-row reader is the reference. Over every GWAS-SSF and Column Map source
the tests hold, the batches must yield the same associations in the same order,
every float bit for bit, drop the same rows, name the same faults, count the
same frequencies made absent and the same indels an `hm_code` placed; a source
the per-row reader refuses, they refuse. The Column Map reader reads through
the same batches.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import batch_fixtures as shared
import pandas as pd
import pytest
import ssf_cases
from batch_fixtures import Built
from finngen_cases import CASES as FINNGEN_CASES
from liftover_fixtures import ALT_LIFTED_ROW, PRIMARY_LIFTED_ROW
from ssf_cases import CASES, MAIN, Case, row, text
from test_column_map_reader import _SSF_FIXTURES
from test_finngen_batches import Read, _columns, _rows, _tally
from test_indel_pairs import _gwas_ssf as _pairs_ssf
from test_indel_pairs import _rows as _pair_rows
from test_infinite_effects import BETAS
from test_infinite_effects import _gwas_ssf as _infinite_ssf

import opengwasdb.readers.columnar as columnar
import opengwasdb.readers.gwas_ssf as gwas_ssf
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.interface import (
    AssociationBatch,
    ImputationScoreDeclaration,
    ImputationScoreKind,
    StreamTally,
    association_batches,
)

Make = Callable[[Path, StoredEffectScale], Any]


def _ssf(path: Path, scale: StoredEffectScale = StoredEffectScale.SD) -> GwasSsfReader:
    return GwasSsfReader(path, scale)


def _mapped(path: Path, scale: StoredEffectScale = StoredEffectScale.SD) -> ColumnMapReader:
    return ColumnMapReader(path, load_column_map("gwas-ssf"), scale)


def _per_row(make: Make, path: Path, scale: StoredEffectScale) -> Read:
    tally = StreamTally()
    try:
        associations = list(make(path, scale).stream_associations(tally))
    except Exception as refused:
        return Read(None, None, type(refused).__name__)
    return Read(_rows(associations), _tally(tally))


def _batched(make: Make, path: Path, scale: StoredEffectScale) -> Read:
    tally = StreamTally()
    try:
        batches = list(make(path, scale).stream_association_batches(tally))
    except Exception as refused:
        return Read(None, None, type(refused).__name__)
    return Read(_columns(batches), _tally(tally))


def _summary(path: Path) -> ssf_cases.Read:
    tally = StreamTally()
    try:
        associations = list(_ssf(path).stream_associations(tally))
    except Exception as refused:
        return ssf_cases.Read(0, 0, 0, 0, type(refused).__name__)
    return ssf_cases.Read(len(associations), *ssf_cases_tally(tally), None)


def ssf_cases_tally(tally: StreamTally) -> tuple[int, int, int]:
    return len(tally.faults.rows), tally.frequencies.made_absent, tally.indel_rows_kept_by_hm_code


# ── the cases hold ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_each_case_holds_what_it_is_named_for(tmp_path: Path, case: Case) -> None:
    assert case.holds(_summary(case.write(tmp_path)))


def test_the_generated_sources_are_big_enough_to_span_blocks(tmp_path: Path) -> None:
    for name in ("generated", "decided"):
        (case,) = (c for c in CASES if c.name == name)
        assert _summary(case.write(tmp_path)).associations > 3_000


# ── every source: the batches are the associations ───────────────────────────

#: Bytes a block reads: the reader's own, about one line, and one byte, so a
#: line is split across blocks and a source spans many. The two generated
#: sources read in 4 KiB blocks instead of the two smallest.
_SMALL = ((None, "default"), (64, "64B"), (1, "1B"))
_LARGE = ((None, "default"), (1 << 12, "4KiB"))
_BIG = {"generated", "decided"}
CASE_READS = [
    pytest.param(case, block, id=f"{case.name}-{label}")
    for case in CASES
    for block, label in (_LARGE if case.name in _BIG else _SMALL)
]


@pytest.mark.parametrize(("case", "block"), CASE_READS)
@pytest.mark.parametrize("make", [_ssf, _mapped], ids=["reader", "column-map"])
def test_the_batches_are_the_per_row_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: Make, case: Case, block: int | None
) -> None:
    if block is not None:
        monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", block)
    path = case.write(tmp_path)
    scale = StoredEffectScale.SD
    assert _per_row(make, path, scale) == _batched(make, path, scale)


@pytest.mark.parametrize("fixture", sorted(_SSF_FIXTURES))
@pytest.mark.parametrize("block", [None, 64, 1])
@pytest.mark.parametrize("make", [_ssf, _mapped], ids=["reader", "column-map"])
@pytest.mark.parametrize("scale", [StoredEffectScale.SD, StoredEffectScale.LOG_OR])
def test_the_batches_are_the_per_row_associations_of_every_existing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture: str, block: int | None, make: Make,
    scale: StoredEffectScale,
) -> None:
    """Every GWAS-SSF fixture the Column Map tests hold, on both scales (a
    signed z is refused on the case-control one by both paths)."""
    if block is not None:
        monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", block)
    path = _SSF_FIXTURES[fixture](tmp_path)
    per_row = _per_row(make, path, scale)
    assert per_row == _batched(make, path, scale)
    if scale is StoredEffectScale.SD:
        assert per_row.columns is not None and per_row.columns["z"]


def _other_fixtures(directory: Path) -> dict[str, Path]:
    (directory / "infinite").mkdir()
    (directory / "pairs").mkdir()
    return {
        "infinite effects": _infinite_ssf(directory / "infinite" / "i.tsv", BETAS),
        **{
            f"indel pair {hm}": _pairs_ssf(directory / "pairs" / f"{hm}.tsv", _pair_rows(True), hm)
            for hm in ("10", "11", "1")
        },
    }


@pytest.mark.parametrize("block", [None, 64, 1])
@pytest.mark.parametrize("make", [_ssf, _mapped], ids=["reader", "column-map"])
def test_the_batches_are_the_per_row_associations_of_other_modules_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: int | None, make: Make
) -> None:
    if block is not None:
        monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", block)
    for name, path in _other_fixtures(tmp_path).items():
        per_row = _per_row(make, path, StoredEffectScale.SD)
        assert per_row.columns is not None and per_row.columns["z"], name
        assert per_row == _batched(make, path, StoredEffectScale.SD), name


@pytest.mark.parametrize("case", FINNGEN_CASES, ids=[c.name for c in FINNGEN_CASES])
@pytest.mark.parametrize("block", [None, 64])
def test_a_finngen_source_through_the_column_map_reads_as_its_translation_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: Any, block: int | None
) -> None:
    """A mapped source declares its reference by convention, of every row: the
    other branch of `_declaration`."""
    if block is not None:
        monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", block)
    path = case.write(tmp_path)

    def make(path: Path, scale: StoredEffectScale) -> ColumnMapReader:
        return ColumnMapReader(path, load_column_map("finngen-r13"), scale)

    scale = StoredEffectScale.LOG_OR
    assert _per_row(make, path, scale) == _batched(make, path, scale)


@pytest.mark.parametrize("make", [_ssf, _mapped], ids=["reader", "column-map"])
def test_a_signed_z_is_refused_on_a_case_control_scale_by_both_paths(
    tmp_path: Path, make: Make
) -> None:
    (case,) = (c for c in CASES if c.name == "signed z")
    path = case.write(tmp_path)
    per_row = _per_row(make, path, StoredEffectScale.LOG_OR)
    assert per_row.refused == "CaseControlZScoreError"
    assert _batched(make, path, StoredEffectScale.LOG_OR) == per_row


def _declared_scores(directory: Path) -> Path:
    header = [*MAIN, "info"]
    scores = ["0", "0.6", "1", "", "NA", "#NA", "bogus", "inf", "-0.1", "1.01"]
    rows = [[*row(pos=str(100 + i)), score] for i, score in enumerate(scores)]
    path = directory / "scored.tsv"
    path.write_text(text(header, rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("make", [_ssf, _mapped], ids=["reader", "column-map"])
def test_a_declared_imputation_score_is_read_as_it_is_per_row(tmp_path: Path, make: Make) -> None:
    path = _declared_scores(tmp_path)
    declaration = ImputationScoreDeclaration(
        "info", ImputationScoreKind.IMPUTATION_INFO, "provider documentation: INFO"
    )

    def declared(path: Path, scale: StoredEffectScale) -> Any:
        reader = make(path, scale)
        return type(reader)(
            **{**reader.__dict__, "imputation_score_declaration": declaration}
        )

    per_row = _per_row(declared, path, StoredEffectScale.SD)
    assert per_row.columns is not None
    assert len(set(per_row.columns["imputation_status"])) >= 5
    assert _batched(declared, path, StoredEffectScale.SD) == per_row


# ── the block bound, and what the columns decide ─────────────────────────────

_BLOCKED = [
    pytest.param(name, block, id=f"{name}-{block}B")
    for name in ("beta", "odds ratio", "signed z", "SE recovered from the row", "decided")
    for block in ((1 << 12,) if name == "decided" else (64, 1))
]


@pytest.mark.parametrize(("name", "block"), _BLOCKED)
def test_a_block_holds_one_read_and_the_line_it_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, block: int
) -> None:
    """A small block size reads a source as many blocks, each at most one read
    and the line the read before it split, and each yields a batch: what bounds
    a worker's memory is the block, not the source."""
    (case,) = (c for c in CASES if c.name == name)
    path = case.write(tmp_path)
    longest = max(len(line) for line in case.body.splitlines(keepends=True)[1:])
    parsed: list[int] = []
    fmt_block = gwas_ssf._Format.block_batches

    def record(self: Any, lines: bytes, *args: Any) -> Iterator[AssociationBatch]:
        parsed.append(len(lines))
        return fmt_block(self, lines, *args)

    monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", block)
    monkeypatch.setattr(gwas_ssf._Format, "block_batches", record)
    batches = list(_ssf(path).stream_association_batches(StreamTally()))

    assert len(batches) > 1 and len(parsed) > 1
    assert max(parsed) <= block + longest


def _refuse(*args: object, **kwargs: object) -> None:
    raise AssertionError("a row the columns decide was read row by row")


def test_a_source_the_columns_decide_is_read_without_the_per_row_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every row of `decided` is parsed as columns: each effect source, an SE
    recovered from an interval and a p-value, a frequency made absent or never
    there. No row reaches the per-row parser, so the identity above is the
    columnar parse's, not the per-row reader's twice."""
    (case,) = (c for c in CASES if c.name == "decided")
    path = case.write(tmp_path)
    expected = _per_row(_ssf, path, StoredEffectScale.SD)
    assert expected.columns is not None and len(expected.columns["z"]) > 4_500
    monkeypatch.setattr(gwas_ssf, "_tabular_row", _refuse)
    monkeypatch.setattr(gwas_ssf, "_BLOCK_BYTES", 1 << 14)
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        assert _batched(_ssf, path, StoredEffectScale.SD) == expected
    assert _fallbacks(caplog) == []


def _fallbacks(caplog: pytest.LogCaptureFixture) -> list[str]:
    return shared.fallbacks(caplog, gwas_ssf)


def _extent(path: Path, row_by_row: int, blocks: int, undecided: int) -> str:
    return shared.extent("GWAS-SSF", path, row_by_row, blocks, undecided)


@pytest.mark.parametrize(
    "cell",
    ["NAN", "Nan", "+nan", "-nan", "nan ", "NaN ", "None ", "NA ", "#NA", "N/A", "null"],
)
def test_pandas_reads_a_nan_only_from_the_tokens_a_statistic_is_missing_for(cell: str) -> None:
    """A frequency the columns read as NaN is one `parse_af` reads as absent
    without counting it, so the batch path needs no cell text to tell: any
    other spelling of NaN makes pandas read the whole column as text."""
    data = f"0.1\n{cell}\n0.2\n".encode()
    frame = pd.read_csv(
        io.BytesIO(data), names=[0], na_values={0: list(columnar.MISSING_STATISTICS)},
        float_precision="round_trip", **columnar.PANDAS,
    )
    assert frame[0].dtype.kind != "f"
    assert cell not in columnar.MISSING_STATISTICS


def test_a_statistic_is_missing_where_the_per_row_reader_reads_it_missing() -> None:
    import opengwasdb.readers.tabular as tabular
    import opengwasdb.stats as stats

    assert set(columnar.MISSING_STATISTICS) == tabular._MISSING == stats._MISSING_AF


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("quoted", "it holds a quote"),
        ("non-ascii", "it holds a byte outside ASCII"),
        ("crlf", "its header holds a carriage return"),
        ("no chromosome column", "its header names no chromosome column"),
    ],
)
def test_a_source_read_row_by_row_says_where_from_and_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, reason: str
) -> None:
    (case,) = (c for c in CASES if c.name == name)
    path = case.write(tmp_path)
    offset = case.body.find(b'"') if "quote" in reason else (
        case.body.find("é".encode()) if "ASCII" in reason else 0
    )
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        list(_ssf(path).stream_association_batches(StreamTally()))
    assert _fallbacks(caplog) == [
        f"GWAS-SSF source {path}: read row by row from byte {offset} on, since {reason}",
        _extent(path, 1, 1, 0),
    ]


def test_rows_the_columns_leave_to_the_per_row_rules_are_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An indel its code declares nothing for is a fault the per-row rules name,
    so its row is read from its own line; the block is still parsed as columns."""
    rows = [row(pos=str(100 * i), effect="AT" if i % 3 == 0 else "G",
                other="A", hm="11" if i % 6 == 0 else "1") for i in range(1, 10)]
    path = tmp_path / "undeclared.tsv"
    path.write_text(text(MAIN, rows), encoding="utf-8")
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        list(_ssf(path).stream_association_batches(StreamTally()))
    assert _fallbacks(caplog) == [_extent(path, 0, 1, 2)]


def test_a_plain_source_is_read_without_a_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (case,) = (c for c in CASES if c.name == "decided")
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        list(_ssf(case.write(tmp_path)).stream_association_batches(StreamTally()))
    assert _fallbacks(caplog) == []


def test_a_declared_imputation_score_is_read_row_by_row_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _declared_scores(tmp_path)
    declaration = ImputationScoreDeclaration(
        "info", ImputationScoreKind.IMPUTATION_INFO, "provider documentation: INFO"
    )
    with caplog.at_level("WARNING", logger=gwas_ssf.__name__):
        list(
            GwasSsfReader(path, imputation_score_declaration=declaration)
            .stream_association_batches(StreamTally())
        )
    assert "since an imputation score is declared" in _fallbacks(caplog)[0]


def test_both_readers_offer_their_associations_as_batches(tmp_path: Path) -> None:
    path = CASES[0].write(tmp_path)
    for reader in (_ssf(path), _mapped(path)):
        assert association_batches(reader, StreamTally()) is not None


# ── through a Dense build's Pass 2 ───────────────────────────────────────────


def _written(name: str) -> Callable[[Path], Path]:
    (case,) = (c for c in CASES if c.name == name)
    return case.write


def _off_primary(directory: Path) -> Path:
    """A source declared hg19 whose first row the alt-target chain lifts onto an
    alt contig, which the Store does not hold (#376), and its second onto a
    primary one."""
    rows = [row(chrom=c, pos=str(p), effect=a, other=r) for c, p, r, a, _alid in
            (ALT_LIFTED_ROW, PRIMARY_LIFTED_ROW)]
    path = directory / "off-primary.tsv"
    path.write_text(text(MAIN, rows), encoding="utf-8")
    return path


#: What a Dense build reads: sources that between them meet every reason a
#: GWAS-SSF row is dropped for, and one declared hg19 whose lift puts a row off
#: the primary assembly.
_BUILDS: dict[str, tuple[Callable[[Path], Path], str]] = {
    "generated": (_written("generated"), "hg38"),
    "decided": (_written("decided"), "hg38"),
    "beta": (_written("beta"), "hg38"),
    "SE recovered": (_written("SE recovered from the row"), "hg38"),
    "layout hm_": (_written("hm_ allele columns"), "hg38"),
    "unharmonised": (_written("no harmoniser column"), "hg38"),
    "off-primary": (_off_primary, "hg19"),
}


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Built, Built]]:
    return shared.paired_builds(
        tmp_path_factory.mktemp("builds"), _BUILDS, reader=GwasSsfReader, module=gwas_ssf,
        capability=GWAS_SSF_CAPABILITY,
    )


@pytest.mark.parametrize("name", list(_BUILDS))
def test_pass2_reads_gwas_ssf_batches_as_it_reads_its_associations(
    builds: dict[str, tuple[Built, Built]], name: str
) -> None:
    shared.assert_identical(*builds[name])


def test_the_builds_meet_every_drop_reason(builds: dict[str, tuple[Built, Built]]) -> None:
    """What the builds compare includes a row dropped for each reason a
    GWAS-SSF row can be: the one-row rule's two (ADR 0054), an indel each
    layout places no reference for (ADR 0079), and a lift onto an alt contig."""
    assert shared.drop_reasons(builds) >= {
        "allele_invalid", "chromosome_missing", "se_unusable", "effect_missing", "z_nonfinite",
        "duplicate_collapsed", "duplicate_conflicting", "lift_off_primary",
        "indel_ssf_hm_code", "indel_ssf_hm_columns", "indel_ssf_unharmonised",
    }

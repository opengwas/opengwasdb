"""FinnGen's association batches are exactly its per-row associations (#318).

The per-row reader is the reference. Over every FinnGen source the tests hold
(`finngen_cases`, and the fixtures other modules write), the columnar batches
must yield the same associations in the same order, every float bit for bit,
drop the same rows, name the same faults and count the same frequencies made
absent; a source the per-row reader refuses, they refuse.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import finngen_cases
import numpy as np
import pytest
from batch_fixtures import (
    Built,
    assert_identical,
    drop_reasons,
    extent,
    fallbacks,
    paired_builds,
    read_logging,
)
from finngen_cases import CASES, GOLDEN, Case, _row
from liftover_fixtures import ALT_LIFTED_ROW, PRIMARY_LIFTED_ROW
from test_column_map_reader import _FINNGEN_FIXTURES
from test_indel_alid import _finngen_text
from test_indel_pairs import _finngen as _pairs_finngen
from test_indel_pairs import _rows as _pair_rows
from test_infinite_effects import BETAS
from test_infinite_effects import _finngen as _infinite_finngen
from test_source_reader_seam import _finngen as _seam_finngen

import opengwasdb.readers.finngen as finngen
import opengwasdb.readers.tabular as tabular
import opengwasdb.stats as stats
from opengwasdb.readers.finngen import FinnGenR13Reader
from opengwasdb.readers.interface import (
    AssociationBatch,
    ReaderAssociation,
    StreamTally,
    association_batches,
)

# ── every FinnGen source ─────────────────────────────────────────────────────


def _generated(directory: Path) -> Path:
    """Six thousand rows of every ordinary kind: both orientations, both cases,
    indels, a missing effect, an unusable SE, a frequency made absent, many
    chromosomes. Plain text, and every row decided by its columns."""
    pick = random.Random(318)
    bases = "ACGTacgt"
    lines = [finngen_cases.HEADER]
    for i in range(6_000):
        ref, alt = pick.sample(bases, 2)
        if ref.upper() == alt.upper():
            alt = "T" if ref.upper() != "T" else "C"
        if i % 7 == 0:
            alt = ref + "".join(pick.choice("ACGT") for _ in range(pick.randint(1, 5)))
        if i % 11 == 0:
            ref, alt = alt, ref
        beta = "NA" if i % 97 == 0 else repr(pick.gauss(0, 0.3))
        se = "0" if i % 89 == 0 else repr(abs(pick.gauss(0.05, 0.02)) + 1e-6)
        af = "1" if i % 79 == 0 else ("-0.5" if i % 83 == 0 else repr(pick.random()))
        chrom = pick.choice(["1", "2", "7", "22", "23", "X"])
        lines.append(finngen_cases._row(chrom, str(1_000 + 37 * i), ref, alt, beta, se, af))
    path = directory / "generated.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _from_text(name: str, text: str) -> Callable[[Path], Path]:
    def write(directory: Path) -> Path:
        path = directory / f"{name}.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    return write


def _named(fixture: Callable[[Path], Path], name: str) -> Callable[[Path], Path]:
    def write(directory: Path) -> Path:
        (directory / name).mkdir()
        return fixture(directory / name)

    return write


#: Every FinnGen source the suite writes, by name: each `finngen_cases` case,
#: and the fixtures of the modules that test the per-row reader.
SOURCES: dict[str, Callable[[Path], Path]] = {
    **{f"case {case.name}": case.write for case in CASES},
    **{f"column-map {name}": _named(write, "cm") for name, write in _FINNGEN_FIXTURES.items()},
    "infinite effects": lambda d: _infinite_finngen(d / "infinite.tsv", BETAS),
    "indel pair": lambda d: _pairs_finngen(d / "pairs.tsv", _pair_rows(True)),
    "indel alid": _from_text("indel-alid", _finngen_text()),
    "seam": _named(_seam_finngen, "seam"),
    "seam lower": _named(lambda d: _seam_finngen(d, lower=True), "seam-lower"),
    "generated": _generated,
}


# ── what each path yields ────────────────────────────────────────────────────


def _bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _strings(values: Sequence[object]) -> list[str]:
    """Each value, which must be a `str` itself (not a numpy string)."""
    assert all(type(value) is str for value in values)
    return [str(value) for value in values]


@dataclass(frozen=True)
class Read:
    """One path's answer for one source: its columns and its tally, or its refusal."""

    columns: dict[str, list[object]] | None
    tally: tuple[object, ...] | None
    refused: str | None = None


def _tally(tally: StreamTally) -> tuple[object, ...]:
    return (
        list(tally.faults.rows),
        tally.frequencies.made_absent,
        tally.indel_rows_kept_by_hm_code,
    )


def _rows(associations: list[ReaderAssociation]) -> dict[str, list[object]]:
    return {
        "chromosome": _strings([a.chromosome for a in associations]),
        "position": [a.position for a in associations],
        "ref": _strings([a.ref for a in associations]),
        "alt": _strings([a.alt for a in associations]),
        "z": [_bits(a.z) for a in associations],
        "se": [_bits(a.se) for a in associations],
        "eaf": [_bits(math.nan if a.eaf is None else a.eaf) for a in associations],
        "imputation_score": [
            _bits(math.nan if a.imputation_score.value is None else a.imputation_score.value)
            for a in associations
        ],
        "imputation_status": [a.imputation_score.status for a in associations],
    }


def _columns(batches: list[AssociationBatch]) -> dict[str, list[object]]:
    def joined(name: str) -> list[object]:
        return [value for batch in batches for value in getattr(batch, name)]

    for batch in batches:
        assert len(batch), "a batch holds at least one association"
        for name in ("z", "se", "eaf", "imputation_score"):
            assert getattr(batch, name).dtype == np.float64, name
    return {
        "chromosome": _strings(joined("chromosome")),
        "position": [int(p) for p in joined("position")],
        "ref": _strings(joined("ref")),
        "alt": _strings(joined("alt")),
        **{name: [_bits(float(v)) for v in joined(name)] for name in
           ("z", "se", "eaf", "imputation_score")},
        "imputation_status": joined("imputation_status"),
    }


def _per_row(path: Path) -> Read:
    tally = StreamTally()
    try:
        associations = list(FinnGenR13Reader(path).stream_associations(tally))
    except Exception as refused:
        return Read(None, None, type(refused).__name__)
    return Read(_rows(associations), _tally(tally))


def _batched(path: Path) -> Read:
    tally = StreamTally()
    try:
        batches = list(FinnGenR13Reader(path).stream_association_batches(tally))
    except Exception as refused:
        return Read(None, None, type(refused).__name__)
    return Read(_columns(batches), _tally(tally))


# ── the cases hold, and the per-row reader is the one it was ─────────────────


def _case_text(case: Case) -> str:
    return (case.prefix + case.text.encode() + case.tail).decode("utf-8", "replace")


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_each_case_holds_what_it_is_named_for(tmp_path: Path, case: Case) -> None:
    path = case.write(tmp_path)
    tally = StreamTally()
    if case.refuses:
        with pytest.raises(UnicodeDecodeError):
            list(FinnGenR13Reader(path).stream_associations(tally))
        return
    associations = list(FinnGenR13Reader(path).stream_associations(tally))
    assert case.holds(_case_text(case), associations, tally)


def test_the_golden_names_every_case() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden) == sorted(case.name for case in CASES)
    assert sum(len(entry.get("associations", ())) for entry in golden.values()) > 150


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_the_per_row_reader_reads_each_case_as_it_did_before_the_batches(
    tmp_path: Path, case: Case
) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[case.name]
    observed = finngen_cases.per_row(case.write(tmp_path))
    if "refused" in golden:
        assert observed["refused"].split(":")[0] == golden["refused"].split(":")[0]
        return
    assert observed == golden


# ── the batches are the associations ─────────────────────────────────────────

#: Bytes a block reads: the reader's own, about one line, and one byte, so a
#: line is split across blocks and a source spans many. The generated source is
#: read in 4 KiB blocks instead of the two smallest: about 200 blocks, each
#: parsed as columns with a line carried over, at a fraction of the cost.
_SMALL_BLOCKS = ((None, "default"), (64, "64B"), (1, "1B"))
_GENERATED_BLOCKS = ((None, "default"), (1 << 12, "4KiB"))
READS = [
    pytest.param(source, block, id=f"{source}-{label}")
    for source in SOURCES
    for block, label in (_GENERATED_BLOCKS if source == "generated" else _SMALL_BLOCKS)
]


@pytest.mark.parametrize(("source", "block"), READS)
def test_the_batches_are_the_per_row_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, block: int | None
) -> None:
    if block is not None:
        monkeypatch.setattr(finngen, "_BLOCK_BYTES", block)
    path = SOURCES[source](tmp_path)
    per_row, batched = _per_row(path), _batched(path)
    assert per_row == batched


#: Sources whose every line is plain, read at block sizes smaller than the source.
_BLOCKED = [
    pytest.param("generated", 1 << 12, id="generated-4KiB"),
    *[
        pytest.param(source, block, id=f"{source}-{block}B")
        for source in ("case alleles", "case se", "case beta", "case af", "case indels")
        for block in (64, 1)
    ],
]


@pytest.mark.parametrize(("source", "block"), _BLOCKED)
def test_a_block_holds_one_read_and_the_line_it_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, block: int
) -> None:
    """A small block size reads a source as many blocks, each at most one read
    and the line the read before it split: what bounds a worker's memory is
    the block, not the source."""
    path = SOURCES[source](tmp_path)
    with finngen._opener(path)(path, "rb") as fh:
        fh.readline()
        longest = max(len(line) for line in fh)
    parsed: list[int] = []
    parse = finngen._block_batches

    def record(lines: bytes, *args: Any) -> Iterator[AssociationBatch]:
        parsed.append(len(lines))
        return parse(lines, *args)

    monkeypatch.setattr(finngen, "_BLOCK_BYTES", block)
    monkeypatch.setattr(finngen, "_block_batches", record)
    batches = list(FinnGenR13Reader(path).stream_association_batches(StreamTally()))

    assert len(batches) > 1 and len(parsed) > 1
    assert max(parsed) <= block + longest


def test_a_plain_source_is_parsed_as_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated source's blocks never reach the per-row parser, so the
    identity above is the columnar parse's, not the per-row reader's twice."""
    path = _generated(tmp_path)
    expected = _per_row(path)
    assert expected.columns is not None and len(expected.columns["z"]) > 5_000

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a plain block was read row by row")

    monkeypatch.setattr(finngen, "_source_row", refuse)
    monkeypatch.setattr(finngen, "_BLOCK_BYTES", 1 << 14)
    assert _batched(path) == expected


def test_a_statistic_is_missing_where_the_per_row_reader_reads_it_missing() -> None:
    """The block parse's missing tokens are the per-row parsers' own."""
    assert set(finngen._MISSING_STATISTICS) == tabular._MISSING == stats._MISSING_AF


_CASES = {case.name: case for case in CASES}


def _fallbacks(caplog: pytest.LogCaptureFixture) -> list[str]:
    return fallbacks(caplog, finngen)


def _extent(path: Path, row_by_row: int, blocks: int, undecided: int) -> str:
    return extent("FinnGen", path, row_by_row, blocks, undecided)


def _read_logging(path: Path, caplog: pytest.LogCaptureFixture) -> list[str]:
    return read_logging(
        caplog, finngen, lambda: FinnGenR13Reader(path).stream_association_batches(StreamTally())
    )


@pytest.mark.parametrize(
    ("case", "byte", "reason"),
    [
        ("quoted", b'"', "it holds a quote"),
        ("non-ascii", "É".encode(), "it holds a byte outside ASCII"),
        ("crlf", None, "its header holds a carriage return"),
        ("no-pos", None, "its header names no pos column"),
        ("byte-order-mark", None, "its header names no #chrom or chrom column"),
    ],
)
def test_a_source_read_row_by_row_says_where_from_and_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str, byte: bytes | None, reason: str
) -> None:
    """The rows read the same, but about five times slower, so the source,
    the byte the per-row reader starts at and the cause are logged, and then
    how much of the source the per-row rules read. Each source here is one
    block, all of it read row by row."""
    source = _CASES[case]
    path = source.write(tmp_path)
    offset = 0 if byte is None else (source.prefix + source.text.encode()).find(byte)
    assert _read_logging(path, caplog) == [
        f"FinnGen source {path}: read row by row from byte {offset} on, since {reason}",
        _extent(path, 1, 1, 0),
    ]


def _with_a_quote(directory: Path, *, last: bool) -> tuple[Path, int]:
    """The generated source with a quoted gene in a line far past the first
    4 KiB block, or in its last line, left unterminated; and that quote's byte."""
    with gzip.open(_generated(directory), "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    at = len(lines) - 1 if last else 3_000
    lines[at] = lines[at].replace("GENE", '"GE NE"')
    text = "\n".join(lines) + ("" if last else "\n")
    path = directory / f"quoted-{'last' if last else 'later'}.tsv"
    path.write_text(text, encoding="utf-8")
    return path, text.encode().find(b'"')


@pytest.mark.parametrize("last", [False, True], ids=["later-block", "unterminated-tail"])
def test_a_fallback_names_its_true_byte_past_the_first_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, last: bool
) -> None:
    path, byte = _with_a_quote(tmp_path, last=last)
    assert byte > 4 * (1 << 12), "the quote must lie past the first blocks"
    monkeypatch.setattr(finngen, "_BLOCK_BYTES", 1 << 12)
    with caplog.at_level("WARNING", logger=finngen.__name__):
        assert _batched(path) == _per_row(path)
    first, extent = _fallbacks(caplog)
    assert first == (
        f"FinnGen source {path}: read row by row from byte {byte} on, since it holds a quote"
    )
    assert extent.startswith(f"FinnGen source {path}: ")


def test_a_source_pandas_refuses_throughout_says_every_block_was_read_row_by_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """pandas refuses a block in which no line reaches the header's last
    column. Here none does, so every block is read row by row: the first is
    logged, and so is the whole extent."""
    path = tmp_path / "short.tsv"
    lines = [finngen_cases.HEADER, *(f"1\t{100 * i}\tA\tG" for i in range(1, 40))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(finngen, "_BLOCK_BYTES", 64)
    with caplog.at_level("WARNING", logger=finngen.__name__):
        assert _batched(path) == _per_row(path)
    blocks = math.ceil((path.stat().st_size - len(lines[0]) - 1) / 64)
    first, extent = _fallbacks(caplog)
    assert blocks > 5
    assert first.startswith(
        f"FinnGen source {path}: the block at byte {len(lines[0]) + 1} is read row by row: "
        "pandas refused it"
    )
    assert extent == _extent(path, blocks, blocks, 0)


def test_rows_read_from_their_own_line_are_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A frequency read as missing leaves its row to the per-row rules, which
    read it from its own line; the block is still parsed as columns."""
    path = tmp_path / "missing-frequencies.tsv"
    rows = [finngen_cases._row(pos=str(100 * i), af="NA" if i % 3 == 0 else "0.2")
            for i in range(1, 10)]
    path.write_text("\n".join([finngen_cases.HEADER, *rows]) + "\n", encoding="utf-8")
    assert _read_logging(path, caplog) == [_extent(path, 0, 1, 3)]


def test_a_plain_source_is_read_without_a_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=finngen.__name__):
        list(FinnGenR13Reader(_generated(tmp_path)).stream_association_batches(StreamTally()))
    assert _fallbacks(caplog) == []


def test_finngen_offers_its_associations_as_batches(tmp_path: Path) -> None:
    reader = FinnGenR13Reader(CASES[0].write(tmp_path))
    assert association_batches(reader, StreamTally()) is not None


# ── through a Dense build's Pass 2 ───────────────────────────────────────────


def _off_primary(directory: Path) -> Path:
    """A source declared hg19 whose first row the alt-target chain lifts onto
    an alt contig, which the Store does not hold (#376), and its second onto a
    primary one."""
    path = directory / "off-primary.tsv"
    rows = [_row(c, str(p), r, a) for c, p, r, a, _alid in (ALT_LIFTED_ROW, PRIMARY_LIFTED_ROW)]
    path.write_text("\n".join([finngen_cases.HEADER, *rows]) + "\n", encoding="utf-8")
    return path


#: What a Dense build reads, and the assembly its manifest declares: four
#: sources that between them meet every reason a FinnGen row is dropped for,
#: and one declared hg19 whose lift puts a row off the primary assembly. The
#: other cases differ only in how a row is read, which the identity tests
#: above hold at three block sizes.
_BUILDS: dict[str, tuple[Callable[[Path], Path], str]] = {
    "generated": (_generated, "hg38"),
    "empty-cells": (_CASES["empty-cells"].write, "hg38"),
    "alleles": (_CASES["alleles"].write, "hg38"),
    "beta": (_CASES["beta"].write, "hg38"),
    "off-primary": (_off_primary, "hg19"),
}



@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Built, Built]]:
    return paired_builds(
        tmp_path_factory.mktemp("builds"), _BUILDS, reader=FinnGenR13Reader, module=finngen,
        capability=finngen.FINNGEN_R13_CAPABILITY,
    )


@pytest.mark.parametrize("name", list(_BUILDS))
def test_pass2_reads_finngen_batches_as_it_reads_its_associations(
    builds: dict[str, tuple[Built, Built]], name: str
) -> None:
    assert_identical(*builds[name])


def test_the_builds_meet_every_drop_reason(builds: dict[str, tuple[Built, Built]]) -> None:
    """What the builds compare includes a row dropped for each reason a FinnGen
    row can be, a lift onto an alt contig among them."""
    assert drop_reasons(builds) >= {
        "allele_invalid", "chromosome_missing", "se_unusable", "effect_missing", "z_nonfinite",
        "duplicate_collapsed", "duplicate_conflicting", "lift_off_primary",
    }

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
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import finngen_cases
import numpy as np
import pytest
from finngen_cases import CASES, GOLDEN, Case
from pass2_fixtures import captured_spills
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
from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.readers import registry
from opengwasdb.readers.finngen import FinnGenR13Reader
from opengwasdb.readers.interface import (
    AssociationBatch,
    ReaderAssociation,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
    StreamTally,
    association_batches,
)
from opengwasdb.readers.rows import MetricsChunk, ReaderBatch

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
    return [r.getMessage() for r in caplog.records if r.name == finngen.__name__]


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
    the byte the per-row reader starts at and the cause are logged once."""
    source = _CASES[case]
    path = source.write(tmp_path)
    offset = 0 if byte is None else (source.prefix + source.text.encode()).find(byte)
    with caplog.at_level("WARNING", logger=finngen.__name__):
        list(FinnGenR13Reader(path).stream_association_batches(StreamTally()))
    assert _fallbacks(caplog) == [
        f"FinnGen source {path}: read row by row from byte {offset} on, since {reason}"
    ]


def test_blocks_pandas_refuses_are_read_row_by_row_and_said_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """pandas refuses a block in which no line reaches the header's last
    column. Each such block is read row by row; the first is logged."""
    path = tmp_path / "short.tsv"
    lines = [finngen_cases.HEADER, *(f"1\t{100 * i}\tA\tG" for i in range(1, 40))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(finngen, "_BLOCK_BYTES", 64)
    with caplog.at_level("WARNING", logger=finngen.__name__):
        assert _batched(path) == _per_row(path)
    (message,) = _fallbacks(caplog)
    assert message.startswith(
        f"FinnGen source {path}: the block at byte {len(lines[0]) + 1} is read row by row: "
        "pandas refused it"
    )


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


@dataclass(frozen=True)
class RowsOnly:
    """The FinnGen reader with its association batches hidden: Pass 2 reads it
    one association at a time."""

    reader: FinnGenR13Reader

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        return self.reader.stream_associations(tally)

    def stream_variants(self) -> Iterator[SourceVariant]:
        return self.reader.stream_variants()

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        return self.reader.stream_batches(tally)

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        return self.reader.stream_metric_chunks()

    def extract_at_sites(
        self, alids: Iterable[str], *, region: str | None = None, regions_file: RegionsFile = None
    ) -> dict[str, SiteMetrics]:
        return self.reader.extract_at_sites(alids, region=region, regions_file=regions_file)


_ROWS_ONLY = "opengwasdb.test-finngen-rows"
#: The cases a build reads. Pass 1 refuses the rest before Pass 2 runs: a
#: position no Routing Key holds, a source that is not UTF-8, and a header
#: that names no chromosome, position or allele column; and a source with no
#: rows is no Analysis.
_REFUSED_BY_PASS_1 = (
    "huge-position", "not-utf8", "empty", "header-only", "byte-order-mark", "no-chrom",
    "no-pos", "no-ref", "no-alt",
)
#: Each built source, by name. One build each: two cases may name one site
#: with two rsids, which a two-pass build refuses (#379).
_BUILT: dict[str, Callable[[Path], Path]] = {
    **{case.name: case.write for case in CASES if case.name not in _REFUSED_BY_PASS_1},
    "generated": _generated,
}


def _build(
    directory: Path, capability: str, source: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[int, dict[str, np.ndarray]], Path]:
    directory.mkdir()
    manifest = directory / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"finngen\t{source}\tFinnGen\t1000\tlog_or\tbinary_trait\t\t{capability}\thg38\n",
        encoding="utf-8",
    )
    store = directory / "store.opengwasdb"
    with captured_spills(monkeypatch) as spills:
        build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")
    return spills, admission_record_for(store)


@pytest.mark.parametrize("name", list(_BUILT))
def test_pass2_reads_finngen_batches_as_it_reads_its_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The spill and the Admission record are the same whether Pass 2 read the
    source's batches or its associations one at a time."""
    monkeypatch.setitem(
        registry._READERS, _ROWS_ONLY,
        lambda path, scale: RowsOnly(FinnGenR13Reader(path, scale)),
    )
    monkeypatch.setattr(finngen, "_BLOCK_BYTES", 1 << 12)
    (tmp_path / "source").mkdir()
    source = _BUILT[name](tmp_path / "source")

    rows, rows_record = _build(tmp_path / "rows", _ROWS_ONLY, source, monkeypatch)
    batches, batches_record = _build(
        tmp_path / "batches", finngen.FINNGEN_R13_CAPABILITY, source, monkeypatch
    )

    assert sorted(rows) == sorted(batches) == [0]
    for array, values in rows[0].items():
        assert values.dtype == batches[0][array].dtype, array
        assert values.tobytes() == batches[0][array].tobytes(), array
    record = (rows_record / "admission.json").read_bytes()
    assert json.loads(record)["analyses"][0]["examined"] > 0
    assert record == (batches_record / "admission.json").read_bytes()
    assert _logs(rows_record) == _logs(batches_record)


def _logs(record: Path) -> dict[str, bytes]:
    """Each dropped-row log a record holds; a record that dropped nothing has none."""
    dropped = record / "dropped"
    return {log.name: log.read_bytes() for log in dropped.iterdir()} if dropped.exists() else {}


def test_the_built_sources_meet_every_drop_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the builds above compare includes a row dropped for each reason a
    FinnGen row can be."""
    reasons: set[str] = set()
    for name in ("alleles", "chromosomes-hash", "se", "beta", "generated"):
        (tmp_path / name).mkdir()
        _spills, record = _build(
            tmp_path / name / "build", finngen.FINNGEN_R13_CAPABILITY,
            _BUILT[name](tmp_path / name), monkeypatch,
        )
        dropped = json.loads((record / "admission.json").read_text())["analyses"][0]["dropped"]
        reasons |= set(dropped)
    assert reasons >= {
        "allele_invalid", "chromosome_missing", "se_unusable", "effect_missing", "z_nonfinite",
        "duplicate_collapsed", "duplicate_conflicting",
    }

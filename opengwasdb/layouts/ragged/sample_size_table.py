"""The per-association sample-size input of a BESD build (issue #302, ADR 0067).

A BESD file set carries beta and SE per (probe, SNP) pair and no sample size,
yet eQTLGen's published `NrSamples` varies per pair from 99 to 31,684. This
module reads that N from a table keyed the way the file set itself names a
pair -- the `.epi` probe ID and the `.esi` SNP ID -- so the build stores each
pair's published N and nothing it had to estimate.

The format is chosen for eQTLGen's 127 million pairs:

* A tab-separated text file, gzipped or not, with a header naming at least
  `probe_id`, `snp_id` and `n`; other columns are ignored.
* **Grouped by probe, in `.epi` order.** Each probe's rows are contiguous and
  the probes appear in the order the `.epi` lists them; within a probe, rows
  may come in any order. A probe the `.epi` does not hold may appear anywhere,
  and its rows count as unmatched.
* `n` is a whole number from 1 to `N_MAX`. A pair whose N is unknown is
  **omitted**, which the build stores as missing.

The grouping is what lets the build stream the table in lock-step with its own
probe loop, holding one probe's rows at a time -- tens of thousands, not 127
million -- in one sequential pass with no index and no sort. A table that breaks
the grouping, repeats a pair, or carries a malformed N fails the build naming
the line: read any other way, it would attach one pair's N to another, which
is a plausible wrong answer rather than an error.
"""

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from opengwasdb.encoding import N_MAX

#: The columns a sample-size table must name, in its header.
SAMPLE_SIZE_TABLE_COLUMNS = ("probe_id", "snp_id", "n")


class SampleSizeTableError(ValueError):
    """The sample-size table cannot be read as one N per (probe, SNP) pair."""


@dataclass
class _Block:
    """One probe's contiguous rows: `{snp_id: n}`, and where they start."""

    probe_id: str
    sizes: dict[str, int]
    line_no: int


@dataclass
class SampleSizeTable:
    """A sample-size table, served one probe at a time in `.epi` order.

    `sizes_for` must be called for every `.epi` probe, in `.epi` order: the
    reader keeps a one-block lookahead, and a block that belongs to a probe the
    build has already passed is a table out of order. `provenance` drains what
    is left and returns the record the Store carries.
    """

    path: Path
    probe_order: list[str]
    _rank: dict[str, int] = field(init=False)
    _blocks: Iterator[_Block] = field(init=False)
    _pending: _Block | None = field(init=False, default=None)
    _served: set[str] = field(init=False, default_factory=set)
    _sha256: str = field(init=False)
    n_rows: int = field(init=False, default=0)
    n_rows_matched: int = field(init=False, default=0)
    n_associations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._rank = {probe_id: rank for rank, probe_id in enumerate(self.probe_order)}
        self._sha256 = _sha256(self.path)
        self._blocks = _iter_blocks(self.path)

    def sizes_for(self, probe_id: str) -> dict[str, int]:
        """`{snp_id: n}` for one probe, or `{}` when the table has no rows for it."""
        rank = self._rank[probe_id]
        while True:
            block = self._peek()
            if block is None:
                return {}
            block_rank = self._rank.get(block.probe_id)
            if block_rank is None:
                self._pending = None
                continue
            if block.probe_id in self._served or block_rank < rank:
                raise self._out_of_order(block, probe_id)
            if block_rank > rank:
                return {}
            self._pending = None
            self._served.add(block.probe_id)
            return block.sizes

    def record_matches(self, n_associations: int, n_matched: int) -> None:
        """Count one probe's stored associations, and how many the table gave an N."""
        self.n_associations += n_associations
        self.n_rows_matched += n_matched

    def provenance(self) -> dict[str, Any]:
        """The `provenance.sample_size_table` block, after reading the rest of the table.

        Drained rather than abandoned: a block left over after the last probe is
        either a probe the `.epi` does not hold (counted, not an error) or one
        out of order, which must fail here rather than be silently skipped.
        """
        while (block := self._peek()) is not None:
            if block.probe_id in self._rank:
                raise self._out_of_order(block, self.probe_order[-1])
            self._pending = None
        return {
            "path": str(self.path),
            "sha256": self._sha256,
            "columns": list(SAMPLE_SIZE_TABLE_COLUMNS),
            "n_rows": self.n_rows,
            "n_rows_matched": self.n_rows_matched,
            "n_rows_unmatched": self.n_rows - self.n_rows_matched,
            "n_associations": self.n_associations,
            "n_associations_missing": self.n_associations - self.n_rows_matched,
        }

    def _peek(self) -> _Block | None:
        if self._pending is None:
            self._pending = next(self._blocks, None)
            if self._pending is not None:
                self.n_rows += len(self._pending.sizes)
        return self._pending

    def _out_of_order(self, block: _Block, current: str) -> SampleSizeTableError:
        return SampleSizeTableError(
            f"{self.path}:{block.line_no}: rows for probe {block.probe_id!r} arrive "
            f"{'a second time' if block.probe_id in self._served else 'after'} the build "
            f"reached probe {current!r}; a sample-size table must be grouped by probe, "
            "in .epi order (ADR 0067)"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path) -> IO[str]:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _column_positions(header: str, path: Path) -> tuple[int, int, int]:
    names = header.rstrip("\r\n").split("\t")
    missing = [name for name in SAMPLE_SIZE_TABLE_COLUMNS if name not in names]
    if missing:
        raise SampleSizeTableError(
            f"{path}: header lacks {missing}; a sample-size table names "
            f"{', '.join(SAMPLE_SIZE_TABLE_COLUMNS)} (ADR 0067)"
        )
    probe, snp, n = (names.index(name) for name in SAMPLE_SIZE_TABLE_COLUMNS)
    return probe, snp, n


def _parse_n(text: str, path: Path, line_no: int) -> int:
    """A whole number from 1 to `N_MAX`, or the build stops naming the line."""
    if not (text.isascii() and text.isdigit()) or not 1 <= int(text) <= N_MAX:
        raise SampleSizeTableError(
            f"{path}:{line_no}: n {text!r} is not a whole number from 1 to {N_MAX}; a "
            "pair whose sample size is unknown is omitted from the table, never "
            "written as a placeholder"
        )
    return int(text)


def _iter_rows(path: Path) -> Iterator[tuple[int, str, str, int]]:
    """`(line number, probe_id, snp_id, n)` for every data row, checked as it is read."""
    with _open_text(path) as handle:
        header = handle.readline()
        if not header:
            raise SampleSizeTableError(f"{path}: the sample-size table is empty")
        probe_col, snp_col, n_col = _column_positions(header, path)
        width = max(probe_col, snp_col, n_col) + 1
        for line_no, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < width or not fields[probe_col] or not fields[snp_col]:
                raise SampleSizeTableError(f"{path}:{line_no}: row has no probe_id, snp_id or n")
            n = _parse_n(fields[n_col], path, line_no)
            yield line_no, fields[probe_col], fields[snp_col], n


def _iter_blocks(path: Path) -> Iterator[_Block]:
    """The table's runs of rows sharing one probe, in file order."""
    block: _Block | None = None
    for line_no, probe_id, snp_id, n in _iter_rows(path):
        if block is None or probe_id != block.probe_id:
            if block is not None:
                yield block
            block = _Block(probe_id, {}, line_no)
        if snp_id in block.sizes:
            raise SampleSizeTableError(
                f"{path}:{line_no}: pair ({probe_id}, {snp_id}) appears twice; a "
                "sample-size table names one N per pair"
            )
        block.sizes[snp_id] = n
    if block is not None:
        yield block

"""What the batch tests of every tabular reader share (#318, #319): a reader with
its batches hidden, Dense builds of one source read both ways, and what is
compared between them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from liftover_fixtures import write_alt_target_chain
from pass2_fixtures import captured_spills

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.readers import interface, registry
from opengwasdb.readers.rows import MetricsChunk, ReaderBatch


@dataclass(frozen=True)
class RowsOnly:
    """A reader with its association batches hidden: Pass 2 reads it one
    association at a time."""

    reader: Any

    def __getattr__(self, name: str) -> Any:
        """What the reader says of itself (its score column, say), not its batches."""
        return getattr(self.reader, name)

    def stream_associations(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[interface.ReaderAssociation]:
        return self.reader.stream_associations(tally)

    def stream_variants(self) -> Iterator[interface.SourceVariant]:
        return self.reader.stream_variants()

    def stream_batches(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[ReaderBatch]:
        return self.reader.stream_batches(tally)

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        return self.reader.stream_metric_chunks()

    def extract_at_sites(
        self, alids: Iterable[str], **where: Any
    ) -> dict[str, interface.SiteMetrics]:
        return self.reader.extract_at_sites(alids, **where)


@dataclass(frozen=True)
class Built:
    """One Dense build: the spill Pass 2 wrote, and its Admission record."""

    spills: dict[int, dict[str, np.ndarray]]
    record: Path


def build(
    directory: Path, capability: str, source: Path, assembly: str, chain: Path,
    score: str | None = None,
) -> Built:
    directory.mkdir()
    manifest = directory / "manifest.tsv"
    header = (
        "trait_id\ttrait_name\tfile_path\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly"
    )
    cells = f"trait\tTrait\t{source}\t1000\tlog_or\tbinary_trait\t\t{capability}\t{assembly}"
    if score is not None:
        header += (
            "\tinfo_score_threshold\timputation_score_column\timputation_score_kind"
            "\timputation_score_provenance"
        )
        cells += f"\t0.5\t{score}\timputation_info\tProvider Table 2 INFO"
    manifest.write_text(header + "\n" + cells + "\n", encoding="utf-8")
    store = directory / "store.opengwasdb"
    lift = {"chain_file": chain, "liftover_failure_threshold": 0.0} if assembly == "hg19" else {}
    # Each build's own context: a capture left in place would chain onto the
    # next build's, which would then overwrite this one's arrays.
    with pytest.MonkeyPatch.context() as patch, captured_spills(patch) as spills:
        build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r", **lift)
    return Built(spills, admission_record_for(store))


def _refuse_rows(self: object, tally: object = None) -> None:
    raise AssertionError("Pass 2 read a reader that offers batches one association at a time")


def paired_builds(
    root: Path,
    sources: dict[str, tuple[Any, ...]],
    *,
    reader: type,
    module: ModuleType,
    capability: str,
) -> dict[str, tuple[Built, Built]]:
    """Each source built twice: once read row by row, once as batches by a
    reader that refuses to be read row by row. ``sources`` give each source's
    writer, the assembly its manifest declares and, optionally, the column it
    declares an imputation score in.

    The registry builds a fresh reader when a score is declared, so the wrapper
    that hides the batches is put where the registry makes its readers, under
    both names. Each build must have read the way it is named for: the rows
    build's reader class counts the sources it streamed as associations, and
    none as batches.
    """
    chain = write_alt_target_chain(root / "alt-target.chain")
    built: dict[str, tuple[Built, Built]] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "_BLOCK_BYTES", 1 << 12)
        for name, (write, assembly, *score) in sources.items():
            (root / name).mkdir()
            source = write(root / name)
            reads = {"rows": 0, "batches": 0}
            by_rows_method, by_batches_method = _counting(reader, reads)
            with pytest.MonkeyPatch.context() as hidden:
                hidden.setattr(reader, "stream_associations", by_rows_method)
                hidden.setattr(reader, "stream_association_batches", by_batches_method)

                def rows_only(path: Any, scale: Any, **declared: Any) -> RowsOnly:
                    return RowsOnly(reader(path, scale, **declared))

                hidden.setitem(registry._READERS, capability, rows_only)
                if getattr(registry, reader.__name__, None) is reader:
                    hidden.setattr(registry, reader.__name__, rows_only)
                by_rows = build(root / name / "rows", capability, source, assembly, chain, *score)
            assert reads == {"rows": 1, "batches": 0}, (name, reads)
            with pytest.MonkeyPatch.context() as refuse:
                refuse.setattr(reader, "stream_associations", _refuse_rows)
                refuse.setattr(reader, "stream_association_batches", by_batches_method)
                by_batches = build(
                    root / name / "batches", capability, source, assembly, chain, *score
                )
            assert reads == {"rows": 1, "batches": 1}, (name, reads)
            built[name] = (by_rows, by_batches)
    return built


def _counting(reader: type, reads: dict[str, int]) -> tuple[Any, Any]:
    """``reader``'s two association streams, each counting the sources it read."""
    rows, batches = reader.stream_associations, reader.stream_association_batches

    def by_rows(self: Any, tally: Any = None) -> Any:
        reads["rows"] += 1
        return rows(self, tally)

    def by_batches(self: Any, tally: Any = None) -> Any:
        reads["batches"] += 1
        return batches(self, tally)

    return by_rows, by_batches


def logs(record: Path) -> dict[str, bytes]:
    """Each dropped-row log a record holds; a record that dropped nothing has none."""
    dropped = record / "dropped"
    return {log.name: log.read_bytes() for log in dropped.iterdir()} if dropped.exists() else {}


def assert_identical(rows: Built, batches: Built) -> None:
    """The spill and the Admission record are the same whether Pass 2 read the
    source's batches or its associations one at a time."""
    assert sorted(rows.spills) == sorted(batches.spills) == [0]
    assert len(rows.spills[0]["rows"]) > 0
    for array, values in rows.spills[0].items():
        assert values.dtype == batches.spills[0][array].dtype, array
        assert values.tobytes() == batches.spills[0][array].tobytes(), array
    record = (rows.record / "admission.json").read_bytes()
    assert json.loads(record)["analyses"][0]["examined"] > 0
    assert record == (batches.record / "admission.json").read_bytes()
    assert logs(rows.record) == logs(batches.record)


def drop_reasons(builds: dict[str, tuple[Built, Built]]) -> set[str]:
    """Every reason a row was dropped, over all the builds' Admission records."""
    reasons: set[str] = set()
    for rows, _batches in builds.values():
        analysis = json.loads((rows.record / "admission.json").read_text())["analyses"][0]
        reasons |= set(analysis["dropped"])
    return reasons


def fallbacks(caplog: pytest.LogCaptureFixture, module: ModuleType) -> list[str]:
    """What a reader logged about reading a source row by row."""
    return [r.getMessage() for r in caplog.records if r.name == module.__name__]


def extent(label: str, path: Path, row_by_row: int, blocks: int, undecided: int) -> str:
    """The line a source's read ends with when the per-row rules read any of it."""
    return (
        f"{label} source {path}: {row_by_row} of {blocks} blocks read row by row, and "
        f"{undecided} rows of the others read from their own line"
    )


def read_logging(
    caplog: pytest.LogCaptureFixture, module: ModuleType, batches: Callable[[], Any]
) -> list[str]:
    """What ``batches()`` logs, at warning level, as it is read."""
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        list(batches())
    return fallbacks(caplog, module)


def shadow_gwas_ssf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have every `GwasSsfReader.stream_associations` that follows also read its
    source as batches, and refuse a difference in the associations (every float
    bit for bit), the tally or the refusal.

    The identity tests name their sources; this covers the sources a module's own
    tests write and read, whatever they are, so a new one cannot be missed.
    """
    from opengwasdb.readers.gwas_ssf import GwasSsfReader

    rows_of = GwasSsfReader.stream_associations
    shadowing = [False]

    def shadowed(self: Any, tally: interface.StreamTally | None = None) -> Any:
        if shadowing[0]:
            yield from rows_of(self, tally)
            return
        from test_finngen_batches import Read, _columns, _rows, _tally

        read = tally if tally is not None else interface.StreamTally()
        refused: Exception | None = None
        try:
            associations = list(rows_of(self, read))
        except Exception as error:
            refused, associations = error, []
        shadowing[0] = True
        batch_tally = interface.StreamTally()
        try:
            batches = list(self.stream_association_batches(batch_tally))
            other = None if refused else Read(_columns(batches), _tally(batch_tally))
            batch_refused: Exception | None = None
        except Exception as error:
            other, batch_refused = None, error
        finally:
            shadowing[0] = False
        expected = None if refused else Read(_rows(associations), _tally(read))
        assert type(refused) is type(batch_refused), (self.path, refused, batch_refused)
        assert expected == other, f"batches differ from the associations of {self.path}"
        if refused is not None:
            raise refused
        yield from associations

    monkeypatch.setattr(GwasSsfReader, "stream_associations", shadowed)

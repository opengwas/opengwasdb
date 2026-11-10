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


def build(directory: Path, capability: str, source: Path, assembly: str, chain: Path) -> Built:
    directory.mkdir()
    manifest = directory / "manifest.tsv"
    manifest.write_text(
        "trait_id\ttrait_name\tfile_path\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"trait\tTrait\t{source}\t1000\tlog_or\tbinary_trait\t\t{capability}\t{assembly}\n",
        encoding="utf-8",
    )
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
    sources: dict[str, tuple[Callable[[Path], Path], str]],
    *,
    reader: type,
    module: ModuleType,
    capability: str,
) -> dict[str, tuple[Built, Built]]:
    """Each source built twice: once read row by row, once as batches by a
    reader that refuses to be read row by row. ``sources`` give each source's
    writer and the assembly its manifest declares."""
    chain = write_alt_target_chain(root / "alt-target.chain")
    rows_only = "opengwasdb.test-rows-only"
    built: dict[str, tuple[Built, Built]] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            registry._READERS, rows_only, lambda path, scale: RowsOnly(reader(path, scale))
        )
        patch.setattr(module, "_BLOCK_BYTES", 1 << 12)
        for name, (write, assembly) in sources.items():
            (root / name).mkdir()
            source = write(root / name)
            by_rows = build(root / name / "rows", rows_only, source, assembly, chain)
            with pytest.MonkeyPatch.context() as refuse:
                refuse.setattr(reader, "stream_associations", _refuse_rows)
                by_batches = build(root / name / "batches", capability, source, assembly, chain)
            built[name] = (by_rows, by_batches)
    return built


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

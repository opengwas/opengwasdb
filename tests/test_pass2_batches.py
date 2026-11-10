"""Pass 2 reads a Source Reader's association batches when it offers them (#318).

A reader may offer `stream_association_batches`, its associations as columns.
Pass 2 then routes those columns, and reads any other reader one association
at a time, as before. Whichever it reads, an Analysis's routed column and its
Admission record are the same.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from pass2_fixtures import captured_spills

from opengwasdb.admission import AnalysisRules, admission_record_for
from opengwasdb.admission.gate import counting_gate
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import registry
from opengwasdb.readers.fake import FakeReader
from opengwasdb.readers.interface import (
    AssociationBatch,
    ImputationScoreStatus,
    ReaderAssociation,
    StreamTally,
    association_batches,
)
from opengwasdb.variants.routing import RoutedColumn, RouteTable

SD = StoredEffectScale.SD


def _association(
    position: int, ref: str, alt: str, z: float, se: float, eaf: float | None = None
) -> ReaderAssociation:
    return ReaderAssociation("1", position, ref, alt, z, se, SD, eaf)


#: One Analysis's associations, meeting every case routing and Admission
#: decide: placed rows, a row off the axis, two agreeing rows (collapsed),
#: two disagreeing ones (dropped), a lower-case spelling, an unusable SE, an
#: effect the source does not report, and a frequency it does not report.
ASSOCIATIONS = [
    _association(100, "A", "G", 2.0, 0.5, 0.2),
    _association(200, "C", "T", -1.5, 0.25, 0.4),
    _association(250, "A", "C", 0.5, 0.1, 0.3),  # on no axis
    _association(300, "G", "A", 1.0, 0.2, 0.6),
    _association(300, "G", "A", 1.0, 0.2, 0.6),  # agrees: collapsed
    _association(400, "T", "G", 3.0, 0.1, 0.5),
    _association(400, "T", "G", -3.0, 0.1, 0.5),  # disagrees: both dropped
    _association(500, "g", "c", 0.7, 0.3),  # lower case, no frequency
    _association(600, "A", "T", 1.1, math.nan, 0.1),  # se unusable
    _association(700, "C", "A", math.nan, 0.4, 0.2),  # effect missing
    _association(800, "AT", "A", -0.25, 0.05, 0.05),  # a deletion
]

#: The sites the table places: every association's but the one at 250.
SITES = {
    ("1", 100, "A", "G"): "1:100:A:G",
    ("1", 200, "C", "T"): "1:200:C:T",
    ("1", 300, "G", "A"): "1:300:A:G",
    ("1", 400, "T", "G"): "1:400:G:T",
    ("1", 500, "G", "C"): "1:500:C:G",
    ("1", 600, "A", "T"): "1:600:A:T",
    ("1", 700, "C", "A"): "1:700:A:C",
    ("1", 800, "AT", "A"): "1:800:AT:A",
}


def _table() -> RouteTable:
    alids = sorted(set(SITES.values()))
    return RouteTable.build(SITES, (), [alids], keep_unmatched=True)


def _batch(rows: Sequence[ReaderAssociation]) -> AssociationBatch:
    """Built here, not by the code under test, so the two cannot share a defect."""
    return AssociationBatch(
        chromosome=np.array([a.chromosome for a in rows], dtype=object),
        position=np.array([a.position for a in rows], dtype=np.int64),
        ref=np.array([a.ref for a in rows], dtype=object),
        alt=np.array([a.alt for a in rows], dtype=object),
        z=np.array([a.z for a in rows], dtype=np.float64),
        se=np.array([a.se for a in rows], dtype=np.float64),
        eaf=np.array([math.nan if a.eaf is None else a.eaf for a in rows], dtype=np.float64),
        imputation_score=np.full(len(rows), math.nan),
        imputation_status=np.full(len(rows), ImputationScoreStatus.UNDECLARED, dtype=object),
    )


class BatchedFake(FakeReader):
    """A fake that offers its associations in batches of ``batch_rows``, and
    refuses to be read one association at a time."""

    def __init__(self, associations: Sequence[ReaderAssociation], batch_rows: int) -> None:
        super().__init__(associations)
        self._rows = list(associations)
        self._batch_rows = batch_rows

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        raise AssertionError("a reader that offers batches was read one association at a time")

    def stream_association_batches(
        self, tally: StreamTally | None = None
    ) -> Iterator[AssociationBatch]:
        for start in range(0, len(self._rows), self._batch_rows):
            yield _batch(self._rows[start : start + self._batch_rows])


def _bits(values: np.ndarray) -> list[int]:
    return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()


def _routed(column: RoutedColumn) -> dict[str, object]:
    """Everything a routed column says, floats by their bits."""
    placed, unmatched = column.placed[0], column.unmatched
    return {
        "index": placed.index.tolist(),
        "z": _bits(placed.z), "se": _bits(placed.se), "eaf": _bits(placed.eaf),
        "conflicting": placed.conflicting.tolist(),
        "dropped": placed.dropped.tolist(),
        "collapsed": placed.collapsed,
        "unmatched": list(unmatched.keys),
        "unmatched_z": _bits(unmatched.z),
        "unmatched_conflicting": list(unmatched.conflicting),
        "n_unmatched": column.n_unmatched,
    }


def _per_row() -> tuple[dict[str, object], object, list[object]]:
    gate = counting_gate(AnalysisRules("a"))
    column = _table().route_column(iter(ASSOCIATIONS), admit=gate.admit)
    return _routed(column), gate.counts, gate.dropped


def test_the_associations_meet_every_case() -> None:
    routed, counts, _dropped = _per_row()
    assert routed["collapsed"] == 1 and len(routed["conflicting"]) == 1
    assert routed["unmatched"] == ["1:250:A:C"]
    assert {reason.value for reason in counts.dropped} >= {"se_unusable", "effect_missing"}
    assert any(a.eaf is None for a in ASSOCIATIONS)
    assert any(a.ref.islower() for a in ASSOCIATIONS)


@pytest.mark.parametrize("batch_rows", [1, 3, len(ASSOCIATIONS)])
def test_a_reader_offering_batches_is_routed_from_them(batch_rows: int) -> None:
    """However the reader blocks its rows, the column and the Admission are the
    ones its associations give one at a time."""
    reader = BatchedFake(ASSOCIATIONS, batch_rows)
    gate = counting_gate(AnalysisRules("a"))
    column = _table().route_reader(reader, StreamTally(), admit=gate.admit)
    assert (_routed(column), gate.counts, gate.dropped) == _per_row()


def test_a_reader_without_batches_is_routed_one_association_at_a_time() -> None:
    reader = FakeReader(ASSOCIATIONS)
    assert association_batches(reader, StreamTally()) is None
    gate = counting_gate(AnalysisRules("a"))
    column = _table().route_reader(reader, StreamTally(), admit=gate.admit)
    assert (_routed(column), gate.counts, gate.dropped) == _per_row()


def test_a_batch_whose_columns_disagree_in_length_is_refused() -> None:
    batch = _batch(ASSOCIATIONS[:3])
    with pytest.raises(ValueError, match="every column must hold the batch's rows"):
        AssociationBatch(
            chromosome=batch.chromosome, position=batch.position[:2], ref=batch.ref,
            alt=batch.alt, z=batch.z, se=batch.se, eaf=batch.eaf,
            imputation_score=batch.imputation_score, imputation_status=batch.imputation_status,
        )


# ── through a Dense build's Pass 2 ───────────────────────────────────────────

_ROWS = "opengwasdb.test-rows"
_BATCHES = "opengwasdb.test-batches"


def _manifest(directory: Path, capability: str) -> Path:
    manifest = directory / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"one\tunused-one\tOne\t1000\tsd\tdeclared_standardised\t\t{capability}\thg38\n"
        f"two\tunused-two\tTwo\t1000\tsd\tdeclared_standardised\t\t{capability}\thg38\n",
        encoding="utf-8",
    )
    return manifest


def _build(
    directory: Path, capability: str
) -> tuple[dict[int, dict[str, np.ndarray]], Path]:
    directory.mkdir()
    # Each build's own context: a capture left in place would chain onto the
    # next build's, which would then overwrite this one's arrays.
    with pytest.MonkeyPatch.context() as patch, captured_spills(patch) as spills:
        build_dense_from_vcf_manifest(
            _manifest(directory, capability), directory / "store.opengwasdb",
            store_id="s", release_id="r",
        )
    return spills, admission_record_for(directory / "store.opengwasdb")


def test_pass2_routes_a_readers_batches_as_it_routes_its_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batched reader cannot be read row by row, so its build consumed its
    batches; its spills and its Admission record are the per-row build's."""
    monkeypatch.setitem(registry._READERS, _ROWS, lambda path, scale: FakeReader(ASSOCIATIONS))
    monkeypatch.setitem(
        registry._READERS, _BATCHES, lambda path, scale: BatchedFake(ASSOCIATIONS, 4)
    )
    rows, rows_record = _build(tmp_path / "rows", _ROWS)
    batches, batches_record = _build(tmp_path / "batches", _BATCHES)

    assert sorted(rows) == [0, 1] and all(len(spill["rows"]) for spill in rows.values())
    for column, spill in rows.items():
        for name, values in spill.items():
            assert values.dtype == batches[column][name].dtype
            assert values.tobytes() == batches[column][name].tobytes(), (column, name)
    record = (rows_record / "admission.json").read_bytes()
    assert b"duplicate_conflicting" in record and b"se_unusable" in record
    assert record == (batches_record / "admission.json").read_bytes()
    for log in sorted((rows_record / "dropped").iterdir()):
        assert log.read_bytes() == (batches_record / "dropped" / log.name).read_bytes()

"""The Variant Axis opens the ALID index its release's `format_version` carries (#467 review).

One key chooses the index: the version (`alid_index_kind`), never the files present.
A 0.2.3 release with a stray, wrong sidecar beside its fixed-width pair is read from the
pair, and `envelope.closed` reports the stray files; a 0.3.0 release missing its sidecar
is refused by name, never read by scanning the table.
"""

from __future__ import annotations

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3

from opengwasdb.query import query_store
from opengwasdb.store_reader.alid_index import FixedWidthAlidIndex, MissingAlidIndex
from opengwasdb.store_reader.alid_sidecar import ALID_SIDECAR_FILENAMES, write_alid_sidecar
from opengwasdb.validation import validate_store


def _rows_by_alid(store) -> dict[str, int]:
    with query_store(store) as q:
        return {row["alid"]: index for index, row in q.variants_table().items()}


def test_a_stray_sidecar_on_a_0_2_3_release_is_not_read(dense_store_path):
    as_0_2_3(dense_store_path)
    rows = _rows_by_alid(dense_store_path)
    alids = [alid for alid, _ in sorted(rows.items(), key=lambda item: item[1])]
    assert len(alids) >= 2, "fixture: two rows to swap"
    # A sidecar naming the first two rows by each other's ALID: read, it would answer
    # each lookup with the other's row.
    write_alid_sidecar(dense_store_path, [alids[1], alids[0], *alids[2:]])

    with query_store(dense_store_path) as q:
        assert isinstance(q._variant_axis.alid_index, FixedWidthAlidIndex)
        for alid in alids[:2]:
            assert q._variant_axis.index_of(alid) == rows[alid]
    result = validate_store(dense_store_path)
    assert result.failed() == {"envelope.closed"}, result.errors
    stray = [e for e in result.errors if "unexpected store entry" in e]
    assert sorted(e.split("'")[1] for e in stray) == sorted(ALID_SIDECAR_FILENAMES)


@pytest.mark.parametrize("missing", ["variant_alid_keys.npy", "all"])
def test_a_0_3_0_release_missing_its_sidecar_is_refused(dense_store_path, missing):
    names = ALID_SIDECAR_FILENAMES if missing == "all" else (missing,)
    for name in names:
        (dense_store_path / name).unlink()
    refused = FileNotFoundError if missing != "all" else MissingAlidIndex
    with pytest.raises(refused, match="ALID sidecar is incomplete|carries an ALID index"):
        with query_store(dense_store_path) as q:
            q.phewas("1:1:A:C")


def test_the_axis_opens_only_the_kind_its_version_names(dense_store_path):
    with query_store(dense_store_path) as q:
        index = q._variant_axis.alid_index
        assert type(index).__name__ == "AlidSidecar"
        assert np.array_equal(index.unindexed_rows(3), np.empty(0, dtype=np.int64))

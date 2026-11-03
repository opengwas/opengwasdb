"""The value rules judge an association sequence over one windowed read (issue #55, ADR 0074).

A Ragged release's sequence and a Hybrid's Overflow are read a window of
consecutive cells at a time: `z`, `se`, `eaf` and `variant_index` once, SE
decoded from the window's own frequency read. Before, the Ragged checks decoded
the frequencies twice -- once to predict SE and once to range-check them -- and
an SE that could not be decoded stopped every check after it, the side table that
explains why included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import residual_stores
import zarr
from store_reads import element_reads, times_read
from test_ragged_residual_completion import build_residual_ssf_source

from opengwasdb.encoding import SE_EXCEPTION, SeExceptionTable
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.validation.scans import analyses_of

#: The rules that fold over a sequence's windows.
SEQUENCE_RULES = {
    "csr.segment_order",
    "top_hits.held",
    "top_hits.z",
    "top_hits.se",
    "top_hits.imputed",
    "top_hits.eaf",
    "top_hits.complete",
    "associations.decodable",
    "associations.se_infinite",
    "associations.se_negative",
    "associations.markers_paired",
    "associations.markers_canonical",
    "associations.eaf_range",
    "side_tables.z_overflow",
    "side_tables.eaf_exceptions",
    "side_tables.se_exceptions",
}

#: An Overflow's own: where its cells sit in the shared variant table.
OVERFLOW_RULES = {"csr.shared_rows", "hybrid.partition"}


@pytest.fixture(scope="module")
def residual_ragged(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An observed Ragged release with residual `se` and `eaf` planes."""
    store, _se = build_residual_ssf_source(tmp_path_factory.mktemp("residual_ragged"))
    return store


def test_the_value_rules_read_a_sequence_once(
    residual_ragged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = open_store(residual_ragged)
    group = opened.arrays(mode="r")["ragged"]
    planes = ("z", "se", "eaf", "variant_index", "eaf_baseline")
    assert opened.manifest.encoding.se.is_residual, "residual se is what shares the eaf read"
    assert opened.manifest.encoding.eaf.is_residual
    assert all(name in group for name in planes), "the fixture must carry every plane"

    with element_reads(monkeypatch) as counts:
        result = validate_store(residual_ragged, rules=SEQUENCE_RULES)

    assert result.ok, result.errors
    assert not result.skipped, result.skipped
    assert {name: times_read(counts, group[name]) for name in planes} == dict.fromkeys(
        planes, 1.0
    )


def test_the_value_rules_read_an_overflow_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = residual_stores.hybrid_completed(tmp_path)
    opened = open_store(store)
    overflow = opened.arrays(mode="r")["ragged"]
    planes = ("z", "se", "eaf", "variant_index")
    assert opened.manifest.encoding.se.is_residual, "an Overflow's residual se reads its eaf"
    assert all(name in overflow for name in planes), "the fixture must carry every plane"
    assert int(overflow["offsets"][-1]) > 0, "the Overflow must hold associations"

    with element_reads(monkeypatch) as counts:
        result = validate_store(store, rules=SEQUENCE_RULES | OVERFLOW_RULES)

    assert result.ok, result.errors
    assert {name: times_read(counts, overflow[name]) for name in planes} == dict.fromkeys(
        planes, 1.0
    )


def test_an_undecodable_se_does_not_hide_the_table_that_explains_it(tmp_path: Path) -> None:
    """An exception code with no table entry: the decode fails, and the table says why.

    The rules that read decoded SE are skipped behind the decode; the rules that
    read only codes, or other planes, still see every window.
    """
    store, _se = build_residual_ssf_source(tmp_path)
    assert validate_store(store).ok, "the planted code must be the only defect"
    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+")
    held = set(SeExceptionTable.read(group).index.tolist())
    orphan = next(p for p in range(int(group["offsets"][-1])) if p not in held)
    assert int(group["se"][orphan]) != SE_EXCEPTION, "the cell must not already be marked"
    group["se"][orphan] = SE_EXCEPTION

    result = validate_store(store)

    assert result.failed() == {"associations.decodable", "side_tables.se_exceptions"}
    (decode,) = result.of("associations.decodable")
    assert decode.message.startswith(
        "data.zarr/ragged/se cannot be decoded under the declared plan: "
    )
    (table,) = result.of("side_tables.se_exceptions")
    assert table.message == (
        "data.zarr/ragged/se: 1 se exception cell(s) have no entry in the se exception "
        f"table (first at flat position {orphan})"
    )
    assert result.skipped["associations.se_negative"] == ("associations.decodable",)
    assert "side_tables.z_overflow" not in result.skipped
    assert "csr.segment_order" not in result.skipped


def test_a_tier_missing_an_association_above_its_cutoff_is_incomplete(tmp_path: Path) -> None:
    """A tier holds every association that clears its cutoff, as a grid's does.

    Raising the tier's threshold lowers its cutoff below cells it does not list.
    A sampled check of the cells a tier does list cannot see that.
    """
    store, _se = build_residual_ssf_source(tmp_path)
    tier = open_store(store).arrays(mode="a")["top_hits"]["p_5e_08"]
    assert tier.attrs["threshold"] == 5e-08, "the fixture's tier must record its threshold"
    tier.attrs["threshold"] = 0.5

    result = validate_store(store)

    assert result.failed() == {"top_hits.complete"}
    assert result.errors == ["top-hit index p_5e_08 does not match stored z values"]


def test_a_tier_cell_the_sequence_does_not_hold_is_reported(tmp_path: Path) -> None:
    """Every tier cell is an association the sequence holds; one it lacks is named."""
    store, _se = build_residual_ssf_source(tmp_path)
    data = open_store(store).arrays(mode="a")
    tier = data["top_hits"]["p_5e_08"]
    rows = np.asarray(tier["variant_index"][:])
    cols = np.asarray(tier["analysis_index"][:])
    i, absent = _an_absent_variant_between_neighbours(data["ragged"], rows, cols)
    rows[i] = absent
    tier["variant_index"][:] = rows

    result = validate_store(store)

    assert result.failed() == {"top_hits.held"}
    assert result.errors == ["top-hit index p_5e_08 references missing association"]


def _an_absent_variant_between_neighbours(
    sequence: zarr.Group, rows: np.ndarray, cols: np.ndarray
) -> tuple[int, int]:
    """A tier cell, and a variant its Analysis does not hold that keeps the tier ordered."""
    offsets = np.asarray(sequence["offsets"][:])
    for i in range(1, len(rows) - 1):
        if cols[i - 1] != cols[i] or cols[i + 1] != cols[i]:
            continue
        a = int(cols[i])
        held = set(np.asarray(sequence["variant_index"][offsets[a] : offsets[a + 1]]).tolist())
        for v in range(int(rows[i - 1]) + 1, int(rows[i + 1])):
            if v not in held:
                return i, v
    raise AssertionError("the fixture's tier must leave a gap its Analysis does not hold")


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0, 7, [0, 0, 0, 2, 2, 2, 2]),
        (2, 5, [0, 2, 2]),
        (3, 3, []),
        (6, 7, [2]),
    ],
)
def test_each_window_cell_knows_its_analysis(lo: int, hi: int, expected: list[int]) -> None:
    """Analysis 1's segment is empty, so no cell is ever said to be its."""
    offsets = np.array([0, 3, 3, 7], dtype=np.int64)
    assert analyses_of(offsets, lo, hi).tolist() == expected

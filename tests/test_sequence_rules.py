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
from test_ragged_residual_completion import RaggedResidualScenario, build_residual_ssf_source

from opengwasdb.encoding import SE_EXCEPTION, Z_MISSING, SeExceptionTable
from opengwasdb.model.analyses import AnalysesTable, read_analyses, write_analyses
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


@pytest.fixture
def fresh_ragged(residual_ragged: Path, tmp_path: Path) -> Path:
    """A private copy of `residual_ragged`, for a test that damages it."""
    return residual_stores.copied(residual_ragged, tmp_path)


@pytest.fixture(scope="module")
def completed_scenario(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    """A Reference-Completed Ragged release with residual planes and imputed cells."""
    scenario = RaggedResidualScenario(tmp_path_factory)
    assert scenario.result.n_imputed > 0, "the fixture must hold imputed cells"
    return scenario


@pytest.fixture
def completed_ragged(completed_scenario: RaggedResidualScenario, tmp_path: Path) -> Path:
    """A private copy of the completed release, for a test that damages it."""
    return residual_stores.copied(completed_scenario.completed, tmp_path)


@pytest.fixture(scope="module")
def completed_hybrid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Reference-Completed Hybrid with residual SE in both components."""
    return residual_stores.hybrid_completed(tmp_path_factory.mktemp("completed_hybrid"))


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


#: The rules about a completed sequence's mask, over the same windows.
COMPLETION_RULES = {"completion.mask", "completion.mask_binary", "completion.imputed_present"}


def test_a_completed_sequence_reads_its_mask_with_its_planes_once(
    completed_scenario: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mask is read once, with the planes, for the frequency substitution and the
    completion rules alike; SE is decoded once, not again to check the mask."""
    store = completed_scenario.completed
    group = open_store(store).arrays(mode="r")["ragged"]
    planes = ("z", "se", "eaf", "imputed", "variant_index", "eaf_baseline", "eaf_reference")
    assert all(name in group for name in planes), "the fixture must carry every plane"

    with element_reads(monkeypatch) as counts:
        result = validate_store(store, rules=SEQUENCE_RULES | COMPLETION_RULES)

    assert result.ok, result.errors
    assert not result.skipped, result.skipped
    assert {name: times_read(counts, group[name]) for name in planes} == dict.fromkeys(
        planes, 1.0
    )


def test_a_sequence_mask_flagging_a_missing_cell_fails_the_completion_rule(
    completed_ragged: Path,
) -> None:
    """A mask that flags a missing association is the completion rule's, on a sequence
    as on a grid, in the words a sequence has always used."""
    store = completed_ragged
    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+")
    z_missing = np.asarray(group["z"][:]) == Z_MISSING
    imputed = np.asarray(group["imputed"][:])
    candidates = np.flatnonzero(z_missing & (imputed == 0))
    assert len(candidates), "the fixture must hold a missing association"
    cell = int(candidates[0])
    group["imputed"][cell] = 1
    _flag_in_the_variant_index(group, cell)

    result = validate_store(store)

    assert result.failed() == {"completion.imputed_present"}, result.errors
    assert result.errors == [
        "imputed=1 rows have missing z-scores",
        "imputed=1 rows have NaN se values",
        "missing z rows have imputed=1 (inconsistent)",
    ]


def test_a_sequence_whose_metadata_denies_its_imputed_cells_fails(
    completed_ragged: Path,
) -> None:
    """Each Analysis holds imputed cells exactly when it says it does, on a sequence too.

    That disagreement is the shape the ancestry-match filter's defect took on a grid
    (ADR 0028); nothing held a Ragged release to it.
    """
    store = completed_ragged
    imputed = np.asarray(zarr.open_group(str(store / "data.zarr" / "ragged"))["imputed"][:])
    table = read_analyses(store / "analyses.tsv")
    row = next(r for r in table.rows if int(r["completion_n_imputed_total"] or 0))
    assert imputed.any(), "the fixture must hold imputed cells"
    row["completion_n_imputed_total"] = "0"
    write_analyses(store / "analyses.tsv", table)

    result = validate_store(store)

    assert result.failed() == {"completion.declaration"}, result.errors
    (finding,) = result.of("completion.declaration")
    assert finding.message.startswith(
        f"analysis {row['analysis_id']!r} has "
    ) and finding.message.endswith(
        " imputed=1 cell(s) in data.zarr/ragged/imputed but declares "
        "completion_n_imputed_total='0'; the release holds imputed cells its metadata "
        "does not account for"
    )


def test_a_completed_analysis_whose_count_is_not_an_integer_fails(
    completed_ragged: Path,
) -> None:
    """A count the declaration rule cannot read is reported, not passed over.

    An Analysis completed against an ancestry is the one the declaration is held to
    the mask for, so its count is the one that must be legible.
    """
    store = completed_ragged
    table = read_analyses(store / "analyses.tsv")
    row = next(r for r in table.rows if r["completed_against"])
    assert int(row["completion_n_imputed_total"] or 0), "the row must declare imputed cells"
    row["completion_n_imputed_total"] = "not-an-integer"
    write_analyses(store / "analyses.tsv", table)

    result = validate_store(store)

    assert result.failed() == {"analyses.completion_declared"}, result.errors
    assert result.errors == [
        f"analyses.tsv row {row['analysis_id']!r} has a non-integer "
        "completion_n_imputed_total 'not-an-integer'"
    ]


def test_a_sequence_without_its_missing_counts_fails(completed_ragged: Path) -> None:
    """A completed sequence records how many cells completion left missing, as a grid does."""
    store = completed_ragged
    table = read_analyses(store / "analyses.tsv")
    assert "completion_n_missing_total" in table.fieldnames
    write_analyses(store / "analyses.tsv", _without_column(table, "completion_n_missing_total"))

    result = validate_store(store)

    assert result.failed() == {"completion.n_missing_column"}, result.errors
    assert result.errors == [
        "analyses.tsv is missing completion_n_missing_total for a reference-completed store"
    ]


def _without_column(table: AnalysesTable, name: str) -> AnalysesTable:
    kept = tuple(field for field in table.fieldnames if field != name)
    return AnalysesTable(kept, tuple({k: row[k] for k in kept} for row in table.rows))


def _flag_in_the_variant_index(group: zarr.Group, cell: int) -> None:
    """Flag the same association imputed in the variant index, so the two copies agree."""
    offsets = np.asarray(group["offsets"][:])
    analysis = int(np.searchsorted(offsets, cell, side="right")) - 1
    variant = int(group["variant_index"][cell])
    index = group["by_variant"]
    lo, hi = (int(x) for x in index["offsets"][variant : variant + 2])
    rows = np.flatnonzero(np.asarray(index["analysis_index"][lo:hi]) == analysis)
    assert len(rows) == 1, "the association must have one row in the variant index"
    index["imputed"][lo + int(rows[0])] = 1


def test_the_value_rules_read_an_overflow_once(
    completed_hybrid: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = completed_hybrid
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


#: The rules about a sequence's variant index that read whole arrays.
VARIANT_INDEX_RULES = {
    "by_variant.present",
    "by_variant.provenance",
    "by_variant.offsets",
    "by_variant.members",
    "by_variant.layout",
    "by_variant.analyses",
    "by_variant.order",
    "by_variant.counts",
}


def test_the_variant_index_rules_read_each_array_once(
    residual_ragged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index's Analyses are read once for its bounds and order, and its runs are
    counted from the sequence's one read, not a second pass over `variant_index`."""
    group = open_store(residual_ragged).arrays(mode="r")["ragged"]
    index = group["by_variant"]

    with element_reads(monkeypatch) as counts:
        result = validate_store(residual_ragged, rules=VARIANT_INDEX_RULES)

    assert result.ok, result.errors
    assert not result.skipped, result.skipped
    assert times_read(counts, group["variant_index"]) == 1.0
    assert times_read(counts, index["analysis_index"]) == 1.0


def test_a_component_defect_does_not_hide_an_overflow_defect(
    completed_hybrid: Path, tmp_path: Path
) -> None:
    """A Hybrid's Dense Component and its Overflow are judged apart.

    Before, any error in the Dense Component stopped the Overflow's checks, so a
    release with a defect in each reported only the first.
    """
    store = residual_stores.copied(completed_hybrid, tmp_path)
    for data in (store / "dense" / "data.zarr", store / "data.zarr"):
        tiers = zarr.open_group(str(data / "top_hits"), mode="r+")
        tier = tiers[sorted(tiers.group_keys())[0]]
        assert tier["z"].shape[0] > 0, "each component's tier must hold a cell"
        tier["z"][0] = 0.0

    result = validate_store(store)

    assert {f.target for f in result.of("top_hits.z")} == {"dense/data.zarr", "data.zarr/ragged"}
    assert result.failed() == {"top_hits.z", "top_hits.abs_z"}, result.errors


def test_an_undecodable_se_does_not_hide_the_table_that_explains_it(fresh_ragged: Path) -> None:
    """An exception code with no table entry: the decode fails, and the table says why.

    The rules that read decoded SE are skipped behind the decode; the rules that
    read only codes, or other planes, still see every window.
    """
    store = fresh_ragged
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


def _decrease_the_first_segment(group: zarr.Group) -> str:
    """Swap the first segment's first two variants; the ordering rule's message for it."""
    offsets = np.asarray(group["offsets"][:])
    assert offsets[1] - offsets[0] >= 2, "the first segment must hold two cells"
    first, second = (int(v) for v in group["variant_index"][0:2])
    group["variant_index"][0:2] = np.array([second, first], dtype=group["variant_index"].dtype)
    return (
        "data.zarr/ragged/variant_index is not non-decreasing within Analysis 0's "
        f"segment: offset 1 holds {first} after {second}"
    )


def test_an_undeclared_n_plane_does_not_hide_a_segment_out_of_order(fresh_ragged: Path) -> None:
    """Ordering reads `offsets` and `variant_index`, not the plan the planes decode under.

    Before, it ran after every plan rule, so an `n` plane the manifest does not
    declare left a segment out of order unreported.
    """
    store = fresh_ragged
    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+")
    assert open_store(store).manifest.encoding.n is None and "n" not in group
    # Laid out as the plane it sits beside, on both sides of the index, so the
    # manifest not declaring it is the plane's one defect.
    for side, length in ((group, group["z"]), (group["by_variant"], group["by_variant/z"])):
        side.create_array(
            "n", shape=length.shape, dtype="uint32", chunks=length.chunks, shards=length.shards
        )[:] = 1000
    order = _decrease_the_first_segment(group)

    result = validate_store(store)

    assert result.failed() == {"encoding.n", "csr.segment_order"}, result.errors
    assert [f.message for f in result.of("csr.segment_order")] == [order]


def test_a_plan_the_value_rules_wait_on_does_not_stop_the_windows_ordering_reads(
    fresh_ragged: Path,
) -> None:
    """A residual `se` without its coefficients fails its plan rule; the windows still
    read, so ordering judges them rather than failing on the decoder's setup."""
    store = fresh_ragged
    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+")
    assert open_store(store).manifest.encoding.se.is_residual
    del group["se_coefficients"]
    order = _decrease_the_first_segment(group)

    result = validate_store(store)

    assert "encoding.se" in result.failed(), result.errors
    assert [f.message for f in result.of("csr.segment_order")] == [order]


def test_a_tier_missing_an_association_above_its_cutoff_is_incomplete(fresh_ragged: Path) -> None:
    """A tier holds every association that clears its cutoff, as a grid's does.

    Raising the tier's threshold lowers its cutoff below cells it does not list.
    A sampled check of the cells a tier does list cannot see that.
    """
    store = fresh_ragged
    tier = open_store(store).arrays(mode="a")["top_hits"]["p_5e_08"]
    assert tier.attrs["threshold"] == 5e-08, "the fixture's tier must record its threshold"
    tier.attrs["threshold"] = 0.5

    result = validate_store(store)

    assert result.failed() == {"top_hits.complete"}
    assert result.errors == ["top-hit index p_5e_08 does not match stored z values"]


def test_a_tier_cell_the_sequence_does_not_hold_is_reported(fresh_ragged: Path) -> None:
    """Every tier cell is an association the sequence holds; one it lacks is named."""
    store = fresh_ragged
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

"""The Top-Hit Index module: one owner for the format's write, read, counts and
structural rules, for every layout (#53)."""

from __future__ import annotations

import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest
from top_hit_fixtures import build_ragged

from opengwasdb import top_hits
from opengwasdb.store import arrays as store_arrays


def _columns(*, imputed: bool = False, eaf: bool = False) -> dict[str, np.ndarray]:
    """Six candidate cells over 3 Analyses, deliberately not in index order.

    |z| 6.0 / 7.0 / 5.0 / 4.0 / 9.0 / 3.6: 5e-8 (|z| >= 5.45) keeps three, 5e-6
    (>= 4.56) four, 5e-4 (>= 3.48) all six.
    """
    columns = {
        "variant_index": np.array([40, 10, 30, 20, 5, 7], dtype=np.int32),
        "analysis_index": np.array([2, 0, 2, 0, 1, 2], dtype=np.int32),
        "z": np.array([6.0, -7.0, 5.0, 4.0, 9.0, -3.6], dtype=np.float32),
        "se": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    }
    if imputed:
        columns["imputed"] = np.array([0, 0, 1, 0, 0, 1], dtype=np.uint8)
    if eaf:
        columns["eaf"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    return columns


def _write(tmp_path, columns, thresholds=(5e-8, 5e-6, 5e-4)):
    root = store_arrays.open_group_for_write(tmp_path / "data.zarr", "a")
    top_hits.write_index(root, columns, thresholds, n_analyses=3)
    return store_arrays.open_group(tmp_path / "data.zarr")


@pytest.mark.parametrize("with_optional", [False, True], ids=["required-only", "imputed-and-eaf"])
def test_an_index_reads_back_the_rows_it_was_written_from(tmp_path, with_optional):
    columns = _columns(imputed=with_optional, eaf=with_optional)
    tiers = top_hits.TopHitTiers(_write(tmp_path, columns))

    rows = tiers.tier(5e-6).read()
    # ordered by (analysis_index, variant_index)
    assert rows.analysis_index.tolist() == [0, 1, 2, 2]
    assert rows.variant_index.tolist() == [10, 5, 30, 40]
    assert rows.z.tolist() == [-7.0, 9.0, 5.0, 6.0]
    assert rows.se is not None and rows.se.tolist() == pytest.approx([0.2, 0.5, 0.3, 0.1])
    if with_optional:
        assert rows.imputed is not None and rows.imputed.tolist() == [0, 0, 1, 0]
        assert rows.eaf is not None and rows.eaf.tolist() == pytest.approx([0.2, 0.5, 0.3, 0.1])
    else:
        assert rows.imputed is None and rows.eaf is None


@pytest.mark.parametrize(
    ("short", "refusal"),
    [
        (
            "variant_index",
            "variant_index of length 5, analysis_index of length 6, z of length 6, "
            "se of length 6, imputed of length 6 and eaf of length 6",
        ),
        (
            "analysis_index",
            "variant_index of length 6, analysis_index of length 5, z of length 6, "
            "se of length 6, imputed of length 6 and eaf of length 6",
        ),
        (
            "z",
            "variant_index of length 6, analysis_index of length 6, z of length 5, "
            "se of length 6, imputed of length 6 and eaf of length 6",
        ),
        (
            "se",
            "variant_index of length 6, analysis_index of length 6, z of length 6, "
            "se of length 5, imputed of length 6 and eaf of length 6",
        ),
        (
            "imputed",
            "variant_index of length 6, analysis_index of length 6, z of length 6, "
            "se of length 6, imputed of length 5 and eaf of length 6",
        ),
        (
            "eaf",
            "variant_index of length 6, analysis_index of length 6, z of length 6, "
            "se of length 6, imputed of length 6 and eaf of length 5",
        ),
    ],
)
def test_an_index_whose_columns_are_not_parallel_is_refused_and_writes_nothing(
    tmp_path, short, refusal
) -> None:
    """Every column of one candidate set describes the same cells: a set that says
    otherwise is refused before the index group is created, naming every column's
    length."""
    columns = _columns(imputed=True, eaf=True)
    columns[short] = columns[short][:-1]
    root = store_arrays.open_group_for_write(tmp_path / "data.zarr", "a")
    before = _files(tmp_path / "data.zarr")
    assert set(before) == {"zarr.json"}, "fixture must be an empty store, not a missing one"
    assert not top_hits.has_index(root), "fixture already has an index, so a rewrite would show"
    with pytest.raises(ValueError, match=f"write_index: {refusal} must be parallel"):
        top_hits.write_index(root, columns, (5e-8, 5e-6, 5e-4), n_analyses=3)
    assert _files(tmp_path / "data.zarr") == before


def _files(path) -> dict[str, bytes]:
    """Every file of a written tree, so a refused write shows as no change."""
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_an_analysis_addressed_read_is_that_analysis_slice(tmp_path):
    tiers = top_hits.TopHitTiers(_write(tmp_path, _columns()))
    tier = tiers.tier(5e-4)
    assert tier.read(2).variant_index.tolist() == [7, 30, 40]
    assert tier.read(0).variant_index.tolist() == [10, 20]
    assert tier.read(3).variant_index.tolist() == []  # past the last Analysis
    assert tier.read(-1).variant_index.tolist() == []
    whole = tier.read()
    assert len(whole.z) == 6


def test_a_threshold_with_no_tier_is_none(tmp_path):
    tiers = top_hits.TopHitTiers(_write(tmp_path, _columns()))
    assert tiers.tier(1e-9) is None


def test_tier_keys_and_attributes_are_the_existing_format(tmp_path):
    root = _write(tmp_path, _columns())
    assert top_hits.threshold_key(5e-8) == "p_5e_08"
    assert top_hits.threshold_key(5e-4) == "p_5e_04"
    assert list(root["top_hits"].attrs["thresholds"]) == [5e-8, 5e-6, 5e-4]
    group = root["top_hits/p_5e_08"]
    assert group.attrs["order"] == "analysis_index,variant_index"
    assert group.attrs["threshold"] == 5e-8
    assert sorted(group.keys()) == sorted(top_hits.TOP_HIT_TIER_ARRAYS)
    assert group["variant_index"].dtype == np.dtype("uint32")
    assert group["p_value"].dtype == np.dtype("float64")


def test_the_module_addresses_its_own_group_and_logs_each_tier(tmp_path, caplog):
    with caplog.at_level("INFO", logger="opengwasdb.top_hits"):
        root = _write(tmp_path, _columns())
    assert "top_hits" in root and sorted(root["top_hits"].group_keys()) == [
        "p_5e_04", "p_5e_06", "p_5e_08"
    ]
    messages = [r.getMessage() for r in caplog.records]
    for key in ("p_5e_08", "p_5e_06", "p_5e_04"):
        assert any(m.startswith(f"Top-hit tier {key}: done in") for m in messages), key


def test_add_hit_counts_sets_and_accumulates_the_counts(tmp_path):
    from opengwasdb.model.analyses import Analysis

    _write(tmp_path, _columns())
    blank = [Analysis(analysis_id=f"a{i}", analysis_index=i) for i in range(3)]
    once = top_hits.add_hit_counts(tmp_path, blank)
    assert [a.n_hits_5e8 for a in once] == ["1", "1", "1"]
    assert [a.n_hits_5e4 for a in once] == ["2", "1", "3"]
    twice = top_hits.add_hit_counts(tmp_path, once)  # a Hybrid adds its two components
    assert [a.n_hits_5e4 for a in twice] == ["4", "2", "6"]


def test_a_tiers_threshold_is_its_attribute_else_its_name(tmp_path):
    _write(tmp_path, _columns())
    writable = store_arrays.open_group_for_write(tmp_path / "data.zarr", "a")
    group = writable["top_hits/p_5e_06"]
    assert top_hits.tier_threshold("p_5e_06", group) == 5e-6
    del group.attrs["threshold"]
    assert top_hits.tier_threshold("p_5e_06", group) == 5e-6  # parsed from the key
    assert top_hits.tier_threshold("p_5e_06", group, missing=0.0) == 0.0  # caller's own
    for threshold in top_hits.TOP_HIT_THRESHOLDS:
        key = top_hits.threshold_key(threshold)
        no_attribute = SimpleNamespace(attrs={})
        assert top_hits.tier_threshold(key, no_attribute) == threshold  # inverse of the key


def test_the_index_group_is_found_by_the_module(tmp_path):
    root = _write(tmp_path, _columns())
    assert top_hits.has_index(root)
    assert sorted(top_hits.index_group(root).group_keys()) == ["p_5e_04", "p_5e_06", "p_5e_08"]
    empty = store_arrays.open_group_for_write(tmp_path / "other.zarr", "a")
    assert not top_hits.has_index(empty)


def test_counts_are_the_per_analysis_slice_lengths(tmp_path):
    _write(tmp_path, _columns())
    counts = top_hits.read_top_hit_counts(tmp_path, 3)
    assert counts == {"n_hits_5e8": [1, 1, 1], "n_hits_5e6": [1, 1, 2], "n_hits_5e4": [2, 1, 3]}


def test_the_structural_rules_name_what_is_wrong(tmp_path):
    root = _write(tmp_path, _columns(imputed=True))
    group = root["top_hits/p_5e_04"]
    arrays = top_hits.load_tier_arrays(group)
    assert top_hits.structural_problem(arrays, group, 3, expects_imputed=True) is None

    swapped = dict(arrays, rows=arrays["rows"][::-1].copy())
    assert "non-genomic" in str(top_hits.structural_problem(swapped, group, 3, True))
    short = dict(arrays, offsets=arrays["offsets"][:-1])
    assert "invalid analysis offsets" in str(top_hits.structural_problem(short, group, 3, True))
    no_imputed = dict(arrays, imputed_values=None)
    assert "missing imputed" in str(top_hits.structural_problem(no_imputed, group, 3, True))


def test_a_ragged_builds_analyses_carry_per_analysis_hit_counts(tmp_path):
    """n_hits_* against a count made by hand from the fixture's z values.

    The BESD fixture's |z| are 5.0 and 6.7 (probe 1), 10.0 (probe 2), 7.5 and 6.0
    (probe 3). 5e-8 needs |z| >= 5.45, 5e-6 needs >= 4.56, 5e-4 needs >= 3.48.
    """
    store = build_ragged(tmp_path)
    with open(store / "analyses.tsv", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [r["n_hits_5e8"] for r in rows] == ["1", "1", "2"]
    assert [r["n_hits_5e6"] for r in rows] == ["2", "1", "2"]
    assert [r["n_hits_5e4"] for r in rows] == ["2", "1", "2"]


def test_the_index_and_the_scan_fallback_cut_at_one_function(tmp_path, monkeypatch):
    """Both paths threshold through `top_hits.z_critical`, so they cannot drift.

    No threshold separates an inline bisection from `erfcinv` by a whole float32
    z, so this does not look for one: it moves the one function and requires the
    index build and the Ragged scan to move with it.
    """
    from opengwasdb.query import query_store

    store = build_ragged(tmp_path)
    indexed = query_store(store).top_hits(threshold=5e-8)
    scanned = query_store(store).top_hits(threshold=1e-9)  # no tier: scans the CSR
    assert len(indexed["z"]) == 4 and len(scanned["z"]) == 3

    monkeypatch.setattr(top_hits, "z_critical", lambda threshold: 100.0)
    assert len(query_store(store).top_hits(threshold=1e-9)["z"]) == 0
    from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes

    build_ragged_top_hit_indexes(store)
    assert len(query_store(store).top_hits(threshold=5e-8)["z"]) == 0


def test_z_critical_is_the_two_sided_normal_cutoff():
    assert top_hits.z_critical(5e-8) == pytest.approx(5.4513, abs=1e-4)
    assert math.erfc(top_hits.z_critical(5e-6) / math.sqrt(2)) == pytest.approx(5e-6)

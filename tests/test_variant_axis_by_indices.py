"""`VariantAxis.by_indices()`/`identity_by_indices()` bulk resolution
strategies (issue #104 follow-up).

`by_indices()`: a random-access `by_index()` seek and a sequential `all()`
scan return the same data; the only thing under test there is that
`by_indices()` picks the right strategy for the request size and that both
strategies agree.

`identity_by_indices()`: chromosome/position/effect_allele/other_allele/alid
recovered from the mmap'd ALID search index alone (no `variants.tsv.gz` I/O
at all) must match what `by_index()` reads from the file.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.variants import VariantAxis
from opengwasdb.variants.axis import (
    _BULK_SCAN_FRACTION,
    variant_alid_bytes_path,
    variant_alid_rows_path,
    write_variant_axis,
)
from opengwasdb.variants.normalise import CanonicalVariant

N_VARIANTS = 1000  # threshold = N_VARIANTS * _BULK_SCAN_FRACTION = 10


def _build_axis(tmp_path):
    variants = [
        CanonicalVariant(chromosome="1", position=1000 + i, effect_allele="A", other_allele="G")
        for i in range(N_VARIANTS)
    ]
    write_variant_axis(tmp_path, variants, {})
    return VariantAxis(tmp_path)


def test_below_threshold_uses_random_access(tmp_path, monkeypatch):
    axis = _build_axis(tmp_path)
    try:
        small = list(range(5))
        assert len(small) <= N_VARIANTS * _BULK_SCAN_FRACTION

        calls = {"by_index": 0, "all": 0}
        real_by_index = axis.by_index

        def spy_by_index(index):
            calls["by_index"] += 1
            return real_by_index(index)

        monkeypatch.setattr(axis, "by_index", spy_by_index)
        monkeypatch.setattr(axis, "all", lambda: (_ for _ in ()).throw(
            AssertionError("full scan should not run below threshold")
        ))

        records = axis.by_indices(small)
        assert sorted(records) == small
        assert calls["by_index"] == len(small)
    finally:
        axis.close()


def test_above_threshold_uses_full_scan(tmp_path, monkeypatch):
    axis = _build_axis(tmp_path)
    try:
        large = list(range(50))
        assert len(large) > N_VARIANTS * _BULK_SCAN_FRACTION

        monkeypatch.setattr(axis, "by_index", lambda index: (_ for _ in ()).throw(
            AssertionError("random access should not run above threshold")
        ))

        records = axis.by_indices(large)
        assert sorted(records) == large
        for i in large:
            assert records[i].variant_index == i
            assert records[i].position == 1000 + i
    finally:
        axis.close()


def test_both_strategies_agree(tmp_path):
    axis = _build_axis(tmp_path)
    try:
        small = list(range(5))
        large = list(range(50))
        via_random_access = axis.by_indices(small)
        via_full_scan = axis.by_indices(large)
        for i in small:
            assert via_random_access[i] == via_full_scan[i]
    finally:
        axis.close()


def test_empty_indices_returns_empty_dict(tmp_path):
    axis = _build_axis(tmp_path)
    try:
        assert axis.by_indices([]) == {}
    finally:
        axis.close()


def test_identity_by_indices_matches_by_index(tmp_path):
    axis = _build_axis(tmp_path)
    try:
        sample = np.array([0, 3, 500, N_VARIANTS - 1], dtype="int64")
        identity = axis.identity_by_indices(sample)
        for i, vi in enumerate(sample):
            record = axis.by_index(int(vi))
            assert identity["chromosome"][i] == record.chromosome
            assert identity["position"][i] == record.position
            assert identity["effect_allele"][i] == record.effect_allele
            assert identity["other_allele"][i] == record.other_allele
            assert identity["alid"][i] == record.alid
    finally:
        axis.close()


def test_identity_by_indices_touches_no_file_io(tmp_path, monkeypatch):
    axis = _build_axis(tmp_path)
    try:
        monkeypatch.setattr(axis, "by_index", lambda index: (_ for _ in ()).throw(
            AssertionError("identity_by_indices() must not seek variants.tsv.gz")
        ))
        monkeypatch.setattr(axis, "all", lambda: (_ for _ in ()).throw(
            AssertionError("identity_by_indices() must not scan variants.tsv.gz")
        ))
        identity = axis.identity_by_indices(np.array([1, 2, 3], dtype="int64"))
        assert list(identity["chromosome"]) == ["1", "1", "1"]
    finally:
        axis.close()


def test_identity_by_indices_empty_returns_empty_arrays(tmp_path):
    axis = _build_axis(tmp_path)
    try:
        identity = axis.identity_by_indices(np.empty(0, dtype="int64"))
        assert all(len(v) == 0 for v in identity.values())
    finally:
        axis.close()


def test_identity_by_indices_with_non_indexable_alids(tmp_path):
    """Long-allele ALIDs (issue #127) leave the mmap'd index a PARTIAL
    permutation of the axis, not a permutation (this is what broke OGS-00010,
    whose 115k long-allele rows sit outside the fixed-width index).

    Identity must still resolve every requested row: index rows from the
    mmap'd index, long-allele rows via the table read `by_index` performs.
    """
    from opengwasdb.variants.axis import _ALID_WIDTH, is_indexable_alid

    long_allele = "A" * (_ALID_WIDTH + 8)  # alid exceeds the fixed width
    variants = [
        CanonicalVariant(
            chromosome="1",
            position=5000 + i,
            effect_allele=long_allele if i % 10 == 0 else "A",
            other_allele="G",
        )
        for i in range(60)
    ]
    write_variant_axis(tmp_path, variants, {})
    axis = VariantAxis(tmp_path)
    try:
        assert not is_indexable_alid(f"1:5000:{long_allele}:G")
        rows = np.arange(60, dtype="int64")
        identity = axis.identity_by_indices(rows)
        assert identity is not None
        for i, vi in enumerate(rows):
            record = axis.by_index(int(vi))
            assert identity["alid"][i] == record.alid
            assert identity["chromosome"][i] == record.chromosome
            assert identity["position"][i] == record.position
            assert identity["effect_allele"][i] == record.effect_allele
            assert identity["other_allele"][i] == record.other_allele
    finally:
        axis.close()


def test_identity_by_indices_none_without_mmap_alid_index(tmp_path):
    axis = _build_axis(tmp_path)
    try:
        variant_alid_bytes_path(tmp_path).unlink()
        variant_alid_rows_path(tmp_path).unlink()
        axis_no_index = VariantAxis(tmp_path)
        try:
            assert axis_no_index.identity_by_indices(np.array([0], dtype="int64")) is None
        finally:
            axis_no_index.close()
    finally:
        axis.close()

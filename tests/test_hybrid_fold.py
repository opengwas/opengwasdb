"""Tests for parallel vectorized off-reference spill folding (ticket #223).

Pins the fold guarantees:
1. Searchsorted lookup drops unresolved keys without leaking them.
2. Rows that share a Variant Index -- an off-reference entry and an existing
   overflow one, or two raw keys of one column -- are resolved by Association
   Admission's duplicate rule (D5, issue #357): agreeing rows collapse to the
   first, conflicting ones are dropped, and a reversed indel whose source
   declared the reference keeps the row whose reference allele the ALID spells
   first. What the fold drops is the Admission record's `fold` stage.
3. Overflow spills are written once atomically via temp file rename, and .unk
   files are unlinked only after successful write.
4. Parallel execution across --n-workers matches serial execution (n_workers=1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from admission_helpers import logged, stage_part

from opengwasdb.admission import AdmissionRecord
from opengwasdb.layouts.hybrid.build import (
    _fold_unknown_spills,
)
from opengwasdb.layouts.hybrid.key_table import (
    RAW_KEY_DTYPE,
    CanonicalRawKeys,
    KeyTable,
)


def _canonical_keys(*pairs: tuple[int, str]) -> CanonicalRawKeys:
    if not pairs:
        return CanonicalRawKeys(
            values=np.empty(0, dtype=np.uint64),
            raw=np.empty(0, dtype=RAW_KEY_DTYPE),
        )
    values = np.array([p[0] for p in pairs], dtype=np.uint64)
    raw = np.array([p[1] for p in pairs], dtype=RAW_KEY_DTYPE)
    order = np.argsort(values)
    return CanonicalRawKeys(values=values[order], raw=raw[order])


#: The shared axis a fold test's indices name: index ``i`` is ``1:<i>:A:G``.
SHARED_ALIDS = [f"1:{i}:A:G" for i in range(1, 501)]


def _record(spill_dir: Path, n_analyses: int = 1) -> AdmissionRecord:
    analyses = [f"A{col}" for col in range(n_analyses)]
    return AdmissionRecord.open(spill_dir.parent / "record", analyses, command="test")


def _run_fold_column(
    spill_dir: Path,
    table: KeyTable,
    canonical: CanonicalRawKeys | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute single-column fold and load resulting overflow arrays."""
    _fold_unknown_spills(
        spill_dir=spill_dir,
        n_analyses=1,
        table=table,
        canonical=canonical if canonical is not None else _canonical_keys(),
        old_to_new=None,
        n_workers=1,
        record=_record(spill_dir),
        shared_alids=SHARED_ALIDS,
    )
    with np.load(spill_dir / "0.ovf.npz") as data:
        return data["variant_index"], data["z"], data["se"], data["eaf"]


def _fold_stage(spill_dir: Path) -> tuple[dict, list[str]]:
    """The fold's part of the record for column 0, and its logged lines."""
    record = spill_dir.parent / "record"
    part = stage_part(record, 0, "fold")
    has_log = (record / "dropped" / "0.fold.tsv.gz").exists()
    logged_rows = logged(record, 0, "fold")[1:] if has_log else []
    return part, ["\t".join(row) for row in logged_rows]


def test_fold_drops_an_off_reference_entry_conflicting_with_existing_overflow(
    tmp_path: Path,
) -> None:
    """D5 (issue #357): an existing overflow entry and an off-reference entry that
    share a Variant Index and disagree are both dropped and logged, where the
    off-reference entry used to win (issue #223)."""
    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()

    # Column 0 has existing overflow with variant index 100, z=1.0, se=0.5, eaf=0.2
    # and variant index 200, z=2.0, se=0.4, eaf=0.3
    np.savez(
        spill_dir / "0.ovf.npz",
        variant_index=np.array([100, 200], dtype=np.int64),
        z=np.array([1.0, 2.0], dtype=np.float32),
        se=np.array([0.5, 0.4], dtype=np.float32),
        eaf=np.array([0.2, 0.3], dtype=np.float32),
        reference=np.zeros(2, dtype=np.int8),
    )

    # Column 0 has off-reference spill with key 1000 resolving to variant index 100,
    # with z=5.0, se=0.1, eaf=0.8, which disagrees with the existing entry
    np.savez(
        spill_dir / "0.unk.npz",
        keys=np.array([1000], dtype=np.uint64),
        z=np.array([5.0], dtype=np.float32),
        se=np.array([0.1], dtype=np.float32),
        eaf=np.array([0.8], dtype=np.float32),
        reference=np.zeros(1, dtype=np.int8),
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([1000], dtype=np.uint64),
        shared_index=np.array([100], dtype=np.int64),
    )

    vi, z, se, eaf = _run_fold_column(spill_dir, table)
    assert not (spill_dir / "0.unk.npz").exists()

    # Neither row of index 100 is kept; the untouched existing entry is.
    assert vi.tolist() == [200]
    assert z.tolist() == pytest.approx([2.0])
    assert se.tolist() == pytest.approx([0.4])
    assert eaf.tolist() == pytest.approx([0.3])
    part, logged = _fold_stage(spill_dir)
    assert (part["admitted"], part["dropped"]) == (-2, {"duplicate_conflicting": 2})
    assert logged == ["1:101:A:G\tduplicate_conflicting"] * 2


def test_fold_drops_unresolved_keys_without_leaking(tmp_path: Path) -> None:
    """Issue #223 AC4: keys not in the KeyTable (e.g. failed liftover or multi-assembly)
    must be dropped cleanly; no association may leak into the overflow spill."""
    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()

    # Column 0 has 3 off-reference associations:
    # row 0: key 10 -> resolves to variant index 10, z=2.0
    # row 1: key 20 -> resolves to variant index 20, z=4.0
    # row 2: key 999 -> unresolved (absent from table), z=-999.0 (MUST DROP)
    np.savez(
        spill_dir / "0.unk.npz",
        keys=np.array([10, 20, 999], dtype=np.uint64),
        z=np.array([2.0, 4.0, -999.0], dtype=np.float32),
        se=np.array([0.2, 0.4, 0.999], dtype=np.float32),
        eaf=np.array([0.1, 0.3, 0.99], dtype=np.float32),
        reference=np.zeros(3, dtype=np.int8),
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([10, 20], dtype=np.uint64),
        shared_index=np.array([10, 20], dtype=np.int64),
    )

    vi, z, se, eaf = _run_fold_column(spill_dir, table)

    assert len(vi) == 2, "unresolved key 999 must be dropped, not leaked"
    np.testing.assert_array_equal(vi, [10, 20])
    np.testing.assert_array_equal(z, [2.0, 4.0])
    np.testing.assert_allclose(se, [0.2, 0.4])
    np.testing.assert_allclose(eaf, [0.1, 0.3])
    assert -999.0 not in z


@pytest.mark.parametrize(
    ("reference", "kept"),
    [([0, 0], []), ([1, 2], [1.5]), ([2, 1], [3.5]), ([1, 1], [])],
    ids=["undeclared", "first-spelt-first", "second-spelt-first", "same-side"],
)
def test_fold_two_raw_keys_mapping_to_one_variant_index_are_d5_duplicates(
    tmp_path: Path, reference: list[int], kept: list[float]
) -> None:
    """D5 (issue #357): two raw keys of one column that resolve to one Variant
    Index and disagree are both dropped, where the later used to win (#223) --
    unless they are a reversed indel whose source declared the reference, which
    keeps the row whose reference allele the ALID spells first."""
    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()

    # Column 0 has 2 off-reference associations:
    # row 0: key 100 -> resolves to variant index 42, z=1.5, se=0.3
    # row 1: key 200 -> resolves to same variant index 42, z=3.5, se=0.1
    np.savez(
        spill_dir / "0.unk.npz",
        keys=np.array([100, 200], dtype=np.uint64),
        z=np.array([1.5, 3.5], dtype=np.float32),
        se=np.array([0.3, 0.1], dtype=np.float32),
        eaf=np.array([0.25, 0.75], dtype=np.float32),
        reference=np.array(reference, dtype=np.int8),
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([100, 200], dtype=np.uint64),
        shared_index=np.array([42, 42], dtype=np.int64),
    )

    vi, z, se, eaf = _run_fold_column(spill_dir, table)

    assert z.tolist() == pytest.approx(kept)
    assert vi.tolist() == [42] * len(kept)
    part, logged = _fold_stage(spill_dir)
    reason = "duplicate_reversed_indel" if kept else "duplicate_conflicting"
    assert part["dropped"] == {reason: 2 - len(kept)}
    assert logged == [f"1:43:A:G\t{reason}"] * (2 - len(kept))


def test_fold_atomic_write_preserves_unk_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #223 AC3: overflow spill is written atomically; off-reference spill
    is deleted only after its column's overflow spill is safely written."""
    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()

    unk_file = spill_dir / "0.unk.npz"
    np.savez(
        unk_file,
        keys=np.array([100], dtype=np.uint64),
        z=np.array([1.0], dtype=np.float32),
        se=np.array([0.1], dtype=np.float32),
        eaf=np.array([0.2], dtype=np.float32),
        reference=np.zeros(1, dtype=np.int8),
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([100], dtype=np.uint64),
        shared_index=np.array([10], dtype=np.int64),
    )
    canonical = _canonical_keys()

    # Simulate write failure during rename/save
    def _exploding_savez(*args: object, **kwargs: object) -> None:
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(np, "savez", _exploding_savez)

    with pytest.raises(OSError, match="Simulated disk write failure"):
        _fold_unknown_spills(
            spill_dir=spill_dir,
            n_analyses=1,
            table=table,
            canonical=canonical,
            old_to_new=None,
            n_workers=1,
            record=_record(spill_dir),
            shared_alids=SHARED_ALIDS,
        )

    # unk file MUST still be intact
    assert unk_file.exists(), ".unk file must not be deleted if write failed"
    # temporary file must not linger
    assert not (spill_dir / "0.ovf.tmp.npz").exists()
    assert not (spill_dir / "0.ovf.npz").exists()


def test_fold_parallel_matches_serial(tmp_path: Path) -> None:
    """Issue #223 AC1: parallel fold across workers produces results identical
    to the serial path (n_workers=1)."""
    # Create 8 columns with different combinations:
    # col 0: only existing overflow
    # col 1: only off-reference
    # col 2: both, with overlapping indices (precedence test)
    # col 3: both, with disjoint indices
    # col 4: off-reference with unresolved keys
    # col 5: two raw keys mapping to one variant index
    # col 6: empty (neither ovf nor unk)
    # col 7: off-reference all unresolved

    table = KeyTable(
        keys=np.array([10, 20, 30, 40, 50], dtype=np.uint64),
        shared_index=np.array([100, 200, 300, 400, 100], dtype=np.int64),  # 10 and 50 map to 100
    )
    canonical = _canonical_keys()
    old_to_new = np.array([0, 100, 200, 300, 400], dtype=np.int64)

    def _setup_dir(base: Path) -> Path:
        s = base / "spills"
        s.mkdir(parents=True)
        # col 0: existing ovf only
        np.savez(
            s / "0.ovf.npz",
            variant_index=np.array([1, 2], dtype=np.int64),
            z=np.array([1.0, 2.0], dtype=np.float32),
            se=np.array([0.1, 0.2], dtype=np.float32),
            eaf=np.array([0.3, 0.4], dtype=np.float32),
            reference=np.zeros(2, dtype=np.int8),
        )
        # col 1: off-ref only
        np.savez(
            s / "1.unk.npz",
            keys=np.array([20, 30], dtype=np.uint64),
            z=np.array([2.5, 3.5], dtype=np.float32),
            se=np.array([0.25, 0.35], dtype=np.float32),
            eaf=np.array([0.45, 0.55], dtype=np.float32),
            reference=np.zeros(2, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 2: both with overlap on index 100 (key 10 maps to 100, existing 1 maps to 100)
        np.savez(
            s / "2.ovf.npz",
            variant_index=np.array([1], dtype=np.int64),
            z=np.array([1.1], dtype=np.float32),
            se=np.array([0.11], dtype=np.float32),
            eaf=np.array([0.21], dtype=np.float32),
            reference=np.zeros(1, dtype=np.int8),
        )
        np.savez(
            s / "2.unk.npz",
            keys=np.array([10], dtype=np.uint64),
            z=np.array([9.9], dtype=np.float32),
            se=np.array([0.99], dtype=np.float32),
            eaf=np.array([0.88], dtype=np.float32),
            reference=np.zeros(1, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 3: both disjoint
        np.savez(
            s / "3.ovf.npz",
            variant_index=np.array([2], dtype=np.int64),
            z=np.array([2.2], dtype=np.float32),
            se=np.array([0.22], dtype=np.float32),
            eaf=np.array([0.33], dtype=np.float32),
            reference=np.zeros(1, dtype=np.int8),
        )
        np.savez(
            s / "3.unk.npz",
            keys=np.array([40], dtype=np.uint64),
            z=np.array([4.4], dtype=np.float32),
            se=np.array([0.44], dtype=np.float32),
            eaf=np.array([0.55], dtype=np.float32),
            reference=np.zeros(1, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 4: unresolved key 999
        np.savez(
            s / "4.unk.npz",
            keys=np.array([20, 999], dtype=np.uint64),
            z=np.array([2.0, -100.0], dtype=np.float32),
            se=np.array([0.2, 1.0], dtype=np.float32),
            eaf=np.array([0.3, 0.5], dtype=np.float32),
            reference=np.zeros(2, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 5: two raw keys mapping to one variant index (10 and 50 both map to 100)
        np.savez(
            s / "5.unk.npz",
            keys=np.array([10, 50], dtype=np.uint64),
            z=np.array([1.0, 5.0], dtype=np.float32),
            se=np.array([0.1, 0.5], dtype=np.float32),
            eaf=np.array([0.2, 0.8], dtype=np.float32),
            reference=np.zeros(2, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 6: empty
        # col 7: all unresolved
        np.savez(
            s / "7.unk.npz",
            keys=np.array([888, 999], dtype=np.uint64),
            z=np.array([8.0, 9.0], dtype=np.float32),
            se=np.array([0.8, 0.9], dtype=np.float32),
            eaf=np.array([0.8, 0.9], dtype=np.float32),
            reference=np.zeros(2, dtype=np.int8),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        return s

    serial_dir = _setup_dir(tmp_path / "serial")
    parallel_dir = _setup_dir(tmp_path / "parallel")

    _fold_unknown_spills(
        spill_dir=serial_dir,
        n_analyses=8,
        table=table,
        canonical=canonical,
        old_to_new=old_to_new,
        n_workers=1,
        record=_record(serial_dir, 8),
        shared_alids=SHARED_ALIDS,
    )

    _fold_unknown_spills(
        spill_dir=parallel_dir,
        n_analyses=8,
        table=table,
        canonical=canonical,
        old_to_new=old_to_new,
        n_workers=4,
        record=_record(parallel_dir, 8),
        shared_alids=SHARED_ALIDS,
    )

    for col in range(8):
        ser_ovf = serial_dir / f"{col}.ovf.npz"
        par_ovf = parallel_dir / f"{col}.ovf.npz"
        assert ser_ovf.exists() == par_ovf.exists(), f"col {col} existence mismatch"
        if ser_ovf.exists():
            with np.load(ser_ovf) as d_ser, np.load(par_ovf) as d_par:
                np.testing.assert_array_equal(d_ser["variant_index"], d_par["variant_index"])
                np.testing.assert_array_equal(d_ser["z"], d_par["z"])
                np.testing.assert_array_equal(d_ser["se"], d_par["se"])
                np.testing.assert_array_equal(d_ser["eaf"], d_par["eaf"])
        assert not (serial_dir / f"{col}.unk.npz").exists()
        assert not (parallel_dir / f"{col}.unk.npz").exists()


def test_fold_counts_and_logs_the_rows_no_key_resolves(tmp_path: Path) -> None:
    """D6 (issue #357): a row whose off-reference key the fold cannot resolve
    reaches no axis. It is counted and logged as `off_axis` by its raw key, a
    packed SNV decoded from its bits and a hashed indel from the side file,
    where it used to vanish after the gates had written their record."""
    from opengwasdb.layouts.hybrid.build import _write_unknown_side_file
    from opengwasdb.layouts.hybrid.unknown_keys import encode_keys

    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()
    raw = ["1:5:A:G", "1:7:C:T", "1:9:A:AT"]
    encoded = encode_keys(raw)
    assert encoded.hashed_index.tolist() == [2], "the fixture must hold a hashed key"
    np.savez(
        spill_dir / "0.unk.npz",
        keys=encoded.values,
        z=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        se=np.full(3, 0.1, dtype=np.float32),
        eaf=np.full(3, np.nan, dtype=np.float32),
        reference=np.zeros(3, dtype=np.int8),
        hashed_index=encoded.hashed_index,
    )
    _write_unknown_side_file(spill_dir, 0, encoded.hashed_raw)
    canonical = _canonical_keys((int(encoded.values[2]), "1:9:A:AT"))
    table = KeyTable(keys=encoded.values[:1], shared_index=np.array([10], dtype=np.int64))

    vi, z, _, _ = _run_fold_column(spill_dir, table, canonical)

    assert (vi.tolist(), z.tolist()) == ([10], [1.0])
    part, logged_rows = _fold_stage(spill_dir)
    assert (part["admitted"], part["dropped"]) == (-2, {"off_axis": 2})
    assert logged_rows == ["1:7:C:T\toff_axis", "1:9:A:AT\toff_axis"]

"""Tests for parallel vectorized off-reference spill folding (ticket #223).

Pins the fold guarantees:
1. Searchsorted lookup drops unresolved keys without leaking them.
2. Precedence: duplicate Variant Indices deduplicate last-wins, with off-reference
   entries winning over existing overflow entries.
3. Two raw keys in the same column mapping to the same Variant Index deduplicate
   last-wins in stream order.
4. Overflow spills are written once atomically via temp file rename, and .unk
   files are unlinked only after successful write.
5. Parallel execution across --n-workers matches serial execution (n_workers=1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
    )
    with np.load(spill_dir / "0.ovf.npz") as data:
        return data["variant_index"], data["z"], data["se"], data["eaf"]


def test_fold_precedence_off_reference_wins_over_existing_overflow(tmp_path: Path) -> None:
    """Issue #223 AC4: when an existing overflow entry and an off-reference entry
    share a Variant Index, the off-reference entry must win (last-wins precedence)."""
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
    )

    # Column 0 has off-reference spill with key 1000 resolving to variant index 100,
    # with z=5.0, se=0.1, eaf=0.8 (must overwrite existing overflow for index 100)
    np.savez(
        spill_dir / "0.unk.npz",
        keys=np.array([1000], dtype=np.uint64),
        z=np.array([5.0], dtype=np.float32),
        se=np.array([0.1], dtype=np.float32),
        eaf=np.array([0.8], dtype=np.float32),
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([1000], dtype=np.uint64),
        shared_index=np.array([100], dtype=np.int64),
    )

    vi, z, se, eaf = _run_fold_column(spill_dir, table)
    assert not (spill_dir / "0.unk.npz").exists()

    assert set(vi.tolist()) == {100, 200}
    idx_100 = np.flatnonzero(vi == 100)[0]
    idx_200 = np.flatnonzero(vi == 200)[0]

    # Off-reference entry won over existing overflow
    assert z[idx_100] == pytest.approx(5.0), "off-reference entry must win over existing overflow"
    assert se[idx_100] == pytest.approx(0.1)
    assert eaf[idx_100] == pytest.approx(0.8)

    # Untouched existing overflow preserved
    assert z[idx_200] == pytest.approx(2.0)
    assert se[idx_200] == pytest.approx(0.4)
    assert eaf[idx_200] == pytest.approx(0.3)


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


def test_fold_two_raw_keys_mapping_to_one_variant_index_last_wins(tmp_path: Path) -> None:
    """Issue #223 AC4: when two different raw keys in the same column resolve to
    the same Variant Index, the later stream occurrence must win (last-wins)."""
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
        hashed_index=np.empty(0, dtype=np.int64),
    )

    table = KeyTable(
        keys=np.array([100, 200], dtype=np.uint64),
        shared_index=np.array([42, 42], dtype=np.int64),
    )

    vi, z, se, eaf = _run_fold_column(spill_dir, table)

    assert len(vi) == 1, "two raw keys mapping to one variant index must deduplicate to 1 entry"
    assert vi[0] == 42
    assert z[0] == pytest.approx(3.5), "later stream occurrence must win (last-wins)"
    assert se[0] == pytest.approx(0.1)
    assert eaf[0] == pytest.approx(0.75)


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
        )
        # col 1: off-ref only
        np.savez(
            s / "1.unk.npz",
            keys=np.array([20, 30], dtype=np.uint64),
            z=np.array([2.5, 3.5], dtype=np.float32),
            se=np.array([0.25, 0.35], dtype=np.float32),
            eaf=np.array([0.45, 0.55], dtype=np.float32),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 2: both with overlap on index 100 (key 10 maps to 100, existing 1 maps to 100)
        np.savez(
            s / "2.ovf.npz",
            variant_index=np.array([1], dtype=np.int64),
            z=np.array([1.1], dtype=np.float32),
            se=np.array([0.11], dtype=np.float32),
            eaf=np.array([0.21], dtype=np.float32),
        )
        np.savez(
            s / "2.unk.npz",
            keys=np.array([10], dtype=np.uint64),
            z=np.array([9.9], dtype=np.float32),
            se=np.array([0.99], dtype=np.float32),
            eaf=np.array([0.88], dtype=np.float32),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 3: both disjoint
        np.savez(
            s / "3.ovf.npz",
            variant_index=np.array([2], dtype=np.int64),
            z=np.array([2.2], dtype=np.float32),
            se=np.array([0.22], dtype=np.float32),
            eaf=np.array([0.33], dtype=np.float32),
        )
        np.savez(
            s / "3.unk.npz",
            keys=np.array([40], dtype=np.uint64),
            z=np.array([4.4], dtype=np.float32),
            se=np.array([0.44], dtype=np.float32),
            eaf=np.array([0.55], dtype=np.float32),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 4: unresolved key 999
        np.savez(
            s / "4.unk.npz",
            keys=np.array([20, 999], dtype=np.uint64),
            z=np.array([2.0, -100.0], dtype=np.float32),
            se=np.array([0.2, 1.0], dtype=np.float32),
            eaf=np.array([0.3, 0.5], dtype=np.float32),
            hashed_index=np.empty(0, dtype=np.int64),
        )
        # col 5: two raw keys mapping to one variant index (10 and 50 both map to 100)
        np.savez(
            s / "5.unk.npz",
            keys=np.array([10, 50], dtype=np.uint64),
            z=np.array([1.0, 5.0], dtype=np.float32),
            se=np.array([0.1, 0.5], dtype=np.float32),
            eaf=np.array([0.2, 0.8], dtype=np.float32),
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
    )

    _fold_unknown_spills(
        spill_dir=parallel_dir,
        n_analyses=8,
        table=table,
        canonical=canonical,
        old_to_new=old_to_new,
        n_workers=4,
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

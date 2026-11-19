"""The companion bundle's ownership boundary, using actual file mappings (ADR 0012; #47)."""

from __future__ import annotations

import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_alid_index import written

from ogld.format import InputError
from ogld.store import StoreReader, layout
from ogld.store import reader as reader_module
from ogld.store.reader import _MappedAlids


@pytest.mark.parametrize("failure_at", ["keys-open", "keys-map", "rows-open", "rows-map"])
def test_failed_construction_closes_maps_with_a_retained_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    store = written(tmp_path / "store")
    companion = layout.alid_companion(zarr.open_group(store, mode="r"))
    assert companion is not None
    real_open, real_map = Path.open, mmap.mmap
    acquired: list[mmap.mmap] = []
    attempts = 0
    failure = OSError("companion attachment failed")

    def opening(path: Path, *args, **kwargs):
        if path == companion.path / (
            "keys.bin" if failure_at == "keys-open" else "rows.i32le"
        ) and failure_at.endswith("open"):
            raise failure
        return real_open(path, *args, **kwargs)

    def mapping(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if failure_at == ("keys-map" if attempts == 1 else "rows-map"):
            raise failure
        mapped = real_map(*args, **kwargs)
        acquired.append(mapped)
        return mapped

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(reader_module.mmap, "mmap", mapping)
    # Keeping both the bundle and exception alive rules out collection as the cleanup mechanism.
    bundle = object.__new__(_MappedAlids)
    with pytest.raises(OSError, match="attachment failed") as stopped:
        bundle.__init__(companion)
    assert stopped.value is failure
    assert stopped.value.__traceback__ is not None
    assert len(acquired) == (1 if failure_at.startswith("rows") else 0)
    assert all(mapped.closed for mapped in acquired), "failed attachment must release immediately"
    bundle.release()
    bundle.release()
    assert all(mapped.closed for mapped in acquired)


def test_lookup_results_survive_idempotent_release(tmp_path: Path) -> None:
    store = written(tmp_path / "store")
    reader = StoreReader.open(store)
    bundle = reader._alid_mmap
    assert bundle is not None
    mappings = tuple(bundle._maps)
    assert len(mappings) == 2
    assert all(not mapped.closed for mapped in mappings)
    keys = np.array([b"22:200:A:C", b"22:300:A:G", b"absent", b"22:200:A:C"], dtype="S32")
    returned = bundle.lookup(keys)
    public = reader.rows_of(["22:200:A:C", "22:300:A:G", "absent", "22:200:A:C"])
    assert returned.tolist() == public.tolist() == [2, 4, -1, 2]
    bundle.release()
    bundle.release()
    assert all(mapped.closed for mapped in mappings)
    assert returned.tolist() == public.tolist() == [2, 4, -1, 2]
    returned[0] = 99
    assert public.tolist() == [2, 4, -1, 2]
    with pytest.raises(InputError, match="released"):
        bundle.lookup(keys)
    with pytest.raises(InputError, match="released"):
        reader.rows_of(["22:200:A:C"])


def test_reader_close_releases_mappings_and_keeps_returned_rows(tmp_path: Path) -> None:
    reader = StoreReader.open(written(tmp_path / "store"))
    bundle = reader._alid_mmap
    assert bundle is not None
    mappings = tuple(bundle._maps)
    assert len(mappings) == 2
    assert all(not mapped.closed for mapped in mappings)
    rows = reader.rows_of(["22:200:A:C", "22:300:A:G"])
    reader.close()
    reader.close()
    assert all(mapped.closed for mapped in mappings)
    assert rows.tolist() == [2, 4]
    with pytest.raises(InputError, match="released"):
        reader.rows_of(["22:200:A:C"])


def test_release_waits_for_a_lookup_then_closes_its_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = StoreReader.open(written(tmp_path / "store"))._alid_mmap
    assert bundle is not None
    mappings = tuple(bundle._maps)
    assert len(mappings) == 2
    entered, unblock, releasing = (threading.Event() for _ in range(3))
    search = np.searchsorted

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.attempts = 0

        def __enter__(self):
            self.attempts += 1
            if self.attempts == 2:
                releasing.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def held_search(*args, **kwargs):
        entered.set()
        assert unblock.wait(2), "test must release the actual search"
        return search(*args, **kwargs)

    monkeypatch.setattr(bundle, "_lock", ObservedLock())
    monkeypatch.setattr(reader_module.np, "searchsorted", held_search)
    with ThreadPoolExecutor(max_workers=2) as workers:
        lookup = workers.submit(bundle.lookup, np.array([b"22:200:A:C"], dtype="S32"))
        try:
            assert entered.wait(2), "lookup holds its mapping before teardown starts"
            release = workers.submit(bundle.release)
            assert releasing.wait(2), "release has reached the same ownership lock"
            assert not release.done()
            assert all(not mapped.closed for mapped in mappings)
        finally:
            unblock.set()
        assert lookup.result(timeout=2).tolist() == [2]
        release.result(timeout=2)
    assert all(mapped.closed for mapped in mappings)
    bundle.release()


def test_replaced_descriptor_file_is_refused_and_releases_prior_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    store = written(tmp_path / "store")
    companion = layout.alid_companion(zarr.open_group(store, mode="r"))
    assert companion is not None
    row_path = companion.path / "rows.i32le"
    replacement = row_path.with_suffix(".replacement")
    replacement.write_bytes(row_path.read_bytes())
    assert replacement.stat().st_ino != row_path.stat().st_ino
    os.replace(replacement, row_path)
    real_map = mmap.mmap
    acquired = []

    def mapping(*args, **kwargs):
        mapped = real_map(*args, **kwargs)
        acquired.append(mapped)
        return mapped

    monkeypatch.setattr(reader_module.mmap, "mmap", mapping)
    with pytest.raises(InputError, match="changed before mapping"):
        _MappedAlids(companion)
    assert len(acquired) == 1, "the keys were mapped before the replaced rows were encountered"
    assert acquired[0].closed

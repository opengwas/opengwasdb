"""Writes refuse a group that consolidated metadata describes (#244 review, finding 1).

zarr-python 3's ``open_group`` reads a Zarr v2 ``.zmetadata`` (or a v3
``consolidated_metadata`` block) in place of the live array metadata whenever
one exists; zarr 2.18 did not. The package never writes consolidated metadata,
and none of its writes -- array creation, deletion, the directory moves `repair`
and the SE fallback make -- update one. Under zarr 3 a write beneath a
consolidated record therefore leaves a release whose next open reads stale
shapes and chunks: the review reproduced a rechunked `eaf_baseline` reopening
with its old `(8,)` chunks and failing to reshape.

So every writable open, and every move, refuses before changing anything when a
consolidated record describes the group, its own or an enclosing group's. Read
opens are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from test_eaf_encoding_layouts import make_store_needing_eaf_repair

from opengwasdb.encoding import EAF_BASELINE
from opengwasdb.repair import repair_eaf_chunks
from opengwasdb.store import open_store
from opengwasdb.store.arrays import (
    ArrayRole,
    ConsolidatedMetadataError,
    create_array,
    move_in_group,
    open_group,
    open_group_for_write,
)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under `root`, by relative path, with its bytes."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _consolidate(group_path: Path) -> None:
    zarr.consolidate_metadata(str(group_path), zarr_format=2)
    assert (group_path / ".zmetadata").is_file(), "consolidation wrote no .zmetadata"
    assert zarr.open_group(str(group_path), mode="r").metadata.consolidated_metadata is not None


def _group_with_subgroup(path: Path) -> Path:
    """A v2 group holding one array and one subgroup with an array of its own."""
    root = open_group_for_write(path, "w")
    create_array(root, "a", ArrayRole.PER_VARIANT, data=np.arange(8, dtype="float32"), hint=2)
    sub = root.create_group("sub")
    create_array(sub, "b", ArrayRole.PER_VARIANT, data=np.arange(4, dtype="float32"), hint=2)
    return path


@pytest.fixture
def consolidated_release(tmp_path: Path) -> Path:
    """A Dense release that needs EAF repair, with its `data.zarr` consolidated."""
    out = make_store_needing_eaf_repair(tmp_path)
    _consolidate(out / "data.zarr")
    # A fresh open reads the record. The repair would rechunk this array, which is
    # what would stale the record.
    assert zarr.open_group(str(out / "data.zarr"), mode="r")[EAF_BASELINE].chunks == (8,)
    return out


def test_repair_refuses_a_consolidated_release_before_changing_anything(
    consolidated_release: Path,
) -> None:
    before = _tree(consolidated_release)
    values = np.asarray(open_store(consolidated_release).arrays()[EAF_BASELINE][:])
    assert np.isfinite(values).any(), (
        "the baseline must carry values for the reread to mean anything"
    )

    with pytest.raises(ConsolidatedMetadataError, match=r"\.zmetadata"):
        repair_eaf_chunks(consolidated_release)

    assert _tree(consolidated_release) == before
    # The release still opens and reads what it held: the record still matches.
    reread = np.asarray(open_store(consolidated_release).arrays()[EAF_BASELINE][:])
    np.testing.assert_array_equal(reread, values)


def test_a_move_refuses_under_consolidated_metadata(tmp_path: Path) -> None:
    path = _group_with_subgroup(tmp_path / "g.zarr")
    # Opened for writing before anything consolidated it, as a long-lived handle
    # would be: the move itself must still refuse.
    group = open_group(path, "r+")
    _consolidate(path)
    before = _tree(path)

    with pytest.raises(ConsolidatedMetadataError, match=r"\.zmetadata"):
        move_in_group(group, "a", "moved")

    assert _tree(path) == before
    assert (path / "a").is_dir() and not (path / "moved").exists()


@pytest.mark.parametrize("mode", ["r+", "a"])
def test_writable_opens_refuse_a_consolidated_group(tmp_path: Path, mode: str) -> None:
    path = _group_with_subgroup(tmp_path / "g.zarr")
    _consolidate(path)

    with pytest.raises(ConsolidatedMetadataError):
        open_group(path, mode)
    if mode == "a":
        with pytest.raises(ConsolidatedMetadataError):
            open_group_for_write(path, mode)


@pytest.mark.parametrize("mode", ["r+", "a", "w"])
def test_a_subgroup_under_a_consolidated_root_refuses(tmp_path: Path, mode: str) -> None:
    """The root's record lists the subgroup's arrays; even wiping the subgroup stales it."""
    path = _group_with_subgroup(tmp_path / "g.zarr")
    _consolidate(path)
    assert not (path / "sub" / ".zmetadata").exists(), "only the root is consolidated"

    with pytest.raises(ConsolidatedMetadataError, match=r"g\.zarr/\.zmetadata"):
        open_group(path / "sub", mode)


def test_reads_and_a_wipe_of_the_consolidated_group_itself_still_work(tmp_path: Path) -> None:
    path = _group_with_subgroup(tmp_path / "g.zarr")
    _consolidate(path)

    np.testing.assert_array_equal(open_group(path)["a"][:], np.arange(8, dtype="float32"))
    # mode="w" deletes the group and its own record with it, so nothing goes stale.
    open_group_for_write(path, "w")
    assert not (path / ".zmetadata").exists()

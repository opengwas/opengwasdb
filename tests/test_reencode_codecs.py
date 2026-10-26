"""`reencode_se_codecs` re-encodes the SE plane and its exception index (#326).

The retrospective re-encode fixes a Blosc `typesize` that was inherited from
the float32 scratch plane: `se` (int8) and `se_exception_index` (int64) are
stored at `typesize: 4` where 1 and 8 are correct.  The re-encode is verified
by full decoded equality (each inner chunk, byte for byte, with the shard-index
CRC validated by the decode itself), staged beside the live array and swapped
in only after verification.  A process that dies at any step leaves a state the
next run settles: the original or the re-encoded array, never a torn mix.

The stores lane owns the release-lock/marker refusal, the
`records/se-reencode.json` writer and the CLI; these tests cover the core
primitive those will call.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree
from typing import Any

import numpy as np
import pytest
from swap_interruption import ProcessDeath, die_on_second_move
from zarr.codecs import BloscCodec

import opengwasdb.reencode as reencode
from opengwasdb.reencode import plan_se_reencode, reencode_se_codecs
from opengwasdb.store import open_store, swap
from opengwasdb.store.arrays import ArrayRole, create_array, open_group, sharded_compressor

SE_TEMPORARY = ".se.reencoding"
SE_BACKUP = ".se.old"


def _write_manifest(tmp_path: Path, *, primary_layout: str = "dense") -> None:
    """A minimal 0.2.0 manifest, so `open_store` reads the fixture."""
    manifest = {
        "store_id": "example",
        "release_id": "observed-1",
        "format_version": "0.2.0",
        "primary_layout": primary_layout,
        "association_coverage": "gwas",
        "completion_state": "observed_only",
        "reference_assembly": "GRCh37",
        "encoding": {
            "version": 3,
            "z": {"kind": "float16"},
            "se": {"kind": "int8_residual", "residual_range": 2.0},
            "eaf": {"kind": "absent"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _buggy_codec(typesize: int) -> BloscCodec:
    """The codec the defect writes: the seam config with a fixed wrong `typesize`."""
    return BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", typesize=typesize)


def _non_default_codec(typesize: int) -> BloscCodec:
    """A non-seam Blosc config with a fixed wrong `typesize` (fix 4)."""
    return BloscCodec(cname="lz4", clevel=5, shuffle="shuffle", typesize=typesize)


def _make_store_with_codec(
    tmp_path: Path, se_codec: BloscCodec, index_codec: BloscCodec
) -> tuple[Path, np.ndarray, np.ndarray]:
    """A Dense fixture whose `se`/`se_exception_index` use the given codecs."""
    _write_manifest(tmp_path)
    group = open_group(tmp_path / "data.zarr", "w")
    rng = np.random.default_rng(0)
    se = rng.integers(-128, 128, size=(2100, 5)).astype(np.int8)
    exception_index = np.arange(12, dtype=np.int64)
    create_array(
        group,
        "se",
        ArrayRole.DENSE_STATISTIC_PLANE,
        data=se,
        compressor=se_codec,
    )
    create_array(
        group,
        "se_exception_index",
        ArrayRole.EXCEPTION_TABLE,
        data=exception_index,
        compressor=index_codec,
    )
    create_array(
        group,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        data=np.zeros((2100, 5), dtype=np.float16),
        compressor=sharded_compressor(),
    )
    return tmp_path, se, exception_index


def make_store_needing_se_reencode(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A Dense fixture with a buggy `se`/`se_exception_index` and a correct `z`."""
    return _make_store_with_codec(tmp_path, _buggy_codec(4), _buggy_codec(4))


def _se(store: Path) -> np.ndarray:
    return np.asarray(open_store(store).arrays()["se"][:])


def _exception_index(store: Path) -> np.ndarray:
    return np.asarray(open_store(store).arrays()["se_exception_index"][:])


def _entries(store: Path) -> set[str]:
    return {path.name for path in (store / "data.zarr").iterdir()}


def _file_snapshot(store: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(store)): path.read_bytes()
        for path in sorted((store).rglob("*"))
        if path.is_file()
    }


def _die_between_renames(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the re-encode and kill it between the two swap renames."""
    with monkeypatch.context() as patch:
        die_on_second_move(patch)
        with pytest.raises(ProcessDeath):
            reencode_se_codecs(store)


def _build_leftover_state(store: Path, names: tuple[str, ...]) -> None:
    """Rearrange `se` into one leftover-state set (copy/backup/canonical)."""
    data = store / "data.zarr"
    for extra in names[1:]:
        copytree(data / "se", data / extra)
    if names[0] != "se":
        (data / "se").rename(data / names[0])


def test_plan_reports_only_the_two_buggy_arrays(tmp_path: Path) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)

    plans = plan_se_reencode(store)

    assert [(p.array, p.current_typesize, p.expected_typesize) for p in plans] == [
        ("se", 4, 1),
        ("se_exception_index", 4, 8),
    ]
    assert all(p.group == "data.zarr" for p in plans)
    assert [p.dtype for p in plans] == ["int8", "int64"]
    assert all(p.before_bytes > 0 for p in plans)
    assert all(p.estimated_after_bytes >= 0 for p in plans)
    assert all(
        p.estimated_saving_bytes == max(0, p.before_bytes - p.estimated_after_bytes)
        for p in plans
    )
    # Planning must not have touched the store.
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)


def test_reencode_is_lossless_and_only_the_two_arrays_change(tmp_path: Path) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)
    before = _file_snapshot(store)

    results = reencode_se_codecs(store)

    assert [(r.array, r.before_bytes, r.after_bytes, r.equality, r.chunks) for r in results] == [
        ("se", results[0].before_bytes, results[0].after_bytes, True, 3),
        ("se_exception_index", results[1].before_bytes, results[1].after_bytes, True, 1),
    ]
    assert all(r.equality for r in results)
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)
    group = open_store(store).arrays()
    assert group["se"].compressors[0].typesize == 1
    assert group["se_exception_index"].compressors[0].typesize == 8

    after = _file_snapshot(store)
    changed = {path for path in before if before[path] != after.get(path)}
    assert changed, "the re-encoded arrays' bytes must change"
    assert all(
        path.startswith(("data.zarr/se/", "data.zarr/se_exception_index/")) for path in changed
    )
    # No staged or backup names may survive a completed swap.
    assert not _entries(store) & {SE_TEMPORARY, SE_BACKUP}


def test_reencode_is_idempotent(tmp_path: Path) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)

    assert len(reencode_se_codecs(store)) == 2
    after_first = _file_snapshot(store)
    assert reencode_se_codecs(store) == []
    assert _file_snapshot(store) == after_first
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)


def test_an_already_correct_store_needs_nothing(tmp_path: Path) -> None:
    store, _, _ = make_store_needing_se_reencode(tmp_path)
    reencode_se_codecs(store)

    assert plan_se_reencode(store) == []
    assert reencode_se_codecs(store) == []


def test_verification_rejects_a_corrupted_staged_array(tmp_path: Path) -> None:
    store, se, _ = make_store_needing_se_reencode(tmp_path)
    group = open_store(store).arrays(mode="r+")
    source = group["se"]
    target = reencode._stage_reencoded(group, source, SE_TEMPORARY)
    target[0:1, 0:1] = np.asarray(se[0:1, 0:1]) + 1

    with pytest.raises(reencode.SeReencodeError, match="not bit-identical"):
        reencode._verify_reencoded(source, target)

    # The live array is untouched; the failed copy is left for the caller to drop.
    assert np.array_equal(np.asarray(source[:]), se)


def test_verification_reads_the_shard_index_crc(tmp_path: Path) -> None:
    store, se, _ = make_store_needing_se_reencode(tmp_path)
    group = open_store(store).arrays(mode="r+")
    source = group["se"]
    target = reencode._stage_reencoded(group, source, SE_TEMPORARY)
    staged_dir = reencode._array_directory(group, SE_TEMPORARY)
    chunk_file = next(
        path for path in staged_dir.rglob("*") if path.is_file() and path.name != "zarr.json"
    )
    corrupted = bytearray(chunk_file.read_bytes())
    corrupted[-1] ^= 0xFF
    chunk_file.write_bytes(corrupted)

    with pytest.raises(ValueError, match="checksum"):
        reencode._verify_reencoded(source, target)

    assert np.array_equal(np.asarray(source[:]), se)


def test_a_death_between_the_renames_is_recovered_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)
    _die_between_renames(store, monkeypatch)
    # The state a death between the renames leaves: no canonical `se`, the
    # original under its backup name and the staged copy still beside it.
    entries = _entries(store)
    assert "se" not in entries and SE_BACKUP in entries and SE_TEMPORARY in entries

    results = reencode_se_codecs(store)

    assert [r.array for r in results] == ["se", "se_exception_index"]
    assert not _entries(store) & {SE_TEMPORARY, SE_BACKUP}
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)


def test_a_death_inside_the_recovery_leaves_a_state_the_next_run_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)
    _die_between_renames(store, monkeypatch)
    names = {"se", SE_TEMPORARY, SE_BACKUP}
    assert names & _entries(store) == {SE_TEMPORARY, SE_BACKUP}, "the run must stop between renames"

    # The recovery restores the backup first; killing after that move leaves the
    # array and its copy, which the next run settles by dropping the copy.
    real_move = swap.move_in_group

    def move_then_die(group: Any, source: str, dest: str) -> None:
        real_move(group, source, dest)
        raise ProcessDeath(f"killed after moving {source} -> {dest}")

    with monkeypatch.context() as patch:
        patch.setattr(swap, "move_in_group", move_then_die)
        with pytest.raises(ProcessDeath):
            reencode_se_codecs(store)
    assert names & _entries(store) == {"se", SE_TEMPORARY}

    results = reencode_se_codecs(store)

    assert [r.array for r in results] == ["se", "se_exception_index"]
    assert not _entries(store) & {SE_TEMPORARY, SE_BACKUP}
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)


def test_a_death_after_the_swap_leaves_a_backup_the_next_run_removes(tmp_path: Path) -> None:
    store, se, exception_index = make_store_needing_se_reencode(tmp_path)
    data = store / "data.zarr"
    copytree(data / "se", tmp_path / "original-se")
    reencode_se_codecs(store)
    reencoded = _se(store)
    # The state a death after the second rename leaves: the re-encoded array in
    # place, the original still under its backup name.
    copytree(tmp_path / "original-se", data / SE_BACKUP)

    assert reencode_se_codecs(store) == []

    assert SE_BACKUP not in _entries(store)
    assert np.array_equal(_se(store), reencoded)
    assert np.array_equal(_exception_index(store), exception_index)


#: The leftover states a single death cannot leave, as the names present.  Each
#: is built from the canonical array, and the re-encode must refuse it untouched.
IMPOSSIBLE_STATES = {
    "copy only": (SE_TEMPORARY,),
    "backup only": (SE_BACKUP,),
    "array, copy and backup": ("se", SE_TEMPORARY, SE_BACKUP),
}


@pytest.mark.parametrize("state", sorted(IMPOSSIBLE_STATES))
def test_a_state_no_single_death_leaves_is_refused_untouched(tmp_path: Path, state: str) -> None:
    store, se, _ = make_store_needing_se_reencode(tmp_path)
    _build_leftover_state(store, IMPOSSIBLE_STATES[state])
    assert {"se", SE_TEMPORARY, SE_BACKUP} & _entries(store) == set(IMPOSSIBLE_STATES[state])
    before = _file_snapshot(store)

    with pytest.raises(reencode.SeReencodeError, match="no single interrupted re-encode leaves"):
        reencode_se_codecs(store)

    assert _file_snapshot(store) == before


def test_plan_refuses_a_death_between_the_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _, _ = make_store_needing_se_reencode(tmp_path)
    _die_between_renames(store, monkeypatch)
    entries = _entries(store)
    assert "se" not in entries and SE_BACKUP in entries and SE_TEMPORARY in entries

    with pytest.raises(
        reencode.SeReencodeError, match="an interrupted re-encode must be settled"
    ):
        plan_se_reencode(store)

    # Planning is read-only: the half-done swap is left for apply mode to settle.
    assert _entries(store) == entries


@pytest.mark.parametrize("state", sorted(IMPOSSIBLE_STATES))
def test_plan_refuses_leftover_swap_artifacts(tmp_path: Path, state: str) -> None:
    store, _, _ = make_store_needing_se_reencode(tmp_path)
    _build_leftover_state(store, IMPOSSIBLE_STATES[state])
    before = _file_snapshot(store)

    with pytest.raises(
        reencode.SeReencodeError, match="an interrupted re-encode must be settled"
    ):
        plan_se_reencode(store)

    assert _file_snapshot(store) == before


@pytest.mark.parametrize("max_shards", [0, -1])
def test_plan_rejects_nonpositive_max_shards(tmp_path: Path, max_shards: int) -> None:
    store, _, _ = make_store_needing_se_reencode(tmp_path)

    with pytest.raises(reencode.SeReencodeError, match="max_shards must be >= 1"):
        plan_se_reencode(store, max_shards=max_shards)


def test_reencode_preserves_a_non_default_compressor(tmp_path: Path) -> None:
    store, se, exception_index = _make_store_with_codec(
        tmp_path, _non_default_codec(4), _non_default_codec(4)
    )

    results = reencode_se_codecs(store)

    assert [r.array for r in results] == ["se", "se_exception_index"]
    group = open_store(store).arrays()
    for name, itemsize in (("se", 1), ("se_exception_index", 8)):
        codec = group[name].compressors[0]
        assert int(codec.typesize) == itemsize, (name, codec.typesize)
        assert str(codec.cname) == "lz4", (name, codec.cname)
        assert int(codec.clevel) == 5, (name, codec.clevel)
        assert str(codec.shuffle) == "shuffle", (name, codec.shuffle)
    assert np.array_equal(_se(store), se)
    assert np.array_equal(_exception_index(store), exception_index)


def _make_hybrid_store_needing_se_reencode(
    tmp_path: Path,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """A Hybrid release: a buggy nested Dense Component and a correct Ragged Overflow."""
    _write_manifest(tmp_path, primary_layout="hybrid")
    dense = tmp_path / "dense"
    dense.mkdir()
    _write_manifest(dense, primary_layout="dense")

    dgroup = open_group(dense / "data.zarr", "w")
    rng = np.random.default_rng(0)
    se = rng.integers(-128, 128, size=(2100, 5)).astype(np.int8)
    exception_index = np.arange(12, dtype=np.int64)
    create_array(
        dgroup, "se", ArrayRole.DENSE_STATISTIC_PLANE, data=se, compressor=_buggy_codec(4)
    )
    create_array(
        dgroup,
        "se_exception_index",
        ArrayRole.EXCEPTION_TABLE,
        data=exception_index,
        compressor=_buggy_codec(4),
    )
    create_array(
        dgroup,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        data=np.zeros((2100, 5), dtype=np.float16),
        compressor=sharded_compressor(),
    )

    (tmp_path / "data.zarr").mkdir(parents=True, exist_ok=True)
    rgroup = open_group(tmp_path / "data.zarr" / "ragged", "w")
    ragged_se = np.zeros(50, dtype=np.float16)
    ragged_z = np.ones(50, dtype=np.float16)
    ragged_variant_index = np.arange(50, dtype=np.int32)
    for name, role, values in (
        ("se", ArrayRole.ASSOCIATION_SEQUENCE, ragged_se),
        ("z", ArrayRole.ASSOCIATION_SEQUENCE, ragged_z),
        ("variant_index", ArrayRole.ASSOCIATION_SEQUENCE, ragged_variant_index),
    ):
        create_array(rgroup, name, role, data=values, compressor=sharded_compressor())
    return tmp_path, se, exception_index, ragged_se


def test_hybrid_plans_and_applies_the_nested_dense_component_only(tmp_path: Path) -> None:
    store, se, exception_index, ragged_se = _make_hybrid_store_needing_se_reencode(tmp_path)
    dense_group = open_store(store / "dense").arrays()
    dense_z_before = np.asarray(dense_group["z"][:])
    ragged = store / "data.zarr" / "ragged"
    ragged_snapshot = _file_snapshot(ragged)
    ragged_before = {
        name: np.asarray(open_group(ragged, "r")[name][:])
        for name in ("se", "z", "variant_index")
    }

    plans = plan_se_reencode(store)

    assert [(p.group, p.array, p.current_typesize, p.expected_typesize) for p in plans] == [
        ("dense/data.zarr", "se", 4, 1),
        ("dense/data.zarr", "se_exception_index", 4, 8),
    ]

    results = reencode_se_codecs(store)

    assert [r.array for r in results] == ["se", "se_exception_index"]
    dense_group = open_store(store / "dense").arrays()
    assert int(dense_group["se"].compressors[0].typesize) == 1
    assert int(dense_group["se_exception_index"].compressors[0].typesize) == 8
    assert np.array_equal(np.asarray(dense_group["se"][:]), se)
    assert np.array_equal(np.asarray(dense_group["se_exception_index"][:]), exception_index)
    assert np.array_equal(np.asarray(dense_group["z"][:]), dense_z_before)

    # The Ragged Overflow and its non-target arrays are byte-for-byte untouched.
    assert _file_snapshot(ragged) == ragged_snapshot
    for name, expected in ragged_before.items():
        assert np.array_equal(np.asarray(open_group(ragged, "r")[name][:]), expected), name
    assert np.array_equal(
        np.asarray(open_group(ragged, "r")["se"][:]), ragged_se
    )

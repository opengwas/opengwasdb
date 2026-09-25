"""Unit tests for the memory-bounded off-reference key-run merge (ticket #222).

A key run carries no raw strings: the build-wide hash guarantee rests on each
hashed key's independent check hash, and the merge must release every input
run once folded. These tests pin the merge's assembly OR, the lowest-origin
rule, the collision refusal, the stream's bounded retention and the spill.
"""

from __future__ import annotations

import pickle
import weakref
from collections.abc import Iterator

import numpy as np
import pytest

from opengwasdb.layouts.hybrid import key_runs
from opengwasdb.layouts.hybrid.key_runs import HG19, HG38
from opengwasdb.layouts.hybrid.unknown_keys import HASH_TAG, UnknownKeyEncodingError

TAG = int(HASH_TAG)


def _run(
    values: list[int],
    bit: int,
    *,
    column: int = 0,
    checks: dict[int, int] | None = None,
) -> key_runs.KeyRun:
    """A column run; every tagged value is hashed, with ``checks[value]`` (default 1)."""
    hashed = [value for value in values if value & TAG]
    checks = checks or {}
    return key_runs.column_run(
        np.array(values, dtype=np.uint64),
        np.array(hashed, dtype=np.uint64),
        np.array([checks.get(value, 1) for value in hashed], dtype=np.uint64),
        column=column,
        assembly_bit=bit,
    )


def test_merge_ors_the_declaring_assemblies() -> None:
    """The same key in an hg19 run and an hg38 run carries both bits, which is
    what the two-assembly drop in ``resolve_keys`` keys off."""
    merged = key_runs.merge_runs(_run([5], HG19), _run([5], HG38))
    assert merged.values.tolist() == [5], "the fixture must be one key declared twice"
    assert merged.assembly_bits.tolist() == [HG19 | HG38]


def test_merge_repeats_one_assembly_without_widening_it() -> None:
    merged = key_runs.merge_runs(_run([5, 9], HG38), _run([1, 5], HG38))
    assert merged.values.tolist() == [1, 5, 9]
    assert merged.assembly_bits.tolist() == [HG38] * 3


def test_merge_keeps_the_lowest_origin_of_a_repeated_hashed_key() -> None:
    """A hashed key's raw string is later read from its lowest declaring
    column; a repeat with the same check is the same key, not a collision."""
    value = TAG | 7
    merged = key_runs.merge_runs(
        _run([value], HG38, column=9), _run([value, TAG | 3], HG38, column=4)
    )
    assert merged.hashed_values.tolist() == [TAG | 3, value]
    assert merged.hashed_origin.tolist() == [4, 4]


def test_merge_refuses_one_hashed_value_with_two_checks_naming_both_columns() -> None:
    """Two distinct raw keys forced onto one hash must fail the build-wide
    merge, wherever in the manifest they sit (#218 review). The run holds no
    raw strings, so it names the two columns whose side files hold them."""
    value = TAG | 7
    with pytest.raises(key_runs.HashedKeyCollision) as excinfo:
        key_runs.merge_runs(
            _run([value], HG38, column=3, checks={value: 11}),
            _run([value], HG38, column=8, checks={value: 12}),
        )
    assert excinfo.value.value == value
    assert excinfo.value.columns == (3, 8)


def test_column_run_refuses_a_collision_inside_one_column() -> None:
    value = TAG | 7
    with pytest.raises(key_runs.HashedKeyCollision) as excinfo:
        key_runs.column_run(
            np.array([value, value], dtype=np.uint64),
            np.array([value, value], dtype=np.uint64),
            np.array([11, 12], dtype=np.uint64),
            column=2,
            assembly_bit=HG38,
        )
    assert excinfo.value.columns == (2, 2)


def test_hashed_key_collision_survives_the_pool_boundary() -> None:
    """A worker's collision is pickled back to the parent, which reads the two
    columns' raw keys to name them; the fields must survive the trip."""
    restored = pickle.loads(pickle.dumps(key_runs.HashedKeyCollision(TAG | 7, (3, 8))))
    assert isinstance(restored, UnknownKeyEncodingError)
    assert (restored.value, restored.columns) == (TAG | 7, (3, 8))


def _random_run(rng: np.random.Generator, column: int) -> key_runs.KeyRun:
    """Keys from a small universe so runs overlap heavily; a value's check is
    a function of the value, so repeats are the same key, never a collision."""
    values = rng.integers(0, 400, size=150).astype(np.uint64)
    values[values % 5 == 0] |= np.uint64(TAG)
    distinct = sorted(set(values.tolist()))
    return _run(
        distinct,
        int(rng.choice([HG19, HG38])),
        column=column,
        checks={value: value ^ 0x5555 for value in distinct},
    )


def _reference_merge(runs: list[key_runs.KeyRun]) -> tuple[dict[int, int], dict[int, int]]:
    """``{value: OR of bits}`` and ``{hashed value: lowest origin}``, by dict."""
    bits: dict[int, int] = {}
    origins: dict[int, int] = {}
    for run in runs:
        for value, bit in zip(run.values.tolist(), run.assembly_bits.tolist(), strict=True):
            bits[value] = bits.get(value, 0) | bit
        for value, origin in zip(
            run.hashed_values.tolist(), run.hashed_origin.tolist(), strict=True
        ):
            origins[value] = min(origins.get(value, origin), origin)
    return bits, origins


def test_merge_key_stream_matches_a_dict_reduction() -> None:
    """The streamed pairwise merge is the same reduction the dicts computed:
    every distinct key, the OR of its assemblies, the lowest origin column."""
    rng = np.random.default_rng(222)
    runs = [_random_run(rng, column) for column in range(13)]
    bits, origins = _reference_merge(runs)
    assert HG19 | HG38 in bits.values(), "the fixture must hold two-assembly keys"
    assert len(set(origins.values())) > 1, "the fixture must spread origins"
    merged = key_runs.merge_key_stream(iter(runs))
    assert merged.values.tolist() == sorted(bits)
    assert merged.assembly_bits.tolist() == [bits[value] for value in sorted(bits)]
    assert merged.hashed_values.tolist() == sorted(origins)
    assert merged.hashed_origin.tolist() == [origins[value] for value in sorted(origins)]
    assert merged.hashed_check.tolist() == [value ^ 0x5555 for value in sorted(origins)]


def test_merge_key_stream_of_nothing_is_empty() -> None:
    merged = key_runs.merge_key_stream(iter([]))
    assert merged.size == 0
    assert len(merged.hashed_values) == 0


def test_merge_key_stream_detects_a_collision_between_distant_runs() -> None:
    value = TAG | 7
    filler = [_run([position], HG38, column=position) for position in range(1, 7)]
    runs = [
        _run([value], HG38, column=0, checks={value: 11}),
        *filler,
        _run([value], HG38, column=7, checks={value: 12}),
    ]
    with pytest.raises(key_runs.HashedKeyCollision) as excinfo:
        key_runs.merge_key_stream(iter(runs))
    assert excinfo.value.columns == (0, 7)


def _tracked(runs: list[key_runs.KeyRun], live_counts: list[int]) -> Iterator[key_runs.KeyRun]:
    """Yield ``runs``, recording before each how many earlier ones survive.

    Only the stream holds a strong reference to a run after yielding it, so a
    surviving earlier run is one the consumer is still retaining.
    """
    alive: list[weakref.ref[np.ndarray]] = []
    while runs:
        live_counts.append(sum(ref() is not None for ref in alive))
        run = runs.pop(0)
        alive.append(weakref.ref(run.values))
        yield run
        del run


def test_merge_key_stream_releases_each_run_once_folded() -> None:
    """Review round 1 blocker: ``merge(list(stream))`` retained every worker
    result at once. With runs of similar size the fold may keep at most one
    unmerged run while the next arrives."""
    rng = np.random.default_rng(7)
    runs = [_random_run(rng, column) for column in range(32)]
    live_counts: list[int] = []
    merged = key_runs.merge_key_stream(_tracked(runs, live_counts))
    assert len(live_counts) == 32, "the fixture must stream every run"
    assert merged.size > 150
    assert max(live_counts) <= 1, f"earlier runs retained: {live_counts}"


def test_run_spill_round_trips(tmp_path) -> None:
    value = TAG | 7
    run = key_runs.merge_runs(_run([5, value], HG19, column=2, checks={value: 99}), _run([5], HG38))
    path = tmp_path / "keyrun.0.npz"
    key_runs.write_run(run, path)
    loaded = key_runs.read_run(path)
    for field in ("values", "assembly_bits", "hashed_values", "hashed_check", "hashed_origin"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(run, field))
        assert getattr(loaded, field).dtype == getattr(run, field).dtype


def test_run_spill_refuses_arrays_that_do_not_pair_up(tmp_path) -> None:
    path = tmp_path / "keyrun.0.npz"
    np.savez(
        path,
        values=np.array([1, 2], dtype=np.uint64),
        assembly_bits=np.array([HG38], dtype=np.int8),
        hashed_values=np.empty(0, dtype=np.uint64),
        hashed_check=np.empty(0, dtype=np.uint64),
        hashed_origin=np.empty(0, dtype=np.int32),
    )
    with pytest.raises(UnknownKeyEncodingError, match="refusing to merge it"):
        key_runs.read_run(path)


def test_sorted_distinct_matches_np_unique() -> None:
    values = np.array([9, 3, 3, 7, 9, 1], dtype=np.uint64)
    np.testing.assert_array_equal(key_runs.sorted_distinct(values), np.unique([9, 3, 3, 7, 9, 1]))


def test_sorted_distinct_of_distinct_values_sorts_them() -> None:
    values = np.array([9, 3, 7, 1], dtype=np.uint64)
    np.testing.assert_array_equal(key_runs.sorted_distinct(values), [1, 3, 7, 9])

"""`repair_eaf_chunks` survives a process that dies mid-swap (#244 review, finding 3).

The repair swaps a rechunked copy in with two renames: `name -> .name.old`, then
`.name.rechunking -> name`, then deletes the backup. Each rename is atomic; the
pair is not. An exception between them rolls back, but a process that dies there
(a kill, an out-of-memory, a power cut) leaves the published release without
`name` at all. Before this fix the next run skipped the array as absent,
reported nothing repaired and left the release broken.

The next run now recovers deterministically from whichever state a death left,
and refuses a state no single death can leave.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_eaf_encoding_layouts import make_store_needing_eaf_repair

import opengwasdb.repair as repair
from opengwasdb.encoding import EAF_BASELINE
from opengwasdb.repair import repair_eaf_chunks
from opengwasdb.store import open_store
from opengwasdb.validation import validate_store

TEMPORARY = f".{EAF_BASELINE}.rechunking"
BACKUP = f".{EAF_BASELINE}.old"


class _ProcessDeath(BaseException):
    """Stands in for a process dying: `except Exception` rollbacks do not see it."""


def _baseline(store: Path) -> np.ndarray:
    return np.asarray(open_store(store).arrays()[EAF_BASELINE][:])


def _entries(store: Path) -> set[str]:
    return {p.name for p in (store / "data.zarr").iterdir()}


def _die_on_second_move(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    real = repair.move_in_group

    def move(group: Any, source: str, dest: str) -> None:
        calls.append((source, dest))
        if len(calls) == 2:
            raise _ProcessDeath(f"killed before moving {source} -> {dest}")
        real(group, source, dest)

    monkeypatch.setattr(repair, "move_in_group", move)


def test_a_death_between_the_renames_is_recovered_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store_needing_eaf_repair(tmp_path)
    original = _baseline(store)
    assert np.isfinite(original).any()

    with monkeypatch.context() as patch:
        _die_on_second_move(patch)
        with pytest.raises(_ProcessDeath):
            repair_eaf_chunks(store)
    # The state a death leaves: no canonical array, the original under its backup name.
    entries = _entries(store)
    assert EAF_BASELINE not in entries and BACKUP in entries and TEMPORARY in entries

    repaired = repair_eaf_chunks(store)

    assert [(item.old_chunk, item.new_chunk) for item in repaired] == [(8, 3)]
    assert not _entries(store) & {BACKUP, TEMPORARY}
    np.testing.assert_array_equal(_baseline(store), original)
    assert validate_store(store).ok


def test_a_death_after_the_swap_leaves_a_backup_the_next_run_removes(tmp_path: Path) -> None:
    store = make_store_needing_eaf_repair(tmp_path)
    data = store / "data.zarr"
    shutil.copytree(data / EAF_BASELINE, tmp_path / "original-baseline")
    repair_eaf_chunks(store)
    repaired = _baseline(store)
    # The state a death after the second rename leaves: the new array in place,
    # the original still under its backup name.
    shutil.copytree(tmp_path / "original-baseline", data / BACKUP)

    assert repair_eaf_chunks(store) == []

    assert BACKUP not in _entries(store)
    np.testing.assert_array_equal(_baseline(store), repaired)
    assert validate_store(store).ok


#: The leftover states a single death cannot leave, as the names present. Each
#: is built from the canonical array, and the repair must refuse it untouched.
#: The copy alone is the review's case (round 2): the old recovery deleted it,
#: the only remaining baseline, and returned [].
IMPOSSIBLE_STATES = {
    "copy only": (TEMPORARY,),
    "backup only": (BACKUP,),
    "array, copy and backup": (EAF_BASELINE, TEMPORARY, BACKUP),
}


@pytest.mark.parametrize("state", sorted(IMPOSSIBLE_STATES))
def test_a_state_no_single_death_leaves_is_refused_untouched(tmp_path: Path, state: str) -> None:
    store = make_store_needing_eaf_repair(tmp_path)
    data = store / "data.zarr"
    names = IMPOSSIBLE_STATES[state]
    for extra in names[1:]:
        shutil.copytree(data / EAF_BASELINE, data / extra)
    if names[0] != EAF_BASELINE:
        (data / EAF_BASELINE).rename(data / names[0])
    assert {EAF_BASELINE, TEMPORARY, BACKUP} & _entries(store) == set(names)
    before = {p: p.read_bytes() for p in sorted(data.rglob("*")) if p.is_file()}

    with pytest.raises(RuntimeError, match="no single interrupted repair leaves"):
        repair_eaf_chunks(store)

    assert {p: p.read_bytes() for p in sorted(data.rglob("*")) if p.is_file()} == before

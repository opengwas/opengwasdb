"""Staged in-place swap for a Store array, with recovery from a death mid-swap.

`opengwasdb.repair` and `opengwasdb.reencode` both replace one array with a
freshly written copy staged beside it in the same group.  The copy is verified
before the swap, then swapped in with two renames -- ``name -> backup``,
``copy -> name`` -- so the original survives until the copy has landed.  A
process that dies between the renames leaves a state the next run settles; a
state no single death leaves is refused rather than guessed.

The staged and backup names differ per operation (``.{name}.rechunking`` /
``.{name}.reencoding`` and ``.{name}.old``), so the caller supplies
``swap_names`` and the wording the refusal and warnings use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from opengwasdb.store.arrays import move_in_group

log = logging.getLogger(__name__)


def swap_staged(group: Any, name: str, temporary: str, backup: str) -> None:
    """Swap `temporary` into `name`, keeping the original under `backup` until it lands."""
    move_in_group(group, name, backup)
    try:
        move_in_group(group, temporary, name)
    except Exception:
        move_in_group(group, backup, name)
        raise
    del group[backup]


def recover_interrupted_swap(
    group: Any,
    name: str,
    label: str,
    *,
    swap_names: Callable[[str], tuple[str, str]],
    refusal: Callable[[str], Exception],
    copied: str,
    operation: str,
) -> None:
    """Put `name` back in one piece after a staged swap interrupted mid-rename.

    Each rename is atomic but the pair is not, and an exception handler cannot
    roll back a process that died.  The one safe reading of each state a death
    leaves is settled here; any other state is refused with nothing touched.
    """
    temporary, backup = swap_names(name)
    roles = {"array": name, "copy": temporary, "backup": backup}
    present = frozenset(role for role, entry in roles.items() if entry in group)
    if present <= frozenset({"array"}):
        return
    if present == frozenset({"array", "copy"}):
        del group[temporary]
        log.warning("%s/%s: removed the unfinished %s copy %s", label, name, copied, temporary)
        return
    if present == frozenset({"copy", "backup"}):
        move_in_group(group, backup, name)
        del group[temporary]
        log.warning(
            "%s/%s: restored the original from %s (a dead %s)", label, name, backup, operation
        )
        return
    if present == frozenset({"array", "backup"}):
        del group[backup]
        log.warning("%s/%s: removed %s left after a completed swap", label, name, backup)
        return
    found = ", ".join(roles[role] for role in sorted(present))
    raise refusal(
        f"{label}/{name}: found {found}, a state no single interrupted {operation} "
        "leaves; it cannot tell which is the original. Inspect them and remove the stale ones."
    )

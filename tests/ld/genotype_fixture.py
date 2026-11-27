"""The committed Genotype Store, placed where a test can open it (opengwas/ogquery#11).

`data/test-panel/genotype-store/` is the Test Panel's Genotype Store, the bytes ogquery's own tests
read (`data/test-panel/genotype-store.SOURCE`). It pins the pgen, psam and pvar beside it by mtime,
which git does not keep, and an open refuses a changed one. `place` copies the store and its
inputs (all five a build read) into a tree and gives each input the recorded mtime; a test opens
that copy and never the committed files. A build test that needs a store built from scratch uses
the session-built `genotype_store` fixture instead.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_PANEL = HERE / "data" / "test-panel"
COMMITTED = TEST_PANEL / "genotype-store"
#: What the store pins, beside it (`../panel.*`), and what a build of it reads besides.
PINNED = ("panel.pgen", "panel.psam", "panel.pvar")
BUILT_FROM = (*PINNED, "panel.keep", "pcs.eigenvec")


def pin_mtimes(tree: Path, mtime_ns: int) -> None:
    """Give every input of the build in `tree` the mtime a store records for it."""
    for name in BUILT_FROM:
        if (tree / name).exists():
            os.utime(tree / name, ns=(mtime_ns, mtime_ns))


def recorded_mtime_ns(store: Path) -> int:
    """The one mtime the store records for its pinned inputs: every pin carries it."""
    record = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"]
    mtimes = {
        fileset[name]["mtime_ns"]
        for fileset in record["genotype_store"]["filesets"]
        for name in ("pgen", "psam", "pvar")
    }
    assert len(mtimes) == 1, f"the store's pins carry more than one mtime: {sorted(mtimes)}"
    return mtimes.pop()


def place(tree: Path) -> Path:
    """A copy of the committed store under `tree`, beside its pinned inputs at the recorded mtime;
    the path of the copy."""
    store = tree / COMMITTED.name
    shutil.copytree(COMMITTED, store)
    for (
        name
    ) in BUILT_FROM:  # a read opens the pins; validation also checks the PC File and keep list
        shutil.copy2(TEST_PANEL / name, tree / name)
    pin_mtimes(tree, recorded_mtime_ns(store))
    return store

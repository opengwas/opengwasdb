"""The derivation at eight workers, on the committed Test Store (partner-rows §6.3; ogld#69).

The bytes a derivation writes must not depend on how many workers made them, and the widest
setting is the one worth holding: eight spawn workers, each an interpreter that imports ogld and
opens the store itself. That costs about a second of start-up a worker, which the default tier's
30 s cannot pay for, so it is held here — the tier `CONTRIBUTING.md` gives a suite that is too
slow for it (`realdata`, which `pixi run -e dev test-realdata` runs and `tests/ld/realdata/run.sh`
names). The narrower settings, one worker and two, stay in the default tier with the rest of the
derivation's tests.

Nothing here reads the panels: the store is the committed Test Store, put back to 0.1.0 by
`tests/ld/store_copies.py`, and the bytes compared are the ones its writer built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from store_copies import TEST_STORE, unpartnered

from opengwasdb.ld.store.assembly import derive_partners

#: The tests directory, which holds `store_copies`: a realdata module is imported by path, so its
#: own directory is on `sys.path` and the tier's parent is not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The Test Store's chromosomes, its Partner Threshold, and the two chunk files whose bytes a
#: derivation must reproduce (partner-rows §3, §5.1).
CHROMOSOMES = ("21", "22")
T_PARTNER = 0.01
FILES = ("frames/c/0", "frames_idx/c/0")


def chunk(store: Path, chrom: str, name: str) -> bytes:
    """One of a chromosome's two chunk files, as bytes."""
    return (store / "partners" / chrom / name).read_bytes()


@pytest.mark.parametrize(("threads", "processes"), [(8, 8)])
def test_eight_workers_write_the_builds_bytes(tmp_path: Path, threads: int, processes: int) -> None:
    """Eight compressor threads on eight chromosome processes write the committed store's frames,
    byte for byte (partner-rows §6.3, §9.1). The pool takes one worker a chromosome at most, so a
    two-chromosome store derives two at once and the other six start and stop."""
    store = unpartnered(tmp_path / "store")
    made = derive_partners(store, t_partner=T_PARTNER, threads=threads, processes=processes)
    assert made is not None
    for chrom in CHROMOSOMES:
        for name in FILES:
            assert chunk(store, chrom, name) == chunk(TEST_STORE, chrom, name), f"{chrom}/{name}"

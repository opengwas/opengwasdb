"""The engine's acceptance of LD Stores on whole panels (ogld#7): every validation rule passes on
each, each holds chromosomes 1 to 22, and the direct decoder returns the same bytes as zarr on each
(ogld#40, ADR 0095). These checks need the engine's internals, so they stay here.

Any LD Store: `$OPENGWASDB_LD_STORE` names one, at any path, such as a Store rebuilt by
opengwas/opengwasdb-stores. Unset, the run checks the five published HGDP+1kGP LD Stores,
`build_hgdp1kgp.store_path` under `$OPENGWASDB_LD_STORES`, or `$OPENGWASDB_LD_STORE_<ANC>` for
one elsewhere. Their build, and the checks of what `opengwasdb ld info` shows of them (panel, n,
scale, k, the PC Check), are opengwasdb-stores' (`tools/ld-stores/`, opengwasdb-stores#541).
Validating all five takes about an hour of the node (`pixi run -e dev ld-acceptance-stores`),
which is why this is an acceptance run and not the realdata tier. The decoder equality reads every
array of each store, so it belongs here too: it took the realdata tier over its five-minute
budget. A Store missing fails the run; it never skips (CONTRIBUTING.md, "Acceptance runs").
"""

from __future__ import annotations

from pathlib import Path

import build_hgdp1kgp as conversion
import numpy as np
import pytest

import opengwasdb.ld
from opengwasdb.ld import env
from opengwasdb.ld.format import VALIDATION_RULES
from opengwasdb.ld.store.layout import chromosomes_listed
from opengwasdb.ld.store.location import ChunkCache, open_group
from opengwasdb.ld.store.validation import validate_at

#: ADR 0088, decision 11: each panel's unrelated samples (`scripts/ld/make_keeps.py`).
PANELS = {"EUR": 665, "AFR": 608, "EAS": 724, "SAS": 672, "AMR": 341}
AUTOSOMES = [str(c) for c in range(1, 23)]


def stores() -> dict[str, str]:
    """The Stores this run checks, by name: `$OPENGWASDB_LD_STORE`, or the five published."""
    one = env.read("OPENGWASDB_LD_STORE", "")
    if one:
        return {Path(one).name: one}
    return {
        panel: env.read(f"OPENGWASDB_LD_STORE_{panel}", str(conversion.store_path(panel)))
        for panel in PANELS
    }


STORES = stores()


def store_path(name: str) -> Path:
    path = Path(STORES[name])
    assert path.exists(), f"the {name} store is not at {path}: set $OPENGWASDB_LD_STORE"
    return path


def keep_count(panel: str) -> int:
    keep = conversion.keep_file(panel)
    assert keep.exists(), f"{keep} is missing: scripts/ld/make_keeps.py writes it"
    return len(conversion.keep_iids(keep))


@pytest.mark.parametrize("panel", STORES)
def test_n_is_the_panels_unrelated_samples(panel: str) -> None:
    """n is the panel's unrelated samples: the literal count, and its keep file's."""
    p = opengwasdb.ld.open(store_path(panel)).provenance
    ancestry = p["panel"]["ancestry"]
    assert p["panel"]["n"] == PANELS[ancestry] == keep_count(ancestry)


@pytest.mark.parametrize("panel", STORES)
def test_sampled_chunks_equal_zarr(panel: str) -> None:
    """The direct decoder's first, middle and last chunk of every array in the store equals
    zarr's own read of it (ogld#40, ADR 0095)."""
    import zarr

    path = store_path(panel)
    cache = ChunkCache(1 << 25)
    root = open_group(path, cache=cache)
    arrays = [a for _, a in root.members(max_depth=None) if isinstance(a, zarr.Array)]
    assert len(arrays) > 100
    for array in arrays:
        size, n = int(array.chunks[0]), int(array.shape[0])
        if n == 0:
            np.testing.assert_array_equal(cache.take(array, []), array[:])
            continue
        last = (n - 1) // size
        for k in sorted({0, last // 2, last}):
            coords = np.arange(k * size, min(n, (k + 1) * size))
            expected = np.asarray(array[k * size : (k + 1) * size])
            got = cache.take(array, coords)
            assert got.dtype == expected.dtype, array.path
            np.testing.assert_array_equal(got, expected, err_msg=f"{panel}/{array.path}/{k}")


@pytest.mark.parametrize("panel", STORES)
def test_every_rule_passes(panel: str) -> None:
    threads = int(env.read("OPENGWASDB_LD_VALIDATE_THREADS", "16"))
    found = validate_at(store_path(panel), threads=threads)
    assert list(found) == list(VALIDATION_RULES)
    failing = {rule: findings[:3] for rule, findings in found.items() if findings}
    assert not failing, f"{panel} breaks {len(failing)} rule(s): {failing}"


@pytest.mark.parametrize("panel", STORES)
def test_it_holds_the_autosomes(panel: str) -> None:
    held = chromosomes_listed(open_group(store_path(panel))["variants"].attrs["chromosomes"])
    assert [name for name, _, _ in held] == AUTOSOMES, f"{panel} holds chromosomes {held}"
    assert all(stop > start for _, start, stop in held), "a chromosome with no variants"

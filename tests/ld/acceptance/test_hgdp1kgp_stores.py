"""Epic #1's acceptance for the five HGDP+1kGP LD Stores (ogld#7): every validation rule passes
on each, each records what a reader of its provenance needs, and the direct decoder returns the
same bytes as zarr on each (ogld#40, ADR 0095).

The stores are the ones `scripts/ld/build_hgdp1kgp.py build` writes under `$OGLD_STORES`; its
constants and paths (`recipe`) say the scale, k, names and roots, and `$OGLD_STORE_<ANC>` names one
elsewhere. `$OGLD_KEEP` holds the keep files, so n is checked against the unrelated samples, not
against what the store says of itself. Each store must hold chromosomes 1 to 22.
Validating all five takes about an hour of the node (`pixi run -e dev acceptance-stores`),
which is why this is an acceptance run and not the realdata tier. The decoder equality reads every
array of each store, so it belongs here too: it took the realdata tier over its five-minute budget.
An input missing fails the run; it never skips (CONTRIBUTING.md, "Acceptance runs").
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import build_hgdp1kgp as recipe
import numpy as np
import pytest

import opengwasdb.ld
from opengwasdb.ld.format import VALIDATION_RULES
from opengwasdb.ld.store.layout import chromosomes_listed
from opengwasdb.ld.store.location import ChunkCache, open_group
from opengwasdb.ld.store.validation import validate_at

#: ADR 0088, decision 11: each panel's unrelated samples (`scripts/ld/make_keeps.py`).
PANELS = {"EUR": 665, "AFR": 608, "EAS": 724, "SAS": 672, "AMR": 341}
K_MIN = 5  # build-inputs §1's default `k_min`
AUTOSOMES = [str(c) for c in recipe.CHROMS]


def store_path(panel: str) -> Path:
    path = Path(os.environ.get(f"OGLD_STORE_{panel}", recipe.store_path(panel)))
    assert path.exists(), f"the {panel} store is not at {path}: build it or set $OGLD_STORE_{panel}"
    return path


def keep_count(panel: str) -> int:
    keep = recipe.keep_file(panel)
    assert keep.exists(), f"{keep} is missing: scripts/make_keeps.py writes it"
    return len(recipe.keep_iids(keep))


@pytest.mark.parametrize("panel", PANELS)
def test_sampled_chunks_of_published_stores_equal_zarr(panel: str) -> None:
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


@pytest.mark.parametrize("panel", PANELS)
def test_every_rule_passes(panel: str) -> None:
    threads = int(os.environ.get("OGLD_VALIDATE_THREADS", "16"))
    found = validate_at(store_path(panel), threads=threads)
    assert list(found) == list(VALIDATION_RULES)
    failing = {rule: findings[:3] for rule, findings in found.items() if findings}
    assert not failing, f"{panel} breaks {len(failing)} rule(s): {failing}"


@pytest.mark.parametrize("panel", PANELS)
def test_provenance_records_what_a_reader_needs(panel: str) -> None:
    p = opengwasdb.ld.open(store_path(panel)).provenance
    assert p["panel"]["id"] == recipe.panel_id(panel)
    assert p["panel"]["ancestry"] == panel
    assert p["panel"]["n"] == PANELS[panel] == keep_count(panel)
    assert p["encoding"]["r_scale"] == recipe.PANEL_SCALE
    assert p["n_variants"] > 0
    held = chromosomes_listed(open_group(store_path(panel))["variants"].attrs["chromosomes"])
    assert [name for name, _, _ in held] == AUTOSOMES, f"{panel} holds chromosomes {held}"
    assert all(stop > start for _, start, stop in held), "a chromosome with no variants"
    assert p["counts"]["near_pairs"] > 0, "the Near Band's pair count"
    # These five are 0.1.0 stores, so they still record the Proxy Index they hold: #73 derives
    # their Partner Rows and drops it, and this assertion goes with the last of them (#72).
    assert p["counts"]["index_entries"] > 0, "a 0.1.0 store's Proxy Index, until #73 drops it"
    t_far = p["t_far"]
    assert isinstance(t_far, float), f"t_far is {t_far!r}"
    assert 0 < t_far < 1, f"t_far is {t_far!r}"
    build = p["build"]
    assert build["wall_seconds"] > 0, "the build time"
    assert build["peak_memory_bytes"] > 0
    assert p["software"]["ogld"], "the builder's version"


@pytest.mark.parametrize("panel", PANELS)
def test_the_pc_check_and_its_diagnostic_are_recorded(panel: str) -> None:
    adjustment = opengwasdb.ld.open(store_path(panel)).provenance["adjustment"]
    assert adjustment["k_target"] == recipe.K
    assert adjustment["k_min"] == K_MIN
    dropped = adjustment["pc_check"]["dropped"]
    assert isinstance(dropped, list), "the PC Check's dropped PCs are recorded, even when none"
    assert K_MIN <= adjustment["k"] <= recipe.K, f"{panel} kept {adjustment['k']} PCs"
    assert len(adjustment["covariates"]) == adjustment["k"]
    diagnostic = opengwasdb.ld.open(store_path(panel)).provenance["pc_diagnostic"]
    assert diagnostic["pairs"]["n_pairs"] > 0
    for rate in (
        "rate_r2_ge_0_01_unadjusted",
        "rate_r2_ge_0_01_adjusted",
        "rate_r2_ge_0_01_sampling",
    ):
        assert math.isfinite(diagnostic[rate]), f"{rate} is {diagnostic[rate]!r}"
    change = diagnostic["near_100kb_change"]
    assert set(change) == {"0", "0.5", "0.9", "0.99", "1"}
    assert all(math.isfinite(v) for v in change.values())

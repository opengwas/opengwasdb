"""The `clumps.*` Validation Rules, each against the corrupted Store it names (spec §20.0, #418).

A clumped Store passes every rule, the re-clump included; each corruption below is one of the
silent failures §20 lists, and fails the rule that names it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import zarr
from clump_stores import LD_STORE, build_dense, build_dense_completed, build_hybrid, copied

from opengwasdb.clumps import clump_store
from opengwasdb.model.analyses import write_analysis_columns
from opengwasdb.store.arrays import create_array, open_group
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.validation import validate_store
from opengwasdb.validation.rules.clumps import RULES

EUR = {"EUR": LD_STORE}
CLUMP_RULES = [rule.name for rule in RULES]
CLUMPS = Path("data.zarr/top_hits/p_5e_08/clumps/default")


@pytest.fixture(scope="module")
def clumped(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("clump-rules")
    stores = {
        "completed": build_dense_completed(root / "completed"),
        "hybrid": build_hybrid(root / "hybrid"),
    }
    for store in stores.values():
        clump_store(store, EUR)
    stores["unclumped"] = build_dense(root / "dense" / "dense.opengwasdb")
    return stores


def _failed(store: Path, ld_stores: Mapping[str, Path] | None = None) -> set[str]:
    return set(validate_store(store, rules=CLUMP_RULES, ld_stores=ld_stores).failed())


def _arrays(store: Path) -> zarr.Group:
    return zarr.open_group(str(store / CLUMPS), mode="r+")


def _row(store: Path, analysis_id: str) -> int:
    rows = read_analyses(store / "analyses.tsv").rows
    return next(int(r["analysis_index"]) for r in rows if r["analysis_id"] == analysis_id)


def _slice(store: Path, analysis_id: str) -> tuple[int, int]:
    offsets = np.asarray(
        zarr.open_group(str(store / "data.zarr/top_hits/p_5e_08"), mode="r")["analysis_offsets"][:]
    )
    index = _row(store, analysis_id)
    return int(offsets[index]), int(offsets[index + 1])


def _set_counts(store: Path, analysis_id: str, n_loci: str) -> None:
    rows = sorted(read_analyses(store / "analyses.tsv").rows,
                  key=lambda r: int(r["analysis_index"]))
    write_analysis_columns(store / "analyses.tsv", {
        "n_loci_5e8": [n_loci if r["analysis_id"] == analysis_id else r["n_loci_5e8"]
                       for r in rows],
    })


# ── a clean Store passes ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["completed", "hybrid"])
def test_a_clumped_store_is_valid_and_its_reclump_agrees(
    clumped: dict[str, Path], name: str
) -> None:
    result = validate_store(clumped[name], ld_stores=EUR)
    assert result.ok, result.errors
    ran = {f.rule for f in result.findings if f.severity == "skip"}
    assert not ran & set(CLUMP_RULES), [f for f in result.findings if f.rule in CLUMP_RULES]


def test_without_ld_stores_the_reclump_is_recorded_as_not_run(clumped: dict[str, Path]) -> None:
    result = validate_store(clumped["completed"], rules=CLUMP_RULES)
    assert result.ok, result.errors
    skips = [f for f in result.findings if f.rule == "clumps.reclump"]
    assert skips and all(f.severity == "skip" and "not run" in f.message for f in skips)


# ── clumps.declared ──────────────────────────────────────────────────────────


def test_a_group_the_manifest_does_not_declare_fails_declared(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["completed"], tmp_path)
    manifest = json.loads((store / "manifest.json").read_text())
    del manifest["clumps"]
    (store / "manifest.json").write_text(json.dumps(manifest))
    assert _failed(store) == {"clumps.declared"}


def test_columns_without_the_clumps_fail_declared(clumped: dict[str, Path], tmp_path: Path) -> None:
    store = copied(clumped["unclumped"], tmp_path)
    assert _failed(store) == set()
    n = len(read_analyses(store / "analyses.tsv").rows)
    write_analysis_columns(store / "analyses.tsv", {"n_loci_5e8": ["1"] * n,
                                                    "n_unclumped_5e8": ["0"] * n})
    assert _failed(store) == {"clumps.declared"}


def test_a_hybrid_declaring_the_clumps_in_one_manifest_fails_declared(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["hybrid"], tmp_path)
    dense = store / "dense" / "manifest.json"
    manifest = json.loads(dense.read_text())
    del manifest["clumps"]
    dense.write_text(json.dumps(manifest))
    assert "clumps.declared" in _failed(store)


# ── clumps.structure ─────────────────────────────────────────────────────────


def test_arrays_of_another_length_than_the_tier_fail_structure(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["completed"], tmp_path)
    group = open_group(store / CLUMPS, "r+")
    lead = np.asarray(group["lead"][:])
    create_array(group, "lead", ArrayRole.TOP_HIT_INDEX, data=lead[:-1], dtype="int32",
                 overwrite=True)
    assert _failed(store) == {"clumps.structure"}


# ── clumps.markers ───────────────────────────────────────────────────────────


def test_an_imputed_row_clumped_fails_markers(clumped: dict[str, Path], tmp_path: Path) -> None:
    store = copied(clumped["completed"], tmp_path)
    a, b = _slice(store, "b")
    imputed = np.asarray(zarr.open_group(str(store / "data.zarr/top_hits/p_5e_08"),
                                         mode="r")["imputed"][a:b]).astype(bool)
    k = int(np.flatnonzero(imputed)[0])
    arrays = _arrays(store)
    arrays["lead"][a + k] = k
    arrays["r"][a + k] = 511
    assert "clumps.markers" in _failed(store)


# ── clumps.greedy ────────────────────────────────────────────────────────────


def test_a_lead_weaker_than_its_member_fails_greedy(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["completed"], tmp_path)
    a, b = _slice(store, "a")
    arrays = _arrays(store)
    lead = np.asarray(arrays["lead"][a:b])
    lead_offset = next(k for k in range(len(lead))
                       if lead[k] == k and ((lead == k).sum() > 1))
    member = int(np.flatnonzero((lead == lead_offset) & (np.arange(len(lead)) != lead_offset))[0])
    swapped = np.where(lead == lead_offset, member, lead)
    arrays["lead"][a:b] = swapped.astype(np.int32)
    assert _failed(store) == {"clumps.greedy"}


# ── clumps.ancestry ──────────────────────────────────────────────────────────


def test_an_analysis_without_an_ld_store_counted_as_0_fails_ancestry(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["completed"], tmp_path)
    _set_counts(store, "c", "0")
    assert _failed(store) == {"clumps.ancestry"}


# ── clumps.counts ────────────────────────────────────────────────────────────


def test_a_count_that_disagrees_with_the_arrays_fails_counts(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clumped["completed"], tmp_path)
    loci = next(r["n_loci_5e8"] for r in read_analyses(store / "analyses.tsv").rows
                if r["analysis_id"] == "a")
    _set_counts(store, "a", str(int(loci) + 1))
    assert _failed(store) == {"clumps.counts"}


# ── clumps.reclump ───────────────────────────────────────────────────────────


def test_a_sign_not_flipped_fails_reclump(clumped: dict[str, Path], tmp_path: Path) -> None:
    store = copied(clumped["completed"], tmp_path)
    a, b = _slice(store, "a")
    arrays = _arrays(store)
    lead = np.asarray(arrays["lead"][a:b])
    member = int(np.flatnonzero((lead >= 0) & (lead != np.arange(len(lead))))[0])
    arrays["r"][a + member] = -int(arrays["r"][a + member])
    assert _failed(store) == set(), "only a re-clump sees a sign"
    assert _failed(store, ld_stores=EUR) == {"clumps.reclump"}


def test_another_ld_store_than_the_one_recorded_fails_reclump(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    other = Path(shutil.copytree(LD_STORE, tmp_path / "ld.store"))
    meta = json.loads((other / "zarr.json").read_text())
    meta["attributes"]["ld_store"]["build"]["created_at"] = "2000-01-01T00:00:00Z"
    (other / "zarr.json").write_text(json.dumps(meta))
    assert _failed(clumped["completed"], ld_stores={"EUR": other}) == {"clumps.reclump"}

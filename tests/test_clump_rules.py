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
from clump_stores import LD_STORE, copied

from opengwasdb.model.analyses import write_analysis_columns
from opengwasdb.store.arrays import create_array, open_group
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.validation import validate_store
from opengwasdb.validation.rules.clumps import RULES

EUR = {"EUR": LD_STORE}
CLUMP_RULES = [rule.name for rule in RULES]
CLUMPS = Path("data.zarr/top_hits/p_5e_08/clumps/default")


#: The rules a clumps/ group inside a tier could upset besides its own: the closed envelope,
#: the Zarr format of every node, and a tier's arrays and structure.
NEIGHBOURS = ["zarr.format", "envelope.closed", "top_hits.arrays", "top_hits.structure"]


@pytest.fixture(scope="module")
def clumped(clump_seeds: dict[str, Path], clumped_seeds: dict[str, Path]) -> dict[str, Path]:
    """The clumped completed Dense and Hybrid seeds, shared with `test_clumps`, and the Dense
    seed unclumped. Never written to: a test copies the one it corrupts."""
    return {"completed": clumped_seeds["completed"], "hybrid": clumped_seeds["hybrid"],
            "unclumped": clump_seeds["dense"]}


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
    result = validate_store(clumped[name], rules=[*CLUMP_RULES, *NEIGHBOURS], ld_stores=EUR)
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
    assert _failed(store) == {"clumps.declared"}


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
    """The imputed row is made a member of a stronger lead on its chromosome, within the
    window and above r2, so only the marker rule can see that it was clumped at all."""
    from opengwasdb.store import open_store

    store = copied(clumped["completed"], tmp_path)
    a, b = _slice(store, "b")
    tier = zarr.open_group(str(store / "data.zarr/top_hits/p_5e_08"), mode="r")
    imputed = np.asarray(tier["imputed"][a:b]).astype(bool)
    abs_z = np.abs(np.asarray(tier["z"][a:b], dtype=np.float64))
    axis = open_store(store).variant_axis(aliases=False)
    found = axis.identity_by_indices(np.asarray(tier["variant_index"][a:b], dtype=np.int64))
    axis.close()
    arrays = _arrays(store)
    lead = np.asarray(arrays["lead"][a:b])
    k = int(np.flatnonzero(imputed)[0])
    reach = ((lead == np.arange(len(lead))) & (found["chromosome"] == found["chromosome"][k])
             & (np.abs(found["position"] - found["position"][k]) < 10_000_000)
             & (abs_z > abs_z[k]))
    assert reach.any(), "the fixture must have a stronger lead near its imputed row"
    arrays["lead"][a + k] = int(np.flatnonzero(reach)[0])
    arrays["r"][a + k] = 400
    assert _failed(store) == {"clumps.markers"}


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


def _sign_flipped(clumped: Path, tmp_path: Path) -> Path:
    """A copy of `clumped` with one member's r negated: a sign not flipped, or flipped twice,
    which only a re-clump can see."""
    store = copied(clumped, tmp_path)
    a, b = _slice(store, "a")
    arrays = _arrays(store)
    lead = np.asarray(arrays["lead"][a:b])
    member = int(np.flatnonzero((lead >= 0) & (lead != np.arange(len(lead))))[0])
    arrays["r"][a + member] = -int(arrays["r"][a + member])
    return store


def test_a_sign_not_flipped_fails_reclump(clumped: dict[str, Path], tmp_path: Path) -> None:
    store = _sign_flipped(clumped["completed"], tmp_path)
    assert _failed(store) == set(), "only a re-clump sees a sign"
    assert _failed(store, ld_stores=EUR) == {"clumps.reclump"}


def test_the_cli_fidelity_seed_seeds_the_reclump_sample(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    store = _sign_flipped(clumped["completed"], tmp_path)
    seeds = {}
    for seed in ("7", None):
        argv = ["validate", str(store), "--format", "json", "--ld-store", f"EUR={LD_STORE}"]
        ran = CliRunner().invoke(app, argv + (["--fidelity-seed", seed] if seed else []))
        assert ran.exit_code == 1, ran.output
        found = [f["message"] for f in json.loads(ran.stdout)["findings"]
                 if f["rule"] == "clumps.reclump"]
        assert len(found) == 1, found
        seeds[seed] = found[0]
    assert "(seed 7, sample" in seeds["7"]
    assert "(seed 0, sample" in seeds[None]


def test_another_ld_store_than_the_one_recorded_fails_reclump(
    clumped: dict[str, Path], tmp_path: Path
) -> None:
    other = Path(shutil.copytree(LD_STORE, tmp_path / "ld.store"))
    meta = json.loads((other / "zarr.json").read_text())
    meta["attributes"]["ld_store"]["build"]["created_at"] = "2000-01-01T00:00:00Z"
    (other / "zarr.json").write_text(json.dumps(meta))
    assert _failed(clumped["completed"], ld_stores={"EUR": other}) == {"clumps.reclump"}

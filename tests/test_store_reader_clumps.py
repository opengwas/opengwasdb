"""The Store Reader's read of the Clumps, `TopHitTier.clumps` (spec §18.1 *Reading*, #418).

#419's Top-Hit Query builds on this read, so each refusal is held here: a group not laid out
as §18.1 says, a group the manifest does not declare, and a declaration with no group. Also
the manifest's in-place entry write the step declares the Clumps with,
`OpenGWASDBStore.set_manifest_entry`.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from clump_stores import copied

from opengwasdb.store import open_store
from opengwasdb.store.arrays import create_array
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.top_hits import MalformedClumps, TopHitTier

TIER = Path("data.zarr/top_hits/p_5e_08")
DEFAULT = TIER / "clumps" / "default"


def _tier(store: Path) -> TopHitTier:
    tier = open_store(store).top_hit_tiers().tier(5e-8)
    assert tier is not None
    return tier


def test_a_declared_tier_reads_its_clumps_and_one_analysis_reads_its_slice(
    clumped_seeds: dict[str, Path],
) -> None:
    tier = _tier(clumped_seeds["completed"])
    assert tier.has_clumps() and tier.clumps_declared
    whole = tier.clumps()
    assert whole is not None
    offsets = np.asarray(tier.group["analysis_offsets"][:], dtype=np.int64)
    assert whole.lead.dtype == np.int32 and whole.r.dtype == np.int16
    assert len(whole.lead) == len(whole.r) == offsets[-1] > 0
    sizes = np.diff(offsets)
    assert (sizes > 0).sum() > 1, "the fixture must have more than one Analysis with rows"
    for index in range(len(offsets) - 1):
        part = tier.clumps(index)
        assert part is not None
        a, b = offsets[index], offsets[index + 1]
        assert part.lead.tolist() == whole.lead[a:b].tolist()
        assert part.r.tolist() == whole.r[a:b].tolist()
        assert part.attribute == whole.attribute


def test_a_tier_without_clumps_and_without_a_declaration_reads_none(
    clump_seeds: dict[str, Path],
) -> None:
    tier = _tier(clump_seeds["dense"])
    assert not tier.has_clumps() and not tier.clumps_declared
    assert tier.clumps() is None


def _undeclare(store: Path) -> None:
    manifest = json.loads((store / "manifest.json").read_text())
    del manifest["clumps"]
    (store / "manifest.json").write_text(json.dumps(manifest))


def _extra_member(store: Path) -> None:
    zarr.open_group(str(store / TIER / "clumps"), mode="r+").create_group("other")


def _missing_member(store: Path) -> None:
    shutil.rmtree(store / DEFAULT / "r")


def _rewritten_lead(data: Callable[[np.ndarray], np.ndarray], dtype: str) -> Callable[[Path], None]:
    def corrupt(store: Path) -> None:
        group = zarr.open_group(str(store / DEFAULT), mode="r+")
        lead = np.asarray(group["lead"][:])
        create_array(group, "lead", ArrayRole.TOP_HIT_INDEX, data=data(lead), dtype=dtype,
                     overwrite=True)
    return corrupt


def _attribute(edit: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def corrupt(store: Path) -> None:
        group = zarr.open_group(str(store / DEFAULT), mode="r+")
        attrs = dict(group.attrs)
        edit(attrs)
        group.attrs.clear()
        group.attrs.update(attrs)
    return corrupt


@pytest.mark.parametrize(("corrupt", "named"), [
    (_undeclare, r"not declared by the release's manifest"),
    (lambda store: shutil.rmtree(store / TIER / "clumps"), r"holds no clumps/ group"),
    (_extra_member, r"holds \['default', 'other'\]"),
    (_missing_member, r"default/r is missing"),
    (_rewritten_lead(lambda lead: lead, "int64"), r"default/lead is int64, not int32"),
    (_rewritten_lead(lambda lead: lead[:-1], "int32"), r"default/lead holds \d+ entries"),
    (_attribute(lambda attrs: attrs.pop("clumps")), r"has no 'clumps' attribute"),
    (_attribute(lambda attrs: attrs["clumps"].pop("kb")), r"attribute lacks \['kb'\]"),
], ids=["undeclared", "declared-no-group", "extra-member", "missing-member", "dtype", "shape",
        "missing-attribute", "missing-key"])
def test_each_malformed_or_undeclared_group_is_refused_by_name(
    clumped_seeds: dict[str, Path], tmp_path: Path, corrupt: Callable[[Path], None], named: str
) -> None:
    store = copied(clumped_seeds["completed"], tmp_path)
    corrupt(store)
    tier = _tier(store)
    with pytest.raises(MalformedClumps, match=named):
        tier.clumps()
    with pytest.raises(MalformedClumps, match=named):
        tier.clumps(0)


# ── the manifest entry ───────────────────────────────────────────────────────


def test_set_manifest_entry_sets_replaces_and_removes_one_entry_and_keeps_the_rest(
    clump_seeds: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(clump_seeds["dense"], tmp_path)
    path = store / "manifest.json"
    os.chmod(path, 0o640)
    before = json.loads(path.read_text())
    assert "clumps" not in before

    opened = open_store(store).set_manifest_entry("clumps", {"tier": "p_5e_08"})
    assert json.loads(path.read_text()) == {**before, "clumps": {"tier": "p_5e_08"}}
    assert opened.manifest.clumps == {"tier": "p_5e_08"}
    assert os.stat(path).st_mode & 0o777 == 0o640

    open_store(store).set_manifest_entry("clumps", {"tier": "p_5e_08", "configuration": "default"})
    assert json.loads(path.read_text())["clumps"] == {"tier": "p_5e_08",
                                                      "configuration": "default"}

    removed = open_store(store).set_manifest_entry("clumps", None)
    assert json.loads(path.read_text()) == before
    assert removed.manifest.clumps is None
    assert os.stat(path).st_mode & 0o777 == 0o640
    assert not list(store.glob(".manifest.json.writing"))


def test_an_in_place_analyses_column_write_keeps_the_table_s_mode(
    clump_seeds: dict[str, Path], tmp_path: Path
) -> None:
    from opengwasdb.model.analyses import write_analysis_columns
    from opengwasdb.store_reader.analyses import read_analyses

    store = copied(clump_seeds["dense"], tmp_path)
    table = store / "analyses.tsv"
    os.chmod(table, 0o640)
    n = len(read_analyses(table).rows)
    write_analysis_columns(table, {"n_loci_5e8": [str(i) for i in range(n)]})
    assert [r["n_loci_5e8"] for r in read_analyses(table).rows] == [str(i) for i in range(n)]
    assert os.stat(table).st_mode & 0o777 == 0o640
    assert not list(store.glob(".analyses.tsv.writing"))

"""On format 0.3.0 the Store Reader never reads `variants.tsv.gz` (#469).

A 0.2 release names a row's rsID, and answers `variants_table()`, from its BGZF Variant
Table, which loads pysam. From 0.3.0 the sidecars answer: the ALID sidecar names each
row, its ALID spells the chromosome, position and alleles, and the rsID index, inverted,
names its rsID (`opengwasdb.store_reader.rsid_index`). The table ships as a convenience.

Every fixture -- the six `store_reader_digests` releases (Dense, Ragged and Hybrid, each
Observed-Only and Reference-Completed) and the three `one_source_releases` -- is given
rsIDs that plant each case a row can be in (`PLANTED`): a row naming none, two rows
naming one rsID, and a row whose cell names several, as a GWAS-VCF `ID` column can. Its
0.3.0 form must answer `variants_table()` and resolved rows exactly as its 0.2.3 form
(`format_0_2_3.as_0_2_3`) does, column by column, whether the rsID index is scanned or
inverted. In a fresh interpreter, with the table and its `.tbi` unreadable, every
question of the rewrite's proof is answered on 0.3.0 without importing pysam.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3, components
from rsid_releases import RELEASE_NAMES, with_rsids

from opengwasdb.query import query_store
from opengwasdb.store_reader import rsid_index
from opengwasdb.store_reader.rsid_index import MalformedRsidIndex, MissingRsidIndex, RsidIndex
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis

#: The rsID cell planted at each of a release's first four rows; every later row `i`
#: names ``rs{1000 + i}``. Row 0 names none (``None``, the table's ``.``).
PLANTED = (None, "rs77", "rs77", "rs11;rs12")

FIXTURES = [*RELEASE_NAMES, "one_source_dense", "one_source_ragged", "one_source_hybrid"]


def _plant(release: Path) -> None:
    axis = VariantAxis(release)
    try:
        alids = [record.alid for record in axis.records()]
    finally:
        axis.close()
    assert len(alids) >= len(PLANTED), f"{release} has too few rows to plant every case"
    names = {alid: f"rs{1000 + i}" for i, alid in enumerate(alids)}
    for alid, name in zip(alids, PLANTED, strict=False):
        if name is None:
            del names[alid]
        else:
            names[alid] = name
    with_rsids(release, names)


@pytest.fixture(scope="module")
def planted(
    digest_releases: dict[str, Path],
    range_releases: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, Path]]:
    """Each fixture, planted (`PLANTED`), as ``(0.3.0, its 0.2.3 form)``. Never written to."""
    root = tmp_path_factory.mktemp("planted")
    sources = {**digest_releases, **{f"one_source_{k}": v for k, v in range_releases.items()}}
    out = {}
    for name in FIXTURES:
        current = Path(shutil.copytree(sources[name], root / name / "0.3.0"))
        _plant(current)
        legacy = as_0_2_3(Path(shutil.copytree(current, root / name / "0.2.3")))
        out[name] = (current, legacy)
    return out


def _reprs(rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> list[dict[str, str]]:
    """Each row's values as `repr` writes them: what the proof digests, so an int and a
    numpy int, or None and ``"."``, differ."""
    return [{name: repr(value) for name, value in row.items()} for row in rows]


def _answers(release: Path) -> dict[str, Any]:
    """`variants_table()`, and the resolved rows of every Analysis, a lookup of every
    variant, and the widest top hits, with variant info."""
    with query_store(release) as q:
        table = q.variants_table()
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        alids = [table[key]["alid"] for key in sorted(table)]
        results = {f"analysis/{a}": q.analysis(a) for a in ids}
        results["lookup"] = q.lookup(alids, ids)
        results["top_hits"] = q.top_hits(threshold=0.2)
        resolved = {
            name: _reprs(q.resolve(result, include_variant_info=True))
            for name, result in results.items()
        }
    return {"variants_table": {k: _reprs([row])[0] for k, row in table.items()}, **resolved}


def test_every_fixture_plants_each_case_a_row_can_be_in(planted):
    """A row naming no rsID, two naming one, one naming several: in the 0.2.3 table,
    and in rows the resolved answers reach."""
    for name, (_, legacy) in planted.items():
        with query_store(legacy) as q:
            table = q.variants_table()
        rsids = [table[key]["rsid"] for key in sorted(table)]
        assert rsids[: len(PLANTED)] == list(PLANTED), name
        assert rsids[len(PLANTED) :] == [
            f"rs{1000 + i}" for i in range(len(PLANTED), len(rsids))
        ], name
        resolved = _answers(legacy)
        reached = {
            row["rsid"]
            for shape, rows in resolved.items()
            if shape != "variants_table"
            for row in rows
        }
        assert {"'.'", "'rs77'", "'rs11;rs12'"} <= reached, (name, reached)


@pytest.mark.parametrize("inverted", [False, True], ids=["scanned", "inverted"])
@pytest.mark.parametrize("name", FIXTURES)
def test_0_3_0_answers_the_variant_table_and_resolved_rows_as_0_2_3(
    planted, monkeypatch: pytest.MonkeyPatch, name: str, inverted: bool
):
    current, legacy = planted[name]
    if inverted:
        monkeypatch.setattr(rsid_index, "_SCANNED_ROWS", 0)
    ours, theirs = _answers(current), _answers(legacy)
    assert ours.keys() == theirs.keys()
    for shape in theirs:
        assert ours[shape] == theirs[shape], (name, shape)
    assert len(ours["variants_table"]) >= len(PLANTED)


PROBE = r"""
import json, sys
from opengwasdb.query import query_store
from opengwasdb.store.query_digests import release_digests

path = sys.argv[1]
digests = release_digests(path)
with query_store(path) as q:
    ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
    resolved = [
        row["rsid"]
        for a in ids
        for row in q.resolve(q.analysis(a), include_variant_info=True)
    ]
print(json.dumps({
    "questions": len(digests["shapes"]) + len(digests["range_probes"]),
    "resolved": sorted(set(resolved)),
    "pysam": "pysam" in sys.modules,
}))
"""


def _unreadable_tables(release: Path) -> list[Path]:
    """Make every component's `variants.tsv.gz` and `.tbi` unreadable; return them."""
    locked = []
    for directory in components(release):
        for name in ("variants.tsv.gz", "variants.tsv.gz.tbi"):
            path = directory / name
            assert path.exists(), path
            path.chmod(0)
            locked.append(path)
    return locked


@pytest.mark.parametrize("name", ["one_source_dense", "one_source_ragged", "one_source_hybrid"])
def test_every_query_shape_on_0_3_0_reads_no_variant_table_and_no_pysam(
    planted, tmp_path: Path, name: str
):
    release = Path(shutil.copytree(planted[name][0], tmp_path / name))
    locked = _unreadable_tables(release)
    with pytest.raises(PermissionError):
        locked[0].open("rb")  # the probe below cannot read the table either
    try:
        out = subprocess.run(
            [sys.executable, "-c", PROBE, str(release)], capture_output=True, text=True
        )
    finally:
        for path in locked:
            path.chmod(0o644)
    assert out.returncode == 0, out.stderr[-3000:]
    found = json.loads(out.stdout)
    assert found["questions"] > 50, "a probe that asked little would prove little"
    assert {".", "rs77", "rs11;rs12"} <= set(found["resolved"])
    assert found["pysam"] is False, "a 0.3.0 query loaded pysam"


def test_a_0_3_0_release_without_its_rsid_index_is_refused(planted, tmp_path: Path):
    """On 0.3.0 the rsID index is how a row's rsID is named: without it every row would
    read as naming none, so the release is refused, not read."""
    release = Path(shutil.copytree(planted["one_source_ragged"][0], tmp_path / "release"))
    (release / "variant_rsid_rows.npy").unlink()
    with pytest.raises(MissingRsidIndex, match="variant_rsid_rows.npy"):
        query_store(release)


def _index(names: list[str | None]) -> RsidIndex:
    """The rsID index a writer would make of one name (or none) per row."""
    named = [(name, row) for row, name in enumerate(names) if name is not None]
    keys = np.array([name for name, _ in named], dtype="|S24")
    rows = np.array([row for _, row in named], dtype="int32")
    order = np.argsort(keys, kind="stable")
    return RsidIndex(keys[order], rows[order])


@pytest.mark.parametrize("asked", [[5], [0, 7, 7, 3], list(range(200)), [199, 0, 50]])
def test_scanned_and_inverted_rows_name_the_same_rsids(asked: list[int]):
    names = [None if i % 7 == 0 else f"rs{i % 13}" for i in range(200)]
    expected = [names[i] for i in asked]
    assert _index(names).rsids(np.array(asked), 200).tolist() == expected
    inverted = _index(names)
    inverted.rsids(np.arange(200), 200)  # past `_SCANNED_ROWS`: inverts, and keeps it
    assert inverted._slot_of_row is not None
    assert inverted.rsids(np.array(asked), 200).tolist() == expected


@pytest.mark.parametrize("width", [3, 200], ids=["scanned", "inverted"])
def test_an_index_naming_a_row_twice_is_refused(width: int):
    index = RsidIndex(np.array([b"rs1", b"rs2"], dtype="|S24"), np.array([4, 4], dtype="int32"))
    with pytest.raises(MalformedRsidIndex, match="row 4 twice"):
        index.rsids((np.arange(width) + 4) % 200, 200)


def test_a_row_outside_the_axis_is_refused():
    with pytest.raises(IndexError, match="row 9 is outside the axis of 9"):
        _index(["rs1"] * 9).rsids(np.array([9]), 9)


def _swap_two_rows_names(release: Path) -> None:
    """Swap the rows of the first two distinct rsIDs in the release's rsID index, so
    every rsID still counts once and two rows are named by each other's rsID."""
    names = np.load(release / "variant_rsid_bytes.npy")
    rows = np.load(release / "variant_rsid_rows.npy")
    first, second = 0, int(np.flatnonzero(names != names[0])[0])
    rows[[first, second]] = rows[[second, first]]
    np.save(release / "variant_rsid_rows.npy", rows)


@pytest.mark.parametrize("version", ["0.3.0", "0.2.3"])
def test_an_rsid_index_naming_rows_otherwise_than_the_table_fails_on_0_3_0(
    planted, tmp_path: Path, version: str
):
    """The index still holds one entry per rsID in the table, so `rsid_index`, which
    counts them, passes; on 0.3.0, which names rows from the index, `rsid_rows` fails."""
    current, legacy = planted["one_source_ragged"]
    release = Path(shutil.copytree(current if version == "0.3.0" else legacy, tmp_path / "r"))
    assert validate_store(release).ok
    _swap_two_rows_names(release)

    result = validate_store(release)

    if version == "0.2.3":
        assert result.ok, result.errors  # `rsid_rows` judges 0.3.0 only
        return
    assert result.failed() == {"variant_table.rsid_rows"}, result.errors
    [error] = result.errors
    assert "the rsID index names 2 row(s) otherwise than variants.tsv.gz" in error, error

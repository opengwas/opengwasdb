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
from rsid_releases import RELEASE_NAMES, rsid_names, with_rsids

from opengwasdb.query import query_store
from opengwasdb.store_reader import rsid_index
from opengwasdb.store_reader.rsid_index import MalformedRsidIndex, MissingRsidIndex, RsidIndex
from opengwasdb.validation import validate_store

#: The rsID cell planted at each of a release's first four rows; every later row `i`
#: names ``rs{1000 + i}``. Row 0 names none (``None``, the table's ``.``).
PLANTED = (None, "rs77", "rs77", "rs11;rs12")

FIXTURES = [*RELEASE_NAMES, "one_source_dense", "one_source_ragged", "one_source_hybrid"]


class _Planted:
    """Each fixture, planted (`PLANTED`), as ``(0.3.0, its 0.2.3 form)``, made on first
    ask and kept: an xdist worker plants only the fixtures its tests name."""

    def __init__(self, sources: dict[str, Path], root: Path) -> None:
        self._sources = sources
        self._root = root
        self._made: dict[str, tuple[Path, Path]] = {}

    def __getitem__(self, name: str) -> tuple[Path, Path]:
        if name not in self._made:
            current = Path(shutil.copytree(self._sources[name], self._root / name / "0.3.0"))
            with_rsids(current, planted=PLANTED)
            legacy = as_0_2_3(Path(shutil.copytree(current, self._root / name / "0.2.3")))
            self._made[name] = (current, legacy)
        return self._made[name]


@pytest.fixture(scope="module")
def planted(
    digest_releases: dict[str, Path],
    range_releases: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> _Planted:
    """The planted fixtures, by name (`_Planted`). Never written to."""
    sources = {**digest_releases, **{f"one_source_{k}": v for k, v in range_releases.items()}}
    return _Planted(sources, tmp_path_factory.mktemp("planted"))


def _reprs(rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> list[dict[str, str]]:
    """Each row's values as `repr` writes them: what the proof digests, so an int and a
    numpy int, or None and ``"."``, differ."""
    return [{name: repr(value) for name, value in row.items()} for row in rows]


def _answers(release: Path) -> dict[str, Any]:
    """`variants_table()`, and the resolved rows of every Analysis, a lookup of every
    variant, and the widest indexed top hits, with variant info."""
    with query_store(release) as q:
        table = q.variants_table()
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        alids = [table[key]["alid"] for key in sorted(table)]
        results = {f"analysis/{a}": q.analysis(a) for a in ids}
        results["lookup"] = q.lookup(alids, ids)
        # A tier every layout stores: a Dense or Hybrid release refuses any other
        # threshold (#483), which `query_digests` compares as a refusal.
        results["top_hits"] = q.top_hits(threshold=5e-4)
        resolved = {
            name: _reprs(q.resolve(result, include_variant_info=True))
            for name, result in results.items()
        }
    return {"variants_table": {k: _reprs([row])[0] for k, row in table.items()}, **resolved}


@pytest.mark.parametrize("name", FIXTURES)
def test_every_fixture_plants_each_case_a_row_can_be_in(planted, name: str):
    """A row naming no rsID, two naming one, one naming several: in the 0.2.3 table,
    and in rows the resolved answers reach."""
    _, legacy = planted[name]
    with query_store(legacy) as q:
        table = q.variants_table()
    alids = [table[key]["alid"] for key in sorted(table)]
    rsids = [table[key]["rsid"] for key in sorted(table)]
    assert rsids[: len(PLANTED)] == list(PLANTED)
    expected = rsid_names(alids, PLANTED)
    assert rsids == [expected.get(alid) for alid in alids]
    resolved = _answers(legacy)
    assert resolved["top_hits"] and resolved["lookup"], "an empty answer would compare nothing"
    reached = {
        row["rsid"]
        for shape, rows in resolved.items()
        if shape != "variants_table"
        for row in rows
    }
    assert {"'.'", "'rs77'", "'rs11;rs12'"} <= reached, reached


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


def _run_unreadable(release: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `argv` in a fresh interpreter with every Variant Table of `release` unreadable."""
    locked = _unreadable_tables(release)
    with pytest.raises(PermissionError):
        locked[0].open("rb")  # the subprocess cannot read the table either
    try:
        return subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    finally:
        for path in locked:
            path.chmod(0o644)


#: The probed releases: each layout over three chromosomes, and each layout
#: Reference-Completed, whose queries also read the presence mask.
PROBED = [
    "one_source_dense", "one_source_ragged", "one_source_hybrid",
    "dense_completed", "ragged_completed", "hybrid_completed",
]


@pytest.mark.parametrize("name", PROBED)
def test_every_query_shape_on_0_3_0_reads_no_variant_table_and_no_pysam(
    planted, tmp_path: Path, name: str
):
    release = Path(shutil.copytree(planted[name][0], tmp_path / name))
    out = _run_unreadable(release, ["-c", PROBE, str(release)])
    assert out.returncode == 0, out.stderr[-3000:]
    found = json.loads(out.stdout)
    assert found["questions"] > 50, "a probe that asked little would prove little"
    assert {".", "rs77", "rs11;rs12"} <= set(found["resolved"])
    assert found["pysam"] is False, "a 0.3.0 query loaded pysam"


CLI = r"""
import contextlib, io, json, sys
from opengwasdb.cli.main import app

out = {}
for argv in json.loads(sys.argv[1]):
    written = io.StringIO()
    with contextlib.redirect_stdout(written):
        try:
            app(argv, prog_name="opengwasdb")
        except SystemExit as exit:
            code = exit.code or 0
        else:
            code = 0
    out[argv[0]] = {"code": code, "stdout": written.getvalue()}
print(json.dumps({"answers": out, "pysam": "pysam" in sys.modules}))
"""


@pytest.mark.parametrize("name", ["one_source_hybrid", "ragged_completed"])
def test_every_cli_query_on_0_3_0_reads_no_variant_table_and_no_pysam(
    planted, tmp_path: Path, name: str
):
    """`opengwasdb query-*` loads no pysam on 0.3.0: Reference Completion's modules, which
    do, are imported by the `complete-*` commands alone. The five commands run in one
    fresh interpreter, which loads nothing a command could have loaded for another."""
    release = Path(shutil.copytree(planted[name][0], tmp_path / name))
    with query_store(release) as q:
        table = q.variants_table()
        analysis_id = q.analyses_table()[0]["analysis_id"]
    first, last = table[0], table[len(table) - 1]
    asked = {
        "query-phewas": [first["alid"]],
        "query-range-phewas": [first["chromosome"], "1", "500000000"],
        "query-analysis": [analysis_id],
        "query-lookup": [f"{first['alid']},rs77,{last['alid']}", analysis_id],
        "query-top-hits": ["--threshold", "5e-4"],  # a stored tier: every layout answers (#483)
    }
    argvs = [[command, str(release), *args, "--variant-info"] for command, args in asked.items()]
    out = _run_unreadable(release, ["-c", CLI, json.dumps(argvs)])
    assert out.returncode == 0, out.stderr[-3000:]
    found = json.loads(out.stdout)
    assert found["answers"].keys() == asked.keys()
    for command, answer in found["answers"].items():
        assert answer["code"] == 0, (command, out.stderr[-3000:])
        assert answer["stdout"].count("\n") > 1, (command, "a query that answered nothing")
    assert "\trs77\t" in found["answers"]["query-lookup"]["stdout"], "the planted rsID"
    assert found["pysam"] is False, "a 0.3.0 CLI query loaded pysam"


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
    assert inverted.inverted
    assert inverted.rsids(np.array(asked), 200).tolist() == expected


def test_the_index_inverts_once_its_scans_would_cost_more_than_one_inversion():
    """Asked 32 rows a call, the index scans while the scans so far cost less than one
    inversion (`_scan_cost`), then inverts, once, and gathers from then on."""
    names = [f"rs{i}" for i in range(400)]
    index = _index(names)
    per_call = rsid_index._scan_cost(32)
    scans = int(1.0 // per_call)
    assert 1 < scans < 10, "the measured costs make a few scans cheaper than inverting"
    for call in range(scans):
        asked = np.arange(32) + 32 * call
        assert index.rsids(asked, 400).tolist() == [names[i] for i in asked.tolist()]
        assert not index.inverted, f"inverted after {call + 1} scan(s) of {scans}"
    index.rsids(np.arange(32), 400)
    assert index.inverted, "the next scan would have cost more than an inversion"


@pytest.mark.parametrize("width", [3, 200], ids=["scanned", "inverted"])
def test_an_index_naming_a_row_twice_is_refused(width: int):
    index = RsidIndex(np.array([b"rs1", b"rs2"], dtype="|S24"), np.array([4, 4], dtype="int32"))
    with pytest.raises(MalformedRsidIndex, match="row 4 twice"):
        index.rsids((np.arange(width) + 4) % 200, 200)


@pytest.mark.parametrize("asked", [[0], list(range(100))], ids=["scanned", "inverted"])
def test_an_index_naming_a_row_outside_the_axis_is_refused_on_either_path(asked: list[int]):
    """One malformed index, the same refusal however many rows are asked."""
    index = RsidIndex(
        np.array([b"rs1", b"rs2"], dtype="|S24"), np.array([0, 100], dtype="int32")
    )
    with pytest.raises(MalformedRsidIndex, match=r"rows 0\.\.100, outside the axis of 100"):
        index.rsids(np.array(asked), 100)


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


@pytest.mark.parametrize(
    "name", ["one_source_dense", "one_source_ragged", "one_source_hybrid", "dense_completed"]
)
def test_the_rewrite_proves_each_planted_fixture_with_its_variant_info(
    planted, tmp_path: Path, name: str
):
    """The rewrite's proof asks `variants_table()` and resolved rows with variant info;
    with the planted rsIDs it passes, and the published release carries no offsets."""
    from opengwasdb.store.rewrite import rewrite_release

    _, legacy = planted[name]
    destination = rewrite_release(legacy, tmp_path / "rewritten")

    for directory in components(destination):
        assert not (directory / "variant_offsets.npy").exists(), directory
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["provenance"]["format_rewrite"]["steps"][-1] == "variant_offsets"
    assert _answers(destination) == _answers(legacy)


def test_a_0_3_0_build_writes_no_offsets_and_a_stray_one_fails_its_envelope(
    planted, tmp_path: Path
):
    current, legacy = planted["one_source_dense"]
    assert not (current / "variant_offsets.npy").exists()
    assert (legacy / "variant_offsets.npy").exists(), "the 0.2.3 form carries them"
    release = Path(shutil.copytree(current, tmp_path / "stray"))
    shutil.copy(legacy / "variant_offsets.npy", release / "variant_offsets.npy")

    result = validate_store(release)

    assert result.failed() == {"envelope.closed"}, result.errors
    assert any("variant_offsets.npy" in error for error in result.errors), result.errors


def test_a_0_2_3_dense_release_without_its_offsets_is_incomplete(planted, tmp_path: Path):
    release = Path(shutil.copytree(planted["one_source_dense"][1], tmp_path / "r"))
    (release / "variant_offsets.npy").unlink()

    result = validate_store(release)

    assert "envelope.required" in result.failed(), result.errors
    assert "missing variant_offsets.npy" in result.errors, result.errors


def test_the_writer_refuses_an_rsid_the_index_cannot_hold(tmp_path: Path):
    """The 0.2 series kept such an rsID in the table and out of the index; on 0.3.0 the
    index alone names a row's rsID, so it would vanish from every answer."""
    from opengwasdb.store_reader.alid import RSID_WIDTH
    from opengwasdb.variants.axis import UnindexableRsidError, write_variant_axis
    from opengwasdb.variants.normalise import CanonicalVariant

    variants = [CanonicalVariant("1", 100, "A", "G"), CanonicalVariant("1", 200, "C", "T")]
    wide = "rs1;rs2;rs3;rs4;rs5;rs6;rs7"
    assert len(wide.encode()) > RSID_WIDTH

    with pytest.raises(UnindexableRsidError, match=r"1 variant\(s\) .* first, 1:200:C:T"):
        write_variant_axis(tmp_path, variants, {variants[0].alid: "rs1", variants[1].alid: wide})
    assert list(tmp_path.iterdir()) == [], "nothing is written"

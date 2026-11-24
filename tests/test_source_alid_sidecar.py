"""Format 0.3.0 names each row's `source_alid` from its own sidecar (#470).

A row's `source_alid` is the ALID its source spelt it by, before a lift. The 0.2 series
holds it in the Variant Table's eighth column alone, so a 0.3.0 Store Reader, which
never reads the table (#469), could not return it. `variant_source_alid_state.npy` gives
each row's state -- none, its own ALID, or another -- and `variant_source_alid_utf8.npy`
and `variant_source_alid_offsets.npy` hold a lifted row's, whole (ADR 0105 §8).

Every fixture -- the six `store_reader_digests` releases (Dense, Ragged and Hybrid, each
Observed-Only and Reference-Completed) -- is given `source_alid`s that plant each case a
row can be in (`plant`): a collision row with none, an unlifted row naming its own ALID,
a lifted row naming another, and lifted rows whose `source_alid` is 24, 32, 33 and 130
bytes long. A row Reference Completion added from the panel keeps none. Its 0.3.0 form
answers `variants_table()` and resolved rows with those values, exactly as its 0.2.3
form (`format_0_2_3.as_0_2_3`) does from its table.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3, components, write_variant_offsets
from rsid_releases import RELEASE_NAMES
from test_variant_table_off_read_path import _answers

from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.store.rewrite import rewrite_release
from opengwasdb.store_reader import source_alid_sidecar as codec
from opengwasdb.store_reader.alid_sidecar import AlidSidecar
from opengwasdb.store_reader.hybrid import DenseComponentSharesAxis
from opengwasdb.store_reader.panel_axis import PanelVariantAxis
from opengwasdb.store_reader.source_alid_sidecar import (
    SOURCE_ALID_SIDECAR_FILENAMES,
    MalformedSourceAlidSidecar,
    MissingSourceAlidSidecar,
    SourceAlidSidecar,
    UnencodableSourceAlidError,
    write_source_alid_sidecar,
)
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

RULE = "variant_table.source_alid_sidecar"

#: The lengths of the planted lifted `source_alid`s that are not the ordinary kind: the
#: ALID index's key width and the 0.2 slot's edges, and an indel well past 100 bytes.
PLANTED_LENGTHS = (24, 32, 33, 130)


# ── the codec ───────────────────────────────────────────────────────────────

LONG = "1:1000:A:" + "ACGT" * 40
#: Six rows: each ALID, and its `source_alid`.
ALIDS = ["1:10:A:G", "1:20:A:G", "1:30:A:G", "1:40:A:G", "1:50:C:T", "1:60:A:G"]
SOURCES = [None, "1:20:A:G", "", LONG, "X:7:C:T", "1:60:A:G"]


def _own(rows: np.ndarray) -> np.ndarray:
    """`ALIDS` at `rows`, as the ALID sidecar would give them."""
    return np.array([ALIDS[int(r)] for r in rows], dtype=object)


def _written(tmp_path: Path, sources=SOURCES, alids=ALIDS) -> SourceAlidSidecar:
    write_source_alid_sidecar(tmp_path, sources, alids)
    sidecar = SourceAlidSidecar.open(tmp_path)
    assert sidecar is not None
    return sidecar


def test_each_row_has_a_state_and_only_a_lifted_row_has_text(tmp_path: Path):
    _written(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(SOURCE_ALID_SIDECAR_FILENAMES)
    # a collision none, an unlifted row its own ALID, a lifted row another, held whole
    assert np.load(tmp_path / codec.SOURCE_ALID_STATE_FILENAME).tolist() == [0, 1, 0, 2, 2, 1]
    assert np.load(tmp_path / codec.SOURCE_ALID_STATE_FILENAME).dtype == np.dtype("|u1")
    text = np.load(tmp_path / codec.SOURCE_ALID_UTF8_FILENAME).tobytes()
    assert text == (LONG + "X:7:C:T").encode()
    offsets = np.load(tmp_path / codec.SOURCE_ALID_OFFSETS_FILENAME)
    assert offsets.tolist() == [0, len(LONG), len(LONG) + 7]


def test_the_codec_round_trips_every_value_whole(tmp_path: Path):
    sidecar = _written(tmp_path)
    expected = [None, "1:20:A:G", None, LONG, "X:7:C:T", "1:60:A:G"]
    assert sidecar.values(np.arange(6), _own).tolist() == expected
    assert sidecar.values([3, 0, 5, 3], _own).tolist() == [LONG, None, "1:60:A:G", LONG]
    assert sidecar.sourced(0, 6) == 4 and sidecar.sourced(2, 4) == 1
    assert list(sidecar.problems(6)) == []
    assert sidecar.differing(0, expected, ALIDS) == []
    # an unlifted row read as lifted, and a lifted row's text another
    assert sidecar.differing(1, ["X:1:A:G", None, LONG + "A"], ALIDS[1:4]) == [1, 3]


def test_the_scan_finds_the_first_row_of_each_value_however_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(codec, "_WINDOW", 2)  # several windows, a value in a later one
    long = "2:9:" + "T" * 120 + ":G"
    alids = [f"1:{i}:A:G" for i in range(1, 10)]
    sources = ["0:1:A:G", None, "0:2:A:G", long, "1:5:A:G", "0:2:A:G", None, "0:4:A:GG", long]
    sidecar = _written(tmp_path, sources, alids)
    def search(asked):
        return np.array([alids.index(a) if a in alids else -1 for a in asked])
    found = sidecar.rows_of(["0:2:A:G", long, "0:4:A:GG", "1:5:A:G", "9:9:A:G", ""], search)
    assert found == {"0:2:A:G": 2, long: 3, "0:4:A:GG": 7, "1:5:A:G": 4}
    # a row whose ALID is asked but whose state is not its own ALID is no match
    assert sidecar.rows_of(["1:1:A:G"], search) == {}


@pytest.mark.parametrize(
    "value", [".", "1:1:A\tG", "1:1:A:G\n", "1:1:\x00:G", "1:1:\ud800:G"],
    ids=["table-blank", "tab", "newline", "nul", "lone-surrogate"],
)
def test_the_writer_refuses_a_value_the_table_could_not_hold_as_given(
    tmp_path: Path, value: str
):
    with pytest.raises(UnencodableSourceAlidError, match="row 1"):
        write_source_alid_sidecar(tmp_path, ["1:1:A:G", value], ["1:1:A:G", "1:2:A:G"])
    assert not list(tmp_path.iterdir()), "nothing is written for a refused value"


def test_a_half_written_sidecar_is_refused(tmp_path: Path):
    _written(tmp_path)
    (tmp_path / codec.SOURCE_ALID_STATE_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        SourceAlidSidecar.open(tmp_path)


@pytest.mark.parametrize("fault", ["start", "end", "dtype", "state-dtype"])
def test_offsets_that_cannot_slice_the_bytes_are_refused_at_open(tmp_path: Path, fault: str):
    _written(tmp_path)
    name = codec.SOURCE_ALID_OFFSETS_FILENAME
    if fault == "state-dtype":
        name = codec.SOURCE_ALID_STATE_FILENAME
    array = np.load(tmp_path / name)
    if fault == "start":
        array[0] = 1
    elif fault == "end":
        array[-1] += 1
    else:
        array = array.astype("<i8")
    np.save(tmp_path / name, array)
    with pytest.raises(MalformedSourceAlidSidecar):
        SourceAlidSidecar.open(tmp_path)


def test_a_row_outside_the_sidecar_is_refused_never_read_as_none(tmp_path: Path):
    sidecar = _written(tmp_path)
    with pytest.raises(IndexError, match="row 6 is outside"):
        sidecar.values([0, 6], _own)


@pytest.mark.parametrize("fault", ["state-3", "text-for-no-row", "empty-text", "short"])
def test_each_structural_fault_is_a_problem_and_a_read_refuses_it(tmp_path: Path, fault: str):
    _written(tmp_path)
    state_path = tmp_path / codec.SOURCE_ALID_STATE_FILENAME
    offsets_path = tmp_path / codec.SOURCE_ALID_OFFSETS_FILENAME
    state, offsets = np.load(state_path), np.load(offsets_path)
    rows = 6
    if fault == "state-3":
        state[1] = 3
        expected = f"{codec.SOURCE_ALID_STATE_FILENAME} holds state 3 at row 1"
    elif fault == "text-for-no-row":
        state[3] = codec.OWN_ALID  # the text of a lifted row now belongs to none
        expected = "has 3 offsets for 1 lifted row(s)"
    elif fault == "empty-text":
        offsets[1] = 0  # the first lifted row's text is empty
        expected = "does not rise at lifted row 0"
    else:
        rows = 7
        expected = "holds 6 rows but variants.tsv.gz has 7"
    np.save(state_path, state)
    np.save(offsets_path, offsets)
    sidecar = SourceAlidSidecar.open(tmp_path)
    assert sidecar is not None
    [problem] = sidecar.problems(rows)
    assert expected in problem, problem
    if fault in {"state-3", "text-for-no-row"}:
        with pytest.raises(MalformedSourceAlidSidecar):
            sidecar.values([0], _own)


# ── the planted fixtures ────────────────────────────────────────────────────


def _spelt(chromosome: str, position: int, length: int) -> str:
    """A lifted-looking `source_alid` of exactly `length` bytes, 1,000 bases before."""
    head = f"{chromosome}:{position - 1000}:A:"
    assert len(head) < length, (head, length)
    return head + "G" * (length - len(head))


def planted_sources(records: Sequence[Any]) -> dict[str, str | None]:
    """Each row's planted `source_alid`, by ALID. Where some rows have one, a row with
    none keeps none (a panel row of a completed release); a release whose build recorded
    none (the Dense digest fixtures, built from records) is planted on every row. Of the
    planted rows, in row order: the first is a collision (none), the second unlifted (its
    own ALID), then one of each `PLANTED_LENGTHS`; every later one is lifted from 1,000
    bases before."""
    sourced = [r for r in records if r.source_alid] or list(records)
    assert len(sourced) >= 2 + len(PLANTED_LENGTHS), "too few observed rows to plant"
    out: dict[str, str | None] = {r.alid: None for r in records}
    for k, r in enumerate(sourced):
        if k == 0:
            continue
        if k == 1:
            out[r.alid] = r.alid
        elif k - 2 < len(PLANTED_LENGTHS):
            out[r.alid] = _spelt(r.chromosome, r.position, PLANTED_LENGTHS[k - 2])
        else:
            out[r.alid] = f"{r.chromosome}:{r.position - 1000}:{r.effect_allele}:{r.other_allele}"
    return out


def _rewrite_axis(store: Path, sources: dict[str, str | None]) -> None:
    axis = VariantAxis(store)
    try:
        records = axis.all()
    finally:
        axis.close()
    write_variant_axis(
        store,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        {r.alid: r.rsid for r in records if r.rsid},
        [sources[r.alid] for r in records],
    )


def plant(release: Path) -> dict[str, str | None]:
    """Plant `planted_sources` in every component of `release` that carries its own
    Variant Axis. A 0.3.0 Hybrid's Dense Component carries none: its rows are the shared
    ones (#472). Returns them."""
    axis = VariantAxis(release)
    try:
        sources = planted_sources(axis.all())
    finally:
        axis.close()
    for directory in components(release):
        if (directory / "variants.tsv.gz").exists():
            _rewrite_axis(directory, sources)
    return sources


class _Planted:
    """Each fixture, planted, as ``(0.3.0, its 0.2.3 form, the planted values)``, made
    on first ask and kept."""

    def __init__(self, sources: dict[str, Path], root: Path) -> None:
        self._sources = sources
        self._root = root
        self._made: dict[str, tuple[Path, Path, dict[str, str | None]]] = {}

    def __getitem__(self, name: str) -> tuple[Path, Path, dict[str, str | None]]:
        if name not in self._made:
            shutil.rmtree(self._root / name, ignore_errors=True)  # a failed attempt's
            current = Path(shutil.copytree(self._sources[name], self._root / name / "0.3.0"))
            values = plant(current)
            legacy = as_0_2_3(Path(shutil.copytree(current, self._root / name / "0.2.3")))
            self._made[name] = (current, legacy, values)
        return self._made[name]


@pytest.fixture(scope="module")
def planted(
    digest_releases: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> _Planted:
    return _Planted(digest_releases, tmp_path_factory.mktemp("planted-source-alid"))


@pytest.mark.parametrize("name", RELEASE_NAMES)
def test_0_3_0_returns_each_rows_source_alid_through_the_table_and_resolved_rows(
    planted, name: str
):
    current, legacy, values = planted[name]
    lengths = {len(v) for v in values.values() if v}
    assert set(PLANTED_LENGTHS) <= lengths, "the fixture plants every length"
    with query_store(current) as q:
        table = q.variants_table()
    got = {row["alid"]: row["source_alid"] for row in table.values()}
    assert got == values
    assert None in got.values(), "a collision row: none"
    assert any(a == s for a, s in got.items()), "an unlifted row: its own ALID"
    rows = _resolved_rows(current)
    long = next(v for v in values.values() if v and len(v) == 130)
    assert {".", long} <= {row["source_alid"] for row in rows}, "rows reach the planted ones"
    assert [row["source_alid"] for row in rows] == [values[row["alid"]] or "." for row in rows]
    assert _answers(current) == _answers(legacy), "0.3.0 answers as its 0.2.3 form does"


def _resolved_rows(release: Path) -> list[dict[str, Any]]:
    """The resolved rows, with variant info, of a lookup of every variant."""
    with query_store(release) as q:
        table = q.variants_table()
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        result = q.lookup([table[key]["alid"] for key in sorted(table)], ids)
        return list(q.resolve(result, include_variant_info=True))


@pytest.mark.parametrize("name", ["hybrid_observed", "hybrid_completed"])
def test_a_0_3_0_hybrid_dense_component_answers_source_alid_from_the_shared_table(
    planted, name: str
):
    """#472's Dense Component carries no sidecar of its own: each panel row's
    `source_alid` is its shared row's, through `PanelVariantAxis`, in every read, and a
    Dense Analysis's resolved rows give it (opengwasdb-stores#499)."""
    current, _, values = planted[name]
    dense_dir = current / "dense"
    assert not any((dense_dir / f).exists() for f in SOURCE_ALID_SIDECAR_FILENAMES)
    assert all((current / f).exists() for f in SOURCE_ALID_SIDECAR_FILENAMES)
    shared = open_store(current).variant_axis(aliases=False)
    dense = open_store(dense_dir).variant_axis(aliases=False)
    try:
        assert isinstance(dense, PanelVariantAxis)
        with pytest.raises(DenseComponentSharesAxis, match="source_alid sidecar"):
            _ = dense.source_alid_sidecar  # the component has none of its own
        mapped = np.asarray(dense.dense_to_shared, dtype=np.int64)
        expected = shared.source_alids_by_indices(mapped).tolist()
        planted_on_panel = [values[a] for a in shared.identity_by_indices(mapped)["alid"]]
        assert expected == planted_on_panel
        assert any(v and len(v) == 130 for v in expected), "a planted long value is on the panel"
        assert any(v is None for v in expected) and any(v for v in expected)
        rows = np.arange(dense.n_variants)
        assert dense.source_alids_by_indices(rows).tolist() == expected
        assert [r["source_alid"] for _, r in sorted(dense.variants_table().items())] == expected
        assert [r.source_alid for r in dense.records()] == expected
        by = dense.by_indices(rows.tolist())
        assert [by[int(k)].source_alid for k in rows] == expected
        lifted = {v: int(k) for k, v in enumerate(expected) if v and v not in {None}}
        found = dense.indices_by_source_alid(set(lifted))
        assert found == {v: k for v, k in lifted.items() if expected.index(v) == k}
    finally:
        dense.close()
    with open_store(current).query() as q:
        ids = [row["analysis_id"] for row in q.analyses_table().values()]
        nested = q._dense
        # The nested Dense query resolves in panel rows; the Hybrid facade in shared rows.
        rows_out = [
            row
            for a in ids
            for query in (nested, q)
            for row in query.resolve(query.analysis(a), include_variant_info=True)
        ]
    assert len({r["alid"] for r in rows_out}) > 5, "the Dense Analyses answer rows"
    assert [r["source_alid"] for r in rows_out] == [values[r["alid"]] or "." for r in rows_out]


def test_a_completed_ragged_release_keeps_the_sidecar_observed_rows_sourced_panel_rows_empty(
    digest_releases: dict[str, Path],
):
    """The Ragged digest pair is a source and its completion, built on GRCh38, so each
    observed row's `source_alid` is its own ALID; completion adds panel rows, which no
    source observed."""
    completed = digest_releases["ragged_completed"]
    observed = digest_releases["ragged_observed"]
    assert all((completed / f).exists() for f in SOURCE_ALID_SIDECAR_FILENAMES)
    with query_store(observed) as q:
        before = {row["alid"] for row in q.variants_table().values()}
    with query_store(completed) as q:
        after = {row["alid"]: row["source_alid"] for row in q.variants_table().values()}
    added = set(after) - before
    assert added and before <= set(after), "completion added panel rows"
    assert all(after[a] == a for a in before), "an observed row lost its source_alid"
    assert all(after[a] is None for a in added), "a panel row was given a source_alid"
    sidecar = SourceAlidSidecar.open(completed)
    assert sidecar is not None
    assert sidecar.sourced(0, sidecar.n_rows) == len(before)


def test_a_lifted_hybrid_release_completes_with_its_sidecar(tmp_path: Path):
    """The shared table's sidecar keeps each lifted row's hg19 ALID through completion,
    and names none for a row completion folds in from the panel (#449's shared carry)."""
    import test_hybrid_completion as thc

    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    source = thc._build_source(tmp_path)
    completed = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(source, completed, thc._make_ld_panel(tmp_path), min_cor=0.0, thresh=0.9)
    with query_store(source) as q:
        before = {row["alid"]: row["source_alid"] for row in q.variants_table().values()}
    with query_store(completed) as q:
        after = {row["alid"]: row["source_alid"] for row in q.variants_table().values()}
    lifted = {a: s for a, s in before.items() if s and s != a}
    assert lifted, "nothing was lifted: the fixture proves nothing"
    assert {a: after[a] for a in before} == before
    assert all(after[a] is None for a in set(after) - set(before))
    table = {r.alid: r.source_alid for r in _table_records(completed)}
    with_sidecar = VariantAxis(completed)
    try:
        assert {r.alid: r.source_alid for r in with_sidecar.records()} == table
    finally:
        with_sidecar.close()
    # The completed Dense Component carries no copy (#472); its rows answer from the
    # shared sidecar, whatever its dropped copy held (opengwasdb-stores#499).
    assert not any((completed / "dense" / f).exists() for f in SOURCE_ALID_SIDECAR_FILENAMES)
    dense = open_store(completed / "dense").variant_axis(aliases=False)
    try:
        panel = {r.alid: r.source_alid for r in dense.records()}
    finally:
        dense.close()
    assert panel and {a: after[a] for a in panel} == panel
    assert any(a in lifted for a in panel), "a lifted row is on the panel"


def test_a_lifted_dense_release_completes_with_its_sidecar(tmp_path: Path):
    """A real lift: the sidecar of the completed release names each observed row's hg19
    ALID, and none for a panel row, as its table does."""
    import test_completion_keeps_source_alid as keeps

    _, completed = keeps.lifted_dense_completion(tmp_path)
    lifted = keeps._strand_source_alids()
    assert any(a != s for a, s in lifted.items()), "nothing moved"
    with query_store(completed) as q:
        got = {row["alid"]: row["source_alid"] for row in q.variants_table().values()}
    assert {a: got[a] for a in lifted} == lifted
    assert any(v is None for v in got.values()), "a panel row"
    table = {r.alid: r.source_alid for r in _table_records(completed)}
    assert table == got
    assert validate_store(completed).failed() == frozenset()


def _table_records(release: Path) -> list[Any]:
    axis = VariantAxis(release)
    try:
        return list(axis.table_records())
    finally:
        axis.close()


# ── the validator ───────────────────────────────────────────────────────────


def _resave(release: Path, edit) -> None:
    """Rewrite `release`'s sidecar from its values and ALIDs as `edit` changes them."""
    sidecar = SourceAlidSidecar.open(release)
    alid_sidecar = AlidSidecar.open(release)
    assert sidecar is not None and alid_sidecar is not None
    rows = np.arange(sidecar.n_rows)
    values = sidecar.values(rows, alid_sidecar.alids).tolist()
    alids = alid_sidecar.alids(rows).tolist()
    write_source_alid_sidecar(release, *edit(values, alids))


def _lifted_row(values: list, alids: list) -> int:
    return next(k for k, (v, a) in enumerate(zip(values, alids, strict=True)) if v and v != a)


def _other_text(values: list, alids: list) -> tuple[list, list]:
    k = _lifted_row(values, alids)
    return [*values[:k], values[k] + "A", *values[k + 1 :]], alids


#: Each way a sidecar can differ from the table, and the row and words the rule reports.
FAULTS = {
    "collision-made-lifted": (lambda v, a: (["1:1:A:G", *v[1:]], a), "names 1 row(s) otherwise"),
    "lifted-text-differs": (_other_text, "names 1 row(s) otherwise"),
    "one-row-short": (lambda v, a: (v[:-1], a[:-1]), "rows but variants.tsv.gz has"),
}


@pytest.mark.parametrize("fault", [*FAULTS, "state-out-of-range"])
def test_the_validator_fails_a_sidecar_that_is_not_the_table(
    planted, tmp_path: Path, fault: str
):
    release = Path(shutil.copytree(planted["ragged_observed"][0], tmp_path / "r"))
    assert validate_store(release).ok, "a valid release to begin with"
    state = np.load(release / codec.SOURCE_ALID_STATE_FILENAME)
    assert {0, 1, 2} <= set(state.tolist()), "every state is planted"
    if fault == "state-out-of-range":
        state[int(np.flatnonzero(state == codec.OWN_ALID)[0])] = 3
        np.save(release / codec.SOURCE_ALID_STATE_FILENAME, state)
        words = "holds state 3 at row"
    else:
        edit, words = FAULTS[fault]
        _resave(release, edit)

    result = validate_store(release)

    assert result.failed() == {RULE}, result.errors
    [error] = result.errors
    assert words in error, error


def test_the_validator_reports_a_missing_sidecar_by_name(planted, tmp_path: Path):
    release = Path(shutil.copytree(planted["dense_observed"][0], tmp_path / "r"))
    (release / codec.SOURCE_ALID_UTF8_FILENAME).unlink()
    result = validate_store(release)
    assert "envelope.required" in result.failed(), result.errors
    assert any(codec.SOURCE_ALID_UTF8_FILENAME in e for e in result.errors), result.errors


def test_a_0_3_0_release_without_its_sidecar_is_refused_never_read_as_none(
    planted, tmp_path: Path
):
    release = Path(shutil.copytree(planted["dense_observed"][0], tmp_path / "r"))
    for name in SOURCE_ALID_SIDECAR_FILENAMES:
        (release / name).unlink()
    with pytest.raises(MissingSourceAlidSidecar, match="variant_source_alid"):
        query_store(release)


def test_a_stray_sidecar_beside_a_0_2_3_release_fails_its_envelope(planted, tmp_path: Path):
    current, legacy, _ = planted["ragged_observed"]
    release = Path(shutil.copytree(legacy, tmp_path / "r"))
    for name in SOURCE_ALID_SIDECAR_FILENAMES:
        shutil.copy(current / name, release / name)
    result = validate_store(release)
    assert result.failed() == {"envelope.closed"}, result.errors


# ── the rewrite ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["dense_completed", "hybrid_observed", "ragged_observed"])
def test_the_rewrite_fills_the_sidecar_from_the_table_and_the_proof_passes(
    planted, tmp_path: Path, name: str
):
    current, legacy, values = planted[name]
    destination = rewrite_release(legacy, tmp_path / "rewritten")
    assert SourceAlidSidecar.open(destination) is not None
    # a Hybrid's Dense Component gets none: its rows are the shared ones (#472)
    assert all(SourceAlidSidecar.open(d) is None for d in components(destination)[1:])
    with query_store(destination) as q:
        got = {row["alid"]: row["source_alid"] for row in q.variants_table().values()}
    assert got == values
    assert _answers(destination) == _answers(legacy)
    for name in SOURCE_ALID_SIDECAR_FILENAMES:
        written, built = np.load(destination / name), np.load(current / name)
        assert written.dtype == built.dtype and np.array_equal(written, built), name


def _as_seven_columns(directory: Path) -> None:
    """Rewrite a 0.2.3 component's Variant Table without its `source_alid` column, as a
    table written before that column was, with its `.tbi` and offsets."""
    import pysam

    table = directory / "variants.tsv.gz"
    with gzip.open(table, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0].endswith("\tsource_alid"), "an eight-column table to begin with"
    plain = directory / "variants.tsv"
    plain.write_text("".join("\t".join(line.split("\t")[:7]) + "\n" for line in lines))
    pysam.tabix_compress(str(plain), str(table), force=True)
    plain.unlink()
    pysam.tabix_index(
        str(table), seq_col=0, start_col=1, end_col=1, meta_char="#", zerobased=False, force=True
    )
    (directory / "variant_offsets.npy").unlink()
    write_variant_offsets(directory)


def test_a_seven_column_table_rewrites_to_a_sidecar_naming_none(planted, tmp_path: Path):
    _, legacy, _ = planted["ragged_observed"]
    source = Path(shutil.copytree(legacy, tmp_path / "source"))
    _as_seven_columns(source)
    with query_store(source) as q:
        assert all(row["source_alid"] is None for row in q.variants_table().values())

    destination = rewrite_release(source, tmp_path / "rewritten")

    sidecar = SourceAlidSidecar.open(destination)
    assert sidecar is not None and sidecar.n_rows > 0
    assert sidecar.sourced(0, sidecar.n_rows) == 0
    assert len(sidecar.utf8) == 0

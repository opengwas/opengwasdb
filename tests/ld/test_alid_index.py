"""ogld#42: the ALID search index, held to the engine's own writer (ADR 0010).

`tests/data/alid_index/index.tsv` holds the engine's answers (opengwasdb at the commit named on
the file's first line) for five Variant Sets, made by `make_fixture.py` from the engine's
`write_variant_axis` and its two sidecars. These tests read it; no expected value here comes
from ogld. The layout's rule — the slot's width, the NUL padding, the bytewise order with a
stable sort, and which ALIDs are left out — is `ogld.store.layout`'s; the store writes it from
its own variant table (`ogld.store.assembly`, behind `StoreWriter`).

Expected bytes come from the fixture's ALIDs through numpy's own `|S32` cast, which is how the
engine's writer pads each key, never from `ogld.store.layout`.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from script_modules import load_script

from ogld.format import FAR_BP, NEAR_BP, REncoding, Sourced
from ogld.identity import orient
from ogld.store import StoreReader, StoreWriter, Variants, assembly, layout

DATA = Path(__file__).resolve().parent / "data" / "alid_index"
TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"


@dataclass(frozen=True)
class Case:
    """One Variant Set the engine indexed: its ALIDs, its keys in order, and their rows."""

    name: str
    alids: list[str]
    keys: list[str]
    rows: list[int]
    left_out: int


def _cases() -> list[Case]:
    with (DATA / "index.tsv").open(newline="") as handle:
        header = handle.readline()
        assert header.startswith("# opengwasdb "), "the fixture must name its engine commit"
        return [
            Case(
                row["case"],
                row["alids"].split("|"),
                row["keys"].split("|") if row["keys"] else [],
                [int(at) for at in row["rows"].split(",")] if row["rows"] else [],
                int(row["left_out"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


CASES = _cases()


def test_the_fixture_is_meaningful() -> None:
    # Asserted before anything is asserted about it: the width's two sides, both orders, a
    # stable sort, and an ALID left out.
    assert len(CASES) >= 5
    assert {case.left_out > 0 for case in CASES} == {True, False}
    every = {alid for case in CASES for alid in case.alids}
    lengths = {len(alid.encode("utf-8")) for alid in every}
    assert 32 in lengths, "an ALID at the slot's width"
    assert 33 in lengths, "an ALID a byte wider"
    wide = {alid[:32] for alid in every if len(alid) > 32}
    assert wide & every, "an ALID at the slot's width that a wider one agrees with over it"
    assert any(len(case.keys) > 20 for case in CASES), "a sort worth taking"
    tied = [(case, key) for case in CASES for key in set(case.keys) if case.keys.count(key) > 1]
    assert tied, "a key two rows share, which only a stable sort orders"
    for case, key in tied:
        rows = [case.rows[at] for at, seen in enumerate(case.keys) if seen == key]
        assert rows == sorted(rows), "a tie's rows must ascend: the engine sorts stably"
    admitted = all(len(case.keys) == len(case.alids) - case.left_out for case in CASES)
    assert admitted, "the keys are the ALIDs the width admits"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_the_index_is_the_engine_s(case: Case) -> None:
    keys, rows = layout.alid_index(case.alids)
    # The engine's writer casts its keys with numpy's own `|S32`; this is that cast, not ogld's.
    expected = np.array(case.keys, dtype=f"S{layout.ALID_INDEX_WIDTH}")
    assert keys.shape == (len(case.keys), layout.ALID_INDEX_WIDTH)
    np.testing.assert_array_equal(keys.view(f"S{layout.ALID_INDEX_WIDTH}").ravel(), expected)
    assert rows.tolist() == case.rows
    assert len(case.alids) - len(rows) == case.left_out


def test_a_wider_alid_is_left_out_never_truncated() -> None:
    alid = "22:100:" + "A" * 50
    keys, rows = layout.alid_index(["22:100:A:C", alid])
    assert rows.tolist() == [0]
    assert keys.view(f"S{layout.ALID_INDEX_WIDTH}").ravel().tolist() == [b"22:100:A:C"]
    # A truncating build would have made the second row's key the first one's, so an index
    # with two equal keys: the width is what keeps them apart.
    assert len(np.unique(keys.view(f"S{layout.ALID_INDEX_WIDTH}"))) == len(rows)


def test_the_index_is_the_variant_table_s_longer_entries() -> None:
    # A real store: the Test Store's own ALIDs, its 8 wider than the slot included.
    reader = StoreReader.open(TEST_STORE)
    alids = reader.alids_of(np.arange(reader.provenance["n_variants"]))
    keys, rows = layout.alid_index(alids)
    wide = [alid for alid in alids.tolist() if len(alid.encode("utf-8")) > 32]
    assert len(wide) == 8, "the Test Store holds ALIDs wider than the slot"
    assert len(rows) == len(alids) - len(wide)
    assert sorted(keys.view(f"S{layout.ALID_INDEX_WIDTH}").ravel().tolist()) == sorted(
        np.array([a for a in alids.tolist() if a not in set(wide)], dtype="|S32").tolist()
    )


# ------------------------------------------------------------------ what a store writes


def table(rows: list[tuple[str, int, str, str]]) -> Variants:
    """A Variant Set from (chromosome, position, ref, alt) records, by the identity function."""
    oriented = [orient(chrom, position, ref, alt) for chrom, position, ref, alt in rows]
    columns: dict[str, npt.NDArray[np.generic]] = {
        name: np.array([getattr(o, name) for o in oriented])
        for name in ("chrom", "alid", "a1", "a2")
    }
    columns["position"] = np.array([o.position for o in oriented], dtype=np.int64)
    columns["a1_frequency"] = np.full(len(oriented), 0.25)
    return Variants(**cast(Any, columns))


LONG = "ACGT" * 6
#: The Test Store's producer keys, for a store this file writes (tests/test_assembly.py does the
#: same): what only a producer knows, with nothing the store derives.
SOURCED: dict[str, Any] = {
    key: value
    for key, value in json.loads((TEST_STORE / "zarr.json").read_text())["attributes"][
        "ld_store"
    ].items()
    if key
    not in {
        "format_version",
        "encoding",
        "n_variants",
        "near_bp",
        "far_bp",
        "t_far",
        "t_far_q",
        "far_code_floor",
        "counts",
    }
}
SOURCED["panel"]["n"] = 771
SOURCED["adjustment"]["k"] = 10
SOURCED["adjustment"]["covariates"] = [f"PC{i}" for i in range(1, 11)]

#: One row's ALID is `22:100:ACGT:<20 bytes>` — the slot exactly; the next is a byte longer and
#: agrees with it over all 32, which is the engine's issue #127 at one position.
WIDTH_ROWS = [
    ("22", 100, LONG[:20], "ACGT"),
    ("22", 100, LONG[:21], "ACGT"),
    ("22", 200, "A", "C"),
    ("22", 300, "A", "G"),
]


def written(out: Path, rows: list[tuple[str, int, str, str]] = WIDTH_ROWS) -> Path:
    """A store this file writes: `rows`, at r = 1 on the diagonal and 0 elsewhere."""
    given = table(rows)
    writer = StoreWriter.create(
        out,
        variants=given,
        encoding=REncoding(511),
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
        n=771,
        k=10,
    )
    writer.write(writer.cut(0, np.eye(len(given))))
    writer.finish(cast(Sourced, SOURCED))
    return out


def test_a_store_writes_its_index_and_counts_what_it_leaves_out(tmp_path: Path) -> None:
    store = written(tmp_path / "store")
    root = zarr.open_group(store, mode="r")
    alid = root[f"{layout.ALID_INDEX}/alid"]
    assert alid.dtype == np.dtype("uint8")
    assert alid.shape == ((len(WIDTH_ROWS) - 1) * layout.ALID_INDEX_WIDTH,)
    assert root[f"{layout.ALID_INDEX}/row"].dtype == np.dtype("int32")
    keys, rows = layout.alid_index([orient(*r).alid for r in WIDTH_ROWS])
    np.testing.assert_array_equal(np.asarray(alid[:]), keys.ravel())
    np.testing.assert_array_equal(np.asarray(root[f"{layout.ALID_INDEX}/row"][:]), rows)
    provenance = StoreReader.open(store).provenance
    assert provenance["counts"]["alids_left_out"] == 1


def test_a_store_derives_its_index_from_its_own_table(tmp_path: Path) -> None:
    # The rule is the table's, not the producer's (ADR 0005): the store reads back its own ALID
    # column and indexes it, so a store written before the index gets the same one.
    store = written(tmp_path / "store")
    shutil.rmtree(store / layout.ALID_INDEX)
    reader = StoreReader.open(store)
    with pytest.raises(ValueError, match="alid_index"):
        reader.rows_of(["22:200:A:C"])
    written_from = zarr.open_group(store, mode="r+")
    left_out = assembly.derive_alid_index(
        written_from["variants"], written_from["variants"].create_group("alid_index")
    )
    assert left_out == 1
    alids = reader.alids_of(np.arange(len(WIDTH_ROWS)))
    found = reader.rows_of(list(alids))
    assert found.tolist() == list(range(len(WIDTH_ROWS)))


def test_a_lost_chunk_of_the_index_is_refused(tmp_path: Path) -> None:
    store = written(tmp_path / "store")
    alid = store / layout.ALID_INDEX / "alid"
    for chunk in alid.rglob("c/*"):
        chunk.unlink()
    reader = StoreReader.open(store)
    with pytest.raises(ValueError, match="alid_index/alid"):
        reader.rows_of(["22:200:A:C"])


# ------------------------------------------------------------------ reads on a crafted store


def rebuild(store: Path, alids: list[str], *, truncating: bool, unsorted: bool) -> None:
    """Write `store`'s index over again from `alids`: a build that truncates at the slot, or
    one whose entries are in no order. Both are ones validation fails; a reader must not be
    made wrong by them."""
    kept = [row for row, alid in enumerate(alids) if truncating or len(alid.encode()) <= 32]
    keys = np.array([alids[row] for row in kept], dtype=f"|S{layout.ALID_INDEX_WIDTH}")
    rows = np.array(kept, dtype=np.int32)
    if unsorted:
        order = np.random.default_rng(0).permutation(len(rows))
        keys, rows = keys[order], rows[order]
    root = zarr.open_group(store, mode="r+")
    root[f"{layout.ALID_INDEX}/alid"][:] = keys.view(np.uint8).ravel()
    root[f"{layout.ALID_INDEX}/row"][:] = rows


def answer(store: Path, alids: list[str]) -> list[int]:
    """Each ALID's row in `store`, read from its variant table alone: the oracle."""
    root = zarr.open_group(store, mode="r")
    rows = {
        str(alid): row for row, alid in enumerate(np.asarray(root["variants/alid"][:]).tolist())
    }
    return [rows.get(alid, -1) for alid in alids]


@dataclass(frozen=True)
class Crafted:
    """A store with one ALID at the slot's width and one a byte wider that shares it."""

    path: Path
    alids: list[str]


def crafted(tmp_path: Path) -> Crafted:
    store = written(tmp_path / "store")
    root = zarr.open_group(store, mode="r")
    alids = [str(alid) for alid in np.asarray(root["variants/alid"][:]).tolist()]
    assert len(alids[0].encode()) == layout.ALID_INDEX_WIDTH, alids[0]
    assert len(alids[1].encode()) == layout.ALID_INDEX_WIDTH + 1, alids[1]
    assert alids[1][:32] == alids[0], "the two indels must agree over the whole slot"
    return Crafted(store, alids)


def test_rows_of_finds_an_alid_at_the_slot_and_one_a_byte_wider(tmp_path: Path) -> None:
    built = crafted(tmp_path)
    assert StoreReader.open(built.path).rows_of(built.alids).tolist() == list(
        range(len(built.alids))
    )


@pytest.mark.parametrize("rebuild_with", [{"truncating": True}, {"unsorted": True}])
def test_rows_of_is_right_on_an_index_a_broken_build_wrote(
    tmp_path: Path, rebuild_with: dict[str, bool]
) -> None:
    built = crafted(tmp_path)
    rebuild(built.path, built.alids, **{"truncating": False, "unsorted": False} | rebuild_with)
    assert answer(built.path, built.alids) == list(range(len(built.alids)))
    assert StoreReader.open(built.path).rows_of(built.alids).tolist() == list(
        range(len(built.alids))
    )


def test_a_present_alid_is_found_without_loading_its_chromosome(tmp_path: Path) -> None:
    # The index is the variant table's, not the positions': a search reads keys and nothing
    # else, so the chromosome's positions and Near Band indptr stay unread (spec §10).
    built = crafted(tmp_path)
    reader = StoreReader.open(built.path)
    assert reader.rows_of([built.alids[2]]).tolist() == [2]
    assert reader._loaded == {}
    # An ALID at the slot's width takes the scan instead, which does read its chromosome.
    assert reader.rows_of([built.alids[0]]).tolist() == [0]
    assert reader._loaded != {}


def test_the_index_is_read_only_for_an_alid_it_could_hold(tmp_path: Path) -> None:
    # An ALID wider than the slot was never indexed (the engine's issue #127), and is found by
    # the exact scan. So a store whose index is lost still answers it.
    built = crafted(tmp_path)
    for chunk in (built.path / layout.ALID_INDEX / "alid").rglob("c/*"):
        chunk.unlink()
    reader = StoreReader.open(built.path)
    assert reader.rows_of([built.alids[1]]).tolist() == [1]


def test_the_derivation_writes_the_group_whole_or_leaves_nothing(tmp_path: Path) -> None:
    # `scripts/derive_alid_index.py` adds the index to a store written before it: the arrays go
    # under a group a reader ignores, and it takes the real name only when the count matches.
    script = load_script("derive_alid_index")
    store = written(tmp_path / "store")
    shutil.rmtree(store / layout.ALID_INDEX)
    # A store written before ADR 0010 records no count of what its index leaves out; the
    # derivation adds it (spec §7).
    meta = store / "zarr.json"
    document = json.loads(meta.read_text())
    del document["attributes"]["ld_store"]["counts"]["alids_left_out"]
    meta.write_text(json.dumps(document))
    assert script.derive(store) == 3
    assert (store / layout.ALID_INDEX / "row").exists()
    assert not (store / "variants" / "alid_index-partial").exists()
    assert StoreReader.open(store).provenance["counts"]["alids_left_out"] == 1
    assert StoreReader.open(store).rows_of(["22:200:A:C"]).tolist() == [2]


def test_the_derivation_leaves_the_store_as_it_was_when_the_count_differs(
    tmp_path: Path,
) -> None:
    script = load_script("derive_alid_index")
    store = written(tmp_path / "store")
    shutil.rmtree(store / layout.ALID_INDEX)
    meta = store / "zarr.json"
    document = json.loads(meta.read_text())
    document["attributes"]["ld_store"]["counts"]["alids_left_out"] = 3
    meta.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="provenance records"):
        script.derive(store)
    assert not (store / "variants" / "alid_index-partial").exists()
    assert not (store / layout.ALID_INDEX).exists()

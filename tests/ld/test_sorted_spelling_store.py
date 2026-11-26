"""The Test Store is spelt sorted, and a lookup in the other spelling is refused (#456, #462).

The committed Test Store (`tests/ld/data/test-panel/store/`, written by ogld before the merge) is
the independent record of the order and the ALIDs an LD build forms: the build's own pass over the
Test Panel and the Test Store's script must give the same Variant Set, row for row. And an ALID in
the engine's reference-first spelling of a deletion names no variant such a Store can hold, so a
lookup refuses it, naming the spelling, rather than reporting it Absent.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt
import pytest
import testdata
import zarr
from contract.support import TEST_STORE
from ld_script_modules import load_script
from ogquery.ld.store import StoreReader

import opengwasdb.ld
from opengwasdb.ld.build import passes
from opengwasdb.ld.build.variant_set import Settings, variant_set
from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.normalise import canonical_alid
from opengwasdb.variants.sorted_spelling import ReferenceFirstIndel


@cache
def committed() -> dict[str, npt.NDArray[np.str_]]:
    """The committed Test Store's variant columns, read straight from its arrays."""
    group = zarr.open_group(str(TEST_STORE), mode="r")
    return {
        name: np.array(group[f"variants/{name}"][:].tolist(), dtype=np.str_)
        for name in ("alid", "a1", "a2")
    }


@cache
def deletions() -> list[tuple[str, int, str, str]]:
    """The Test Store's deletions: passing Test Panel records whose REF is longer than ALT."""
    v = testdata.variants()
    rows = np.flatnonzero(v.passes_filters)
    found = [(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])) for r in rows]
    return [r for r in found if len(r[2]) > len(r[3])]


def test_the_committed_store_is_meaningful() -> None:
    alids = committed()["alid"]
    assert len(alids) > 1000
    assert len({a.split(":")[0] for a in alids}) == 2
    # Indels on both sides, and every deletion's REF sorts after its ALT, so each is spelt
    # differently by the engine's reference-first rule.
    assert len(deletions()) == 70
    assert all(ref > alt for _, _, ref, alt in deletions())
    a1, a2 = committed()["a1"], committed()["a2"]
    assert sum(len(a) != len(b) for a, b in zip(a1, a2, strict=True)) > 100


def test_the_committed_order_is_the_engine_s_order() -> None:
    alids = committed()["alid"].tolist()
    assert sorted(alids, key=alid_sort_key) == alids


def test_the_script_forms_the_committed_variant_set() -> None:
    table = load_script("make_test_store").variant_set().variants
    for name, column in committed().items():
        assert np.array_equal(getattr(table, name), column), name


def test_the_build_s_pass_forms_the_committed_variant_set() -> None:
    panel = testdata.panel()
    plan = passes.plan((panel.pgen,), panel.keep, panel.pcs, "test-panel")
    found = variant_set(plan.filesets, Settings("test-panel", 20, None, False), plan.samples.n)
    oriented = [o for c in found.chromosomes for o in c.oriented]
    assert [o.alid for o in oriented] == committed()["alid"].tolist()
    assert [o.a1 for o in oriented] == committed()["a1"].tolist()
    assert [o.a2 for o in oriented] == committed()["a2"].tolist()


@pytest.fixture(scope="module")
def reader() -> StoreReader:
    return StoreReader.open(TEST_STORE)


@pytest.mark.parametrize("alid", ["1:827587:GGGGCC:G", "21:28100086:AAATTT:A"])
def test_a_reference_first_deletion_is_refused_naming_the_spelling(
    reader: StoreReader, alid: str
) -> None:
    with pytest.raises(ReferenceFirstIndel, match=r"spells indels\s+sorted.*#462") as refused:
        reader.rows_of(["22:16050075:A:G", alid])
    assert refused.value.reason == "indel_spelt_reference_first"
    with pytest.raises(ReferenceFirstIndel):
        opengwasdb.ld.open(TEST_STORE).matrix([alid])


def test_the_refusal_of_a_held_deletion_names_the_row_that_holds_it(reader: StoreReader) -> None:
    # 21:28100086 AAATTT>A is in the Test Store, spelt sorted, and no insertion shares its site:
    # the refusal must point at that row, never say the Store lacks the variant (review of #456).
    held = "21:28100086:A:AAATTT"
    assert reader.rows_of([held]).tolist() == [51]
    with pytest.raises(ReferenceFirstIndel) as refused:
        reader.rows_of(["21:28100086:AAATTT:A"])
    words = " ".join(str(refused.value).split())
    assert f"would hold this variant as {held}" in words
    assert "is this deletion unless the panel also has the insertion at that site" in words
    assert "r there is signed against the other allele, A, not AAATTT (#462)" in words
    assert "holds no ALID" not in words and "if anything" not in words


def test_every_deletion_of_the_store_is_refused_in_the_engine_s_spelling(
    reader: StoreReader,
) -> None:
    # Each alone, by the short list's scan; then all at once, a list long enough for the index.
    engine = [canonical_alid(chrom, pos, ref, alt) for chrom, pos, ref, alt in deletions()]
    assert all(a not in set(committed()["alid"]) for a in engine)
    for alid in engine:
        with pytest.raises(ReferenceFirstIndel):
            reader.rows_of([alid])
    with pytest.raises(ReferenceFirstIndel):
        reader.rows_of(committed()["alid"].tolist() + engine)


def test_a_sorted_lookup_answers_as_before(reader: StoreReader) -> None:
    sorted_alids = [f"{c}:{p}:{alt}:{ref}" for c, p, ref, alt in deletions()]
    rows = reader.rows_of(sorted_alids)
    assert (rows >= 0).all()
    assert committed()["alid"][rows].tolist() == sorted_alids
    # An ALID neither spelling forms, or one the store does not hold, is Absent as before.
    absent = ["1:827587:G:GGGGCC", "22:100:G:A", "22:1:A:G", "rs123", "22:403:GAT:GT"]
    assert reader.rows_of(absent).tolist() == [-1] * len(absent)

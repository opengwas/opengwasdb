"""Tests for the OGS-00009/OGS-00010 comparison's fidelity classifier (#312).

The classifier is the part of that benchmark that can produce a wrong answer
looking like a right one: a difference filed under "the lift did it" that the
lift did not do would pass a Store with a defect. So every category is pinned on
a fixture that holds one of each, the re-orientation category is shown to depend
on the chain-derived expectation rather than on the data alone, and the chain
arithmetic that expectation rests on is checked against the package's own lift.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import _query_shapes
from benchmarks.benchmark_ogs00009_vs_ogs00010 import (
    Cells,
    ChainStrands,
    LiftMap,
    OtherStore,
    Tally,
    Tolerance,
    _blosc_of,
    _expected_orientations,
    _pair_one_sided,
    _relation,
    agreement,
    chain_orientation,
    compare_reads,
    complement_alid,
    complement_swaps,
    earlier_runs,
    gate_failures,
    is_palindromic,
    tolerance,
    typesize_section,
)
from benchmarks.compare_store_releases import discover_arrays
from opengwasdb.build.liftover import lifted_rows

TOL = Tolerance(se_log=0.004, eaf_logit=0.004)
ANALYSES = ["a1", "a2", "a3"]
NAN = np.nan


def _cells(rows: list[tuple[str, list[float], list[float], list[float]]]) -> Cells:
    """`(alid, z, se, eaf)` per row; a NaN z is an unobserved cell."""
    z = np.array([r[1] for r in rows], dtype=np.float32)
    se = np.array([r[2] for r in rows], dtype=np.float32)
    eaf = np.array([r[3] for r in rows], dtype=np.float32)
    return Cells(np.array([r[0] for r in rows]), z, se, eaf, np.isfinite(z) & np.isfinite(se))


SE = [0.1, 0.2, 0.3]
EAF = [0.3, 0.4, 0.05]
FLIPPED_EAF = [0.7, 0.6, 0.95]


def _fixture() -> tuple[Cells, Cells, LiftMap]:
    """One old/new read holding every category the classifier assigns."""
    old = _cells([
        ("1:100:A:G", [1.0, -2.0, NAN], SE, EAF),        # identical; a3 unobserved
        ("1:200:A:T", [1.5, 0.5, -3.0], SE, EAF),        # reverse-strand palindrome
        ("2:300:A:C", [2.0, 1.0, 0.25], SE, EAF),        # renamed by the lift
        ("7:500:C:T", [-1.0, 0.75, 2.5], SE, EAF),       # collapses onto 7:500:A:G
        ("7:500:A:G", [4.0, 4.0, 4.0], SE, EAF),         # the collapse target's own
        ("3:400:C:G", [0.5, 0.5, 0.5], SE, EAF),         # one cell will not match
        ("4:600:C:T", [0.2, 0.3, 0.4], SE, EAF),         # a collision both Stores share
        ("9:1:A:C", [NAN, NAN, NAN], SE, [NAN] * 3),     # a panel-only row
    ])
    new = _cells([
        ("1:100:A:G", [1.0, -2.0, 6.0], SE, EAF),        # a3 observed only in new
        ("1:200:A:T", [-1.5, -0.5, 3.0], SE, FLIPPED_EAF),
        ("2:300:G:T", [-2.0, -1.0, -0.25], SE, FLIPPED_EAF),
        ("7:500:A:G", [1.0, -0.75, -2.5], SE, FLIPPED_EAF),
        ("3:400:C:G", [0.5, 9.0, 0.5], SE, EAF),
        ("4:600:C:T", [0.2, 0.3, 0.4], SE, EAF),
    ])
    old_axis = {str(a) for a in old.alid[:-1]}
    new_axis = {str(a) for a in new.alid}
    # The parents' blank source_alids: the collapse target in the new Store,
    # and a collision both Stores carry.
    lift = LiftMap(old_axis, new_axis, collided_old={"4:600:C:T"},
                   collided_new={"7:500:A:G", "4:600:C:T"}, expected={"1:200:A:T": "flip"})
    _pair_one_sided(lift, old_axis - new_axis, new_axis - old_axis)
    return old, new, lift


def test_fixture_pairs_a_rename_and_a_collapse():
    """The fixture must reach the rename and collapse paths, or it proves nothing."""
    _, _, lift = _fixture()
    assert lift.renamed == {"2:300:A:C": "2:300:G:T"}
    assert lift.collapsed == {"7:500:C:T": "7:500:A:G"}
    assert not lift.dropped and not lift.unpaired_new


def test_every_category_is_assigned_and_every_old_cell_counted_once():
    old, new, lift = _fixture()
    tally = Tally()
    compare_reads(old, new, lift, TOL, ANALYSES, tally)

    assert tally.cells == {
        "identical": 4,          # 1:100 a1,a2 and 3:400 a1,a3
        "reoriented": 3,
        "renamed": 3,
        "collision": 9,          # both old rows of the collapse, and the shared collision
        "unexplained": 1,        # 3:400:C:G a2
        "unexplained_new_only": 1,
        "collision_new_only": 0,
    }
    # Of the (d) cells the new Store reproduces the renamed source's (3) and the
    # shared collision's (3), not the collapse target's own.
    assert tally.collision_agreeing == 6
    assert sum(v for k, v in tally.cells.items() if not k.endswith("_new_only")) == (
        tally.old_observed_cells
    ) == 20
    reasons = sorted(example["reason"] for example in tally.examples)
    assert reasons == ["identical: mismatch", "observed only in new"]
    mismatch = next(e for e in tally.examples if e["reason"] == "identical: mismatch")
    assert mismatch["analysis_id"] == "a2"
    assert (mismatch["old"]["z"], mismatch["new"]["z"]) == (0.5, 9.0)


def test_a_flipped_palindrome_is_unexplained_without_the_chain_expectation():
    """Re-orientation is filed under (a) because the chain says so, not because it fits."""
    old, new, lift = _fixture()
    lift.expected.clear()
    tally = Tally()
    compare_reads(old, new, lift, TOL, ANALYSES, tally)
    assert tally.cells["reoriented"] == 0
    assert tally.cells["unexplained"] == 1 + 3


def test_a_shared_collision_is_d_even_when_both_stores_agree():
    """A collision both Stores carry is invisible to the diff, so it is (d), not identical."""
    old, new, lift = _fixture()
    lift.collided_old.clear()
    lift.collided_new.discard("4:600:C:T")
    tally = Tally()
    compare_reads(old, new, lift, TOL, ANALYSES, tally)
    assert tally.cells["identical"] == 4 + 3
    assert tally.cells["collision"] == 6


def test_a_collapse_the_parents_do_not_mark_as_a_collision_fails_loudly():
    """Two old rows landing on one new row must be a recorded build collision."""
    old, new, lift = _fixture()
    lift.collided_new.discard("7:500:A:G")
    with pytest.raises(SystemExit, match="does not record as a build collision"):
        compare_reads(old, new, lift, TOL, ANALYSES, Tally())


def test_a_dropped_variant_without_lift_evidence_is_unexplained():
    old, new, lift = _fixture()
    lift.renamed.pop("2:300:A:C")
    lift.dropped.add("2:300:A:C")
    lift.unpaired_new.add("2:300:G:T")
    tally = Tally()
    compare_reads(old, new, lift, TOL, ANALYSES, tally)
    assert tally.cells["renamed"] == 0
    assert tally.cells["unexplained"] == 1 + 3
    assert tally.cells["unexplained_new_only"] == 1 + 3
    assert any(e["reason"].startswith("present only in old") for e in tally.examples)


def test_a_palindrome_the_chain_cannot_orient_is_counted_apart_from_a():
    old, new, lift = _fixture()
    lift.expected["1:200:A:T"] = "ambiguous"
    tally = Tally()
    compare_reads(old, new, lift, TOL, ANALYSES, tally)
    assert tally.cells["reoriented"] == 0
    assert tally.cells["palindrome_ambiguous"] == 3
    assert tally.record()["summary"]["palindrome_strand_ambiguous"] == 3


def test_a_panel_row_holding_an_observed_cell_fails_loudly():
    old, new, lift = _fixture()
    panel = int(np.flatnonzero(old.alid == "9:1:A:C")[0])
    old.observed[panel, 0] = True
    old.z[panel, 0] = 1.0
    with pytest.raises(SystemExit, match="panel-only row"):
        compare_reads(old, new, lift, TOL, ANALYSES, Tally())


def test_agreement_tolerates_one_step_of_se_and_eaf_but_not_of_z():
    a = {"z": np.float32([1.0]), "se": np.float32([0.2]), "eaf": np.float32([0.3])}
    within = {"z": np.float32([1.0]), "se": np.float32([0.2 * np.exp(0.0039)]),
              "eaf": np.float32([0.3005])}
    assert agreement(a, within, flipped=False, tol=TOL)[0]
    for name, value in (("z", 1.0 + 1 / 1024), ("se", 0.2 * np.exp(0.005)), ("eaf", 0.302)):
        beyond = dict(within, **{name: np.float32([value])})
        assert not agreement(a, beyond, flipped=False, tol=TOL)[0], name


def test_agreement_compares_frequencies_of_zero_and_one_and_nan_exactly():
    a = {"z": np.float32([1.0, 1.0, 1.0]), "se": np.float32([0.2] * 3),
         "eaf": np.float32([0.0, 1.0, NAN])}
    flipped = {"z": np.float32([-1.0] * 3), "se": np.float32([0.2] * 3),
               "eaf": np.float32([1.0, 0.0, NAN])}
    assert agreement(a, flipped, flipped=True, tol=TOL).all()
    assert not agreement(a, flipped, flipped=False, tol=TOL).any()
    one_sided = dict(flipped, eaf=np.float32([1.0, 0.0, 0.5]))
    assert not agreement(a, one_sided, flipped=True, tol=TOL)[2]


def test_tolerance_is_the_sum_of_both_stores_half_steps():
    residual = {"kind": "int8_residual", "residual_range": 0.5}
    tol = tolerance([{"se": residual, "eaf": residual}] * 2)
    assert tol.se_log == pytest.approx(0.5 / 127 + 1e-6)
    with pytest.raises(SystemExit, match="float16"):
        tolerance([{"se": {"kind": "float16"}, "eaf": residual}])


def test_complement_rules_match_the_lift():
    assert complement_alid("2:300:A:C") == "2:300:G:T"
    assert complement_alid("1:200:A:T") == "1:200:A:T"
    assert is_palindromic("1:200:A:T") and not is_palindromic("2:300:A:C")
    # Every reverse-strand SNP swaps its A1, palindromes included (#288).
    assert all(complement_swaps(f"1:1:{a}:{b}") for a, b in
               (("A", "C"), ("A", "G"), ("C", "T"), ("G", "T"), ("A", "T"), ("C", "G")))


def _write_chain(path: Path) -> Path:
    """A reverse chain of two blocks with a gap, offset in both assemblies, and a forward one."""
    path.write_text(
        "chain 1000 chr5 100000 + 1000 3000 chr5 90000 - 2000 4000 1\n"
        "1000 100 100\n900\n\n"
        "chain 1000 chr6 100000 + 0 5000 chr6 100000 + 10000 15000 2\n5000\n\n",
        encoding="utf-8",
    )
    return path


def test_chain_strands_agree_with_the_package_lift(tmp_path):
    """The hg38-side reading of a reverse block must invert pyliftover's own mapping."""
    chain_path = _write_chain(tmp_path / "test.chain")
    chain = ChainStrands(chain_path)
    sources = [("chr5", p, "A", "C") for p in (1001, 1500, 2000, 2101, 2999)]
    rows, _, failures = lifted_rows(sources, chain_file=chain_path)
    assert failures == 0 and len(rows) == len(sources)
    for key, row in rows.items():
        assert row.strand == "-"
        assert chain.covered("-", np.array([row.chromosome]), np.array([row.position]))[0]
        assert not chain.covered("+", np.array([row.chromosome]), np.array([row.position]))[0]
        assert (key[0].removeprefix("chr"), key[1]) in chain.reverse_sources(
            row.chromosome, row.position)
    forward = lifted_rows([("chr6", 100, "A", "C")], chain_file=chain_path)[0]
    (row,) = forward.values()
    assert chain.covered("+", np.array(["6"]), np.array([row.position]))[0]
    assert not chain.covered("-", np.array(["6"]), np.array([row.position]))[0]
    assert chain.sources("6", row.position) == [("6", 100, "+")]
    gap = 90000 - 3000  # the hg38 base under the chain's 100-base gap, 1-based edge
    assert not chain.covered("-", np.array(["5"]), np.array([gap - 50]))[0]


def test_common_query_patterns_times_the_regional_alids_it_is_given():
    """Two Stores resolve one window to different variants; the caller pins them (#312)."""
    calls: list[tuple[list[str], list[str]]] = []

    class FakeQuery:
        _variant_axis = None

        def lookup(self, alids: list[str], analyses: list[str]) -> dict[str, np.ndarray]:
            calls.append((alids, analyses))
            return {"z": np.zeros(len(alids))}

    patterns = _query_shapes.common_query_patterns(
        FakeQuery(), exposure="e", phewas_alid="p", region=("1", 1, 2),
        random_alids=["r"], random_analyses=["a"], regional=["1:1:A:C", "1:2:A:G"],
    )
    patterns["regional_one_analysis"]()
    assert calls == [(["1:1:A:C", "1:2:A:G"], ["e"])]


def test_a_rare_frequency_flipped_near_one_agrees_within_float32_rounding():
    """The real renamed cell (1:145973038, ukb-b-19756) the half-step alone rejected.

    Flipped to 0.99855, the frequency keeps only ~4e-5 of logit precision in
    float32, so the comparison must allow for it -- while a frequency 10% off
    is still a difference.
    """
    a = {"z": np.float32([-1.0712890625]), "se": np.float32([0.037820689]),
         "eaf": np.float32([0.0014518456300720572])}
    b = {"z": np.float32([1.0712890625]), "se": np.float32([0.03789634257]),
         "eaf": np.float32([0.9985538721084595])}
    residual = {"kind": "int8_residual", "residual_range": 0.5}
    tol = tolerance([{"se": residual, "eaf": residual}] * 2)  # both Stores' encodings
    fa, fb = float(a["eaf"][0]), float(b["eaf"][0])
    gap = abs(np.log(fb / (1 - fb)) + np.log(fa / (1 - fa)))
    assert gap > tol.eaf_logit, "the fixture must exceed the half-steps on their own"
    assert agreement(a, b, flipped=True, tol=tol)[0]
    off = dict(b, eaf=np.float32([1 - 0.0016]))
    assert not agreement(a, off, flipped=True, tol=tol)[0]


def test_typesize_section_flags_an_int8_plane_stored_at_typesize_four(tmp_path, monkeypatch):
    """The 0.2.0 SE plane inherits its float32 scratch plane's typesize (#312)."""
    import zarr

    from benchmarks import benchmark_ogs00009_vs_ogs00010 as module

    monkeypatch.setattr(module, "TYPESIZE_SAMPLE_CHUNKS", 4)
    root = zarr.open_group(tmp_path / "store" / "data.zarr", mode="w", zarr_format=3)
    codes = np.random.default_rng(1).integers(-20, 20, size=(200, 128)).astype("int8")
    for name, typesize in (("se", 4), ("eaf", 1)):
        array = root.create_array(
            name, shape=codes.shape, chunks=(100, 64), shards=(200, 128), dtype="int8",
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle",
                                               typesize=typesize),
        )
        array[:] = codes

    class Query:
        _root = root

    structure = {"new": discover_arrays(tmp_path / "store")}
    assert structure["new"]["data.zarr/se"]["compressor"]["typesize"] == 4, "fixture must reach it"
    rows = {r["plane"]: r for r in typesize_section({"new": Query()}, structure, 0)["new"]}

    assert "recompressed" not in rows["eaf"]
    sample = rows["se"]["recompressed"]
    assert rows["se"]["stored_typesize"] == 4 and rows["se"]["itemsize"] == 1
    assert sample["chunks"] == 4
    assert sample["bytes_at_stored_typesize"] > sample["bytes_at_element_size"]
    assert sample["ratio"] > 1.0


def _write_two_way_chain(path: Path) -> Path:
    """hg19 chr8:[0, 5000) maps forward onto hg38 chr8:[10000, 15000) and also, through a
    second chain, in reverse onto chr8:[20000, 25000) -- a region the chain maps twice."""
    path.write_text(
        "chain 1000 chr8 100000 + 0 5000 chr8 100000 + 10000 15000 1\n5000\n\n"
        "chain 900 chr8 100000 + 0 5000 chr8 100000 - 75000 80000 2\n5000\n\n"
        "chain 1000 chr5 100000 + 1000 3000 chr5 90000 - 2000 4000 3\n1000 100 100\n900\n\n",
        encoding="utf-8",
    )
    return path


def test_chain_orientation_decides_from_the_source_alone(tmp_path):
    """`either` palindromes take their orientation from the chain, never from the cells."""
    from pyliftover import LiftOver

    engine = LiftOver(str(_write_two_way_chain(tmp_path / "two-way.chain")))
    (reverse,) = lifted_rows([("chr5", 1500, "A", "T")], chain_file=tmp_path / "two-way.chain")[
        0].values()
    assert reverse.strand == "-", "the fixture must reach a reverse-only block"
    alid = f"5:{reverse.position}:A:T"
    assert chain_orientation(engine, "5:1500:A:T", alid) == "flip"
    assert chain_orientation(engine, None, alid) == "ambiguous"
    assert chain_orientation(engine, ".", alid) == "ambiguous"
    assert chain_orientation(engine, "5:1500:A:T", "5:1:A:T") == "ambiguous"  # lands elsewhere
    mapped = engine.convert_coordinate("chr8", 99)
    assert {m[2] for m in mapped} == {"+", "-"}, "the fixture must map one source both ways"
    first = mapped[0]
    assert chain_orientation(engine, "8:100:A:T", f"8:{first[1] + 1}:A:T") == "ambiguous"


def test_agreement_rejects_an_eaf_just_past_the_bound():
    """A mid-range frequency, where float32 rounding is negligible, either side of the bound."""
    from scipy.special import expit, logit

    residual = {"kind": "int8_residual", "residual_range": 0.5}
    tol = tolerance([{"se": residual, "eaf": residual}] * 2)
    f = 0.3
    bound = tol.eaf_logit + 2 * 2.0**-24 * (2 / f)
    a = {"z": np.float32([1.0]), "se": np.float32([0.2]), "eaf": np.float32([f])}
    for scale, agrees in ((0.995, True), (1.005, False)):
        b = dict(a, eaf=np.float32([expit(logit(f) + scale * bound)]))
        assert bool(agreement(a, b, flipped=False, tol=tol)[0]) is agrees, scale


def test_blosc_of_refuses_a_codec_it_does_not_know():
    base = {"cname": "zstd", "clevel": 3, "shuffle": "bitshuffle", "blocksize": 0}
    assert _blosc_of(base).cname == "zstd"
    for bad, match in (({"codecs": ["gzip"]}, "no Blosc codec"),
                       ({**base, "shuffle": "delta"}, "unknown Blosc shuffle"),
                       ({**base, "cname": "brotli"}, "unknown Blosc compressor")):
        with pytest.raises(SystemExit, match=match):
            _blosc_of(bad)


def _gated(**changes: object) -> dict[str, object]:
    lift_axis = {"one_sided_outside_reverse_block": [], "renamed_pairs": 9,
                 "lift_verified_renamed": 9, "collapsed": [{}, {}], "lift_verified_collapsed": 2,
                 "expected_reoriented_total": 5, "lift_verified_flip": 5}
    return {"lift_axis": {**lift_axis, **changes}}


@pytest.mark.parametrize(("changes", "named"), [
    ({"one_sided_outside_reverse_block": ["1:1:A:C"]}, "outside every reverse-strand"),
    ({"lift_verified_renamed": 8}, "lift_verified_renamed: 8 of 9"),
    ({"lift_verified_collapsed": 1}, "lift_verified_collapsed: 1 of 2"),
    ({"lift_verified_flip": 4}, "lift_verified_flip: 4 of 5"),
])
def test_gate_failures_names_each_lift_check_that_falls_short(changes, named):
    assert gate_failures(_gated()) == []
    failures = gate_failures(_gated(**changes))
    assert len(failures) == 1 and named in failures[0]


def _other_store() -> OtherStore:
    rows = {alid: SimpleNamespace(rsid=rsid) for alid, rsid in (
        ("7:100:A:G", "rs848673"), ("10:5:C:T", "rsX"), ("4:9:C:T", "rsY"), ("3:1:A:C", "rsP"))}
    return OtherStore(observed={"7:100:A:G", "10:5:C:T", "4:9:C:T"},
                      collisions={"10:5:C:T": "rsX", "4:9:C:T": "rsY"}, axis_row=rows.get)


def test_relation_looks_on_the_other_stores_whole_axis():
    """A new collision whose ALID the other Store held, uncollided, must say so (#312 r2)."""
    other = _other_store()
    shared = _relation("4:9:C:T", other, "new")
    assert (shared["relation"], shared["other_store_collision"]) == ("shared", True)
    renamed = _relation("10:5:A:G", other, "new")
    assert (renamed["relation"], renamed["other_store_alid"]) == ("shared, renamed by #288",
                                                                  "10:5:C:T")
    held = _relation("7:100:A:G", other, "new")
    assert held == {"relation": "new", "other_store_alid": "7:100:A:G",
                    "other_store_rsid": "rs848673", "other_store_row": "observed",
                    "other_store_collision": False}
    panel = _relation("3:1:A:C", other, "old")
    assert (panel["relation"], panel["other_store_row"]) == ("gone", "panel-only")
    absent = _relation("5:5:A:C", other, "new")
    assert (absent["relation"], absent["other_store_alid"]) == ("new", None)


def _dual_orientation_chain(path: Path) -> Path:
    """hg38 chr8:[10000, 15000) under a forward block (from hg19 chr8) and a reverse one
    (from hg19 chr9); hg38 chr7:[49000, 50000) under a reverse block only."""
    path.write_text(
        "chain 1000 chr8 100000 + 0 5000 chr8 100000 + 10000 15000 1\n5000\n\n"
        "chain 900 chr9 100000 + 0 5000 chr8 100000 - 85000 90000 2\n5000\n\n"
        "chain 800 chr7 100000 + 0 1000 chr7 100000 - 50000 51000 3\n1000\n\n",
        encoding="utf-8")
    return path


class _StubLift:
    """`convert_coordinate` answers for three hg19 sources, one per verdict."""

    MAPPED = {("chr8", 499): [("chr8", 10499, "+", 1)],
              ("chr9", 999): [("chr8", 10999, "-", 1)],
              ("chr8", 1999): [("chr8", 11999, "+", 1), ("chr8", 11999, "-", 1)]}

    def convert_coordinate(self, chrom: str, pos: int) -> list[tuple[str, int, str, int]]:
        return self.MAPPED.get((chrom, pos), [])


def test_each_chain_verdict_maps_to_the_orientation_the_cells_are_held_to(tmp_path):
    chain = ChainStrands(_dual_orientation_chain(tmp_path / "dual.chain"))
    shared = pd.DataFrame({
        "chrom": ["7", "8", "8", "8", "8", "8"],
        "pos": [49500, 10500, 11000, 12000, 13000, 14000],
        "alid": ["7:49500:A:T", "8:10500:A:T", "8:11000:C:G", "8:12000:A:T", "8:13000:A:G",
                 "8:14000:A:T"],
    })
    sources = {"8:10500:A:T": "8:500:A:T", "8:11000:C:G": "9:1000:C:G",
               "8:12000:A:T": "8:2000:A:T"}
    lift = LiftMap(set(), set(), collided_old={"8:14000:A:T"})
    counts = _expected_orientations(lift, shared, chain, sources.get, _StubLift())
    assert lift.expected == {"7:49500:A:T": "flip", "8:11000:C:G": "flip",
                             "8:12000:A:T": "ambiguous"}
    assert {k: counts[k] for k in counts if k.startswith("expected_")} == {
        "expected_flip": 1, "expected_either_flip": 1, "expected_either_same": 1,
        "expected_either_ambiguous": 1, "expected_either_collision": 1, "expected_same": 1,
        "expected_reoriented_total": 2}


def test_earlier_runs_copies_each_runs_warm_timings_with_its_provenance(tmp_path):
    path = tmp_path / "r0.json"
    timings = [{"query": "phewas", "median_ms": 25.0, "warm_ms": [24.3, 25.0, 27.4]}]
    path.write_text(json.dumps({"commit": "abc1234", "measured_at": "2026-10-24T11:20:14",
                                "stores": [{"label": "new", "timings": timings}]}),
                    encoding="utf-8")
    (run,) = earlier_runs([f"round 0={path}"])
    assert (run["label"], run["commit"]) == ("round 0", "abc1234")
    assert run["timings"]["new"]["phewas"] == {"median_ms": 25.0, "min_ms": 24.3,
                                               "max_ms": 27.4, "reps": 3}
    assert len(run["sha256"]) == 64
    with pytest.raises(SystemExit, match="LABEL=PATH"):
        earlier_runs([str(path)])

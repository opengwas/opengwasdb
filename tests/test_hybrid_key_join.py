"""A Hybrid build's Pass 2 and fold, pinned against the spills they wrote at `0055dd6`.

Each build in `hybrid_pass2_fixtures` meets the cases the key join has to keep:
Overflow routing, off-reference routing, a site spelt in two cases, and
duplicate rows judged as Admission judges them (ADR 0072, D5). Each case is
shown present before the spills are compared, so a golden that lost one fails
here rather than passing on nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import hybrid_pass2_fixtures as fixtures
import numpy as np
import pytest
from admission_helpers import record_analyses
from extraction_fixtures import extract_from_manifest
from liftover_fixtures import write_two_strand_chain
from pass2_fixtures import _manifest

from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.variants.reference import write_variant_reference

GOLDEN = json.loads(fixtures.GOLDEN.read_text(encoding="utf-8"))
STAGES = ("dense", "overflow", "off_reference", "folded")


@pytest.fixture(scope="module")
def builds(tmp_path_factory) -> dict[str, tuple[Path, fixtures.Spills]]:
    root = tmp_path_factory.mktemp("hybrid-pass2")
    built = {}
    for which in fixtures.BUILDS:
        with pytest.MonkeyPatch.context() as patch, fixtures.captured_spills(patch) as spills:
            built[which] = (fixtures.build(root / which, which), spills)
    return built


def _length(spills: fixtures.Spills, stage: str, col: int) -> int:
    arrays = spills[stage][col]
    return len(arrays[next(iter(arrays))])


def _dropped(store: Path) -> dict[str, dict[str, int]]:
    return {entry["analysis_id"]: entry["dropped"] for entry in record_analyses(store)}


# ── the fixtures hold their cases ────────────────────────────────────────────


def test_two_pass_sends_every_indel_to_the_overflow_and_nothing_off_reference(builds):
    _, spills = builds["two_pass"]
    assert [_length(spills, "overflow", col) for col in (0, 1)] == [5, 5]
    assert [_length(spills, "dense", col) for col in (0, 1)] == [5, 2]
    assert [_length(spills, "off_reference", col) for col in (0, 1)] == [0, 0]
    assert spills["folded"] == {}


def test_single_pass_leaves_the_second_analyses_own_sites_off_reference(builds):
    store, spills = builds["single_pass"]
    assert [_length(spills, "off_reference", col) for col in (0, 1)] == [1, 3]
    # Its deletion's key is hashed, so it carries a raw string; its two SNVs pack.
    assert spills["off_reference"][1]["hashed_raw"].tolist() == ["3:13000:AT:A"]
    assert {"3:12000:C:T", "3:13000:AT:A"} <= set(fixtures.axis(store))
    # The fold put two of its off-reference rows onto the axis; the rest left it.
    assert _length(spills, "folded", 1) == 2


def test_single_pass_with_a_panel_routes_on_reference_indels_to_the_overflow(builds):
    _, spills = builds["single_pass_panel"]
    assert [_length(spills, "overflow", col) for col in (0, 1)] == [5, 3]
    assert [_length(spills, "folded", col) for col in (0, 1)] == [5, 5]
    # The fold added the off-reference rows to the Overflow the column already had.
    assert _length(spills, "folded", 1) > _length(spills, "overflow", 1)


@pytest.mark.parametrize("which", fixtures.BUILDS)
def test_a_site_spelt_in_two_cases_is_one_variant_and_duplicates_are_judged(builds, which):
    store, _ = builds[which]
    assert fixtures.axis(store).count("3:7000:C:T") == 1
    dropped = _dropped(store)["alpha"]
    assert dropped["duplicate_collapsed"] == 2
    assert dropped["duplicate_conflicting"] == 2


# ── the spills are the ones written at 0055dd6 ───────────────────────────────


@pytest.mark.parametrize("which", fixtures.BUILDS)
def test_hybrid_spills_are_the_ones_the_build_wrote_before_the_join(builds, which):
    store, spills = builds[which]
    pinned = GOLDEN[which]
    assert fixtures.axis(store) == pinned["alids"]
    for stage in STAGES:
        assert sorted(map(str, spills[stage])) == sorted(pinned[stage]), (which, stage)
        for col, arrays in spills[stage].items():
            for name, array in arrays.items():
                expected = pinned[stage][str(col)][name]
                where = (which, stage, col, name)
                assert str(array.dtype) == expected["dtype"], where
                np.testing.assert_array_equal(
                    array, np.array(expected["values"], dtype=expected["dtype"]),
                    err_msg=str(where),
                )


# ── a refusal a route table makes reaches the Hybrid build unchanged ─────────


def test_a_reference_sending_one_routing_key_two_ways_refuses_the_hybrid_build(tmp_path):
    """Two spellings of one site are one Routing Key (ADR 0075), so a Variant
    Reference sending them to different variants is refused, and no Store is
    left."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    manifest = _manifest(tmp_path, ["alpha"])
    reference = tmp_path / "ambiguous.variant-ref.tsv.gz"
    extract_from_manifest(manifest, reference, chain_file=chain, liftover_failure_threshold=0.5)
    ambiguous = tmp_path / "ambiguous-spelling.variant-ref.tsv.gz"
    write_variant_reference(
        ambiguous,
        ["3:1000:A:G", "3:2000:A:G"],
        {("3", 1000, "a", "g"): "3:1000:A:G", ("3", 1000, "A", "G"): "3:2000:A:G"},
        reoriented=(),
    )
    store = tmp_path / "refused.opengwasdb"

    with pytest.raises(ValueError, match="conflicting variants after case folding") as refused:
        hybrid_build.build_hybrid_from_vcf_manifest(
            manifest, store, variant_reference=ambiguous, store_id="s", release_id="r",
            chain_file=chain, liftover_failure_threshold=0.5,
        )
    assert "'3:1000:A:G'" in str(refused.value) or "3:1000:" in str(refused.value)
    assert not store.exists()

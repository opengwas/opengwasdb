"""Reference Completion writes the Store it wrote before Phase 3 streamed (#424).

Every completed release here is compared, file by file and array by array, with
the digest the code at `1ab2b09` wrote for the same fixture
(`tests/golden/completion_stores.json`, see `store_digest`). Ragged completion's
merge, assembly and write were rebuilt to stream, and Dense's fill shards and
duplicate rule moved into `opengwasdb.completion`; neither may change a byte a
reader decodes.

The fixtures are the existing completion tests' own, so each pin covers what
those tests already reason about: a residual SE plane coded against its
source's model (#301), with imputed cells outside the residual range and an
observed exception carried exactly; the ancestry-match filter admitting one
Analysis or none; a release whose only frequencies are the panel's; an Analysis
with no Trait position; and LD blocks that fill one variant twice, agreeing at
one and disagreeing at another (D5).

Two manifest entries are pinned as the code writes them since #381 rather than
as `1ab2b09` wrote them (ADR 0079): `format_version` is 0.2.1, and the
completion record states how the panel placed its rows (`panel_indels`,
`panel_snvs_unplaced`). Each fixture's panel places every row, so those counts
hold no unplaced row, and every other value in a pin is `1ab2b09`'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_dense_completion as tdc
import test_ragged_completion as trc
import test_ragged_completion_keeps_observed_se as keeps
from region_tables import NO_SIGNIFICANT_REGIONS, every_analysis, with_regions
from store_digest import GOLDEN, matches_pin
from stored_rows import stored_rows

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.variants.axis import VariantAxis

# The fixtures these pins reuse, bound here so pytest finds them.
completions = keeps.completions
off_grid = keeps.off_grid
signal_source_path = tdc.signal_source_path
signal_observed_store = tdc.signal_observed_store
signal_panel_npz_only = tdc.signal_panel_npz_only

#: The panel-only variants of the overlapping-block panel. Blocks A and B are one
#: block written twice, so both fill `AGREED` with one value; block C also holds
#: `DISPUTED`, with its own LD, so its fill there disagrees with theirs.
AGREED = "1:1050500:C:T"
DISPUTED = "1:1125000:A:G"


def _ragged_totals(result) -> dict[str, int]:
    return {
        name: int(getattr(result, name))
        for name in ("n_variants", "n_analyses", "n_associations", "n_imputed", "n_missing")
    }


def _observed_rich(root: Path) -> Path:
    out = root / "rich_obs.opengwasdb"
    build_ragged_from_besd(
        trc._make_rich_besd_fixture(root), out, store_id="test", release_id="obs-v1",
        tissue="Blood", association_coverage="cis_and_signals",
    )
    return out


def _overlapping_panel(root: Path) -> Path:
    """Three chr-1 blocks over the rich fixture's variants (`AGREED`, `DISPUTED`).

    The seeds are ones whose LD imputes both variants in A and B and `DISPUTED`
    in C: at 1ab2b09 the completion logs three fills disagreeing at `DISPUTED`.
    """
    observed = [(alid, eaf, bp) for alid, eaf, bp in trc._RICH_OBSERVED]
    agreed, disputed = (AGREED, 0.40, 1_050_500), (DISPUTED, 0.25, 1_125_000)
    blocks = {
        "900000-1200000": (sorted([*observed, agreed, disputed], key=lambda r: r[2]), 4),
        "900000-1250000": (sorted([*observed, agreed, disputed], key=lambda r: r[2]), 4),
        "900000-1210000": (sorted([*observed, disputed], key=lambda r: r[2]), 54),
    }
    chromosome = root / "panel" / "EUR" / "1"
    chromosome.mkdir(parents=True)
    for name, (rows, seed) in blocks.items():
        scratch = root / "scratch" / name
        keeps._write_panel(scratch, rows, seed)
        for suffix in (".tsv", ".ldeig.npz"):
            (scratch / f"skewed{suffix}").rename(chromosome / f"{name}{suffix}")
    return root / "panel"


@pytest.fixture(scope="module")
def besd_completion(tmp_path_factory: pytest.TempPathFactory):
    """`test_ragged_completion`'s completed release, built with its arguments."""
    root = tmp_path_factory.mktemp("identity-besd")
    observed = root / "obs.opengwasdb"
    build_ragged_from_besd(
        trc._make_besd_fixture(root), observed, store_id="test", release_id="obs-v1",
        tissue="Blood", association_coverage="cis",
    )
    dst = root / "comp.opengwasdb"
    result = complete_ragged_store(
        observed, dst, trc._make_ld_panel(root, "1", 900_000, 1_300_000),
        ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
    )
    return dst, _ragged_totals(result)


@pytest.fixture(scope="module")
def trait_less_completion(tmp_path_factory: pytest.TempPathFactory):
    """`TestGeneTargetLessAnalyses`'s source and panel: an Analysis with no Trait
    position (issue #102). It was completed from the block its own variants fall
    in; it is now completed in the significant region its `regions.tsv` records
    (issue #430), which spans that block's panel variants."""
    root = tmp_path_factory.mktemp("identity-trait-less")
    observed = _observed_rich(root)
    trc.TestGeneTargetLessAnalyses._drop_trait_positions(observed)
    with_regions(observed, every_analysis(observed, "significant_trans", "1", 900_000, 1_150_000))
    dst = root / "comp.opengwasdb"
    result = complete_ragged_store(
        observed, dst, trc._make_rich_ld_panel(root), ancestry="EUR", min_cor=0.0,
        release_id="comp-v1",
    )
    return dst, _ragged_totals(result)


@pytest.fixture(scope="module")
def overlapping_completion(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("identity-overlap")
    dst = root / "comp.opengwasdb"
    result = complete_ragged_store(
        with_regions(_observed_rich(root), NO_SIGNIFICANT_REGIONS), dst,
        _overlapping_panel(root), ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        release_id="comp-v1",
    )
    return dst, _ragged_totals(result)


@pytest.fixture
def dense_completion(tmp_path, signal_observed_store, signal_panel_npz_only):
    """`test_dense_completion`'s imputing release: one Analysis, one AR(1) block,
    every position observed but one, which completion imputes."""
    dst = tmp_path / "comp.opengwasdb"
    result = complete_dense_store(
        signal_observed_store, dst, signal_panel_npz_only, ancestry="EUR", min_cor=0.0,
        release_id="comp-signal-v1",
    )
    totals = {
        name: int(getattr(result, name))
        for name in (
            "n_variants", "n_analyses", "n_imputed", "n_missing_off_panel",
            "n_missing_imputation_failed",
        )
    }
    return dst, totals


def _pins() -> dict:
    import json

    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_pins_are_not_vacuous():
    """Each pinned release holds what its test is about: imputed cells, missing
    cells, imputed SEs outside the residual range, and both an agreeing and a
    disagreeing pair of fills."""
    pins = _pins()
    def completion(name: str) -> dict:
        return pins[name]["store"]["manifest.json"]["provenance"]["completion"]

    for name in (
        "ragged_residual", "ragged_out_of_range", "ragged_off_grid", "ragged_trait_less",
        "ragged_overlap", "ragged_panel_only_eaf",
    ):
        assert completion(name)["n_imputed"] > 0, name
    for name in ("ragged_besd", "ragged_residual", "ragged_off_grid", "ragged_overlap"):
        assert completion(name)["n_missing"] > 0, name
    assert completion("ragged_nothing_imputed")["n_imputed"] == 0
    assert completion("ragged_out_of_range")["se_exceptions"]["over_budget"] == ["b"]
    assert pins["dense"]["totals"]["n_imputed"] > 0
    for name in pins:
        assert _unplaced(completion(name)) == {}, name


def _completion_record(store: Path) -> dict:
    """A completed release's `provenance.completion`."""
    import json

    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return manifest["provenance"]["completion"]


def _unplaced(completion: dict) -> dict[tuple[str, str], int]:
    """The panel rows a completion record says were not placed, by kind and reason."""
    counts = {
        **{("snv", reason): n for reason, n in completion["panel_snvs_unplaced"].items()},
        **{("indel", reason): n for reason, n in completion["panel_indels"]["unplaced"].items()},
    }
    return {key: n for key, n in counts.items() if n}


def test_the_overlapping_blocks_fill_one_variant_twice(overlapping_completion):
    """Both D5 outcomes are reached: `AGREED` is imputed once, and `DISPUTED`,
    which the blocks disagree on, is held missing."""
    store, _ = overlapping_completion
    assert _unplaced(_completion_record(store)) == {}, "the panel must place every row"
    rows = stored_rows(store, 0)
    axis = VariantAxis(store)
    try:
        alids = [axis.by_index(int(v)).alid for v in rows.variant_index]
    finally:
        axis.close()
    at = {alid: i for i, alid in enumerate(alids)}
    assert np.isfinite(rows.z[at[AGREED]]), "the agreeing fills must be one imputed cell"
    assert np.isnan(rows.z[at[DISPUTED]]), "the disagreeing fills must all be dropped"


@pytest.mark.parametrize(
    ("name", "run"),
    [
        ("ragged_residual", "in_range"),
        ("ragged_out_of_range", "out_of_range"),
        ("ragged_nothing_imputed", "nothing_imputed"),
    ],
)
def test_a_residual_completion_matches_its_pin(name, run, completions):
    completed = completions.runs[run]
    matches_pin(name, completed.path, {"n_imputed": int(completed.n_imputed)})


def test_a_carried_observed_exception_matches_its_pin(off_grid):
    matches_pin("ragged_off_grid", off_grid.completed.path, {
        "n_imputed": int(off_grid.completed.n_imputed)
    })


def test_a_panel_only_frequency_completion_matches_its_pin(panel_only_eaf_ragged_store):
    matches_pin("ragged_panel_only_eaf", panel_only_eaf_ragged_store, {})


@pytest.mark.parametrize(
    ("name", "fixture"),
    [
        ("ragged_besd", "besd_completion"),
        ("ragged_trait_less", "trait_less_completion"),
        ("ragged_overlap", "overlapping_completion"),
        ("dense", "dense_completion"),
    ],
)
def test_a_completion_matches_its_pin(name, fixture, request):
    store, totals = request.getfixturevalue(fixture)
    matches_pin(name, store, totals)

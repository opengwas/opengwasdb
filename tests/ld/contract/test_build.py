"""ogld#5's contract: the builder, on the Test Panel, against the independent reference r.

The Test Panel's `reference_r(k)` is a short numpy computation in lane 2d's script, checked
against plink2's `--r-unphased` at k = 0 by tests/test_testdata.py. The builder must match it
within the quantisation step: in the Near Band pair by pair, and in which pairs beyond 3 Mb
it keeps, except where a pair's code lies on the far code floor's rounding edge.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import support
from support import alids, contract, panel_data, passing_rows, reference

import ogld
from ogld import cli
from ogld.format import far_code_floor, t_far

OGLD_5 = contract("ogld#5")


def inputs(k: int = support.K, **overrides: object) -> ogld.BuildInputs:
    panel = panel_data().panel()
    fields: dict[str, object] = {
        "pgen": (panel.pgen,),
        "keep": panel.keep,
        "pcs": panel.pcs if k else None,
        "pc_source": "gnomAD v3.1.2 hgdp_tgp_meta.subcontinental_pca.pca_scores" if k else None,
        "panel_id": "test-panel-hgdp1kgp-EUR",
        "ancestry": "EUR",
        "genotype_source": "gnomAD v3.1.2 HGDP+1kG, the Test Panel's slice",
        "k": k,
        "info_min": None,
    }
    return ogld.BuildInputs(**{**fields, **overrides})


def built(tmp_path: Path, k: int) -> tuple[ogld.Provenance, ogld.LDStore]:
    provenance = ogld.build(inputs(k), tmp_path / f"k{k}")
    return provenance, ogld.open(tmp_path / f"k{k}")


def near_band_matches(store: ogld.LDStore, k: int) -> int:
    """Assert every reference pair within 3 Mb reads back within half a step; return how many."""
    ref = reference(k)
    near = ref.where(ref.distance <= 3_000_000, "pairs within 3 Mb")
    names = alids(passing_rows())
    m = store.matrix(names)
    where = {name: a for a, name in enumerate(names)}
    rows = alids(near.row_i)
    cols = alids(near.row_j)
    got = np.array([m.r[where[a], where[b]] for a, b in zip(rows, cols, strict=True)])
    worst = np.abs(got - near.r).max()
    assert worst <= support.HALF_STEP + support.SLACK, (
        f"a Near Band pair is {worst} from the reference"
    )
    return len(got)


@OGLD_5
def test_the_near_band_at_k10_matches_the_reference(tmp_path: Path) -> None:
    _, store = built(tmp_path, 10)
    assert near_band_matches(store, 10) > 1000


@OGLD_5
def test_the_near_band_at_k0_matches_the_reference(tmp_path: Path) -> None:
    # The reference at k = 0 is held to plink2 --r-unphased by tests/test_testdata.py.
    _, store = built(tmp_path, 0)
    assert near_band_matches(store, 0) > 1000


@OGLD_5
def test_the_far_pairs_kept_are_the_reference_s(tmp_path: Path) -> None:
    _, store = built(tmp_path, 10)
    ref = reference(10)
    far = ref.where((ref.distance > 3_000_000) & (ref.distance <= 10_000_000), "3–10 Mb pairs")
    names = alids(passing_rows())
    m = store.matrix(names)
    where = {name: a for a, name in enumerate(names)}
    kept = np.array(
        [
            not np.isnan(m.r[where[a], where[b]])
            for a, b in zip(alids(far.row_i), alids(far.row_j), strict=True)
        ]
    )
    expected = np.abs(far.code) >= support.FLOOR
    assert expected.any(), "the Test Panel must hold real Far Pairs at k = 10"
    assert (~expected).any(), "the Test Panel must hold 3–10 Mb pairs below t_far"
    # Only a pair whose r × 32767 lies within float error of the rounding edge may differ.
    edge = np.abs(np.abs(far.r) * 32767 - (support.FLOOR - 0.5)) < 1e-6
    assert ((kept == expected) | edge).all(), "a Far Pair was kept or dropped against the reference"


@OGLD_5
def test_the_provenance_records_what_was_built(tmp_path: Path) -> None:
    provenance, _ = built(tmp_path, 10)
    panel = panel_data().panel()
    assert provenance["panel"]["n"] == support.N_SAMPLES == panel.n_samples
    assert provenance["adjustment"]["k"] == 10
    assert provenance["adjustment"]["covariates"] == [f"PC{i}" for i in range(1, 11)]
    pc_file = provenance["adjustment"]["pc_file"]
    assert pc_file is not None
    assert pc_file["sha256"] == hashlib.sha256(Path(panel.pcs).read_bytes()).hexdigest()
    assert provenance["adjustment"]["pc_check"] is not None
    assert provenance["t_far"] == support.T_FAR == t_far(771, 10)
    assert provenance["far_code_floor"] == far_code_floor(support.T_FAR)
    assert provenance["source"]["kind"] == "genotypes"
    assert provenance["variant_filters"]["mac_min"] == 20
    assert provenance["n_variants"] == len(passing_rows())
    assert "zero_variance" in provenance["variants_dropped"]
    assert provenance["counts"]["index_entries"] >= 2 * provenance["counts"]["far_pairs"]


@OGLD_5
def test_the_variant_set_is_the_test_panel_s_passing_rows(tmp_path: Path) -> None:
    _, store = built(tmp_path, 10)
    v = panel_data().variants()
    failing = np.flatnonzero(~v.passes_filters)
    assert len(failing), "the Test Panel must hold variants the filters remove"
    m = store.matrix(alids(np.arange(len(v.pos))))
    assert set(m.absent) == set(alids(failing))


@OGLD_5
def test_a_pgen_without_info_refuses_an_info_floor(tmp_path: Path) -> None:
    # A sequenced panel has no INFO: info_min must be None, said on purpose, not skipped.
    with pytest.raises(ValueError, match="INFO"):
        ogld.build(inputs(10, info_min=0.8), tmp_path / "store")


@OGLD_5
def test_ogld_build_writes_a_store_that_validates(tmp_path: Path) -> None:
    out = tmp_path / "store"
    panel = panel_data().panel()
    argv = ["build", str(out), "--pgen", str(panel.pgen), "--keep", str(panel.keep)]
    argv += ["--pcs", str(panel.pcs), "--pc-source", "gnomAD v3.1.2 subcontinental_pca"]
    argv += ["--panel-id", "test", "--ancestry", "EUR", "--genotype-source", "Test Panel"]
    assert cli.main([*argv, "--info-min", "none"]) == 0
    assert not any(ogld.validate(out).values())

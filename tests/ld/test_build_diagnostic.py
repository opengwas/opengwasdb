"""ogld#5: the PC diagnostic, dosages and PC Adjustment's edge, each against a computation of its
own (spec §7; build-inputs §6, §7, §9).

- `near_100kb_change` is the change in the Test Panel's independent reference r from k = 0 to
  k = 10, signed to A1, over its pairs within 100 kb: its quantiles to within one bin.
- The chance-LD rates are the share of #405's seeded pairs at r² ≥ 0.01 by numpy's own r; a
  Variant Set on one chromosome has no such pair, and its rates are null.
- A pgen of dosages gives Pearson's r of the dosages, as plink2 computes it.
- A variant whose genotypes the covariates span has no variance left, and is dropped.
- The Long-Range LD Regions in the package are the spec's table.
"""

from __future__ import annotations

import math
from pathlib import Path

import build_fixtures as made
import numpy as np
import pytest
import testdata

import ogld
from ogld.build import adjust, diagnostic, record
from ogld.format import InputError

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ROOT / "docs" / "spec" / "long-range-ld-regions-grch38.tsv"


@pytest.fixture(scope="module")
def k10(tmp_path_factory: pytest.TempPathFactory) -> ogld.Provenance:
    return made.test_panel_build(10, tmp_path_factory)[0]


def test_the_change_within_100kb_is_the_reference_s(k10: ogld.Provenance) -> None:
    i, j, before = made.reference_pairs(0)
    _, _, after = made.reference_pairs(10)
    pos = testdata.variants().pos
    near = pos[j] - pos[i] <= 100_000  # chr21 and chr22 hold no Long-Range LD Region
    change = (after - before)[near]
    assert len(change) > 10_000, "the fixture must hold many pairs within 100 kb"
    diagnostic_ = k10["pc_diagnostic"]
    assert diagnostic_ is not None
    recorded = diagnostic_["near_100kb_change"]
    width = 2.0**-20
    for q in (0.5, 0.9, 0.99):
        exact = float(np.quantile(change, q, method="lower"))
        assert abs(recorded[f"{q:g}"] - exact) <= width / 2 + made.SLACK, q
    assert recorded["0"] == pytest.approx(change.min(), abs=made.SLACK)
    assert recorded["1"] == pytest.approx(change.max(), abs=made.SLACK)
    assert recorded["0"] < 0 < recorded["1"], "PC Adjustment moved real LD both ways"


def numpy_r(rows: np.ndarray, k: int) -> np.ndarray:
    """Pearson r of Test Panel `rows`' ALT counts after regressing [1, PC1..PCk] out, by lstsq."""
    g = made.hard_calls()[rows].astype(np.float64).T
    design = np.column_stack([np.ones(len(g)), testdata.pcs().values[:, :k]])
    resid = g - design @ np.linalg.lstsq(design, g, rcond=None)[0]
    return np.corrcoef(resid.T)


def test_the_chance_rates_are_numpy_s_on_405_s_pairs(k10: ogld.Provenance) -> None:
    # The Test Panel's PC File lists the samples in its psam's order, so numpy_r's are aligned.
    assert testdata.pcs().iids[:3] == tuple(
        line.split("\t")[0] for line in testdata.panel().psam.read_text().splitlines()[1:4]
    )
    passing = np.flatnonzero(testdata.variants().passes_filters)
    labels = testdata.variants().chrom[passing]
    used = diagnostic.draw_variants(len(passing))
    first, second = diagnostic.draw_pairs(labels[used])
    assert len(first) == 200_000
    assert (labels[used][first] != labels[used][second]).all(), "every pair is cross-chromosome"
    d = k10["pc_diagnostic"]
    assert d is not None
    assert d["pairs"]["n_pairs"] == 200_000
    for key, k in (("rate_r2_ge_0_01_unadjusted", 0), ("rate_r2_ge_0_01_adjusted", 10)):
        r = numpy_r(passing[used], k)[first, second]
        assert d[key] == pytest.approx(float((r * r >= 0.01).mean()), abs=1e-12), key
    assert d["rate_r2_ge_0_01_sampling"] == pytest.approx(
        math.erfc(math.sqrt(0.01 * (771 - 11) / 2))
    )


def test_405_s_draws_are_seeded() -> None:
    labels = np.array(["1"] * 4_000 + ["2"] * 1_000)
    used = diagnostic.draw_variants(len(labels))
    assert len(used) == 3_000
    assert np.array_equal(used, diagnostic.draw_variants(len(labels)))
    assert (np.diff(used) > 0).all()
    one, two = diagnostic.draw_pairs(labels[used])
    again = diagnostic.draw_pairs(labels[used])
    assert np.array_equal(one, again[0])
    assert np.array_equal(two, again[1])


def test_one_chromosome_has_no_chance_rate(tmp_path: Path) -> None:
    on22 = np.flatnonzero(testdata.variants().chrom == "22")
    pgen = made.write_fileset(tmp_path / "chr22", on22)
    provenance = ogld.build(made.inputs(pgen), tmp_path / "store")
    d = provenance["pc_diagnostic"]
    assert d is not None
    assert d["pairs"]["n_pairs"] == 0
    assert d["rate_r2_ge_0_01_unadjusted"] is None
    assert d["rate_r2_ge_0_01_adjusted"] is None
    # The sampling rate is a formula of n and k, measured on no pair: always a number
    # (orchestrator note 2 on ogld#5, which corrected note 1's "three").
    assert d["rate_r2_ge_0_01_sampling"] == pytest.approx(math.erfc(math.sqrt(0.01 * 770 / 2)))
    assert "one chromosome" in d["pairs"]["source"]
    # At k = 0 nothing is adjusted, so nothing changes.
    assert set(d["near_100kb_change"].values()) == {0.0}


def test_dosages_give_pearson_r_of_the_dosages(tmp_path: Path) -> None:
    v = testdata.variants()
    rows = np.flatnonzero((v.chrom == "22") & v.passes_filters)[:150]
    calls = made.hard_calls()[rows]
    rng = np.random.default_rng(7)
    dosages = calls.astype(np.float32)
    soft = rng.random(dosages.shape) < 0.3
    dosages[soft] = np.clip(dosages[soft] + rng.uniform(-0.4, 0.4, int(soft.sum())), 0, 2)
    pgen = made.write_fileset(tmp_path / "dosage", rows, genotypes=calls, dosages=dosages)
    import pgenlib

    reader = pgenlib.PgenReader(str(pgen).encode())
    held = np.empty(dosages.shape, dtype=np.float64)
    reader.read_dosages_range(0, len(rows), held)
    reader.close()
    assert np.abs(held - np.rint(held)).max() > 0.1, "the fixture must hold fractional dosages"
    ogld.build(made.inputs(pgen), tmp_path / "store")
    sign = made.a1_signs()[rows]
    expected = np.corrcoef(held) * sign[:, None] * sign[None, :]
    ours = made.stored_r(tmp_path / "store", rows)
    pos = v.pos[rows]
    near = np.abs(pos[:, None] - pos[None, :]) <= 3_000_000
    assert np.abs(ours[near] - expected[near]).max() <= made.HALF_STEP + made.SLACK
    assert (
        np.abs(
            expected[near] - np.corrcoef(calls)[near] * (sign[:, None] * sign[None, :])[near]
        ).max()
        > 0.01
    )


def test_a_variant_the_covariates_span_has_no_variance_left() -> None:
    rng = np.random.default_rng(3)
    pcs = rng.standard_normal((200, 3))
    spanned = 1.0 + 2.0 * pcs[:, 1]
    free = rng.binomial(2, 0.4, 200).astype(np.float64)
    covariates = adjust.covariates(["PC1", "PC2", "PC3"], pcs, 200)
    _, flat = adjust.residuals(np.vstack([spanned, free]), covariates)
    assert flat.tolist() == [True, False]
    with pytest.raises(ValueError, match="no residual variance"):
        adjust.unit_vectors(np.vstack([spanned, free]), covariates, np.array([True, True]))


def test_the_package_s_regions_are_the_spec_s() -> None:
    rows = [
        line.split("\t")
        for line in REGIONS.read_text().splitlines()
        if line and not line.startswith("#")
    ][1:]
    table = tuple((r[0], r[1], int(r[2]), int(r[3])) for r in rows)
    assert table == diagnostic.LONG_RANGE_LD_REGIONS
    assert len(table) == 27


def test_a_region_s_variants_are_left_out() -> None:
    position = np.array([45_761_252, 45_761_253, 46_765_892, 46_765_893])
    assert diagnostic.outside_regions("17", position).tolist() == [True, False, False, True]
    assert diagnostic.outside_regions("22", position).all()


def test_the_histogram_reads_quantiles_to_within_a_bin() -> None:
    values = np.random.default_rng(1).normal(0, 0.02, 100_001)
    histogram = diagnostic.ChangeHistogram()
    for part in np.array_split(values, 7):
        histogram.add(part)
    histogram.add(np.empty(0))
    got = histogram.quantiles()
    assert got["0"] == values.min()
    assert got["1"] == values.max()
    for q in (0.5, 0.9, 0.99):
        assert abs(got[f"{q:g}"] - np.quantile(values, q, method="lower")) <= 2.0**-21


def test_an_empty_histogram_is_refused() -> None:
    with pytest.raises(InputError, match="cannot be measured"):
        diagnostic.ChangeHistogram().quantiles()


def test_the_cost_is_this_build_s() -> None:
    cost = record.Cost.start()
    assert cost.record(2)["peak_memory_bytes"] > 0
    assert record.Cost().record(1)["peak_memory_bytes"] > 0, "without the reset: the process's"

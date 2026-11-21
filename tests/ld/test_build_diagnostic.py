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
from ogld import genotypes as adjust
from ogld.build import diagnostic, record
from ogld.format import InputError
from ogld.store import StoreReader


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
        adjust.prepare(np.vstack([spanned, free]), covariates, np.array([True, True]))


def test_the_package_owns_the_long_range_ld_regions() -> None:
    # The table is package data (ogld/build/long-range-ld-regions-grch38.tsv), read once by
    # diagnostic.long_range_ld_regions, which the build and both scripts call. Held here to
    # build-inputs §9's own numbers.
    table = diagnostic.long_range_ld_regions()
    assert len(table) == 27
    named = {name: (chrom, start, end) for name, chrom, start, end in table}
    assert named["17q21.31_inversion"] == ("17", 45761253, 46765892)
    assert named["price2008_03_LCT"] == ("2", 133742429, 137242430)
    assert named["MHC"][0] == named["price2008_12_MHC_wide"][0] == "6"
    assert all(start <= end for _, _, start, end in table)


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


def test_an_empty_histogram_is_a_builder_fault() -> None:
    # Unreachable from a build, which refuses a Variant Set with no 100 kb pair before writing.
    with pytest.raises(ValueError, match="builder fault"):
        diagnostic.ChangeHistogram().quantiles()


def test_the_cost_is_this_build_s() -> None:
    cost = record.Cost.start()
    assert cost.record(2)["peak_memory_bytes"] > 0
    assert record.Cost().record(1)["peak_memory_bytes"] > 0, "without the reset: the process's"


def test_a_variant_set_inside_the_long_range_regions_is_refused_before_a_write(
    tmp_path: Path,
) -> None:
    # Review of #5 (standards a3): the PC diagnostic's change in r is measured on pairs within
    # 100 kb outside the Long-Range LD Regions. An MHC-only Variant Set has none, which its
    # positions say before anything is written, so the build refuses there, naming the panel,
    # and leaves no half-written store to block a rerun.
    v = testdata.variants()
    on22 = np.flatnonzero((v.chrom == "22") & (v.pos >= 20_000_000) & (v.pos < 24_000_000))
    assert len(on22) > 100, "the fixture must hold a dense stretch to move into the MHC"
    shift = 28_600_000 - int(v.pos[on22[0]])
    text = made.pvar_lines()
    lines = [text[0]]
    for row in on22:
        fields = text[1 + int(row)].split("\t")
        moved = str(int(fields[1]) + shift)
        lines.append("\t".join(["6", moved, fields[2], *fields[3:]]))
    assert (
        diagnostic.outside_regions("6", np.array([int(x.split("\t")[1]) for x in lines[1:]])).sum()
        == 0
    )
    pgen = made.write_fileset(tmp_path / "mhc", on22, lines=lines)
    with pytest.raises(InputError, match=rf"panel {made.PANEL_ID}.*100 kb.*Long-Range LD Regions"):
        ogld.build(made.inputs(pgen), tmp_path / "store")
    assert not (tmp_path / "store").exists(), "nothing is written"


def standardised(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def test_a_variant_the_covariates_span_is_dropped_and_counted(tmp_path: Path) -> None:
    # Review of #5 (standards a5): filter 4 after PC Adjustment had no test. PC10 here is 30% one
    # variant's genotypes and 70% ancestry PC1, as the contract's kept PC2 is: one window explains
    # much of it, the rest of the genome too, so the PC Check keeps it, and with PC1 also a
    # covariate the variant lies in their span: no variance is left.
    row = 1050
    pcs = testdata.pcs().values
    mix = np.sqrt(0.3) * standardised(made.hard_calls()[row].astype(np.float64))
    mix += np.sqrt(0.7) * standardised(pcs[:, 0])
    lines = testdata.panel().pcs.read_text().splitlines()
    out = [lines[0]]
    for i, line in enumerate(lines[1:]):
        fields = line.split("\t")
        fields[10] = repr(float(mix[i]))
        out.append("\t".join(fields))
    pc_file = tmp_path / "pcs.eigenvec"
    pc_file.write_text("\n".join(out) + "\n")
    passing = testdata.variants().passes_filters
    assert passing[row]
    twins = (made.hard_calls()[passing] == made.hard_calls()[row]).all(axis=1).sum()
    assert twins == 1, "no other variant shares its genotypes, so exactly one is flat"
    provenance = ogld.build(made.inputs(testdata.panel().pgen, 10, pcs=pc_file), tmp_path / "s")
    assert provenance["adjustment"]["covariates"][-1] == "PC10", "the PC Check keeps the mix"
    assert provenance["variants_dropped"]["zero_variance"] == 1
    assert provenance["n_variants"] == int(passing.sum()) - 1
    reader = StoreReader.open(tmp_path / "s")
    assert reader.rows_of(made.alids([row]))[0] == -1


def test_flat_is_a_float_error_bar_not_a_variance_floor() -> None:
    # A residual 1e-6 of the centred dosages' sum of squares is real variance, kept; one the
    # covariates span exactly is flat. FLAT is 1e-12: at 1e-3 the first would be dropped.
    rng = np.random.default_rng(4)
    pcs = rng.standard_normal((300, 2))
    spanned = 1.0 + pcs[:, 0] - 0.5 * pcs[:, 1]
    covariates = adjust.covariates(["PC1", "PC2"], pcs, 300)
    noise = rng.standard_normal(300)
    noise -= covariates.basis @ (covariates.basis.T @ noise)
    centred = spanned - spanned.mean()
    noise *= np.sqrt(1e-6 * (centred @ centred) / (noise @ noise))
    _, flat = adjust.residuals(np.vstack([spanned, spanned + noise]), covariates)
    assert flat.tolist() == [True, False]

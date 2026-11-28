"""The stored-z audit flags an Analysis implausible as a genome-wide scan (#510).

A published Store keeps no source p-value, so ADR 0069's check cannot be repeated
on it. It keeps every z, and in a genome-wide scan most are small: an effect on
the wrong scale (an odds ratio in `beta`, OGS-00032's ten; a mis-scaled `beta`,
OGS-00033's GCST90278111) puts nearly every one at 5e-8. The rule counts, per
Analysis, the observed associations and those at 5e-8, and flags more than 30% at
5e-8. A source published only below a p threshold has every |z| above 3 and is
sound, so a share above |z| 3 is not judged.

Each expectation here is a literal about cells written by hand, or about a Store
built from a source whose every |z| is known.
"""

from __future__ import annotations

import dataclasses
import gzip
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.store_reader.enums import AssociationCoverage
from opengwasdb.validation.rule import Kind, Severity, Target
from opengwasdb.validation.rules import z_plausibility as zp
from opengwasdb.validation.scans import Block
from opengwasdb.validation.validate import validate_store


def _counts(*columns: np.ndarray) -> np.ndarray:
    z = np.stack(columns, axis=1)
    return zp.block_counts(Block.of(z, np.full_like(z, 0.1)))


def test_a_block_is_counted_per_analysis_observed_and_at_5e8():
    z = np.array([0.1, -3.5, 6.0, np.nan, 5.4])
    counts = _counts(z, -z)
    # observed: 4 (NaN missing); at 5e-8 (|z| >= 5.4513): 6.0 alone, not 5.4.
    assert counts.tolist() == [[4, 4], [1, 1]]


def test_imputed_cells_are_not_counted():
    z = np.array([[10.0], [0.5]])
    counts = zp.block_counts(Block.of(z, np.full_like(z, 0.1), imputed=np.array([[True], [False]])))
    assert counts[:, 0].tolist() == [1, 0]


def test_the_bound_flags_more_than_thirty_percent_at_5e8_and_spares_a_strong_scan():
    # Columns: (observed, at 5e-8). Literal shares, and the rule's verdict.
    counts = np.array([[10_000, 10_000, 10_000, 999], [3_001, 3_000, 970, 999]])
    flagged = [m.analysis_index for m in zp.implausible(counts)]
    # 0: 30.01%, flagged; 1: exactly 30%, not; 2: 9.7% (GCST90245992's, the largest
    # sound share in the published Stores), not; 3: 999 observed, below MIN_OBSERVED,
    # so not judged at all.
    assert flagged == [0]


SE = 0.05
N_ROWS = 1_200  # above MIN_OBSERVED, in one Dense component


def _source(path: Path, z: np.ndarray) -> Path:
    """A GWAS-SSF source with no p-value column, so the build's check skips it."""
    lines = ["chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error"]
    lines += [f"1\t{1000 + i}\tA\tC\t{z[i] * SE:.10g}\t{SE:g}" for i in range(len(z))]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _manifest(tmp_path: Path, analyses: dict[str, np.ndarray]) -> Path:
    """A build manifest of one GWAS-SSF source per Analysis, written from its z."""
    header = ("trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
              "\tsource_assembly\tsource_reader_capability")
    lines = [header]
    for name, z in analyses.items():
        path = _source(tmp_path / f"{name}.tsv.gz", z)
        lines.append(f"{name}\t{path}\t{name}\t1000\tsd\tdeclared_standardised\thg38"
                     "\topengwasdb.gwas-ssf")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _store(tmp_path: Path, **analyses: np.ndarray) -> Path:
    manifest = _manifest(tmp_path, analyses)
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="zp", release_id="r1")
    return store


_RNG = np.random.default_rng(510)
#: A null scan: |z| > 3 on about 0.3% of rows.
_NULL = _RNG.standard_normal(N_ROWS)
#: An odds ratio over a log-scale SE: |z| about 1/SE = 20 on every row.
_INFLATED = np.exp(_RNG.standard_normal(N_ROWS) * SE) / SE


def test_the_fixture_scans_are_what_they_are_said_to_be():
    assert np.mean(np.abs(_NULL) > 3) < 0.01
    assert np.mean(np.abs(_INFLATED) >= 5.4513) == 1.0


def test_a_store_holding_an_inflated_analysis_fails_naming_it(tmp_path: Path):
    store = _store(tmp_path, clean=_NULL, inflated=_INFLATED)
    result = validate_store(store)
    assert result.failed() == {zp.RULE}, result.errors
    [message] = [f.message for f in result.of(zp.RULE)]
    assert "'inflated': 100.0% of 1,200 observed associations at 5e-8" in message
    assert "'clean'" not in message


def test_a_clean_store_passes(tmp_path: Path):
    store = _store(tmp_path, clean=_NULL, also_clean=-_NULL)
    result = validate_store(store)
    assert result.failed() == set(), result.errors
    assert result.of(zp.RULE) == []


#: The Hybrid case: 999 on-panel rows, every one at |z| 8 -- a Dense part under
#: MIN_OBSERVED -- and 1,500 off-panel rows, 300 (20%) at |z| 8. Judged per component,
#: neither fails: the Dense part is not judged, the Overflow is under 30%. Summed, it
#: is (999 + 300) / 2,499 = 52.0% at 5e-8.
_ON_PANEL, _OFF_PANEL = 999, 1_500
_SPLIT = np.concatenate([
    np.full(_ON_PANEL, 8.0), np.full(300, 8.0), _RNG.standard_normal(1_200) * 0.5,
])
_SPLIT_CLEAN = _RNG.standard_normal(_ON_PANEL + _OFF_PANEL)


def _hybrid_store(tmp_path: Path, **analyses: np.ndarray) -> Path:
    """A Hybrid whose panel holds the first `_ON_PANEL` rows of each source."""
    manifest = _manifest(tmp_path, analyses)
    panel = tmp_path / "panel.txt"
    panel.write_text("".join(f"1:{1000 + i}:A:C\n" for i in range(_ON_PANEL)), "utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="zp", release_id="r1"
    )
    return store


def test_the_split_fixture_passes_each_component_and_fails_summed():
    on, off = np.abs(_SPLIT[:_ON_PANEL]), np.abs(_SPLIT[_ON_PANEL:])
    assert len(on) < zp.MIN_OBSERVED and np.all(on >= 5.4513)
    assert len(off) >= zp.MIN_OBSERVED and np.mean(off >= 5.4513) == pytest.approx(0.2)
    assert np.mean(np.abs(_SPLIT) >= 5.4513) == pytest.approx(1299 / 2499)


def test_a_hybrid_analysis_is_judged_on_both_components_summed(tmp_path: Path):
    store = _hybrid_store(tmp_path, split=_SPLIT, clean=_SPLIT_CLEAN)
    result = validate_store(store)
    assert result.failed() == {zp.RULE}, result.errors
    [message] = [f.message for f in result.of(zp.RULE)]
    assert "in data.zarr/ragged and dense/data.zarr" in message
    assert "'split': 52.0% of 2,499 observed associations at 5e-8" in message
    assert "'clean'" not in message


def _grid_target(store: Path, **manifest: object) -> Target:
    """The store's grid as a target, its manifest's `manifest` fields replaced."""
    opened = OpenGWASDBStore.open(store)
    replaced = OpenGWASDBStore(opened.path, dataclasses.replace(opened.manifest, **manifest))
    release = Target(Kind.RELEASE, ".", replaced)
    return Target(Kind.GRID, "data.zarr", replaced, parent=release)


#: 420 of 1,200 rows (35%) at |z| 8, the rest null: flagged on the share at 5e-8 alone,
#: with fewer than half above |z| 3.
_MOSTLY_GENOME_WIDE = np.concatenate([np.full(420, 8.0), _NULL[:780]])
#: A source published only below p = 0.0027 (GCST90271757 publishes only p <= 0.00067):
#: every |z| between 3 and 5, none at 5e-8. Sound effects, a threshold-published file.
_THRESHOLD_PUBLISHED = np.linspace(3.05, 5.2, N_ROWS) * np.where(np.arange(N_ROWS) % 2, 1, -1)


def test_the_share_fixtures_are_what_they_are_said_to_be():
    assert np.mean(np.abs(_MOSTLY_GENOME_WIDE) >= 5.4513) == pytest.approx(0.35)
    assert np.mean(np.abs(_MOSTLY_GENOME_WIDE) > 3) < 0.5
    assert np.all(np.abs(_THRESHOLD_PUBLISHED) > 3)
    assert not np.any(np.abs(_THRESHOLD_PUBLISHED) >= 5.4513)


def test_an_analysis_mostly_at_5e8_fails_on_that_share_alone(tmp_path: Path):
    store = _store(tmp_path, clean=_NULL, mostly=_MOSTLY_GENOME_WIDE)
    result = validate_store(store)
    assert result.failed() == {zp.RULE}, result.errors
    [message] = [f.message for f in result.of(zp.RULE)]
    assert "'mostly': 35.0% of 1,200 observed associations at 5e-8" in message
    assert "|z| 3" not in message


def test_a_threshold_published_source_with_every_z_above_three_passes(tmp_path: Path):
    """The |z| > 3 share flagged GCST90444395 and GCST90271757, sound sources published
    only below a p threshold; the rule judges the share at 5e-8 alone (#510)."""
    store = _store(tmp_path, clean=_NULL, threshold=_THRESHOLD_PUBLISHED)
    result = validate_store(store)
    assert result.failed() == set(), result.errors
    assert result.of(zp.RULE) == []


def test_a_release_before_the_current_builders_is_warned_not_failed(tmp_path: Path):
    """`severity_owed`: a release this package's builders did not write is warned (#367)."""
    target = _grid_target(_store(tmp_path, clean=_NULL), format_version="0.2.1")
    counts = np.array([[2_000], [2_000]])
    assert [f.severity for f in zp.RULES[0].judged(target, counts)] == [Severity.WARNING]
    current = _grid_target(tmp_path / "store.opengwasdb")
    assert [f.severity for f in zp.RULES[0].judged(current, counts)] == [Severity.ERROR]


@pytest.mark.parametrize("coverage", ["cis", "signals_only", "cis_and_signals"])
def test_only_a_genome_wide_release_is_judged(tmp_path: Path, coverage: str):
    store = _store(tmp_path, clean=_NULL)
    assert zp.RULES[0].when(_grid_target(store))
    narrower = _grid_target(store, association_coverage=AssociationCoverage(coverage))
    assert not zp.RULES[0].when(narrower)

"""Unit tests for the ragged imputation kernel (issue 038)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opengwasdb.completion.impute import (
    elastic_net_impute,
    impute_z_block,
    ld_pca,
    poly_rescale,
    scalar_n_se,
)


def _random_pd_ld(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n) * n * 0.1


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_ld(rng):
    return _random_pd_ld(20, rng)


class TestLdPca:
    def test_eigenvalues_descending(self, small_ld):
        vals, _ = ld_pca(small_ld, thresh=0.9)
        assert np.all(vals[:-1] >= vals[1:])

    def test_cumvar_reaches_thresh(self, small_ld):
        thresh = 0.85
        vals, _ = ld_pca(small_ld, thresh=thresh)
        all_vals, _ = ld_pca(small_ld, thresh=1.0)
        cumvar = np.cumsum(vals) / all_vals.sum()
        assert cumvar[-1] >= thresh

    def test_fewer_components_than_variants(self, small_ld):
        vals, _ = ld_pca(small_ld, thresh=0.5)
        assert len(vals) < small_ld.shape[0]

    def test_eigenvector_shape(self, small_ld):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        assert vecs.shape == (small_ld.shape[0], len(vals))

    def test_non_negative_eigenvalues(self, small_ld):
        vals, _ = ld_pca(small_ld, thresh=0.9)
        assert np.all(vals >= 0)


class TestElasticNetImpute:
    def test_returns_full_length(self, small_ld, rng):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = rng.standard_normal(small_ld.shape[0]).astype(np.float64)
        z[::4] = np.nan
        result = elastic_net_impute(z, vecs, len(vals))
        assert result is not None
        assert len(result) == small_ld.shape[0]

    def test_no_nans_in_result(self, small_ld, rng):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = rng.standard_normal(small_ld.shape[0]).astype(np.float64)
        z[:5] = np.nan
        result = elastic_net_impute(z, vecs, len(vals))
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_returns_none_all_missing(self, small_ld):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = np.full(small_ld.shape[0], np.nan)
        assert elastic_net_impute(z, vecs, len(vals)) is None

    def test_returns_none_constant_z(self, small_ld):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = np.ones(small_ld.shape[0])
        z[:3] = np.nan
        assert elastic_net_impute(z, vecs, len(vals)) is None

    def test_returns_none_single_observed(self, small_ld):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = np.full(small_ld.shape[0], np.nan)
        z[0] = 1.5
        assert elastic_net_impute(z, vecs, len(vals)) is None


class TestPolyRescale:
    def test_high_correlation(self, rng):
        n = 100
        pred = rng.standard_normal(n)
        truth = 2.5 * pred + rng.standard_normal(n) * 0.1
        truth[10:20] = np.nan
        _, corr = poly_rescale(truth, pred, npoly=1)
        assert np.isfinite(corr) and corr > 0.9

    def test_output_length_matches_input(self, rng):
        n = 50
        pred = rng.standard_normal(n)
        truth = pred * 1.5
        truth[:5] = np.nan
        adj, _ = poly_rescale(truth, pred)
        assert len(adj) == n

    def test_degenerate_too_few_points(self):
        truth = np.array([1.0, np.nan, np.nan, np.nan])
        pred = np.array([0.5, 1.0, 1.5, 2.0])
        adj, corr = poly_rescale(truth, pred, npoly=3)
        assert len(adj) == 4

    def test_all_missing_truth_returns_nan_corr(self, rng):
        truth = np.full(20, np.nan)
        pred = rng.standard_normal(20)
        _, corr = poly_rescale(truth, pred)
        assert not np.isfinite(corr)


class TestImputeZBlock:
    def test_returns_imputed_array_and_corr(self, small_ld, rng):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = rng.standard_normal(small_ld.shape[0]).astype(np.float64)
        z[:5] = np.nan
        z_imp, corr = impute_z_block(z, vecs, vals, min_cor=0.0)
        assert z_imp is not None
        assert np.isfinite(corr)
        assert len(z_imp) == small_ld.shape[0]

    def test_quality_gate_rejects_low_corr(self, small_ld, rng):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = rng.standard_normal(small_ld.shape[0]).astype(np.float64)
        z[:5] = np.nan
        z_imp, _ = impute_z_block(z, vecs, vals, min_cor=1.0)
        assert z_imp is None

    def test_clamps_to_observed_range(self, small_ld, rng):
        vals, vecs = ld_pca(small_ld, thresh=0.9)
        z = rng.standard_normal(small_ld.shape[0]).astype(np.float64) * 3
        z[:3] = np.nan
        obs_max = float(np.nanmax(np.abs(z)))
        z_imp, _ = impute_z_block(z, vecs, vals, min_cor=0.0)
        if z_imp is not None:
            assert float(np.max(np.abs(z_imp))) <= obs_max + 1e-4


class TestScalarNSe:
    def test_basic_output(self):
        se_obs = np.array([0.05, 0.06, 0.04])
        eaf_obs = np.array([0.3, 0.4, 0.5])
        eaf_ref = np.array([0.2, 0.3, 0.5])
        se_ref = scalar_n_se(se_obs, eaf_obs, eaf_ref)
        assert len(se_ref) == 3
        assert np.all(np.isfinite(se_ref))
        assert np.all(se_ref > 0)

    def test_nan_for_eaf_zero_or_one(self):
        se_obs = np.array([0.05])
        eaf_obs = np.array([0.3])
        eaf_ref = np.array([0.0, 0.5, 1.0])
        se_ref = scalar_n_se(se_obs, eaf_obs, eaf_ref)
        assert not np.isfinite(se_ref[0])
        assert np.isfinite(se_ref[1])
        assert not np.isfinite(se_ref[2])

    def test_nan_when_no_valid_observed(self):
        se_obs = np.array([0.05])
        eaf_obs = np.array([np.nan])
        eaf_ref = np.array([0.3])
        se_ref = scalar_n_se(se_obs, eaf_obs, eaf_ref)
        assert not np.isfinite(se_ref[0])

    def test_scale_consistent_with_eaf_formula(self):
        # If se_obs * sqrt(2*eaf*(1-eaf)) = C for all obs,
        # then se_ref = C / sqrt(2*eaf_ref*(1-eaf_ref))
        eaf_obs = np.array([0.3, 0.4])
        C = 0.5
        se_obs = C / np.sqrt(2 * eaf_obs * (1 - eaf_obs))
        eaf_ref = np.array([0.2, 0.6])
        se_ref = scalar_n_se(se_obs, eaf_obs, eaf_ref)
        expected = C / np.sqrt(2 * eaf_ref * (1 - eaf_ref))
        np.testing.assert_allclose(se_ref, expected, rtol=1e-3)


class TestBlocksOverVariants:
    """Issue #102's block-enumeration rule, unit-level.

    An Analysis with no Trait position is completed over the blocks it already
    holds enough of a region in. "Enough" is `impute.min_observed_points()` --
    the number `poly_rescale` actually requires before it returns anything but
    NaN -- and "holds" counts the Analysis's observations **at the block's own
    panel SNPs**, which is what `completion.block.run_block` counts when it
    decides whether it can fit. Counting positions inside the block's
    base-pair extent instead would include off-panel variants and enumerate
    blocks that go on to impute nothing.

    Spec §17's "do not expand singleton suggestive associations" falls out of
    the same threshold.
    """

    @staticmethod
    def _panel_dir(tmp_path, blocks: dict[str, list[str]]) -> Path:
        """A panel of one chromosome, each block listing the given ALIDs.

        The LD matrix beside each TSV is not incidental: `load_block` returns
        None without either a matrix or an eigendecomposition, so a panel of
        bare TSVs reads as no blocks at all.
        """
        import gzip
        import io

        chrom_dir = tmp_path / "panel" / "EUR" / "1"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        for name, alids in blocks.items():
            lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
            for alid in alids:
                chrom, pos, a1, a2 = alid.split(":")
                lines.append(f"{chrom}\t{chrom}:{pos}_{a1}_{a2}\t{a2}\t{a1}\t0.3\t{pos}")
            (chrom_dir / f"{name}.tsv").write_text("\n".join(lines) + "\n")
            n = len(alids)
            rng = np.random.default_rng(0)
            A = rng.standard_normal((n, n))
            ld = A @ A.T + np.eye(n) * n * 0.1
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                for row in ld:
                    gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
            (chrom_dir / f"{name}.unphased.vcor1.gz").write_bytes(buf.getvalue())
        return tmp_path / "panel"

    @staticmethod
    def _alids(*positions: int) -> list[str]:
        return [f"1:{p}:A:G" for p in positions]

    def test_a_block_the_analysis_has_enough_of_is_enumerated(self, tmp_path):
        from opengwasdb.completion.impute import min_observed_points
        from opengwasdb.completion.ld_panel import blocks_over_variants

        n = min_observed_points()
        panel = self._panel_dir(tmp_path, {"1-1000": self._alids(*range(100, 100 + 10 * n, 10))})
        observed = {alid: [7] for alid in self._alids(*range(100, 100 + 10 * n, 10))}

        found = blocks_over_variants(panel, "EUR", observed, ["1"])
        assert [b.block_id for b in found[7]] == ["1/1-1000"]

    def test_one_observation_short_is_not_enumerated(self, tmp_path):
        """The boundary, asserted rather than assumed: at `n - 1` observed
        points `poly_rescale` returns NaN and imputation is rejected, so
        enumerating the block would add its panel variants as missing rows and
        fill none of them."""
        from opengwasdb.completion.impute import min_observed_points
        from opengwasdb.completion.ld_panel import blocks_over_variants

        n = min_observed_points()
        all_positions = list(range(100, 100 + 10 * n, 10))
        panel = self._panel_dir(tmp_path, {"1-1000": self._alids(*all_positions)})
        observed = {alid: [7] for alid in self._alids(*all_positions[: n - 1])}

        assert blocks_over_variants(panel, "EUR", observed, ["1"]) == {}

    def test_an_off_panel_variant_does_not_count_towards_the_threshold(self, tmp_path):
        """The distinction the position-based version got wrong: a variant
        inside the block's extent but absent from its SNP list is not
        something imputation can fit on."""
        from opengwasdb.completion.impute import min_observed_points
        from opengwasdb.completion.ld_panel import blocks_over_variants

        n = min_observed_points()
        on_panel = list(range(100, 100 + 10 * (n - 1), 10))
        panel = self._panel_dir(tmp_path, {"1-1000": self._alids(*on_panel)})
        observed = {alid: [7] for alid in self._alids(*on_panel, 555)}  # 555 is off-panel

        assert blocks_over_variants(panel, "EUR", observed, ["1"]) == {}

    def test_each_analysis_is_judged_on_its_own_observations(self, tmp_path):
        from opengwasdb.completion.impute import min_observed_points
        from opengwasdb.completion.ld_panel import blocks_over_variants

        n = min_observed_points()
        positions = list(range(100, 100 + 10 * n, 10))
        panel = self._panel_dir(tmp_path, {"1-1000": self._alids(*positions)})
        observed: dict[str, list[int]] = {}
        for j, alid in enumerate(self._alids(*positions)):
            observed[alid] = [1] if j == 0 else [1, 2]  # analysis 2 is one short

        found = blocks_over_variants(panel, "EUR", observed, ["1"])
        assert set(found) == {1}

    def test_a_panel_with_no_blocks_for_the_chromosome_is_not_an_error(self, tmp_path):
        from opengwasdb.completion.ld_panel import blocks_over_variants

        (tmp_path / "panel" / "EUR").mkdir(parents=True)
        assert blocks_over_variants(
            tmp_path / "panel", "EUR", {"1:100:A:G": [0]}, ["1"]
        ) == {}

    def test_no_gene_target_less_analyses_reads_no_panel_at_all(self, tmp_path):
        from opengwasdb.completion.ld_panel import blocks_over_variants

        assert blocks_over_variants(tmp_path / "nonexistent", "EUR", {}, ["1"]) == {}

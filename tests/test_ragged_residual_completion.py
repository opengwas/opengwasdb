"""Ragged Reference Completion of a residual-`se` source, end to end (issue #140).

Issue #140 asks that a complete Ragged Store Release "select, write, complete,
query, derive, and validate residual-coded SE" on the same terms as Dense:
observed cells must round-trip under the completed plan and imputed cells --
whose frequencies are the LD panel's -- must decode to a finite EAF and a
finite, positive physical SE. The Dense and Hybrid layouts each have an
end-to-end test that completes a residual source against a panel; Ragged had
none, even though `complete.py` refits `se_coefficients` against the
post-completion decoded EAF (imputed cells included) and re-encodes the CSR
from scratch -- a path no fixture exercised (issue #140 AC5/AC9).

The fixture below is meaningful on three axes:

* The source genuinely selects `int8_residual`: each Analysis's SE tracks its
  frequency-predicted value closely, so the measured decision is the residual
  coding -- except for one cell per Analysis that deviates by a full log-unit
  and can only be held as an exact exception.
* The completed release gains imputed cells: one Analysis observes every
  block variant while the other leaves four in-block positions unobserved, and
  the LD panel adds four positions the store never held -- three carry a panel
  frequency and impute, one carries none and must stay explicitly missing
  rather than receiving a fabricated SE (ADR 0037 §4, #159).
* Queries and the precomputed top-hit index are checked against freshly
  decoded CSR values, so a completed release whose index was built from
  anything other than what the plane decodes fails here rather than passing
  `validate_store`'s sampled check.

`format_version` is the source's; completion writes into the source's arrays
and therefore its encoding (ADR 0038 §4). Ragged completion differs from Dense
completion in one respect the assertions below deliberately exercise: it
*re-fits* coefficients over the completed association list (observed plus
imputed) rather than reusing the source's, so the round-trip tolerance is
against the source's own decoded values, not a reused model.
"""

from __future__ import annotations

import csv
import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest

from opengwasdb import StoreManifest, open_store
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis

# One thousand base pairs apart, so the whole block of interest sits inside a
# single cis window around the trait position used in the manifest.
N_SOURCE = 600
_BP_FIRST = 1_000_000
_BP_STEP = 1_000
_TRAIT_BP = 1_020_000
_FREQS = np.linspace(0.05, 0.95, N_SOURCE, dtype=np.float32)

# Block positions analysis `b` leaves unobserved (analysis `a` still observes
# them, so they exist on the source's axis and only `b` needs them imputed).
_B_MISSING = frozenset({36, 37, 38, 39})
# Panel-only positions the source never holds: three that impute and one whose
# panel frequency is absent, which must remain an explicit missing row.
_NEW_BPS = (1_000_500, 1_001_500, 1_002_500)
_EAF_LESS_BP = 1_003_500
# The exact-exception cells: one per Analysis, a full log-unit off the model.
_EXACT_EXCEPTION = {"a": 10, "b": 12}


def _expected_se(frequencies: np.ndarray, offset: float) -> np.ndarray:
    """SE that tracks the MAF model plus a small periodic wobble.

    The wobble keeps the fixture honest -- perfectly on-model values would all
    land on one residual code -- while staying far inside the ±0.5 range the
    build selects, so nearly every cell codes as an ordinary residual.
    """
    return np.exp(
        (-3.0 + offset)
        - 0.5 * np.log(2 * frequencies * (1 - frequencies))
        + 0.12 * np.sin(np.arange(len(frequencies)) * (0.07 + offset))
    ).astype(np.float32)


def _source_bp(k: int) -> int:
    return _BP_FIRST + k * _BP_STEP


def _source_z(k: int) -> float:
    """The z-score a source row carries.

    Every tenth variant is a decisive hit (|z| = 8, p ~ 1e-15) so the store's
    top-hit tiers are populated, while the LD block's own variants vary enough
    for the completion kernel's elastic-net fit to clear its correlation gate:
    a block whose z values were nearly constant would let imputation depend on
    the arbitrary sign of one LD matrix draw.
    """
    if k % 10 == 0:
        return 8.0
    if k < 60:
        return (0.0, 1.4, -0.9, 2.3, -1.7, 3.0, -2.1, 1.1, -1.2, 2.8)[k % 10]
    return 1.0


def _write_filtered_ssf(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    """One filtered GWAS-SSF file: (bp, beta, se, eaf) rows for chr 1 A/G."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele"
            "\tbeta\tstandard_error\teffect_allele_frequency\n"
        )
        for bp, beta, se, eaf in rows:
            fh.write(f"1\t{bp}\tA\tG\t{beta:.7g}\t{se:.7g}\t{eaf:.7g}\n")


def _write_source_manifest(path: Path) -> None:
    path.write_text(
        "analysis_index\tanalysis_id\ttrait_id\ttrait_chr\ttrait_bp\tn\tfiltered_file\n"
        f"0\ta\tT1\t1\t{_TRAIT_BP}\t50000\ta.tsv.gz\n"
        f"1\tb\tT2\t1\t{_TRAIT_BP}\t50000\tb.tsv.gz\n",
        encoding="utf-8",
    )


def build_residual_ssf_source(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """A residual-codable Ragged observed store from EAF-carrying GWAS-SSF.

    Returns the store path and, per Analysis, the physical SE array the source
    rows carry (already perturbed for the two exact-exception cells), so a
    caller can assert against what it put in.
    """
    per_analysis_se = {"a": _expected_se(_FREQS, 0.0), "b": _expected_se(_FREQS, 0.2)}
    for analysis_id, k in _EXACT_EXCEPTION.items():
        per_analysis_se[analysis_id][k] *= float(np.exp(1.0))

    filtered_dir = root / "filtered"
    filtered_dir.mkdir(parents=True)
    rows_a: list[tuple[int, float, float, float]] = []
    rows_b: list[tuple[int, float, float, float]] = []
    for k in range(N_SOURCE):
        z = _source_z(k)
        rows_a.append(
            (_source_bp(k), z * per_analysis_se["a"][k], per_analysis_se["a"][k], _FREQS[k])
        )
        if k not in _B_MISSING:
            rows_b.append(
                (_source_bp(k), z * per_analysis_se["b"][k], per_analysis_se["b"][k], _FREQS[k])
            )
    _write_filtered_ssf(filtered_dir / "a.tsv.gz", rows_a)
    _write_filtered_ssf(filtered_dir / "b.tsv.gz", rows_b)
    _write_source_manifest(root / "manifest.tsv")

    store = root / "obs.opengwasdb"
    build_ragged_from_ssf(
        root / "manifest.tsv",
        filtered_dir,
        store,
        store_id="residual-rag",
        release_id="obs",
        allow_unverified_eaf=True,
    )
    return store, per_analysis_se


def write_eaf_carrying_panel(panel_root: Path) -> Path:
    """An LD panel whose single chr-1 block adds four positions to the source.

    The block's variant table lists the first 40 source variants (the cis
    window the manifest's trait position sits inside), the three panel-only
    positions that impute, and one panel-only position with no frequency --
    which is a panel this pipeline is required to complete against (issue
    #113) but cannot derive an imputed SE from. The eigendecomposition is
    precomputed and stored, which is the artifact completion actually reads
    (ADR 0031, spec §13.1).
    """
    rows: list[tuple[str, float, int]] = [
        (f"1:{_source_bp(k)}:A:G", float(_FREQS[k]), _source_bp(k)) for k in range(40)
    ]
    rows += [(f"1:{bp}:A:G", float(_FREQS[k]), bp) for k, bp in enumerate(_NEW_BPS)]
    rows.append((f"1:{_EAF_LESS_BP}:A:G", float("nan"), _EAF_LESS_BP))

    block_dir = panel_root / "EUR" / "1"
    block_dir.mkdir(parents=True)
    tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in rows:
        chrom, _, ea, oa = alid.split(":")
        eaf_text = "" if np.isnan(eaf) else f"{eaf:.7g}"
        tsv_lines.append(f"{chrom}\t{alid}\t{oa}\t{ea}\t{eaf_text}\t{bp}")
    (block_dir / "residual-block.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    n = len(rows)
    rng = np.random.default_rng(140)
    noise = rng.normal(size=(n, n))
    ld = noise @ noise.T + np.eye(n) * n * 0.1
    eigenvalues, eigenvectors = np.linalg.eigh((ld + ld.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    np.savez(
        block_dir / "residual-block.ldeig.npz",
        values=eigenvalues[order],
        vectors=eigenvectors[:, order],
    )
    return panel_root


def _panel_eaf_by_alid(panel: Path) -> dict[str, float]:
    with open(panel / "EUR" / "1" / "residual-block.tsv", encoding="utf-8") as fh:
        return {
            row["SNP"]: float(row["EAF"])
            for row in csv.DictReader(fh, delimiter="\t")
            if row["EAF"]
        }


class RaggedResidualScenario:
    """The observed source, its residual-codable fixture values and the
    completed release built from it -- shared by every test in this module."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        root = tmp_path_factory.mktemp("ragged_residual_140")
        self.source, self.expected_se = build_residual_ssf_source(root)
        panel = write_eaf_carrying_panel(root / "ld_panel")
        self.panel_eaf = _panel_eaf_by_alid(panel)
        self.completed = root / "comp.opengwasdb"
        self.result = complete_ragged_store(
            self.source,
            self.completed,
            panel,
            ancestry="EUR",
            cis_window_bp=1_000_000,
            min_cor=0.0,
            release_id="comp",
        )

    def expected_observed(self) -> dict[str, dict[str, tuple[float, float]]]:
        """The source's decoded observed cells per Analysis, keyed by ALID.

        What the completed release has to reproduce: the physical `se` and
        `eaf` the source store itself reads back (fixture source values after
        the source's own quantisation).
        """
        src = RaggedCSRReader(self.source)
        axis = VariantAxis(self.source)
        alid_by_index = {v.variant_index: v.alid for v in axis.all()}
        axis.close()
        out: dict[str, dict[str, tuple[float, float]]] = {}
        for ai, analysis_id in enumerate(("a", "b")):
            row = src.get_analysis(ai)
            out[analysis_id] = {
                alid_by_index[int(vi)]: (float(se), float(eaf))
                for vi, se, eaf in zip(row.variant_index, row.se, row.eaf, strict=True)
            }
        return out


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    return RaggedResidualScenario(tmp_path_factory)


def _completed_root(scenario: RaggedResidualScenario):
    return open_store(scenario.completed).arrays(mode="r")["ragged"]


def _result_positions(result: dict[str, np.ndarray], reader: RaggedCSRReader) -> np.ndarray:
    """The flat CSR ordinal of every row of an analysis-shaped query result.

    Each Analysis's CSR slice is sorted by variant index and the result rows
    are the slice in order, so one searchsorted per result row recovers the
    ordinal the plane's exception table and overflow table are keyed on.
    """
    offsets = reader._offsets[:]
    positions = np.empty(len(result["variant_index"]), dtype=np.int64)
    for ai in np.unique(result["analysis_index"]):
        start, end = int(offsets[ai]), int(offsets[ai + 1])
        want = result["variant_index"][result["analysis_index"] == ai]
        slice_variants = reader._variant_index[start:end].astype(np.int64)
        pos = np.searchsorted(slice_variants, want)
        assert np.all(slice_variants[pos] == want), "result row not found in its CSR slice"
        positions[result["analysis_index"] == ai] = start + pos
    return positions


def test_source_is_residual_codable_and_leaves_panel_cells_unobserved(scenario) -> None:
    """The fixture only tests anything if the source chose the residual coding
    and one Analysis genuinely left block positions unobserved."""
    encoding = StoreManifest.load(scenario.source).encoding
    assert encoding.se.is_residual, "fixture must select int8_residual to be meaningful"
    assert encoding.se.kind == "int8_residual"

    root = open_store(scenario.source).arrays(mode="r")["ragged"]
    for side in ("se_coefficients", "se_exception_index", "se_exception_value"):
        assert side in root, f"source residual se plane is missing {side}"
    # The two deliberate off-model cells are what the fixture's exception
    # assertions will pin after completion; they must be exceptions already.
    assert len(root["se_exception_index"][:]) == len(_EXACT_EXCEPTION)

    src = RaggedCSRReader(scenario.source)
    rows_a, rows_b = src.get_analysis(0), src.get_analysis(1)
    assert len(rows_a.variant_index) == N_SOURCE
    assert len(rows_b.variant_index) == N_SOURCE - len(_B_MISSING)
    axis = VariantAxis(scenario.source)
    alids = {v.variant_index: v.alid for v in axis.all()}
    axis.close()
    b_has = {alids[int(vi)] for vi in rows_b.variant_index}
    missing_b = {f"1:{_source_bp(k)}:A:G" for k in _B_MISSING}
    assert missing_b <= (set(alids.values()) - b_has), (
        "analysis b must leave in-block positions unobserved for completion to impute"
    )


def test_completion_preserves_the_residual_plan_and_side_arrays(scenario) -> None:
    """ADR 0038 §4: completion writes into the source's arrays, so the
    completed release keeps the source's SE kind and range, and adds only the
    `eaf_reference` the panel's frequencies buy it."""
    source_encoding = StoreManifest.load(scenario.source).encoding
    completed_encoding = StoreManifest.load(scenario.completed).encoding
    assert completed_encoding.se == source_encoding.se
    assert completed_encoding.se.is_residual
    assert completed_encoding.se.residual_range == source_encoding.se.residual_range
    assert completed_encoding.eaf.reference, "EAF-carrying panel must add eaf_reference"

    root = _completed_root(scenario)
    for side in (
        "se_coefficients",
        "se_exception_index",
        "se_exception_value",
        "eaf_reference",
        "imputed",
    ):
        assert side in root, f"completed store is missing {side}"

    # Imputation actually happened, and the frequency-less panel position
    # stayed missing rather than silently receiving a value (ADR 0037 §4).
    assert scenario.result.n_imputed >= 8, "fixture must impute for this to mean anything"
    assert scenario.result.n_missing == 2, (
        "exactly the EAF-less panel position should stay missing per Analysis"
    )
    imputed = root["imputed"][:]
    assert int(imputed.sum()) == scenario.result.n_imputed


def test_observed_cells_round_trip_through_the_completed_plan(scenario) -> None:
    """Every observed cell decodes back within the residual contract, and the
    two exact-exception cells come back bit-for-bit (their values are held in
    the exception table, not approximated)."""
    expected = scenario.expected_observed()
    root = _completed_root(scenario)
    assert len(root["se_exception_index"][:]) == len(_EXACT_EXCEPTION), (
        "completion must carry the source's exact-exception cells across"
    )

    with query_store(scenario.completed) as query:
        for analysis_id in ("a", "b"):
            result = query.analysis(analysis_id)
            axis = VariantAxis(scenario.completed)
            alid_by_index = {v.variant_index: v.alid for v in axis.all()}
            axis.close()
            for vi, se, eaf in zip(
                result["variant_index"], result["se"], result["eaf"], strict=True
            ):
                alid = alid_by_index[int(vi)]
                if alid not in expected[analysis_id]:
                    continue  # a panel-added row, checked by the imputed test
                exp_se, exp_eaf = expected[analysis_id][alid]
                if _is_exact_exception(analysis_id, alid):
                    assert float(se) == exp_se, f"{alid}: exception cell must stay exact"
                else:
                    assert exp_se > 0
                    rel = abs(float(se) - exp_se) / exp_se
                    assert rel <= 0.01, f"{alid}: observed se round-trip off by {rel:.4f}"
                np.testing.assert_allclose(float(eaf), exp_eaf, rtol=1e-6, err_msg=alid)


def _is_exact_exception(analysis_id: str, alid: str) -> bool:
    k = _EXACT_EXCEPTION[analysis_id]
    return alid == f"1:{_source_bp(k)}:A:G"


def test_imputed_cells_decode_panel_eaf_and_finite_positive_se(scenario) -> None:
    """Imputed cells read the panel's frequency, and cells the panel cannot
    serve stay explicitly missing instead of being substituted with one."""
    with query_store(scenario.completed) as query:
        results = {analysis_id: query.analysis(analysis_id) for analysis_id in ("a", "b")}

    axis = VariantAxis(scenario.completed)
    alid_by_index = {v.variant_index: v.alid for v in axis.all()}
    axis.close()

    any_imputed = False
    for analysis_id, result in results.items():
        imputed = result["association_status"] == "imputed"
        missing = result["association_status"] == "missing"
        assert imputed.sum() > 0, f"analysis {analysis_id} must gain imputed cells"
        any_imputed = True
        assert missing.sum() == 1
        for vi, se, eaf, z, status in zip(
            result["variant_index"],
            result["se"],
            result["eaf"],
            result["z"],
            result["association_status"],
            strict=True,
        ):
            alid = alid_by_index[int(vi)]
            if status == "imputed":
                assert np.isfinite(z), alid
                assert np.isfinite(se) and se > 0, f"{alid}: imputed se not finite positive"
                assert np.isfinite(eaf), f"{alid}: imputed cell has no decodable EAF (#159)"
                np.testing.assert_allclose(
                    float(eaf), scenario.panel_eaf[alid], rtol=1e-6, err_msg=alid
                )
            elif status == "missing":
                assert alid == f"1:{_EAF_LESS_BP}:A:G", (
                    f"{alid}: only the EAF-less panel position may stay missing"
                )
                assert np.isnan(z) and np.isnan(se) and np.isnan(eaf), (
                    f"{alid}: a missing cell must read NaN, never a substituted value"
                )
    assert any_imputed


def test_queries_and_top_hits_agree_with_the_decoded_csr(scenario) -> None:
    """The query facade and the precomputed top-hit index return the values
    the plane decodes at the same CSR ordinals -- every row, not the sample
    `validate_store` checks."""
    reader = RaggedCSRReader(scenario.completed)
    with query_store(scenario.completed) as query:
        for analysis_id in ("a", "b"):
            result = query.analysis(analysis_id)
            positions = _result_positions(result, reader)
            np.testing.assert_array_equal(reader.se_at(positions), result["se"])
            np.testing.assert_array_equal(reader.eaf_at(positions), result["eaf"])
            np.testing.assert_array_equal(reader.z_at(positions), result["z"])

        for threshold in (5e-8, 5e-6, 5e-4):
            top = query.top_hits(threshold=threshold)
            assert len(top["z"]) > 0, "fixture must produce top hits at every threshold"
            positions = _result_positions(top, reader)
            np.testing.assert_array_equal(reader.se_at(positions), top["se"])
            np.testing.assert_array_equal(reader.eaf_at(positions), top["eaf"])
            np.testing.assert_array_equal(reader.z_at(positions), top["z"])


def test_top_hit_index_is_consistent_with_the_decoded_plane_even_when_stale(
    scenario, tmp_path
) -> None:
    """An index holding pre-quantisation source values -- the stale-index
    defect class issue #163 closed for Hybrid -- fails the exhaustive
    index-vs-plane comparison above, which checks every returned row where
    `validate_store` samples only a bounded subset of each tier."""
    from opengwasdb.layouts.dense.top_hits import threshold_key

    stale_store = tmp_path / "stale.opengwasdb"
    shutil.copytree(scenario.completed, stale_store)
    source_expected = scenario.expected_observed()

    root = open_store(stale_store).arrays(mode="r+")
    group = root[f"top_hits/{threshold_key(5e-8)}"]
    reader = RaggedCSRReader(stale_store)
    offsets = reader._offsets[:]
    vi = group["variant_index"][:].astype(np.int64)
    ai = group["analysis_index"][:].astype(np.int64)
    stale = np.empty(len(vi), dtype=np.float32)
    for i, (v, a) in enumerate(zip(vi, ai, strict=True)):
        start, end = int(offsets[a]), int(offsets[a + 1])
        slice_variants = reader._variant_index[start:end].astype(np.int64)
        pos = np.searchsorted(slice_variants, v)
        assert slice_variants[pos] == v
        alid = _variant_alid(stale_store, int(slice_variants[pos]))
        stale[i] = source_expected["a" if a == 0 else "b"][alid][0]
    decoded = group["se"][:].astype(np.float32)
    assert not np.array_equal(stale, decoded), (
        "fixture must differ from the plane to test the check"
    )

    group["se"][:] = stale

    with pytest.raises(AssertionError):
        with query_store(stale_store) as query:
            top = query.top_hits(threshold=5e-8)
            positions = _result_positions(top, RaggedCSRReader(stale_store))
            np.testing.assert_array_equal(
                RaggedCSRReader(stale_store).se_at(positions), top["se"]
            )


def _variant_alid(store: Path, variant_index: int) -> str:
    axis = VariantAxis(store)
    alid = axis.by_index(int(variant_index)).alid
    axis.close()
    return alid


def test_completed_store_validates(scenario) -> None:
    result = validate_store(scenario.completed)
    assert result.ok, result.errors
    # Sanity: the observed source validates too, so any failure above is
    # specific to the completed release's own path.
    assert validate_store(scenario.source).ok


def test_completed_store_follows_the_ragged_layout_contract(scenario) -> None:
    """Reference Completion writes the same ragged group shapes as the observed
    builder, plus the completion-only arrays a reader and validator expect."""
    assert (scenario.completed / "data.zarr" / "ragged").is_dir()
    root = _completed_root(scenario)
    assert root.attrs["completion_state"] == "reference_completed"
    assert root.attrs["n_analyses"] == 2
    assert root["se"].dtype == np.dtype("int8")
    flat_length = len(root["se"])
    assert all(len(root[name]) == flat_length for name in ("z", "eaf", "imputed", "variant_index"))
    assert len(root["offsets"]) == root.attrs["n_analyses"] + 1
    imp = root["imputed"][:]
    assert np.all((imp == 0) | (imp == 1))

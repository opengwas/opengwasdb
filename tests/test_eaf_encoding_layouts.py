"""Residual-coded `eaf` end to end, on the layouts that store it (#116).

ADR 0037 §2 and §4 are about stores, not arithmetic: a frequency put in by a
builder has to come back out of a query to within half a step of the range the
store declares, the ones the coding cannot express have to come back *exactly*,
and an Analysis that reported none has to come back as NaN rather than as a
residual of zero.

The fixtures are deliberately wider than `test_eaf_end_to_end.py`'s single
cell: the encoding tree only chooses the residual coding when the per-variant
baseline amortises over enough EAF-bearing cells, so a one-variant store is
(correctly) left in `float32` and would prove nothing about the codec.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.encoding import (
    EAF_ABSENT,
    EAF_BASELINE,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    EAF_REFERENCE,
    UnsupportedEncoding,
)
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import (
    read_analyses,
    read_analysis_records,
    write_analysis_records,
)
from opengwasdb.model.enums import EafScope
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

# Eight variants across four Analyses. The frequencies agree closely between
# Analyses -- which is what makes an 8-bit residual work at all -- except at
# 1:600, where one Analysis is three orders of magnitude away and lands in the
# exception table, and 1:700, where one is monomorphic.
_POSITIONS = [100, 200, 300, 400, 500, 600, 700, 800]
_BASE_EAF = [0.05, 0.12, 0.30, 0.47, 0.62, 0.008, 0.21, 0.91]
_ANALYSES = ["a1", "a2", "a3", "a4"]

#: The panel's frequency at the one variant no Analysis observed. Distinct from
#: every stored frequency, so "the imputed cell read the panel" and "the
#: imputed cell read a neighbour" cannot be confused.
PANEL_ONLY_EAF = 0.4

#: Per Analysis, `{position: source effect-allele frequency}`. The effect
#: allele is A throughout, which sorts first, so no orientation flip applies
#: and the stored frequency is the source's.
_EAF: dict[str, dict[int, float]] = {}
for _col, _analysis in enumerate(_ANALYSES):
    _EAF[_analysis] = {
        pos: round(base * (1.0 + 0.01 * (_col - 1)), 6)
        for pos, base in zip(_POSITIONS, _BASE_EAF, strict=True)
    }
# a3 disagrees wildly at 1:600 -- a founder-effect-sized difference, far
# outside any candidate range, so it must be stored exactly.
_EAF["a3"][600] = 0.85
# a4 is monomorphic at 1:700: no logit, so this cell is an exception too.
_EAF["a4"][700] = 0.0
# a2 reports no frequency at 1:300, which must read back as NaN and not as a
# residual of zero.
_EAF["a2"][300] = float("nan")

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FILTER=<ID=PASS,Description="All filters passed">\n'
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
    '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
    "##SAMPLE=<ID={sample},StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
)


def _z_for(analysis: str, position: int) -> float:
    """A distinct, ordinary z for every cell, so nothing collides by accident."""
    return round(0.5 + 0.1 * _ANALYSES.index(analysis) + 0.001 * position, 4)


def _vcf(tmp_path: Path, analysis: str) -> Path:
    path = tmp_path / f"{analysis}.vcf"
    lines = [_VCF_HEADER.format(sample=analysis.upper())]
    for pos in _POSITIONS:
        eaf = _EAF[analysis][pos]
        if np.isnan(eaf):
            lines.append(f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.3\n")
        else:
            z = _z_for(analysis, pos)
            lines.append(
                f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t{z * 0.3:.6f}:0.3:{1 - eaf:.6f}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _vcf_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.tsv"
    rows = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for analysis in _ANALYSES:
        rows.append(
            f"{analysis}\t{_vcf(tmp_path, analysis)}\t{analysis}\t10000\tsd"
            "\tdeclared_standardised\thg38"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _ssf(tmp_path: Path) -> tuple[Path, Path]:
    filtered = tmp_path / "ssf"
    filtered.mkdir()
    manifest_rows = ["analysis_index\tanalysis_id\tfiltered_file\tn"]
    for index, analysis in enumerate(_ANALYSES):
        with gzip.open(filtered / f"{analysis}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
                "beta\tstandard_error\teffect_allele_frequency\n"
            )
            for pos in _POSITIONS:
                eaf = _EAF[analysis][pos]
                fh.write(
                    f"1\t{pos}\tA\tG\t{_z_for(analysis, pos) * 0.3:.6f}\t0.3\t"
                    + ("" if np.isnan(eaf) else f"{eaf:.6f}")
                    + "\n"
                )
        manifest_rows.append(f"{index}\t{analysis}\t{analysis}.tsv.gz\t10000")
    manifest = tmp_path / "ssf_manifest.tsv"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return manifest, filtered


@pytest.fixture
def dense_store(tmp_path: Path) -> Path:
    out = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(
        _vcf_manifest(tmp_path), out, store_id="eafenc", release_id="v1",
        allow_unverified_eaf=True,
    )
    return out


@pytest.fixture
def ragged_store(tmp_path: Path) -> Path:
    manifest, filtered = _ssf(tmp_path)
    out = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, out, store_id="eafenc", release_id="v1",
        allow_unverified_eaf=True,
    )
    return out


#: The panel the Hybrid fixture is built against: half the fixture's variants
#: sit on it (Dense Component), half do not (Ragged Overflow).
_ON_PANEL = _POSITIONS[:4]


@pytest.fixture
def hybrid_store(tmp_path: Path) -> Path:
    panel = tmp_path / "panel.txt"
    panel.write_text(
        "\n".join(f"1:{pos}:A:G" for pos in _ON_PANEL) + "\n", encoding="utf-8"
    )
    out = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _vcf_manifest(tmp_path), out, reference_panel=panel,
        store_id="eafenc", release_id="v1", allow_unverified_eaf=True,
    )
    return out


def _observed(store: Path) -> dict[tuple[int, str], float]:
    """`{(position, analysis_id): decoded eaf}` over the whole store."""
    with query_store(store) as query:
        variants = query.variants_table()
        analyses = query.analyses_table()
        out: dict[tuple[int, str], float] = {}
        for analysis in analyses.values():
            analysis_id = str(analysis["analysis_id"])
            result = query.analysis(analysis_id)
            for vi, eaf in zip(result["variant_index"], result["eaf"], strict=True):
                out[(int(variants[int(vi)]["position"]), analysis_id)] = float(eaf)
    return out


# ── The plan a real build settles on ────────────────────────────────────────


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_a_build_with_frequencies_declares_the_residual_coding(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    manifest = json.loads((store / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"]["kind"] == "int8_residual"
    assert manifest["encoding"]["eaf"]["residual_range"] in (0.5, 1.0, 2.0)
    assert manifest["format_version"] == "2.0"


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_the_plane_is_int8_with_a_baseline_and_an_exception_table(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    group = root["ragged"] if layout == "ragged" else root
    assert str(group["eaf"].dtype) == "int8"
    assert len(group[EAF_BASELINE]) == len(_POSITIONS)
    # Written even when empty, so "residual-coded" and "has a table" are the
    # same statement (ADR 0037 §1, applied to eaf).
    assert EAF_EXCEPTION_INDEX in group
    assert EAF_EXCEPTION_VALUE in group


# ── The acceptance criteria ─────────────────────────────────────────────────


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_every_unclipped_frequency_round_trips_within_half_a_step(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    tolerance = open_store(store).manifest.encoding.eaf.worst_case_relative_error
    observed = _observed(store)
    checked = 0
    for analysis in _ANALYSES:
        for pos in _POSITIONS:
            expected = _EAF[analysis][pos]
            if np.isnan(expected) or expected in (0.0, 1.0):
                continue
            if analysis == "a3" and pos == 600:  # the exception cell
                continue
            got = observed[(pos, analysis)]
            worst = max(
                abs(got - expected) / expected, abs(got - expected) / (1.0 - expected)
            )
            assert worst <= tolerance * 1.05, (analysis, pos, got, expected)
            checked += 1
    assert checked == 29  # the fixture really does exercise the codec


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_a_clipped_cell_resolves_to_its_exact_value_not_to_the_baseline(
    layout: str, dense_store: Path, ragged_store: Path
):
    """a3 is 100x from the baseline at 1:600 -- the founder-effect shape that
    makes clipping to the range unacceptable (ADR 0037 §2)."""
    store = dense_store if layout == "dense" else ragged_store
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    group = root["ragged"] if layout == "ragged" else root
    assert len(group[EAF_EXCEPTION_INDEX]) >= 2  # a3 at 1:600, a4 at 1:700
    assert _observed(store)[(600, "a3")] == pytest.approx(0.85, abs=1e-6)


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_a_monomorphic_frequency_is_stored_exactly(
    layout: str, dense_store: Path, ragged_store: Path
):
    """0 has no logit, so it cannot be a residual; it is held exactly rather
    than nudged to a representable neighbour."""
    store = dense_store if layout == "dense" else ragged_store
    assert _observed(store)[(700, "a4")] == 0.0


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_an_analysis_with_no_frequency_reads_nan_not_a_zero_residual(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    observed = _observed(store)
    assert np.isnan(observed[(300, "a2")])
    # The neighbouring cells at the same variant are real, so "absent" is not
    # standing in for "the whole variant is missing".
    assert observed[(300, "a1")] == pytest.approx(_EAF["a1"][300], rel=1e-2)

    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    group = root["ragged"] if layout == "ragged" else root
    codes = np.asarray(group["eaf"][:])
    assert EAF_ABSENT in codes.ravel().tolist()
    # A residual of zero is a different code, and both occur in this fixture.
    assert 0 in codes.ravel().tolist()


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_the_store_validates_against_its_declared_plan(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    result = validate_store(store)
    assert result.ok, result.errors


@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_analyses_tsv_still_declares_eaf_scope_association(
    layout: str, dense_store: Path, ragged_store: Path
):
    store = dense_store if layout == "dense" else ragged_store
    for row in read_analyses(store / "analyses.tsv").rows:
        assert row["eaf_scope"] == EafScope.ASSOCIATION.value


# ── What a store must not be able to say ────────────────────────────────────


def test_an_eaf_plane_that_contradicts_the_manifest_is_rejected(dense_store: Path):
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["eaf"] = {"kind": "float32"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(dense_store)
    assert not result.ok
    assert any("dtype int8" in error for error in result.errors), result.errors


def test_a_release_declaring_an_unknown_eaf_kind_is_refused(dense_store: Path):
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["eaf"] = {"kind": "int16_log_maf"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsupportedEncoding, match="int16_log_maf"):
        open_store(dense_store)


def test_a_residual_plane_with_no_baseline_is_rejected(dense_store: Path):
    """An `int8` residual plane is meaningless without its baseline -- and
    "meaningless" must not decode to a plausible frequency."""
    root = zarr.open_group(str(dense_store / "data.zarr"), mode="a")
    del root[EAF_BASELINE]

    result = validate_store(dense_store)
    assert not result.ok
    assert any(EAF_BASELINE in error for error in result.errors), result.errors


def test_an_exception_table_that_lost_a_cell_is_rejected(dense_store: Path):
    """The lost value is the frequency furthest from its baseline -- the rare
    variant a user is filtering on."""
    root = zarr.open_group(str(dense_store / "data.zarr"), mode="a")
    kept_index = np.asarray(root[EAF_EXCEPTION_INDEX][:])[1:]
    kept_value = np.asarray(root[EAF_EXCEPTION_VALUE][:])[1:]
    for name, data, dtype in (
        (EAF_EXCEPTION_INDEX, kept_index, "int64"),
        (EAF_EXCEPTION_VALUE, kept_value, "float32"),
    ):
        del root[name]
        root.create_dataset(name, data=data, chunks=(max(1, len(data)),), dtype=dtype)

    result = validate_store(dense_store)
    assert not result.ok
    assert any("no entry in the eaf exception table" in e for e in result.errors), result.errors


def test_a_store_whose_eaf_scope_contradicts_its_plan_is_rejected(dense_store: Path):
    """`eaf_scope` (per Analysis) and the plan (per store) describe the same
    fact at two granularities; their disagreement is the defect that got
    through review on #106."""
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["eaf"] = {"kind": "absent"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(dense_store)
    assert not result.ok
    assert any("eaf_scope=association" in error for error in result.errors), result.errors


# ── Reference EAF for imputed cells (ADR 0037 §4) ───────────────────────────


def _ld_panel(tmp_path: Path) -> Path:
    """A one-block panel covering the store's variants plus one it lacks.

    The panel's frequencies deliberately differ from the store's: substituting
    them for an observed cell is the 3000x error ADR 0037 §4 exists to prevent,
    so the test can only tell the two apart if they disagree. `1:900:A:G` is on
    the panel and in no Analysis, which is what gives completion a cell to
    impute -- and therefore a cell whose frequency must come from the panel.
    """
    root = tmp_path / "ld"
    block_dir = root / "EUR" / "1"
    block_dir.mkdir(parents=True)
    panel = [(pos, min(base * 2.5, 0.97)) for pos, base in zip(_POSITIONS, _BASE_EAF, strict=True)]
    panel.append((900, PANEL_ONLY_EAF))
    rows = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for pos, eaf in panel:
        rows.append(f"1\t1:{pos}_A_G\tG\tA\t{eaf:.6f}\t{pos}")
    (block_dir / "block1.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # A positive-definite LD matrix over the block, so completion has something
    # to impute from -- without it the block is skipped and the test could not
    # fail (CONTRIBUTING: a test that cannot fail is worse than no test).
    n = len(panel)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    ld = a @ a.T + np.eye(n) * n * 0.1
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (block_dir / "block1.unphased.vcor1.gz").write_bytes(buf.getvalue())
    return root


@pytest.fixture
def completed_store(tmp_path: Path, dense_store: Path) -> Path:
    out = tmp_path / "dense-completed.opengwasdb"
    complete_dense_store(
        dense_store, out, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    return out


def test_a_completed_release_declares_and_carries_reference_eaf(completed_store: Path):
    manifest = json.loads((completed_store / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"]["reference"] is True
    root = zarr.open_group(str(completed_store / "data.zarr"), mode="r")
    assert EAF_REFERENCE in root
    assert len(root[EAF_REFERENCE]) == root["eaf"].shape[0]


def test_imputed_cells_read_the_panel_frequency(completed_store: Path):
    """An imputed cell's EAF *is* the panel's, identical for every Analysis
    imputed at that variant -- which is why it is stored once per variant."""
    with query_store(completed_store) as query:
        result = query.phewas("1:900:A:G")
    imputed = result["association_status"] == "imputed"
    assert imputed.any(), "the fixture imputed nothing, so this test could not fail"
    np.testing.assert_allclose(result["eaf"][imputed], PANEL_ONLY_EAF, atol=1e-6)


def test_observed_cells_never_take_the_panel_frequency(completed_store: Path):
    """The central negative result: a cohort's own frequency is not the
    panel's, and an EAF-less observed cell stays NaN rather than borrowing one
    (ADR 0037 §4)."""
    observed = _observed(completed_store)
    assert observed[(100, "a1")] == pytest.approx(_EAF["a1"][100], rel=1e-2)
    assert observed[(100, "a1")] != pytest.approx(min(_BASE_EAF[0] * 2.5, 0.97), rel=1e-3)
    assert np.isnan(observed[(300, "a2")])


def test_a_completed_release_validates(completed_store: Path):
    result = validate_store(completed_store)
    assert result.ok, result.errors


def test_a_reference_array_the_plan_does_not_declare_is_rejected(completed_store: Path):
    manifest_path = completed_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoding"]["eaf"].pop("reference")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_store(completed_store)
    assert not result.ok
    assert any(EAF_REFERENCE in error for error in result.errors), result.errors


# ── A release with no frequencies of its own (issue #113) ───────────────────


def _vcf_without_frequencies(tmp_path: Path, analysis: str) -> Path:
    """The same Analysis, with no `AF` field on any record."""
    path = tmp_path / f"{analysis}-nofreq.vcf"
    lines = [_VCF_HEADER.format(sample=analysis.upper())]
    for pos in _POSITIONS:
        z = _z_for(analysis, pos)
        lines.append(f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE\t{z * 0.3:.6f}:0.3\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def store_with_no_frequencies(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "nofreq-manifest.tsv"
    rows = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for analysis in _ANALYSES:
        rows.append(
            f"{analysis}\t{_vcf_without_frequencies(tmp_path, analysis)}\t{analysis}"
            "\t10000\tsd\tdeclared_standardised\thg38"
        )
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "nofreq.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest_path, out, store_id="nofreq", release_id="v1", allow_unverified_eaf=True
    )
    return out


@pytest.fixture
def completed_store_with_no_source_frequencies(
    tmp_path: Path, store_with_no_frequencies: Path
) -> Path:
    out = tmp_path / "nofreq-completed.opengwasdb"
    complete_dense_store(
        store_with_no_frequencies, out, _ld_panel(tmp_path),
        ancestry="EUR", min_cor=0.0, thresh=0.99,
    )
    return out


def test_a_source_with_no_frequencies_declares_absent_and_no_reference(
    store_with_no_frequencies: Path,
):
    """The fixture is asserted meaningful before anything is asserted about
    what completion does to it: an observed store with no frequencies, so the
    completed one's frequencies can only have come from the panel."""
    manifest = json.loads((store_with_no_frequencies / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"]["kind"] == "absent"
    assert "reference" not in manifest["encoding"]["eaf"]
    root = zarr.open_group(str(store_with_no_frequencies / "data.zarr"), mode="r")
    assert "eaf" not in root


def test_completing_a_source_with_no_frequencies_still_carries_panel_eaf(
    completed_store_with_no_source_frequencies: Path,
):
    """Issue #113's whole point, and the case it is easiest to skip: the
    release has no `eaf` plane to put anything in, and its imputed cells still
    have a frequency -- the panel's. A completion that stored none here would
    hand every imputed cell NaN for a value it holds."""
    store = completed_store_with_no_source_frequencies
    manifest = json.loads((store / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"]["kind"] == "absent"
    assert manifest["encoding"]["eaf"]["reference"] is True

    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    assert "eaf" not in root, "there are no observed frequencies to store"
    assert EAF_REFERENCE in root

    with query_store(store) as query:
        result = query.phewas("1:900:A:G")
    imputed = result["association_status"] == "imputed"
    assert imputed.any(), "the fixture imputed nothing, so this test could not fail"
    np.testing.assert_allclose(result["eaf"][imputed], PANEL_ONLY_EAF, atol=1e-6)


def test_an_observed_cell_of_a_source_with_no_frequencies_stays_nan(
    completed_store_with_no_source_frequencies: Path,
):
    """The other half of the criterion, and the one that costs users if it
    fails: an observed cell whose source reported no frequency does not
    quietly acquire the panel's (ADR 0037 §4)."""
    observed = _observed(completed_store_with_no_source_frequencies)
    on_panel = [(pos, a) for (pos, a) in observed if pos in _POSITIONS]
    assert on_panel, "no observed cells in the fixture"
    assert all(np.isnan(observed[key]) for key in on_panel)


def test_a_release_with_no_plane_but_panel_frequencies_validates(
    completed_store_with_no_source_frequencies: Path,
):
    result = validate_store(completed_store_with_no_source_frequencies)
    assert result.ok, result.errors


def test_completion_stamps_eaf_scope_when_the_source_declared_none(
    completed_store_with_no_source_frequencies: Path,
):
    """`eaf_scope` follows what the release holds. An Analysis that gained
    imputed cells now stores a frequency for them even though its source
    reported none, and saying otherwise is #106's defect (declared metadata
    disagreeing with the arrays) in a new place."""
    rows = read_analyses(
        completed_store_with_no_source_frequencies / "analyses.tsv"
    ).rows
    scopes = {row["analysis_id"]: row["eaf_scope"] for row in rows}
    assert any(scope == EafScope.ASSOCIATION.value for scope in scopes.values()), scopes


@pytest.fixture
def ragged_store_with_no_frequencies(tmp_path: Path) -> Path:
    """The same source with no frequencies, in the other CSR pipeline.

    Dense and Ragged completion are separately implemented, and the Ragged one
    is where this was found: covering only Dense would leave the half that
    failed uncovered.
    """
    filtered = tmp_path / "ssf-nofreq"
    filtered.mkdir()
    manifest_rows = ["analysis_index\tanalysis_id\tfiltered_file\tn"]
    for index, analysis in enumerate(_ANALYSES):
        with gzip.open(filtered / f"{analysis}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
                "beta\tstandard_error\teffect_allele_frequency\n"
            )
            for pos in _POSITIONS:
                fh.write(f"1\t{pos}\tA\tG\t{_z_for(analysis, pos) * 0.3:.6f}\t0.3\t\n")
        manifest_rows.append(f"{index}\t{analysis}\t{analysis}.tsv.gz\t10000")
    manifest = tmp_path / "ssf_nofreq_manifest.tsv"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    out = tmp_path / "ragged-nofreq.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, out, store_id="nofreq", release_id="v1",
        allow_unverified_eaf=True,
    )
    return out


def test_completing_a_ragged_source_with_no_frequencies_carries_panel_eaf(
    tmp_path: Path, ragged_store_with_no_frequencies: Path
):
    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    source_manifest = json.loads(
        (ragged_store_with_no_frequencies / "manifest.json").read_text()
    )
    assert source_manifest["encoding"]["eaf"]["kind"] == "absent"

    out = tmp_path / "ragged-nofreq-completed.opengwasdb"
    complete_ragged_store(
        ragged_store_with_no_frequencies, out, _ld_panel(tmp_path),
        ancestry="EUR", min_cor=0.0, thresh=0.99,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["encoding"]["eaf"]["reference"] is True
    assert validate_store(out).ok, validate_store(out).errors

    with query_store(out) as query:
        result = query.phewas("1:900:A:G")
    imputed = result["association_status"] == "imputed"
    assert imputed.any(), "the fixture imputed nothing, so this test could not fail"
    np.testing.assert_allclose(result["eaf"][imputed], PANEL_ONLY_EAF, atol=1e-6)

    observed = _observed(out)
    on_panel = [key for key in observed if key[0] in _POSITIONS]
    assert on_panel and all(np.isnan(observed[key]) for key in on_panel)

    # The pair-resolving read shape too (top hits go through it), which has its
    # own short circuit on "this component stores no eaf array" -- true here,
    # and not the same question as "this component has a frequency to report".
    from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader

    reader = RaggedCSRReader(out)
    pairs = [
        (vi, ai) for ai in range(len(_ANALYSES)) for vi in range(len(_POSITIONS) + 1)
    ]
    rows = np.array([vi for vi, _ in pairs], dtype=np.int64)
    cols = np.array([ai for _, ai in pairs], dtype=np.int64)
    by_pair = reader.eaf_pairs(rows, cols)
    assert np.any(np.isfinite(by_pair)), (
        "eaf_pairs reported no frequency at all for a release that holds the panel's"
    )
    np.testing.assert_allclose(by_pair[np.isfinite(by_pair)], PANEL_ONLY_EAF, atol=1e-6)


def test_a_completed_hybrid_whose_only_frequencies_are_the_panels_validates(
    tmp_path: Path,
):
    """`eaf_reference` is per component, and a Hybrid release's top-level plan
    describes its observed-only Ragged Overflow, which has no imputed cells and
    so declares none. Judging "does this release hold frequencies" from that
    plan alone calls the release a contradiction of its own `analyses.tsv`
    (spec §6a, §9)."""
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    manifest_path = tmp_path / "hybrid-nofreq-manifest.tsv"
    rows = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for analysis in _ANALYSES:
        rows.append(
            f"{analysis}\t{_vcf_without_frequencies(tmp_path, analysis)}\t{analysis}"
            "\t10000\tsd\tdeclared_standardised\thg38"
        )
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    panel_file = tmp_path / "hybrid-nofreq-panel.txt"
    # Wider than `_ON_PANEL`: the Dense Component has to hold enough observed
    # variants for the LD block to impute the one it lacks, or the fixture
    # imputes nothing and the rule under test is never reached. One position is
    # left off so there is still a Ragged Overflow.
    panel_file.write_text(
        "\n".join(f"1:{pos}:A:G" for pos in _POSITIONS[:-1]) + "\n", encoding="utf-8"
    )
    observed = tmp_path / "hybrid-nofreq.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest_path, observed, reference_panel=panel_file,
        store_id="nofreq", release_id="v1", allow_unverified_eaf=True,
    )
    assert json.loads((observed / "manifest.json").read_text())["encoding"]["eaf"][
        "kind"
    ] == "absent"

    out = tmp_path / "hybrid-nofreq-completed.opengwasdb"
    complete_hybrid_store(
        observed, out, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    dense_plan = json.loads(
        (dense_component_path(out) / "manifest.json").read_text()
    )["encoding"]["eaf"]
    assert dense_plan["reference"] is True, "the Dense Component imputed nothing"
    assert "reference" not in json.loads((out / "manifest.json").read_text())[
        "encoding"
    ]["eaf"], "the observed-only Ragged Overflow has no imputed cells to describe"

    scopes = [row["eaf_scope"] for row in read_analyses(out / "analyses.tsv").rows]
    assert EafScope.ASSOCIATION.value in scopes, (
        "no Analysis gained imputed cells, so the rule under test is never reached"
    )

    result = validate_store(out)
    assert result.ok, result.errors


def _panel_without_frequencies(tmp_path: Path) -> Path:
    """The same LD panel with its `EAF` column removed.

    `completion.ld_panel` has always read a missing panel frequency as NaN, so
    a panel without one is a panel this pipeline supports (issue #113) -- not
    an error, and not a reason to refuse to complete a store.
    """
    root = _ld_panel(tmp_path)
    block = root / "EUR" / "1" / "block1.tsv"
    lines = block.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    keep = [i for i, name in enumerate(header) if name != "EAF"]
    block.write_text(
        "\n".join("\t".join(line.split("\t")[i] for i in keep) for line in lines) + "\n",
        encoding="utf-8",
    )
    return root


def test_a_panel_with_no_eaf_column_completes_and_keeps_the_stores_own_frequencies(
    tmp_path: Path, dense_store: Path
):
    """Such a panel completes; it does not fail the run.

    Nothing is imputed against it, because the imputed `se` is scaled by the
    panel's heterozygosity and there is none (`completion/impute.py`, and a
    non-finite `se` is skipped in `completion/block.py`). That is deliberate --
    an `se` derived from a substituted heterozygosity is a fabricated one --
    and the spec says so (§6a), so "no imputed frequency" here is "no imputed
    cell", not a frequency dropped. What is under test is that asking the panel
    for reference EAF does not turn a supported panel into a failed completion,
    taking the store's *own* frequencies down with it.
    """
    out = tmp_path / "nofreq-panel-completed.opengwasdb"
    complete_dense_store(
        dense_store, out, _panel_without_frequencies(tmp_path),
        ancestry="EUR", min_cor=0.0, thresh=0.99,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert "reference" not in manifest["encoding"]["eaf"]
    assert EAF_REFERENCE not in zarr.open_group(str(out / "data.zarr"), mode="r")
    assert validate_store(out).ok, validate_store(out).errors

    observed = _observed(out)
    assert observed[(100, "a1")] == pytest.approx(_EAF["a1"][100], rel=1e-2)
    with query_store(out) as query:
        result = query.phewas("1:900:A:G")
    assert np.all(np.isnan(result["eaf"])), "no panel frequency, no imputed frequency"


def test_ogdb_info_reports_the_eaf_encoding(dense_store: Path):
    """CLI surface, which `opengwasdb-stores`' walkthrough quotes verbatim
    (CONTRIBUTING, "The walkthrough lives in the other repository")."""
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    result = CliRunner().invoke(app, ["info", str(dense_store)])
    assert result.exit_code == 0, result.output
    line = next(
        line for line in result.output.splitlines() if line.startswith("encoding:")
    )
    assert "eaf=int8_residual" in line
    assert "range" in line


# ── Hybrid: two components, two variant axes, one plan ──────────────────────
#
# The trap issue #119 named for this change: `dense/` is panel-sized and the
# shared table is the union, so a baseline written against the wrong axis
# returns frequencies that are wrong and plausible -- the same shape as the
# crossover bug in #99 and the EAF bug found in review of #106.


def test_hybrid_components_share_one_plan_and_size_their_own_baselines(
    hybrid_store: Path,
):
    hybrid = json.loads((hybrid_store / "manifest.json").read_text())
    dense = json.loads((dense_component_path(hybrid_store) / "manifest.json").read_text())
    assert hybrid["encoding"]["eaf"] == dense["encoding"]["eaf"]
    assert hybrid["encoding"]["eaf"]["kind"] == "int8_residual"

    root = zarr.open_group(str(hybrid_store / "data.zarr"), mode="r")
    dense_root = zarr.open_group(str(dense_component_path(hybrid_store) / "data.zarr"), mode="r")
    # The Dense Component's baseline is panel-sized; the Ragged Overflow's
    # covers the shared union. They are different lengths, and a component
    # holding the other's would decode to plausible nonsense.
    assert len(dense_root[EAF_BASELINE]) == len(_ON_PANEL)
    assert len(root["ragged"][EAF_BASELINE]) == len(_POSITIONS)


def test_hybrid_round_trips_frequencies_in_both_components(hybrid_store: Path):
    tolerance = open_store(hybrid_store).manifest.encoding.eaf.worst_case_relative_error
    observed = _observed(hybrid_store)
    on_panel_checked = off_panel_checked = 0
    for analysis in _ANALYSES:
        for pos in _POSITIONS:
            expected = _EAF[analysis][pos]
            if np.isnan(expected) or expected in (0.0, 1.0):
                continue
            if analysis == "a3" and pos == 600:
                continue
            got = observed[(pos, analysis)]
            worst = max(
                abs(got - expected) / expected, abs(got - expected) / (1.0 - expected)
            )
            assert worst <= tolerance * 1.05, (analysis, pos, got, expected)
            if pos in _ON_PANEL:
                on_panel_checked += 1
            else:
                off_panel_checked += 1
    # Both components really were exercised, in both directions.
    assert on_panel_checked and off_panel_checked


def test_hybrid_validates_against_its_declared_plan(hybrid_store: Path):
    result = validate_store(hybrid_store)
    assert result.ok, result.errors


def test_a_hybrid_component_with_no_frequencies_still_carries_the_declared_plane(
    tmp_path: Path,
):
    """The Dense Component and the Ragged Overflow share one plan, so a
    component with no frequency of its own still carries the plane that plan
    declares -- otherwise its manifest promises an array it does not have, and
    the release fails its own validation. An all-absent `int8` plane costs
    essentially nothing compressed.
    """
    # 1:100 is the only on-panel variant and no Analysis reports a frequency
    # there, so the Dense Component holds EAF for nothing while the Ragged
    # Overflow holds it for everything.
    positions = [100, 200, 300, 400, 500, 600]
    manifest_rows = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for index, analysis in enumerate(_ANALYSES):
        path = tmp_path / f"split_{analysis}.vcf"
        lines = [_VCF_HEADER.format(sample=analysis.upper())]
        for pos in positions:
            z = 0.5 + 0.1 * index + 0.001 * pos
            if pos == 100:
                lines.append(f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE\t{z * 0.3:.6f}:0.3\n")
            else:
                eaf = 0.2 + 0.01 * index
                lines.append(
                    f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t"
                    f"{z * 0.3:.6f}:0.3:{1 - eaf:.6f}\n"
                )
        path.write_text("".join(lines), encoding="utf-8")
        manifest_rows.append(
            f"{analysis}\t{path}\t{analysis}\t10000\tsd\tdeclared_standardised\thg38"
        )
    manifest = tmp_path / "split_manifest.tsv"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    panel = tmp_path / "split_panel.txt"
    panel.write_text("1:100:A:G\n", encoding="utf-8")

    out = tmp_path / "hybrid-split.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, out, reference_panel=panel,
        store_id="eafenc", release_id="v1", allow_unverified_eaf=True,
    )

    dense_root = zarr.open_group(str(dense_component_path(out) / "data.zarr"), mode="r")
    assert "eaf" in dense_root
    assert str(dense_root["eaf"].dtype) == "int8"
    assert np.all(np.asarray(dense_root["eaf"][:]) == EAF_ABSENT)
    result = validate_store(out)
    assert result.ok, result.errors


def test_a_completed_hybrid_records_the_panel_it_was_completed_against(
    tmp_path: Path, hybrid_store: Path
):
    """One panel per completed store, recorded in `manifest.json` (#116).

    Load-bearing once `eaf_reference` holds that panel's frequencies: a reader
    of the top-level Hybrid store has to be able to see which panel supplied
    them without descending into the Dense Component's own manifest.
    """
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    out = tmp_path / "hybrid-completed.opengwasdb"
    complete_hybrid_store(
        hybrid_store, out, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    completion = json.loads((out / "manifest.json").read_text())["provenance"]["completion"]
    dense_completion = json.loads(
        (dense_component_path(out) / "manifest.json").read_text()
    )["provenance"]["completion"]

    assert completion["ancestry"] == "EUR"
    assert completion["ld_panel_id"] == dense_completion["ld_panel_id"]
    assert completion["method"] == dense_completion["method"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m["provenance"].pop("completion"), "records no completion provenance"),
        (lambda m: m["provenance"]["completion"].pop("ld_panel_id"), "without ld_panel_id"),
        (lambda m: m["provenance"]["completion"].update(ancestry=None), "without ancestry"),
    ],
    ids=["no-completion-block", "no-panel-id", "null-ancestry"],
)
def test_a_hybrid_refuses_to_write_a_manifest_that_names_no_panel(
    tmp_path: Path, hybrid_store: Path, mutate, expected: str
):
    """The copy-up fails the completion; it does not write nulls and warn.

    A completed Hybrid manifest with a null `ld_panel_id` is not a smaller
    answer than one naming the panel -- it is a release that cannot say where
    the frequencies on its imputed cells came from, and nothing downstream
    reads it strictly enough to notice (CONTRIBUTING.md, "fail loudly over
    degrading quietly"; #116's one-panel constraint).
    """
    from opengwasdb.layouts.hybrid.complete import (
        HybridCompletionError,
        _dense_completion_provenance,
        complete_hybrid_store,
    )

    out = tmp_path / "hybrid-completed-broken.opengwasdb"
    complete_hybrid_store(
        hybrid_store, out, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    # Fixture meaningfulness: it reads cleanly before the mutation.
    assert _dense_completion_provenance(out)["ld_panel_id"]

    component_manifest = dense_component_path(out) / "manifest.json"
    manifest = json.loads(component_manifest.read_text())
    mutate(manifest)
    component_manifest.write_text(json.dumps(manifest))

    with pytest.raises(HybridCompletionError, match=expected):
        _dense_completion_provenance(out)


def test_a_1_0_release_that_stores_no_frequencies_still_validates(dense_store: Path):
    """`1.0` releases stay readable (spec §21). Their `eaf` kind is
    `float32_optional`, whose whole point is that the plane may or may not be
    there -- so the plan-versus-`eaf_scope` cross-check must not read it as a
    promise that some Analysis declares `association`."""
    manifest_path = dense_store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "1.0"
    manifest["encoding"] = {
        "version": 1,
        "z": manifest["encoding"]["z"],
        "se": manifest["encoding"]["se"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = zarr.open_group(str(dense_store / "data.zarr"), mode="a")
    for name in ("eaf", EAF_BASELINE, "eaf_exception_index", "eaf_exception_value"):
        if name in root:
            del root[name]
    rows = read_analyses(dense_store / "analyses.tsv")
    write_analysis_records(
        dense_store / "analyses.tsv",
        [replace(a, eaf_scope=EafScope.ABSENT.value, eaf_orientation="")
         for a in read_analysis_records(dense_store / "analyses.tsv")],
    )
    assert rows  # the fixture had analyses to rewrite

    result = validate_store(dense_store)
    assert result.ok, result.errors


def test_completion_stamps_eaf_scope_on_an_analysis_that_gains_imputed_cells(
    tmp_path: Path,
):
    """An Analysis whose source reported no frequency stores one for its
    imputed cells once the release carries reference EAF, so its `eaf_scope`
    follows what the release holds rather than what its source reported
    (ADR 0037 §4). Both completion pipelines call one helper for this."""
    manifest_rows = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for index, analysis in enumerate(_ANALYSES):
        path = tmp_path / f"scope_{analysis}.vcf"
        lines = [_VCF_HEADER.format(sample=analysis.upper())]
        for pos in _POSITIONS:
            z = 0.5 + 0.1 * index + 0.001 * pos
            if analysis == "a4":  # reports no frequency anywhere
                lines.append(f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE\t{z * 0.3:.6f}:0.3\n")
            else:
                eaf = 0.2 + 0.01 * index
                lines.append(
                    f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t"
                    f"{z * 0.3:.6f}:0.3:{1 - eaf:.6f}\n"
                )
        path.write_text("".join(lines), encoding="utf-8")
        manifest_rows.append(
            f"{analysis}\t{path}\t{analysis}\t10000\tsd\tdeclared_standardised\thg38"
        )
    manifest = tmp_path / "scope_manifest.tsv"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    observed = tmp_path / "scope.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, observed, store_id="eafenc", release_id="v1", allow_unverified_eaf=True
    )
    before = {
        r["analysis_id"]: r["eaf_scope"] for r in read_analyses(observed / "analyses.tsv").rows
    }
    assert before["a4"] == EafScope.ABSENT.value, before

    completed = tmp_path / "scope-completed.opengwasdb"
    complete_dense_store(
        observed, completed, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    with query_store(completed) as query:
        status = query.analysis("a4")["association_status"]
    assert int((status == "imputed").sum()) > 0, (
        "a4 gained no imputed cells, so this test could not fail"
    )
    after = {
        r["analysis_id"]: r["eaf_scope"] for r in read_analyses(completed / "analyses.tsv").rows
    }
    assert after["a4"] == EafScope.ASSOCIATION.value, after
    result = validate_store(completed)
    assert result.ok, result.errors

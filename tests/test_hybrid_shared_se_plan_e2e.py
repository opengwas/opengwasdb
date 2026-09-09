"""Literal end-to-end Hybrid shared-SE-plan coverage (issue #141, AC9).

The Hybrid builder selects one shared `se` encoding for the Dense Component
and the Ragged Overflow (ADR 0037, issue #119/#141). The joint-selection
fixtures in ``test_se_residual_encoding.py`` drive ``optimise_dense_se_joint``
directly and assert the returned plan, and the codec fixtures there prove
exact exceptions round-trip at the codec level. Issue #141's acceptance
criterion 9 asks for the same behaviour *through a real build*: a supported
Hybrid builder run whose fixture really traverses both components, a
persisted shared plan, and the public query/top-hit surface answering from it
-- with validation on top.

These tests build one Hybrid store each and assert:

- a residual shared plan is genuinely reachable end to end when every finite
  cell in both components carries an EAF (the control each veto is compared
  against, which is AC9's "residual eligibility" half);
- a *single* cell whose standard error has no EAF -- on the Dense Component
  only, or on the Ragged Overflow only -- vetoes residual eligibility for the
  pair, and the shared plan that persists in *both* component manifests is
  `float16` (AC9's "each component forcing fallback");
- a residual Hybrid's exact-exception cells (a Dense flat-position one and an
  Overflow CSR-ordinal one) decode exactly through ``lookup``/``analysis`` and
  the top-hit index, under a validating store (AC9's "exact exceptions");
- a residual Hybrid's missing Dense cells are held by the reserved missing
  code, decode as NaN, stay absent from the public query surface, and never
  become an exact exception (AC9's "missingness").

The imputed-Dense-cell half of AC9 is exercised end to end in
``test_hybrid_completion.py``
(``test_residual_hybrid_crossover_rebuilds_index_with_completed_encoding``):
completion, the imputed plane, the public query and the rebuilt index under
one shared residual plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import residual_fixtures as fixtures

from opengwasdb.encoding.plan import SE_EXCEPTION, SE_MISSING, SeEncoding
from opengwasdb.encoding.planes import DenseSePlane, DenseZPlane
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import VariantAxis

#: Panel variants (Dense Component rows) are 1:1000 .. 1:400000; the Ragged
#: Overflow starts at 1:500000 so the two position spans can never collide no
#: matter how many rows a fixture uses (a collision silently shrinks one
#: component and turns a "traverses both components" fixture into a lie).
OFF_BASE = 500_000
N_PANEL = 400
N_OFF_PANEL = 400


def panel_alid(row: int) -> str:
    return f"1:{(row + 1) * 1000}:A:G"


def overflow_alid(row: int) -> str:
    return f"1:{OFF_BASE + row * 1000}:A:G"


def _se_value(column: int, freq: float, phase: int) -> float:
    """SE that tracks ``log(se) ~ a + b * log(2f(1-f))`` per Analysis closely.

    The same deterministic model the decision/codec fixtures use, with a small
    per-cell wobble so the fit is not degenerate: every EAF-bearing cell lands
    inside the ±0.5 candidate, so a Hybrid build that measures this data end
    to end selects the residual coding (a float16 outcome on this data is
    therefore always caused by a veto, never by an uncodeable fixture).
    """
    return float(
        np.exp(
            (-3.0 + column * 0.2)
            - 0.5 * np.log(2.0 * freq * (1.0 - freq))
            + 0.12 * np.sin(phase * (0.07 + column * 0.01))
        )
    )


def _analysis_rows(
    column: int,
    *,
    drop_eaf_at: str | None = None,
    exact_exception_at: tuple[str, ...] = (),
    missing_dense_from: int | None = None,
    exception_factor: float = 50.0,
) -> list[str]:
    """One Analysis's VCF rows: every panel variant, then every overflow one.

    ``drop_eaf_at`` names a single ALID (on the calling component) whose row
    is written without the AF field: a finite standard error with no EAF,
    which is the one cell that vetoes residual eligibility for the whole pair
    (issue #141 AC2/AC3). ``exact_exception_at`` names ALIDs whose SE is
    scaled by ``exception_factor``, pushing the residual far beyond every
    candidate range so each must be stored exactly. ``missing_dense_from``
    leaves the tail panel variants unobserved, so their Dense cells stay
    missing. Rows are emitted in position order.
    """
    frequencies = np.linspace(0.05, 0.95, N_PANEL, dtype=np.float64)
    off_frequencies = np.linspace(0.10, 0.90, N_OFF_PANEL, dtype=np.float64)
    rows: list[str] = []
    for row in range(N_PANEL):
        if missing_dense_from is not None and row >= missing_dense_from:
            continue
        rows.append(_row((row + 1) * 1000, frequencies[row], row, column,
                         drop_eaf=(panel_alid(row) == drop_eaf_at),
                         scale=exception_factor if panel_alid(row) in exact_exception_at else 1.0))
    for row in range(N_OFF_PANEL):
        in_exceptions = overflow_alid(row) in exact_exception_at
        rows.append(_row(OFF_BASE + row * 1000, off_frequencies[row], row, column,
                         drop_eaf=(overflow_alid(row) == drop_eaf_at),
                         scale=exception_factor if in_exceptions else 1.0))
    return rows


def _row(
    position: int, freq: float, phase: int, column: int, *, drop_eaf: bool, scale: float
) -> str:
    """One ``1:POS A G`` hg38 row; REF=A is the stored effect allele, so the
    z the store reports is -ES/SE (the two cancel in every assertion below,
    which is about `se`)."""
    se = _se_value(column, freq, phase) * scale
    # Row 0 and every 50th row carries |z| = 8 (a p < 5e-8 top hit), which is
    # what lets the exception rows appear in the top-hit index; the rest are
    # filler that keeps the dense plane's byte gates honest.
    z = 8.0 if phase % 50 == 0 else 1.0
    effect = f"{z * se:.6f}:{se:.6f}"
    if drop_eaf:
        return f"1\t{position}\t.\tA\tG\t.\tPASS\t.\tES:SE\t{effect}\n"
    return f"1\t{position}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t{effect}:{freq:.6f}\n"


def _build_hybrid(tmp_path: Path, label: str, analyses: list[dict]) -> Path:
    """A real Hybrid build (hg38 rows, no liftover) from per-Analysis configs.

    Each entry in ``analyses`` is the keyword dict for ``_analysis_rows`` of
    one Analysis, named ``trait_{column}`` in manifest order. Returns the
    store path; the caller reads the persisted plan, query results and
    validation from the public API.
    """
    manifest_rows: list[str] = []
    for column, config in enumerate(analyses):
        vcf = fixtures.write_gwas_vcf_with_eaf(
            tmp_path / f"{label}_trait_{column}.vcf", _analysis_rows(column, **config)
        )
        manifest_rows.append(
            f"trait_{column}\t{vcf}\tTrait {column}\t1000\tsd\tdeclared_standardised\thg38"
        )
    manifest = tmp_path / f"{label}.manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\tsource_assembly\n" + "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )
    panel = tmp_path / f"{label}.panel.txt"
    panel.write_text("\n".join(panel_alid(row) for row in range(N_PANEL)) + "\n",
                     encoding="utf-8")
    store = tmp_path / f"{label}.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r"
    )
    return store


def _assert_spans_both_components(store: Path, encoding: StoreManifest) -> None:
    """The fixture genuinely holds associations in the Dense Component and the
    Ragged Overflow -- without which any shared-plan assertion is vacuous."""
    hybrid = encoding.provenance["hybrid"]
    assert hybrid["n_panel"] == N_PANEL
    assert hybrid["n_off_panel"] == N_OFF_PANEL
    # Both Analyses observe every variant, so the Overflow CSR holds 2 x 400.
    assert hybrid["n_overflow_associations"] == 2 * N_OFF_PANEL
    with query_store(store) as query:
        dense = query.lookup([panel_alid(50)], ["trait_0"])
        overflow = query.lookup([overflow_alid(50)], ["trait_0"])
        assert len(dense["z"]) == 1 and np.isfinite(dense["se"]).all()
        assert len(overflow["z"]) == 1 and np.isfinite(overflow["se"]).all()


def _dense_row_by_alid(store: Path) -> dict[str, int]:
    """Panel ALID -> row index on the Dense Component's own variant axis."""
    axis = VariantAxis(store / "dense")
    rows = {record.alid: int(record.variant_index) for record in axis.all()}
    axis.close()
    return rows


# -- one component alone vetoes residual eligibility --------------------------


@pytest.mark.parametrize(
    ("drop_eaf_dense", "veto_alid", "codeable_alid"),
    [
        # trait_0's Dense row at 1:50000 (row 49) has no EAF; its Overflow
        # cell at 1:550000 does. The Ragged Overflow is entirely codeable.
        (True, panel_alid(49), overflow_alid(50)),
        # trait_0's Overflow row at 1:550000 (row 50) has no EAF; the Dense
        # Component is entirely codeable.
        (False, overflow_alid(50), panel_alid(49)),
    ],
    ids=["dense-veto", "overflow-veto"],
)
def test_one_component_alone_vetoes_residual_and_both_persist_float16(
    tmp_path: Path, drop_eaf_dense: bool, veto_alid: str, codeable_alid: str
) -> None:
    """Issue #141 AC3: failure of either component's eligibility gate persists
    float16 for *both* components.

    The Dense Component and the Ragged Overflow partition one Analysis's
    associations, so a fallback that rewrote only the vetoing component would
    leave the two storing different encodings under one shared manifest. The
    fixture's control -- the same data with every finite cell carrying an EAF
    -- selects the residual coding, so the float16 outcome here is caused by
    the single no-EAF cell, and the veto sits on one component while the other
    is entirely codeable (the "either component alone" case AC9 names).
    """
    veto_config = {"drop_eaf_at": veto_alid}
    # The two fixtures are identical except for the vetoed cell's AF field.
    control = _build_hybrid(tmp_path, "control", [dict(), dict()])
    veto = _build_hybrid(tmp_path, "veto", [veto_config, dict()])

    # AC9 "residual eligibility": the control reaches the residual plan end to
    # end, and its two components persist one shared plan. Without this, the
    # veto store's float16 assertion could hold of a fixture that could never
    # code residually at all.
    control_encoding = StoreManifest.load(control)
    assert control_encoding.encoding.se.is_residual
    assert control_encoding.encoding.se == StoreManifest.load(control / "dense").encoding.se
    assert validate_store(control).ok
    _assert_spans_both_components(control, control_encoding)

    veto_encoding = StoreManifest.load(veto)
    dense_encoding = StoreManifest.load(veto / "dense")
    # The *shared* persisted plan: both component manifests agree, and both
    # physical planes are the float16 their manifests declare.
    assert veto_encoding.encoding.se == SeEncoding("float16")
    assert dense_encoding.encoding.se == SeEncoding("float16")
    assert dense_encoding.encoding.se == veto_encoding.encoding.se
    assert _dense_group(veto)["se"].dtype == np.dtype("float16")
    assert open_store(veto).arrays(mode="r")["ragged"]["se"].dtype == np.dtype("float16")
    assert validate_store(veto).ok
    _assert_spans_both_components(veto, veto_encoding)

    # The veto cell must really be stored on its component with a finite
    # standard error and no EAF: if the build had silently dropped the row,
    # the float16 outcome would not be an eligibility veto at all. The other
    # component's cell at the parallel variant is codeable, so the
    # ineligibility is confined to the vetoing component.
    with query_store(veto) as query:
        vetoed = query.lookup([veto_alid], ["trait_0"])
        assert len(vetoed["z"]) == 1
        assert np.isfinite(vetoed["se"]).all()
        assert np.isnan(vetoed["eaf"]).all()
        codeable = query.lookup([codeable_alid], ["trait_0"])
        assert len(codeable["z"]) == 1
        assert np.isfinite(codeable["se"]).all()
        assert np.isfinite(codeable["eaf"]).all()


def _dense_group(store: Path):
    return open_store(store).dense_component().arrays(mode="r")


# -- exact exceptions reach the public query and top-hit surface -------------


def test_residual_hybrid_exact_exception_cells_decode_exactly(tmp_path: Path) -> None:
    """AC9 "exact exceptions", end to end.

    The Dense Component writes its exception at a flat grid position and the
    Overflow writes its at a CSR ordinal, and the public query surface decodes
    the exact float32 the persisted side table holds -- never a clipped or
    quantised stand-in. The raw int8 plane at each exception cell holds the
    reserved ``SE_EXCEPTION`` code, so the decoded values come from the
    exception tables; validation confirms tables, planes and top-hit index
    agree.
    """
    dense_exception = panel_alid(50)        # 1:51000, |z|=8 -> a Dense top hit
    overflow_exception = overflow_alid(50)  # 1:550000, |z|=8 -> an Overflow top hit
    store = _build_hybrid(
        tmp_path,
        "exceptions",
        [
            {"exact_exception_at": (dense_exception, overflow_exception)},
            dict(),
        ],
    )
    encoding = StoreManifest.load(store)
    assert encoding.encoding.se.is_residual
    assert encoding.encoding.se == StoreManifest.load(store / "dense").encoding.se
    assert validate_store(store).ok
    _assert_spans_both_components(store, encoding)

    # Dense side: the exception sits at flat position row*width + column
    # (row 50, column 0 -> flat 100), its code is SE_EXCEPTION, and decoding
    # the plane returns the exact float32 the side table holds.
    dense_row = _dense_row_by_alid(store)[dense_exception]
    dense_group = _dense_group(store)
    n_analyses = int(dense_group["se"].shape[1])
    assert int(np.asarray(dense_group["se"][dense_row, 0])) == SE_EXCEPTION
    assert np.asarray(dense_group["se_exception_index"][:]).tolist() == [
        dense_row * n_analyses + 0
    ]
    exact_dense = float(np.asarray(dense_group["se_exception_value"][:])[0])
    decoded_dense = DenseSePlane.open(dense_group, encoding.encoding).band(
        dense_row, dense_row + 1
    )[0, 0]
    assert decoded_dense == exact_dense

    # Overflow side: the exception index is the CSR ordinal 50, which lies in
    # trait_0's segment (offsets 0..400); the plane there holds SE_EXCEPTION.
    ragged_group = open_store(store).arrays(mode="r")["ragged"]
    offsets = np.asarray(ragged_group["offsets"][:])
    assert offsets[0] == 0 and offsets[1] == N_OFF_PANEL
    assert np.asarray(ragged_group["se_exception_index"][:]).tolist() == [50]
    assert int(np.asarray(ragged_group["se"][:])[50]) == SE_EXCEPTION
    exact_overflow = float(np.asarray(ragged_group["se_exception_value"][:])[0])

    with query_store(store) as query:
        # Public lookups and the whole-Analysis result decode the exact table
        # values from both components.
        for alid, exact in (
            (dense_exception, exact_dense),
            (overflow_exception, exact_overflow),
        ):
            cell = query.lookup([alid], ["trait_0"])
            assert len(cell["z"]) == 1
            assert cell["se"][0] == exact, (
                f"{alid}: lookup returned {cell['se'][0]!r}; the side table holds {exact!r}"
            )
        analysis = query.analysis("trait_0")
        assert len(analysis["z"]) == N_PANEL + N_OFF_PANEL
        assert np.isfinite(analysis["se"]).all()

        # The top-hit index carries the plane's decoded exact values (ADR
        # 0040), so a hit that was stored exactly reads back exactly.
        hits = query.top_hits(threshold=5e-8, analysis_id="trait_0")
        variants = query.variants_table()
        hit_alids = {variants[int(vi)]["alid"] for vi in hits["variant_index"]}
        assert dense_exception in hit_alids
        assert overflow_exception in hit_alids
        for alid, exact in (
            (dense_exception, exact_dense),
            (overflow_exception, exact_overflow),
        ):
            mask = np.array(
                [variants[int(vi)]["alid"] == alid for vi in hits["variant_index"]]
            )
            assert mask.any()
            assert float(hits["se"][mask][0]) == exact


# -- missing cells stay missing under a residual plan -------------------------


def test_residual_hybrid_missing_cells_stay_missing(tmp_path: Path) -> None:
    """AC9 "missingness", end to end.

    A Dense cell an Analysis never observed is stored under the reserved
    ``SE_MISSING`` code -- not a residual of zero and not an exact exception
    -- decodes to NaN, and stays absent from the public query surface under a
    validating store.
    """
    missing_dense_from = N_PANEL - 20  # trait_1 leaves the last 20 panel rows unobserved
    observed_alid = panel_alid(missing_dense_from - 1)
    missing_alid = panel_alid(missing_dense_from)
    store = _build_hybrid(
        tmp_path, "missing", [dict(), {"missing_dense_from": missing_dense_from}]
    )
    encoding = StoreManifest.load(store)
    assert encoding.encoding.se.is_residual
    assert encoding.encoding.se == StoreManifest.load(store / "dense").encoding.se
    assert validate_store(store).ok
    _assert_spans_both_components(store, encoding)

    dense_group = _dense_group(store)
    raw = np.asarray(dense_group["se"])
    row = _dense_row_by_alid(store)[missing_alid]
    # trait_1's unobserved tail (column 1) is held by the missing code while
    # trait_0's cells at the same rows (column 0) are ordinary codes: a plane
    # that collapsed missingness into zero residuals would decode a made-up
    # tiny SE here.
    assert int(raw[row, 1]) == SE_MISSING
    assert int(raw[row, 0]) != SE_MISSING

    # A missing cell is not an exact exception: no exception-table entry may
    # point at its flat position.
    missing_flats = [r * int(raw.shape[1]) + 1 for r in range(missing_dense_from, N_PANEL)]
    exceptions = np.asarray(dense_group["se_exception_index"][:]).tolist()
    assert not (set(missing_flats) & set(exceptions))

    # Decode agrees: the missing cells are NaN in both paired planes, and the
    # observed cells around them are untouched.
    n_rows = int(raw.shape[0])
    decoded = DenseSePlane.open(dense_group, encoding.encoding).band(0, n_rows)
    zdecoded = DenseZPlane.open(dense_group, encoding.encoding).band(0, n_rows)
    assert np.isnan(decoded[missing_dense_from:, 1]).all()
    assert np.isnan(zdecoded[missing_dense_from:, 1]).all()
    assert np.isfinite(decoded[missing_dense_from:, 0]).all()

    with query_store(store) as query:
        # The public surface reports absence, never a fabricated number.
        assert query.lookup([missing_alid], ["trait_1"])["z"].size == 0
        observed = query.lookup([observed_alid], ["trait_1"])
        assert len(observed["z"]) == 1 and np.isfinite(observed["se"]).all()
        present = query.lookup([missing_alid], ["trait_0"])
        assert len(present["z"]) == 1 and np.isfinite(present["se"]).all()
        analysis = query.analysis("trait_1")
        assert len(analysis["z"]) == (N_PANEL - 20) + N_OFF_PANEL
        assert np.isfinite(analysis["se"]).all()
        # No top hit can carry a missing cell's fabricated SE.
        hits = query.top_hits(threshold=5e-8)
        assert np.isfinite(hits["se"]).all()

"""Every writer holds what `storable` returns, not the cells it was handed (#350).

Today's gate stores every cell unchanged, so a writer that ignored its result
would write the same bytes and nothing would show it. Issue #338 changes only
the gate, so each writer must already honour a result that leaves an
association out: here a gate is injected that refuses one chosen association,
blanking its cells as a refusal does, and the Store must hold neither the
association nor its frequency (#350 D5).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import test_dense_completion as tdc
import test_hybrid_build as hb
import test_hybrid_completion as thc
import test_ragged_completion as trc
import zarr
from dense_sources import build_dense_fixture
from stored_rows import stored_rows
from test_dense_vcf_build import HG19_POS_1, HG19_POS_3, _make_manifest, _make_vcf

from opengwasdb.encoding import (
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    EafEncoding,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
    association_codec,
)
from opengwasdb.encoding.association_codec import CellValues, Storable
from opengwasdb.layouts.dense import build_vcf
from opengwasdb.layouts.dense import complete as dense_complete
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged import zarr_csr
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRWriter
from opengwasdb.store.open import open_store
from opengwasdb.variants import VariantAxis

#: The SE of the association the injected gate refuses, in fixtures that use it.
REFUSED_SE = 0.4375

Refusal = Callable[[CellValues, str], np.ndarray]


def _se_is(value: float) -> Refusal:
    def refused(cells: CellValues, where: str) -> np.ndarray:
        return np.abs(np.asarray(cells.se, dtype=np.float64) - value) < 1e-3 * value

    return refused


def _in_fold(refused: Refusal) -> Refusal:
    def in_fold(cells: CellValues, where: str) -> np.ndarray:
        return refused(cells, where) & ("crossover fold" in where)

    return in_fold


def _inject(
    monkeypatch: pytest.MonkeyPatch, refused: Refusal, *modules: object
) -> list[tuple[str, CellValues]]:
    """Make `storable`, as each module calls it, refuse the cells `refused` names.

    Returns the batches the gate judges, as `(where, cells)`, in call order.
    """
    real = association_codec.storable
    judged: list[tuple[str, CellValues]] = []

    def gate(cells: CellValues, analyses, n_analyses: int, *, where: str) -> Storable:
        judged.append((where, cells))
        kept = real(cells, analyses, n_analyses, where=where)
        out = refused(cells, where)

        def blanked(values: np.ndarray, fill: float) -> np.ndarray:
            copy = np.array(values, copy=True)
            copy[out] = fill
            return copy

        cells_kept = CellValues(
            blanked(cells.z, np.nan),
            blanked(cells.se, np.nan),
            None if cells.eaf is None else blanked(cells.eaf, np.nan),
            None if cells.imputed is None else blanked(cells.imputed, 0),
        )
        return Storable(cells_kept, kept.stored & ~out, kept.se_out_of_range)

    for module in modules:
        monkeypatch.setattr(module, "storable", gate, raising=False)
    return judged


def _judged_at(judged: list[tuple[str, CellValues]], where: str, se: float) -> list[float]:
    """The frequencies the gate was handed, under `where`, with each cell whose
    z is present and whose SE is `se`."""
    out: list[float] = []
    for at, cells in judged:
        if where not in at or cells.eaf is None:
            continue
        hit = np.isfinite(cells.z) & np.asarray(_se_is(se)(cells, at))
        out.extend(np.asarray(cells.eaf, dtype=np.float64)[hit].tolist())
    return out


def _dense_cell(opened, alid: str, analysis: int) -> tuple[float, float, float, int]:
    """`(z, se, eaf, imputed)` of one cell of a Dense grid (a Store or a Component)."""
    root, encoding = opened.arrays(mode="r"), opened.manifest.encoding
    axis = VariantAxis(opened.path)
    try:
        record = axis.by_identifier(alid)
    finally:
        axis.close()
    assert record is not None, f"fixture must hold {alid}"
    rows, cols = np.array([record.variant_index]), np.array([analysis])
    imputed = int(root["imputed"][int(rows[0]), analysis]) if "imputed" in root else 0
    eaf = DenseEafPlane.open(root, encoding).points(rows, cols)[0] if "eaf" in root else np.nan
    return (
        float(DenseZPlane.open(root, encoding).points(rows, cols)[0]),
        float(DenseSePlane.open(root, encoding).points(rows, cols)[0]),
        float(eaf),
        imputed,
    )


def _absent(cell: tuple[float, float, float, int]) -> bool:
    z, se, eaf, imputed = cell
    return bool(np.isnan(z) and np.isnan(se) and np.isnan(eaf) and imputed == 0)


# ── the Ragged writer ─────────────────────────────────────────────────────────


def test_the_ragged_writer_keeps_only_the_rows_it_stores(tmp_path, monkeypatch) -> None:
    _inject(monkeypatch, _se_is(REFUSED_SE), zarr_csr)
    writer = RaggedCSRWriter(6)
    writer.add_analysis(
        np.array([0, 2, 5]),
        np.array([1.0, 2.0, 3.0], np.float32),
        np.array([0.1, REFUSED_SE, 0.2], np.float32),
        eaf=np.array([0.3, 0.4, 0.5], np.float32),
    )
    plan = StoreEncoding(
        z=ZEncoding("int16_fixed", scale=1024), se=SeEncoding("float16"), eaf=EafEncoding("float32")
    )
    writer.flush(tmp_path, plan)

    group = zarr.open_group(str(tmp_path / RAGGED_ZARR_PATH), mode="r")
    assert group["variant_index"][:].tolist() == [0, 5]
    assert group["eaf"][:].tolist() == pytest.approx([0.3, 0.5])


# ── the Dense VCF and Hybrid spills ───────────────────────────────────────────


def test_the_dense_vcf_spill_keeps_only_the_rows_it_stores(tmp_path, monkeypatch) -> None:
    _inject(monkeypatch, _se_is(REFUSED_SE), build_vcf)
    vcf_a = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:{REFUSED_SE}:0.2\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
        ],
    )
    vcf_b = _make_vcf(
        tmp_path, "trait_b", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n"]
    )
    manifest = _make_manifest(tmp_path, [("trait_a", vcf_a, "A"), ("trait_b", vcf_b, "B")])
    store = tmp_path / "dense.opengwasdb"
    build_vcf.build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    stored = _dense_cell(open_store(store), "1:100000:A:G", 1)
    assert not _absent(stored), "fixture: trait_b's cell is stored"
    assert _absent(_dense_cell(open_store(store), "1:100000:A:G", 0))


def test_the_hybrid_spill_keeps_only_the_rows_it_stores(tmp_path, monkeypatch) -> None:
    """trait_a's refused SE sits on the panel (Dense) and off it (Overflow)."""
    _inject(monkeypatch, _se_is(REFUSED_SE), hybrid_build)
    vcf_a = hb._make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{hb.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:{REFUSED_SE}:0.2\n",
            f"1\t{hb.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:{REFUSED_SE}:0.3\n",
            f"1\t{hb.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
        ],
    )
    vcf_b = hb._make_vcf(
        tmp_path, "trait_b", [f"1\t{hb.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.0:0.5:0.25\n"]
    )
    manifest = hb._make_manifest(tmp_path, [("trait_a", vcf_a, "A"), ("trait_b", vcf_b, "B")])
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=hb._panel(tmp_path), store_id="s", release_id="r",
        n_workers=1,
    )

    trait_a, trait_b = stored_rows(store, 0), stored_rows(store, 1)
    assert len(trait_b.variant_index) == 1, "fixture: trait_b's off-panel row is stored"
    assert len(trait_a.variant_index) == 0
    assert _absent(_dense_cell(open_store(store).dense_component(), "1:100000:A:G", 0))


# ── the completions ───────────────────────────────────────────────────────────


def test_dense_completion_writes_only_what_it_stores(tmp_path, monkeypatch) -> None:
    """a1's observed cell at 1:1000000 is refused: missing, unflagged, no frequency."""
    rows = [row.split("\t") for row in tdc.SOURCE_ROWS]
    rows[2][9] = str(REFUSED_SE)  # a1 at 1:1000000
    source = tmp_path / "associations.tsv"
    source.write_text(
        tdc.SOURCE_HEADER + "\teaf\n" + "\n".join("\t".join(r) + "\t0.3" for r in rows) + "\n",
        encoding="utf-8",
    )
    observed = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source], observed, store_id="t", release_id="obs", reference_assembly="GRCh38"
    )
    assert "eaf" in open_store(observed).arrays(mode="r"), "fixture carries frequencies"
    _inject(monkeypatch, _se_is(REFUSED_SE), dense_complete)
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(observed, completed, tdc._make_ld_panel(tmp_path), min_cor=0.0)

    stored = _dense_cell(open_store(completed), "1:900000:A:G", 0)
    assert not _absent(stored), "fixture: a1's other cell is stored"
    assert _absent(_dense_cell(open_store(completed), "1:1000000:A:G", 0))


@pytest.mark.parametrize(
    ("first_probe", "se_kind"),
    [([(0, 0.1, 0.02)], "float16"), ([(0, 0.1, 0.02), (1, -0.2, 0.03)], "int8_residual")],
    ids=["float16-se", "residual-se"],
)
def test_ragged_completion_keeps_only_the_rows_it_stores(
    tmp_path, monkeypatch, first_probe: list[tuple], se_kind: str
) -> None:
    """The second probe's observed association at rs1002 (SE 0.05) is refused;
    its others, at rs1001 and rs1003, are the controls.

    A first probe with one SE leaves the build no residual SE model, so it
    stores `float16` SEs; with two, the build fits one, and completion codes
    the rows it stores against it (issue #301).
    """
    prefix = trc._make_besd_fixture(
        tmp_path,
        freqs=(0.3, 0.45, 0.2),
        probe_assocs=[first_probe, [(0, 0.3, 0.04), (1, 0.5, 0.05), (2, -0.15, 0.025)]],
    )
    observed = tmp_path / "obs.opengwasdb"
    trc.build_ragged_from_besd(
        prefix, observed, store_id="t", release_id="obs", tissue="Blood",
        association_coverage="cis", store_esi_eaf=True, allow_unverified_eaf=True,
    )
    assert open_store(observed).manifest.encoding.se.kind == se_kind, "fixture: the SE plan"
    probe_observed = stored_rows(observed, 1)
    assert len(probe_observed.variant_index) == 3, "fixture: the probe observes three variants"
    assert np.isfinite(probe_observed.eaf).all(), "fixture: each carries a frequency"
    refused = np.isclose(probe_observed.se, 0.05, rtol=1e-3)
    observed_alid = VariantAxis(observed).by_indices(probe_observed.variant_index.tolist())
    alid_of = {int(v): observed_alid[int(v)].alid for v in probe_observed.variant_index}

    # Completion's rows are judged by the writer it adds them to (#424).
    judged = _inject(monkeypatch, _se_is(0.05), zarr_csr, association_codec)
    completed = tmp_path / "comp.opengwasdb"
    complete_ragged_store(
        observed, completed, trc._make_ld_panel(tmp_path, "1", 900_000, 1_300_000),
        ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp",
    )

    seen = _judged_at(judged, "complete-ragged", 0.05)
    assert seen and np.isfinite(seen).all(), "the gate judged rs1002, present, with its frequency"
    probe = stored_rows(completed, 1)
    completed_alid = VariantAxis(completed).by_indices(probe.variant_index.tolist())
    held = {
        completed_alid[int(v)].alid: eaf
        for v, eaf in zip(probe.variant_index, probe.eaf.tolist(), strict=True)
    }
    kept = ~refused
    for v, eaf in zip(probe_observed.variant_index[kept], probe_observed.eaf[kept], strict=True):
        assert held[alid_of[int(v)]] == pytest.approx(float(eaf))
    assert alid_of[int(probe_observed.variant_index[refused][0])] not in held


def test_build_dense_writes_only_what_it_stores(tmp_path, monkeypatch) -> None:
    """`build-dense` writes the codec's cells, so it held this before #350's fix."""
    rows = [row.split("\t") for row in tdc.SOURCE_ROWS]
    rows[2][9] = str(REFUSED_SE)
    source = tmp_path / "associations.tsv"
    source.write_text(
        tdc.SOURCE_HEADER + "\teaf\n" + "\n".join("\t".join(r) + "\t0.3" for r in rows) + "\n",
        encoding="utf-8",
    )
    _inject(monkeypatch, _se_is(REFUSED_SE), association_codec)
    store = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source], store, store_id="t", release_id="obs", reference_assembly="GRCh38"
    )

    stored = _dense_cell(open_store(store), "1:900000:A:G", 0)
    assert not _absent(stored), "fixture: a1's other cell is stored"
    assert _absent(_dense_cell(open_store(store), "1:1000000:A:G", 0))


def _crossover_source(tmp_path) -> Path:
    """`thc`'s Hybrid source with frequencies, and an off-panel observation of
    1:2000000 in both Analyses: trait_a's (SE `REFUSED_SE`) and trait_c's."""
    return thc.build_two_trait_source(
        tmp_path,
        {
            "trait_a": [
                (100_000, "2.0:0.5:0.2"),
                (750_000, "1.0:0.4:0.3"),
                (1_000_000, "1.5:0.3:0.4"),
                (1_500_000, "0.6:0.2:0.5"),
                (2_000_000, f"1.2:{REFUSED_SE}:0.2"),
            ],
            "trait_c": [
                (100_000, "2.2:0.5:0.2"),
                (750_000, "1.1:0.4:0.3"),
                (1_000_000, "1.6:0.3:0.4"),
                (2_000_000, "0.9:0.3:0.25"),
            ],
        },
        fields="ES:SE:AF",
    )


def test_the_crossover_fold_writes_only_what_it_stores(tmp_path, monkeypatch) -> None:
    """The LD panel adds 1:2068561 (hg19 1:2000000) to the Dense axis, so the fold
    moves both Analyses' observations there. trait_a's is refused; trait_c's is
    the control."""
    src = _crossover_source(tmp_path)
    judged = _inject(monkeypatch, _in_fold(_se_is(REFUSED_SE)), association_codec)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, thc._make_ld_panel_with_crossover(tmp_path), min_cor=0.0)

    seen = _judged_at(judged, "crossover fold", REFUSED_SE)
    assert seen and np.isfinite(seen).all(), "the fold judged trait_a's, with its frequency"
    dense = open_store(dst).dense_component()
    z, _se, eaf, imputed = _dense_cell(dense, thc.OFF_PANEL_ALID, 1)
    # ALT G sorts after REF A, so the Store's effect allele is A: z -0.9/0.3, EAF 1 - 0.25.
    assert (z, eaf, imputed) == (pytest.approx(-3.0, rel=5e-3), pytest.approx(0.75, abs=1e-3), 0)
    assert _absent(_dense_cell(dense, thc.OFF_PANEL_ALID, 0))

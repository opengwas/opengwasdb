"""ogld#5's contract: the builder, on the Test Panel, against the independent reference r.

The Test Panel's `reference_r(k)` is a short numpy computation in lane 2d's script, checked
against plink2's `--r-unphased` at k = 0 by tests/test_testdata.py. The builder must match it
within the quantisation step: in the Near Band pair by pair, and in which pairs beyond 3 Mb
it keeps, except where a pair's code lies on the far code floor's rounding edge.

The builder also refuses an input with a missing call, and its PC Check drops a local PC,
keeps one the rest of the genome explains, and uses the first up to 10 PCs kept, refusing
with fewer than 5 (owner, 7 November 2026; ADR 0002).

Every test calls `ogld.build` before it reads the Test Panel: a build is given `PANEL`'s
paths, which need no loader, and each k is built once per worker (`built`). The tests that
make their own pgen or PC File read and write pgen with pgenlib, which ogld#5 adds, after
that first build.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import support
from support import (
    PANEL,
    alids,
    check_panel_paths,
    contract,
    panel_data,
    passing_rows,
    reference,
    reference_matrix,
)

import ogld
from ogld import cli
from ogld.format import far_code_floor, t_far

OGLD_5 = contract("ogld#5")
PANEL_ID = "test-panel-hgdp1kgp-EUR"
_BUILT: dict[int, tuple[ogld.Provenance, Path]] = {}


def inputs(k: int = support.K, **overrides: Any) -> ogld.BuildInputs:
    """The Test Panel as BuildInputs, from paths alone: nothing here reads a file."""
    fields: dict[str, Any] = {
        "pgen": (PANEL.pgen,),
        "keep": PANEL.keep,
        "pcs": PANEL.pcs if k else None,
        "pc_source": "the Test Panel's PC File (tests/data/test-panel/README.md)" if k else None,
        "panel_id": PANEL_ID,
        "ancestry": "EUR",
        "genotype_source": "HGDP+1kGP phased v2, the Test Panel's slice",
        "k": k,
        "info_min": None,
    }
    return ogld.BuildInputs(**{**fields, **overrides})


def built(k: int, tmp_path_factory: pytest.TempPathFactory) -> tuple[ogld.Provenance, Path]:
    """The Test Panel built at k, once per worker; the first call is where a test stops
    until ogld#5 lands."""
    if k not in _BUILT:
        out = tmp_path_factory.mktemp(f"built-k{k}") / "store"
        _BUILT[k] = (ogld.build(inputs(k), out), out)
        check_panel_paths()
    return _BUILT[k]


def near_band_matches(store: ogld.LDStore, k: int) -> int:
    """Assert every reference pair within 3 Mb reads back within half a step; return how many."""
    rows = tuple(int(row) for row in passing_rows())
    m = store.matrix(alids(rows))
    ref = reference_matrix(k, rows)
    v = panel_data().variants()
    pos, chrom = v.pos[list(rows)].astype(np.int64), v.chrom[list(rows)]
    near = (np.abs(pos[:, None] - pos[None, :]) <= 3_000_000) & (chrom[:, None] == chrom[None, :])
    assert not np.isnan(ref[near]).any(), "the reference holds every pair within 3 Mb"
    worst = np.abs(m.r[near] - ref[near]).max()
    assert worst <= support.HALF_STEP + support.SLACK, f"a Near Band pair is {worst} off"
    return int(near.sum())


@OGLD_5
def test_the_near_band_at_k10_matches_the_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _, out = built(10, tmp_path_factory)
    assert near_band_matches(ogld.open(out), 10) > 1000


@OGLD_5
def test_the_near_band_at_k0_matches_the_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # The reference at k = 0 is held to plink2 --r-unphased by tests/test_testdata.py.
    _, out = built(0, tmp_path_factory)
    assert near_band_matches(ogld.open(out), 0) > 1000


@OGLD_5
def test_the_far_pairs_kept_are_the_reference_s(tmp_path_factory: pytest.TempPathFactory) -> None:
    _, out = built(10, tmp_path_factory)
    ref = reference(10)
    far = ref.where((ref.distance > 3_000_000) & (ref.distance <= 10_000_000), "3–10 Mb pairs")
    rows = tuple(int(row) for row in passing_rows())
    m = ogld.open(out).matrix(alids(rows))
    where = np.full(int(max(rows)) + 1, -1)
    where[list(rows)] = np.arange(len(rows))
    kept = ~np.isnan(m.r[where[far.row_i], where[far.row_j]])
    expected = np.abs(far.code) >= support.FLOOR
    assert expected.any(), "the Test Panel must hold real Far Pairs at k = 10"
    assert (~expected).any(), "the Test Panel must hold 3–10 Mb pairs below t_far"
    # Only a pair whose r × 32767 lies within float error of the rounding edge may differ.
    edge = np.abs(np.abs(far.r) * 32767 - (support.FLOOR - 0.5)) < 1e-6
    assert ((kept == expected) | edge).all(), "a Far Pair was kept or dropped against the reference"


@OGLD_5
def test_the_provenance_records_what_was_built(tmp_path_factory: pytest.TempPathFactory) -> None:
    provenance, _ = built(10, tmp_path_factory)
    assert provenance["panel"]["n"] == support.N_SAMPLES == panel_data().panel().n_samples
    adjustment = provenance["adjustment"]
    assert (adjustment["k_target"], adjustment["k_min"], adjustment["k"]) == (10, 5, 10)
    pc_file = adjustment["pc_file"]
    assert pc_file is not None
    assert pc_file["sha256"] == hashlib.sha256(PANEL.pcs.read_bytes()).hexdigest()
    assert isinstance(pc_file["source"], str)
    assert pc_file["source"]
    check = adjustment["pc_check"]
    assert check is not None
    names, _, _ = read_pcs(PANEL.pcs)
    assert [scan["pc"] for scan in check["scan"]] == names
    kept = [name for name in names if name not in check["dropped"]]
    assert adjustment["covariates"] == kept[:10]
    # The reference r at k = 10 regresses out PC1–PC10, so it is the builder's only if the
    # PC Check keeps all ten on the Test Panel. If it does not, the reference must change.
    assert adjustment["covariates"] == [f"PC{i}" for i in range(1, 11)]
    assert provenance["t_far"] == support.T_FAR == t_far(771, 10)
    assert provenance["far_code_floor"] == far_code_floor(support.T_FAR)
    assert provenance["source"]["kind"] == "genotypes"
    assert provenance["variant_filters"]["mac_min"] == 20
    assert provenance["n_variants"] == len(passing_rows())
    assert "zero_variance" in provenance["variants_dropped"]
    assert provenance["counts"]["index_entries"] >= 2 * provenance["counts"]["far_pairs"]


@OGLD_5
def test_the_provenance_records_the_diagnostic_and_the_cost(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    provenance, _ = built(10, tmp_path_factory)
    diagnostic = provenance["pc_diagnostic"]
    assert diagnostic is not None
    rates = [
        diagnostic["rate_r2_ge_0_01_unadjusted"],
        diagnostic["rate_r2_ge_0_01_adjusted"],
        diagnostic["rate_r2_ge_0_01_sampling"],
    ]
    assert all(isinstance(rate, float) and 0.0 <= rate <= 1.0 for rate in rates)
    assert diagnostic["pairs"]["n_pairs"] > 0
    assert set(diagnostic["near_100kb_change"]) == {"0", "0.5", "0.9", "0.99", "1"}
    build = provenance["build"]
    assert isinstance(build["created_at"], str)
    assert build["created_at"].endswith("Z")
    assert build["wall_seconds"] > 0
    assert build["peak_memory_bytes"] > 0
    assert build["threads"] >= 1


@OGLD_5
def test_the_variant_set_is_the_test_panel_s_passing_rows(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _, out = built(10, tmp_path_factory)
    v = panel_data().variants()
    failing = np.flatnonzero(~v.passes_filters)
    assert len(failing), "the Test Panel must hold variants the filters remove"
    m = ogld.open(out).matrix(alids(np.arange(len(v.pos))))
    assert set(m.absent) == set(alids(failing))


@OGLD_5
def test_a_pgen_without_info_refuses_an_info_floor(tmp_path: Path) -> None:
    # A sequenced panel has no INFO: info_min must be None, said on purpose, not skipped.
    with pytest.raises(ogld.InputError, match="INFO"):
        ogld.build(inputs(10, info_min=0.8), tmp_path / "store")


@OGLD_5
def test_ogld_build_writes_a_store_that_validates(tmp_path: Path) -> None:
    out = tmp_path / "store"
    argv = ["build", str(out), "--pgen", str(PANEL.pgen), "--keep", str(PANEL.keep)]
    argv += ["--pcs", str(PANEL.pcs), "--pc-source", "the Test Panel's PC File"]
    argv += ["--panel-id", PANEL_ID, "--ancestry", "EUR", "--genotype-source", "Test Panel"]
    assert cli.main([*argv, "--info-min", "none"]) == 0
    assert not any(ogld.validate(out).values())


# ------------------------------------------------------------- complete genotypes, local PCs


def read_genotypes(pgen: Path) -> npt.NDArray[np.int8]:
    """Every hard call of a pgen, variants by samples: 0, 1, 2, or −9 where missing."""
    import pgenlib  # the builder's pgen reader, which ogld#5 adds

    reader = pgenlib.PgenReader(str(pgen).encode())
    out = np.empty((reader.get_variant_ct(), reader.get_raw_sample_ct()), dtype=np.int8)
    reader.read_range(0, reader.get_variant_ct(), out)
    reader.close()
    return out


def write_pgen(genotypes: npt.NDArray[np.int8], prefix: Path) -> Path:
    """The Test Panel's pvar and psam beside a pgen of `genotypes`; the new .pgen's path."""
    import pgenlib

    pgen = prefix.with_name(prefix.name + ".pgen")
    writer = pgenlib.PgenWriter(str(pgen).encode(), genotypes.shape[1], genotypes.shape[0], False)
    for row in genotypes:
        writer.append_biallelic(np.ascontiguousarray(row))
    writer.close()
    shutil.copyfile(PANEL.pvar, prefix.with_name(prefix.name + "".join(PANEL.pvar.suffixes)))
    shutil.copyfile(PANEL.psam, prefix.with_name(prefix.name + ".psam"))
    return pgen


def read_pcs(path: Path) -> tuple[list[str], list[str], npt.NDArray[np.float64]]:
    """A PC File's PC names, each row's IID, and its values."""
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    n_id = sum(not column.startswith("PC") for column in header)
    rows = [line.split() for line in lines[1:]]
    values = np.array([[float(x) for x in row[n_id:]] for row in rows])
    return header[n_id:], [row[n_id - 1] for row in rows], values


def psam_iids(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    return [line.split()[header.index("IID")] for line in lines[1:]]


def window_axes() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The genotype axis of chr22's densest 4 Mb window, and of chr21's, in PC File order.

    Each axis is the first principal component of its window's standardised genotypes.
    """
    v = panel_data().variants()
    genotypes = read_genotypes(PANEL.pgen).astype(np.float64)
    order = [psam_iids(PANEL.psam).index(iid) for iid in read_pcs(PANEL.pcs)[1]]
    axes = []
    for chrom in ("22", "21"):
        rows = passing_rows()[v.chrom[passing_rows()] == chrom]
        pos = v.pos[rows].astype(np.int64)
        counts = np.searchsorted(pos, pos + 4_000_000, side="left") - np.arange(len(pos))
        start = int(np.argmax(counts))
        window = rows[start : start + int(counts[start])]
        assert len(window) >= 20, f"the Test Panel must hold a dense 4 Mb window on chr{chrom}"
        z = genotypes[window].T
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        axes.append(np.linalg.svd(z, full_matrices=False)[0][:, 0][order])
    return axes[0], axes[1]


def pc_file(tmp_path: Path, columns: list[npt.NDArray[np.float64]]) -> Path:
    """A PC File of `columns`, PC1 first, for the Test Panel's samples."""
    _, iids, _ = read_pcs(PANEL.pcs)
    out = tmp_path / "pcs.eigenvec"
    header = ["#IID", *[f"PC{i}" for i in range(1, len(columns) + 1)]]
    lines = ["\t".join(header)]
    for row, iid in enumerate(iids):
        lines.append("\t".join([iid, *(f"{column[row]:.8g}" for column in columns)]))
    out.write_text("\n".join(lines) + "\n")
    return out


def ancestry_pcs(count: int) -> list[npt.NDArray[np.float64]]:
    """The Test Panel's own PC1…PC<count>: ancestry PCs."""
    _, _, values = read_pcs(PANEL.pcs)
    return [values[:, i] for i in range(count)]


def scan_of(provenance: ogld.Provenance) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The PC Check's scan by PC, and its dropped PCs, each held to the rule."""
    check = provenance["adjustment"]["pc_check"]
    assert check is not None
    assert (check["window_bp"], check["min_share"]) == (4_000_000, 0.1)
    assert check["null"], "the null the rest of the genome was judged against is recorded"
    for scan in check["scan"]:
        rule = scan["window_share"] >= 0.1 and scan["rest_share"] <= scan["rest_null"]
        assert scan["local"] is rule, f"{scan['pc']}: the scan disagrees with the rule"
        assert (scan["pc"] in check["dropped"]) is scan["local"]
    return {scan["pc"]: dict(scan) for scan in check["scan"]}, list(check["dropped"])


@OGLD_5
def test_an_input_with_a_missing_call_is_refused(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    built(0, tmp_path_factory)
    genotypes = read_genotypes(PANEL.pgen)
    assert (genotypes >= 0).all(), "the Test Panel's genotypes must be complete"
    row = int(passing_rows()[0])
    genotypes[row, 0] = -9
    pgen = write_pgen(genotypes, tmp_path / "one-missing")
    with pytest.raises(ogld.InputError, match="missing") as refused:
        ogld.build(inputs(0, pgen=(pgen,)), tmp_path / "store")
    assert str(panel_data().variants().id[row]) in str(refused.value), "it names the variant"
    assert PANEL_ID in str(refused.value), "it names the panel"


@OGLD_5
def test_a_local_pc_is_dropped_and_a_pc_the_rest_explains_is_kept(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    built(10, tmp_path_factory)
    chr22, chr21 = window_axes()
    # PC1 is one window's axis: that window explains it and the rest only by chance.
    # PC2 is two windows' axes on two chromosomes: one window explains about half of it,
    # and so does the rest of the genome, as it would an ancestry PC.
    two_regions = (chr22 / chr22.std() + chr21 / chr21.std()) / np.sqrt(2)
    pcs = pc_file(tmp_path, [chr22, two_regions, *ancestry_pcs(20)])
    provenance = ogld.build(inputs(10, pcs=pcs), tmp_path / "store")
    scans, dropped = scan_of(provenance)
    local, kept = scans["PC1"], scans["PC2"]
    assert local["window_share"] >= 0.1
    assert local["rest_share"] <= local["rest_null"]
    assert "PC1" in dropped
    assert kept["window_share"] >= 0.1, "the second half of the rule must be what keeps PC2"
    assert kept["rest_share"] > kept["rest_null"]
    assert "PC2" not in dropped
    covariates = provenance["adjustment"]["covariates"]
    assert covariates[:2] == ["PC2", "PC3"], "PC2 and the Test Panel's ancestry PC1 are kept"
    assert provenance["adjustment"]["k"] == len(covariates) == 10


@OGLD_5
def test_seven_surviving_pcs_are_used_with_t_far_at_k7(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Owner, 7 November 2026: the first up to 10 PCs kept; fewer are used if at least 5 are.
    built(10, tmp_path_factory)
    chr22, _ = window_axes()
    pcs = pc_file(tmp_path, [chr22, *ancestry_pcs(7)])
    provenance = ogld.build(inputs(10, pcs=pcs), tmp_path / "store")
    _, dropped = scan_of(provenance)
    assert dropped == ["PC1"]
    adjustment = provenance["adjustment"]
    assert adjustment["covariates"] == [f"PC{i}" for i in range(2, 9)]
    assert (adjustment["k_target"], adjustment["k_min"], adjustment["k"]) == (10, 5, 7)
    assert provenance["t_far"] == t_far(771, 7)
    assert provenance["far_code_floor"] == far_code_floor(t_far(771, 7))


@OGLD_5
def test_fewer_than_five_surviving_pcs_are_refused(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    built(10, tmp_path_factory)
    chr22, _ = window_axes()
    pcs = pc_file(tmp_path, [chr22, *ancestry_pcs(4)])
    with pytest.raises(ogld.InputError, match=r"\b4\b") as refused:
        ogld.build(inputs(10, pcs=pcs), tmp_path / "store")
    assert "5" in str(refused.value), "it names the minimum it needed"

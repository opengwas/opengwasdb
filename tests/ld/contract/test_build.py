"""ogld#5's contract: the builder, on the Test Panel, against the independent reference r.

The Test Panel's `reference_r(k)` is a short numpy computation in lane 2d's script, checked
against plink2's `--r-unphased` at k = 0 by tests/test_testdata.py. The builder must match it
within the quantisation step: in the Near Band pair by pair, and in which pairs beyond 3 Mb
it keeps, except where a pair's code lies on the far code floor's rounding edge.

The builder also refuses an input with a missing call, and its PC Check drops a local PC and
keeps an ancestry one (owner, 7 November 2026; ADR 0002). Those tests read and write pgen
files with pgenlib, the reader ogld#5 adds, so each first calls `ogld.build` on the Test
Panel as it is: until ogld#5 lands, that call is where the test stops.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import support
from support import alids, contract, panel_data, passing_rows, reference

import ogld
from ogld import cli
from ogld.format import far_code_floor, t_far

OGLD_5 = contract("ogld#5")
PANEL_ID = "test-panel-hgdp1kgp-EUR"


def inputs(k: int = support.K, **overrides: object) -> ogld.BuildInputs:
    panel = panel_data().panel()
    fields: dict[str, object] = {
        "pgen": (panel.pgen,),
        "keep": panel.keep,
        "pcs": panel.pcs if k else None,
        "pc_source": "the Test Panel's PC File (tests/data/test-panel/README.md)" if k else None,
        "panel_id": PANEL_ID,
        "ancestry": "EUR",
        "genotype_source": "HGDP+1kGP phased v2, the Test Panel's slice",
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
    pc_file = provenance["adjustment"]["pc_file"]
    assert pc_file is not None
    assert pc_file["sha256"] == hashlib.sha256(Path(panel.pcs).read_bytes()).hexdigest()
    check = provenance["adjustment"]["pc_check"]
    assert check is not None
    names, _, _ = read_pcs(Path(panel.pcs))
    assert [scan["pc"] for scan in check["scan"]] == names
    kept = [name for name in names if name not in check["dropped"]]
    assert provenance["adjustment"]["covariates"] == kept[:10]
    # The reference r at k = 10 regresses out PC1–PC10, so it is the builder's only if the
    # PC Check keeps all ten on the Test Panel. If it does not, the reference must change.
    assert provenance["adjustment"]["covariates"] == [f"PC{i}" for i in range(1, 11)]
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
    argv += ["--pcs", str(panel.pcs), "--pc-source", "the Test Panel's PC File"]
    argv += ["--panel-id", "test", "--ancestry", "EUR", "--genotype-source", "Test Panel"]
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

    panel = panel_data().panel()
    pgen = prefix.with_name(prefix.name + ".pgen")
    writer = pgenlib.PgenWriter(str(pgen).encode(), genotypes.shape[1], genotypes.shape[0], False)
    for row in genotypes:
        writer.append_biallelic(np.ascontiguousarray(row))
    writer.close()
    pvar_suffix = "".join(Path(panel.pvar).suffixes)
    shutil.copyfile(panel.pvar, prefix.with_name(prefix.name + pvar_suffix))
    shutil.copyfile(panel.psam, prefix.with_name(prefix.name + ".psam"))
    return pgen


def read_pcs(path: Path) -> tuple[list[str], list[list[str]], npt.NDArray[np.float64]]:
    """A PC File's PC names, each row's identifier columns, and its values."""
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    n_id = sum(not column.startswith("PC") for column in header)
    rows = [line.split() for line in lines[1:]]
    values = np.array([[float(x) for x in row[n_id:]] for row in rows])
    return header[n_id:], [row[:n_id] for row in rows], values


def psam_iids(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    return [line.split()[header.index("IID")] for line in lines[1:]]


def local_pc_file(tmp_path: Path) -> Path:
    """The Test Panel's PC File with one more PC in front: one 4 Mb window's genotype axis.

    The axis is the first principal component of the standardised genotypes of the
    densest 4 Mb window on chr22, so that window explains all of it and the rest of the
    genome explains it only by chance: a local PC. The Test Panel's own PC1, an ancestry
    PC, becomes PC2.
    """
    v = panel_data().variants()
    rows = passing_rows()
    chr22 = rows[v.chrom[rows] == "22"]
    pos = v.pos[chr22].astype(np.int64)
    counts = np.searchsorted(pos, pos + 4_000_000, side="left") - np.arange(len(pos))
    start = int(np.argmax(counts))
    window = chr22[start : start + int(counts[start])]
    assert len(window) >= 20, "the Test Panel must hold a dense 4 Mb window"
    genotypes = read_genotypes(Path(panel_data().panel().pgen))[window].astype(np.float64).T
    z = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    axis = np.linalg.svd(z, full_matrices=False)[0][:, 0]
    by_iid = dict(zip(psam_iids(Path(panel_data().panel().psam)), axis, strict=True))
    names, ids, values = read_pcs(Path(panel_data().panel().pcs))
    out = tmp_path / "local.eigenvec"
    header = ["#IID", *[f"PC{i}" for i in range(1, len(names) + 2)]]
    lines = ["\t".join(header)]
    for row_ids, row in zip(ids, values, strict=True):
        local = by_iid[row_ids[-1]]
        lines.append("\t".join([row_ids[-1], f"{local:.8g}", *(f"{x:.8g}" for x in row)]))
    out.write_text("\n".join(lines) + "\n")
    return out


@OGLD_5
def test_an_input_with_a_missing_call_is_refused(tmp_path: Path) -> None:
    ogld.build(inputs(0), tmp_path / "plain")
    genotypes = read_genotypes(Path(panel_data().panel().pgen))
    assert (genotypes >= 0).all(), "the Test Panel's genotypes must be complete"
    row = int(passing_rows()[0])
    genotypes[row, 0] = -9
    pgen = write_pgen(genotypes, tmp_path / "one-missing")
    with pytest.raises(ValueError, match="missing") as refused:
        ogld.build(inputs(0, pgen=(pgen,)), tmp_path / "store")
    assert str(panel_data().variants().id[row]) in str(refused.value), "it names the variant"
    assert PANEL_ID in str(refused.value), "it names the panel"


@OGLD_5
def test_a_local_pc_is_dropped_and_an_ancestry_pc_is_kept(tmp_path: Path) -> None:
    ogld.build(inputs(10), tmp_path / "plain")
    pcs = local_pc_file(tmp_path)
    provenance = ogld.build(inputs(10, pcs=pcs), tmp_path / "store")
    check = provenance["adjustment"]["pc_check"]
    assert check is not None
    assert (check["window_bp"], check["min_share"]) == (4_000_000, 0.1)
    local = check["scan"][0]
    assert local["pc"] == "PC1"
    assert local["local"]
    assert local["window_share"] >= 0.1
    assert "PC1" in check["dropped"]
    assert "PC2" not in check["dropped"], "the Test Panel's ancestry PC1, now PC2, is kept"
    names = [scan["pc"] for scan in check["scan"]]
    kept = [name for name in names if name not in check["dropped"]]
    assert provenance["adjustment"]["covariates"] == kept[:10]
    assert provenance["adjustment"]["covariates"][0] == "PC2"

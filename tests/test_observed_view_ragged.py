"""The Observed-View Check for a Ragged Store (issue #296, stores ADR 0037).

Every test goes through the CLI. The fixture is an Observed-Only Ragged Store of four
Analyses and the `complete-ragged` output made from it, with the one-block LD panel of
`test_ragged_completion`. Its expected counts are worked by hand from the source rows and
the panel below -- never counted with the code under test.

The panel has six SNPs in one chr1 block (900,000-1,150,000, 1:1050500:C:T among them).
The Analyses (`afr_other` reports no EAF; see `NO_EAF`):

=============== ========== ============================= =====================================
Analysis        ancestry   source associations          completed against the one block
=============== ========== ============================= =====================================
`eur_rich`      EUR, cis   5 on the block + 1:400000     the panel-only SNP is imputed (1)
`eur_sparse`    EUR, cis   2 on the block                too few to fit: 4 NaN rows
`eur_traitless` EUR, none  5 on the block                block found by its own variants; 1 imputed
`afr_other`     AFR        5 on the block                not admitted: carried through
=============== ========== ============================= =====================================
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import test_check_observed_view as tcov
import test_dense_completion as tdc
import test_ragged_completion as trc
import zarr
from cli_output import normalize_cli_output
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRReader
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.variants.axis import VariantAxis

COLUMNS = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "beta", "standard_error", "effect_allele_frequency",
]
BLOCK = [
    (900_000, "A", "G"), (950_000, "A", "C"), (1_000_000, "A", "G"),
    (1_100_000, "C", "T"), (1_150_000, "A", "G"),
]
OFF_BLOCK = (400_000, "A", "G")
PANEL_ONLY = "1:1050500:C:T"
SOURCES = {
    "eur_rich": [*BLOCK, OFF_BLOCK],
    "eur_sparse": BLOCK[:2],
    "eur_traitless": BLOCK,
    "afr_other": BLOCK,
}
ANALYSES = [
    {"analysis_id": "eur_rich", "trait_chr": "1", "trait_bp": 1_050_000, "ancestry": "EUR"},
    {"analysis_id": "eur_sparse", "trait_chr": "1", "trait_bp": 1_050_000, "ancestry": "EUR"},
    {"analysis_id": "eur_traitless", "ancestry": "EUR"},
    {"analysis_id": "afr_other", "trait_chr": "1", "trait_bp": 1_050_000, "ancestry": "AFR"},
]
# Hand counts: source associations 6 + 2 + 5 + 5; the one panel-only SNP imputed for the two
# Analyses that observe five block SNPs; the four block positions `eur_sparse` cannot fit are
# NaN rows; one retained region (the block) for each of the three admitted Analyses.
EXPECTED = {"observed_in_source": 18, "imputed": 2, "nan_rows": 4, "regions": 3}
N_ANALYSES = 4
# `afr_other` reports no frequency. One Analysis without EAF keeps SE as float16 for the whole
# Store (ADR 0037 section 3), which completion then carries exactly; with residual-coded SE the
# refit re-quantises the observed SE of every Analysis it imputes into, see the spec's Ragged scope.
NO_EAF = "afr_other"


def _write_filtered(path: Path, rows: list[tuple[int, str, str]], *, with_eaf: bool) -> None:
    columns = COLUMNS if with_eaf else COLUMNS[:-1]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for i, (bp, effect, other) in enumerate(rows):
            z = 0.5 + 0.37 * i * (-1) ** i
            eaf = round(0.1 + 0.07 * i + (bp % 7) * 0.01, 4)
            cells = [1, bp, effect, other, round(z * 0.15, 6), 0.15, eaf][: len(columns)]
            fh.write("\t".join(str(cell) for cell in cells) + "\n")


def _build_source(
    root: Path,
    sources: dict[str, list[tuple[int, str, str]]],
    analyses: list[dict] = ANALYSES,
) -> Path:
    filtered = root / "filtered"
    filtered.mkdir(parents=True)
    for name, rows in sources.items():
        _write_filtered(filtered / f"{name}.tsv.gz", rows, with_eaf=name != NO_EAF)
    manifest = root / "manifest.tsv"
    trc._write_ssf_manifest(manifest, [
        {
            "analysis_index": i, "analysis_id": a["analysis_id"], "trait_id": f"T{i}",
            "trait_chr": a.get("trait_chr", ""), "trait_bp": a.get("trait_bp", ""),
            "n": 5000, "mhc": "FALSE", "filtered_file": f"{a['analysis_id']}.tsv.gz",
            "assigned_ancestry": a["ancestry"],
        }
        for i, a in enumerate(analyses)
    ])
    observed = root / "obs.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, observed, store_id="test", release_id="obs-v1")
    return observed


@pytest.fixture(scope="module")
def _pair_seed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("observed-view-ragged")
    observed = _build_source(root, SOURCES)
    panel = trc._make_rich_ld_panel(root)
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        observed, completed, panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        release_id="comp-v1",
    )
    return observed, completed, panel


@pytest.fixture(scope="module")
def _longer_source_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The source with one more observation: `afr_other` also observes the panel-only SNP."""
    sources = {**SOURCES, "afr_other": [*BLOCK, (1_050_500, "C", "T")]}
    return _build_source(tmp_path_factory.mktemp("observed-view-ragged-longer"), sources)


@pytest.fixture(scope="module")
def _shorter_source_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The source without `eur_rich`'s observation at 1:900000."""
    sources = {**SOURCES, "eur_rich": [*BLOCK[1:], OFF_BLOCK]}
    return _build_source(tmp_path_factory.mktemp("observed-view-ragged-shorter"), sources)


@pytest.fixture
def pair(tmp_path: Path, _pair_seed: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    observed, completed, panel = _pair_seed
    return (
        shutil.copytree(observed, tmp_path / "obs.opengwasdb"),
        shutil.copytree(completed, tmp_path / "comp.opengwasdb"),
        shutil.copytree(panel, tmp_path / "panel"),
    )


def _run(source, target, out, *extra, seed=7, panel=None, ancestry="EUR"):
    options = ["--ld-panel", str(panel)] if panel is not None else []
    if panel is not None and ancestry:
        options += ["--ancestry", ancestry]
    return CliRunner().invoke(
        app,
        ["check-observed-view", str(source), str(target), "--seed", str(seed),
         "--output", str(out), *options, *extra],
    )


def _verdict(result, out: Path) -> dict:
    assert out.exists(), result.output
    return json.loads(out.read_text())


def _check(pair, tmp_path, *extra, **kwargs):
    observed, completed, panel = pair
    out = tmp_path / "verdict.json"
    result = _run(observed, completed, out, *extra, panel=panel, **kwargs)
    return result, _verdict(result, out)


# --- the fixture is meaningful before anything is asserted about it ---------


def _flat(store: Path, analysis_index: int, alid: str) -> int:
    """The flat position of one Analysis's association at `alid`."""
    axis = VariantAxis(store)
    try:
        index = {v.alid: v.variant_index for v in axis.all()}[alid]
    finally:
        axis.close()
    reader = RaggedCSRReader(store)
    start, _ = reader._span(analysis_index)
    row = reader.variant_indices(analysis_index).tolist().index(index)
    return start + row


def test_the_fixture_has_every_cell_class_the_check_distinguishes(_pair_seed):
    observed, completed, _ = _pair_seed
    source_counts = [len(RaggedCSRReader(observed).variant_indices(i)) for i in range(N_ANALYSES)]
    assert source_counts == [6, 2, 5, 5]
    root = zarr.open_group(str(completed / RAGGED_ZARR_PATH), mode="r")
    imputed = root["imputed"][:]
    offsets = root["offsets"][:]
    reader = RaggedCSRReader(completed)
    per_analysis = []
    for i in range(N_ANALYSES):
        z = reader.get_analysis(i).z
        flags = imputed[offsets[i]:offsets[i + 1]]
        per_analysis.append((int(flags.sum()), int(np.isnan(z).sum())))
    assert per_analysis == [(1, 0), (0, 4), (1, 0), (0, 0)]
    records = read_analysis_records(completed / "analyses.tsv")
    completed_against = [a.completed_against for a in records]
    assert completed_against == ["EUR", "EUR", "EUR", ""]


# --- seam 1: the pair passes, with hand-counted totals ----------------------


def test_a_ragged_pair_passes_with_hand_counted_totals(tmp_path, pair):
    result, verdict = _check(pair, tmp_path)

    assert result.exit_code == 0, result.output
    assert verdict["passed"] is True and verdict["failures"] == []
    assert verdict["seed"] == 7
    totals = verdict["totals"]
    assert {key: totals[key] for key in EXPECTED} == EXPECTED
    assert verdict["sample"]["analyses_checked"] == N_ANALYSES
    assert verdict["sample"]["non_admitted"] == 1
    listing = {a["analysis_id"]: a for a in verdict["sample"]["analyses"]}
    assert "non_admitted" in listing["afr_other"]["strata"]
    assert listing["afr_other"]["admitted"] is False
    assert listing["eur_rich"]["admitted"] is True
    assert verdict["analyses"]["identical"] is True
    assert verdict["axis"]["parent_rows_unmapped"] == 0
    assert all(totals[name] == 0 for name in verdict["failing_counters"]), totals


def test_a_smaller_n_random_still_samples_the_non_admitted_analysis(tmp_path, pair):
    result, verdict = _check(pair, tmp_path, "--n-random", "1")

    assert result.exit_code == 0, result.output
    names = {a["analysis_id"] for a in verdict["sample"]["analyses"]}
    assert "afr_other" in names
    assert verdict["sample"]["by_stratum"]["random"] == 1


# --- seam 2: negative controls ---------------------------------------------


def _target_group(store: Path):
    return zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+")


def _failing(pair, tmp_path, counter: str, expected: int = 1):
    result, verdict = _check(pair, tmp_path)
    assert result.exit_code == 1, result.output
    assert verdict["passed"] is False
    assert verdict["totals"][counter] == expected, verdict["totals"]
    assert counter in verdict["failures"]
    return verdict


def test_a_changed_observed_z_fails(tmp_path, pair):
    _, completed, _ = pair
    position = _flat(completed, 0, "1:900000:A:G")
    before = RaggedCSRReader(completed).get_analysis(0).z.copy()
    _target_group(completed)["z"][position] += 1
    assert not np.array_equal(RaggedCSRReader(completed).get_analysis(0).z, before)

    _failing(pair, tmp_path, "z_mismatch")


def test_a_changed_observed_eaf_fails(tmp_path, pair):
    _, completed, _ = pair
    position = _flat(completed, 0, "1:900000:A:G")
    before = RaggedCSRReader(completed).get_analysis(0).eaf.copy()
    _target_group(completed)["eaf"][position] += 1
    assert not np.array_equal(RaggedCSRReader(completed).get_analysis(0).eaf, before)

    _failing(pair, tmp_path, "eaf_mismatch")


def test_a_dropped_observed_association_fails(tmp_path, pair, _longer_source_seed):
    """The source holds an association for `afr_other` that the completed Store lacks."""
    _, completed, panel = pair
    out = tmp_path / "verdict.json"

    result = _run(_longer_source_seed, completed, out, panel=panel)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["totals"]["observed_missing_in_target"] == 1, verdict["totals"]
    assert "observed_missing_in_target" in verdict["failures"]


def test_an_added_observed_association_fails(tmp_path, pair, _shorter_source_seed):
    """The completed Store holds an observed association `eur_rich`'s source lacks."""
    _, completed, panel = pair
    out = tmp_path / "verdict.json"

    result = _run(_shorter_source_seed, completed, out, panel=panel)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["totals"]["target_observed_not_in_source"] == 1, verdict["totals"]
    assert "target_observed_not_in_source" in verdict["failures"]


def test_an_imputed_association_outside_the_retained_regions_fails(tmp_path, pair):
    """The panel the retained regions are rebuilt from lacks the SNP completion imputed."""
    observed, completed, panel = pair
    block = panel / "EUR" / "1" / "900000-1300000.tsv"
    lines = block.read_text().splitlines()
    block.write_text("\n".join(line for line in lines if "1050500" not in line) + "\n")

    result, verdict = _check((observed, completed, panel), tmp_path)

    assert result.exit_code == 1, result.output
    assert verdict["totals"]["imputed_outside_retained"] == 2
    assert "imputed_outside_retained" in verdict["failures"]


def test_an_imputed_cell_in_a_non_admitted_analysis_fails(tmp_path, pair):
    _, completed, _ = pair
    position = _flat(completed, 3, "1:900000:A:G")
    _target_group(completed)["imputed"][position] = 1

    verdict = _failing(pair, tmp_path, "imputed_in_non_admitted")
    assert verdict["totals"]["observed_marked_imputed"] == 1


def test_an_imputed_mask_value_other_than_0_or_1_fails(tmp_path, pair):
    _, completed, _ = pair
    _target_group(completed)["imputed"][_flat(completed, 0, "1:900000:A:G")] = 2

    _failing(pair, tmp_path, "imputed_mask_invalid")


# --- seam 3: refusals ------------------------------------------------------


def _assert_refused(result, out: Path, reason: str) -> None:
    assert result.exit_code == 2, result.output
    assert reason in normalize_cli_output(result.output)
    assert not out.exists(), "nothing was checked, so no verdict is written"


def test_a_ragged_pair_without_the_ld_panel_is_refused(tmp_path, pair):
    observed, completed, _ = pair
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out), out, "--ld-panel")


def test_an_ld_panel_without_the_ancestry_is_refused(tmp_path, pair):
    observed, completed, panel = pair
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out, panel=panel, ancestry="AFR")

    _assert_refused(result, out, "completed against ancestry EUR")


def test_an_ld_panel_without_the_ancestry_directory_is_refused(tmp_path, pair):
    observed, completed, panel = pair
    shutil.rmtree(panel / "EUR")
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out, panel=panel), out, "LD panel")


def test_an_ld_panel_whose_blocks_are_not_those_completed_against_fails(tmp_path, pair):
    """A panel with no block reconstructs no region: every imputed cell lies outside one."""
    observed, completed, panel = pair
    for block in (panel / "EUR" / "1").iterdir():
        block.unlink()

    result, verdict = _check((observed, completed, panel), tmp_path)

    assert result.exit_code == 1, result.output
    assert verdict["totals"]["imputed_outside_retained"] == 2
    assert verdict["totals"]["quality_blocks_not_in_regions"] == 3
    assert verdict["totals"]["regions"] == 0


def test_a_mixed_layout_pair_is_refused(tmp_path, pair):
    observed, completed, panel = pair
    root = tmp_path / "dense"
    root.mkdir()
    dense_observed, dense_completed = tcov._build_pair(root, tcov.ROWS, tdc._make_ld_panel(root))
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, dense_completed, out, panel=panel), out, "mixed-layout pair")
    _assert_refused(_run(dense_observed, completed, out, panel=panel), out, "mixed-layout pair")


def test_a_source_that_is_not_observed_only_is_refused(tmp_path, pair):
    _, completed, panel = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(completed, completed, out, panel=panel), out,
        "is not Observed-Only (completion_state=reference_completed)",
    )


def test_a_target_that_is_not_reference_completed_is_refused(tmp_path, pair):
    observed, _, panel = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(observed, observed, out, panel=panel), out,
        "is not Reference-Completed (completion_state=observed_only)",
    )


def _dense_pair(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dense"
    root.mkdir()
    return tcov._build_pair(root, tcov.ROWS, tdc._make_ld_panel(root))


def _hybrid_pair(tmp_path: Path) -> tuple[Path, Path]:
    import test_observed_view_hybrid as toh

    root = tmp_path / "hybrid"
    root.mkdir()
    observed = toh._build_observed(root, chunk_shape=(2, 1))
    return observed, toh._complete(observed, root)


@pytest.mark.parametrize("layout", ["dense", "hybrid"])
@pytest.mark.parametrize("option", ["--ld-panel", "--ancestry"])
def test_a_ragged_only_option_is_refused_for_a_dense_or_hybrid_pair(
    tmp_path, pair, layout, option
):
    _, _, panel = pair
    source, target = (_dense_pair if layout == "dense" else _hybrid_pair)(tmp_path)
    out = tmp_path / "verdict.json"
    value = str(panel) if option == "--ld-panel" else "EUR"

    result = CliRunner().invoke(
        app,
        ["check-observed-view", str(source), str(target), "--seed", "7", "--output", str(out),
         option, value],
    )

    _assert_refused(result, out, "only for a Ragged pair")


def test_a_target_without_its_ragged_group_is_refused(tmp_path, pair):
    observed, completed, panel = pair
    shutil.rmtree(completed / RAGGED_ZARR_PATH)
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out, panel=panel), out, "cannot be opened")


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("array", ["eaf_baseline", "eaf"])
def test_a_store_missing_an_eaf_array_is_refused(tmp_path, pair, role, array):
    observed, completed, panel = pair
    shutil.rmtree((observed if role == "source" else completed) / RAGGED_ZARR_PATH / array)
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out, panel=panel), out, "cannot be opened")


def test_a_target_without_the_reference_eaf_is_refused(tmp_path, pair):
    observed, completed, panel = pair
    shutil.rmtree(completed / RAGGED_ZARR_PATH / "eaf_reference")
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out, panel=panel), out, "cannot be opened")


def test_a_target_missing_an_array_is_refused_with_the_reason(tmp_path, pair):
    observed, completed, panel = pair
    shutil.rmtree(completed / RAGGED_ZARR_PATH / "se")
    out = tmp_path / "verdict.json"

    _assert_refused(_run(observed, completed, out, panel=panel), out, "cannot be opened")


def test_an_analysis_emptied_in_the_target_reports_its_associations_as_missing(tmp_path, pair):
    """`afr_other` is the last Analysis, so dropping its five rows is a truncation."""
    _, completed, _ = pair
    root = _target_group(completed)
    offsets = root["offsets"][:]
    assert offsets[-1] - offsets[-2] == 5
    for name in ("variant_index", "z", "se", "eaf", "imputed"):
        root[name].resize((int(offsets[-2]),))
    root["offsets"][-1] = offsets[-2]
    root.attrs["n_associations"] = int(offsets[-2])

    verdict = _failing(pair, tmp_path, "observed_missing_in_target", expected=5)

    assert verdict["totals"]["observed_in_source"] == 18


# --- seam 4: reproducibility ------------------------------------------------


def _without_timing(verdict: dict) -> dict:
    return {k: v for k, v in verdict.items() if k != "seconds"}


def test_the_same_seed_gives_the_same_verdict(tmp_path, pair):
    verdicts = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        verdicts.append(_check(pair, tmp_path / name, "--n-random", "2")[1])

    assert _without_timing(verdicts[0]) == _without_timing(verdicts[1])
    assert verdicts[0]["seed"] == 7 and verdicts[0]["n_random"] == 2


# --- retained regions on a multi-block panel, expected regions written by hand -----------------
#
# Six chr1 blocks, named for their extent, each with five panel SNPs (positions below). The cis
# window is 50,000 bp, so a Trait at 250,000 reaches [200,000, 300,000]. The bounds are
# inclusive, and each is tested from both sides, so a window shifted by one base either way
# changes the regions:
#   1/100000-199999  ends one base before the window starts     -> outside
#   1/150000-200000  ends exactly where the window starts       -> region
#   1/220000-280000  inside the window                          -> region
#   1/300000-380000  starts exactly where the window ends       -> region
#   1/300001-350001  starts one base after the window ends      -> outside
#   1/400000-480000  far outside                                -> outside
# Without a Trait position a block is a region once the Analysis holds `min_observed_points()`
# = 4 observations at its panel SNPs.
BLOCKS = {
    "100000-199999": [100_000, 110_000, 130_000, 190_000, 199_999],
    "150000-200000": [150_000, 160_000, 170_000, 180_000, 200_000],
    "220000-280000": [220_000, 235_000, 250_000, 265_000, 280_000],
    "300000-380000": [300_000, 320_000, 340_000, 360_000, 380_000],
    "300001-350001": [300_001, 310_001, 320_001, 330_001, 350_001],
    "400000-480000": [400_000, 420_000, 440_000, 460_000, 480_000],
}
CIS_WINDOW_BP = 50_000
REGION_ANALYSES = [
    {"analysis_id": "cis", "trait_chr": "1", "trait_bp": 250_000, "ancestry": "EUR"},
    {"analysis_id": "tl_at_threshold", "ancestry": "EUR"},
    {"analysis_id": "tl_below_threshold", "ancestry": "EUR"},
    {"analysis_id": "tl_off_panel", "ancestry": "EUR"},
    {"analysis_id": NO_EAF, "trait_chr": "1", "trait_bp": 250_000, "ancestry": "AFR"},
]


def _snps(positions: list[int]) -> list[tuple[int, str, str]]:
    return [(bp, "A", "G") for bp in positions]


REGION_SOURCES = {
    "cis": _snps(BLOCKS["220000-280000"]),
    # 4 panel SNPs of the first block: exactly at the threshold
    "tl_at_threshold": _snps(BLOCKS["100000-199999"][:4]),
    # 3 panel SNPs of the first block: one below it
    "tl_below_threshold": _snps(BLOCKS["100000-199999"][:3]),
    # 3 panel SNPs of the last block plus one variant inside its extent that is not a panel SNP
    "tl_off_panel": _snps([*BLOCKS["400000-480000"][:3], 450_000]),
    NO_EAF: _snps(BLOCKS["220000-280000"]),
}
EXPECTED_BLOCKS = {
    "cis": ["1/150000-200000", "1/220000-280000", "1/300000-380000"],
    "tl_at_threshold": ["1/100000-199999"],
    "tl_below_threshold": [],
    "tl_off_panel": [],
    NO_EAF: [],
}


def _write_multi_block_panel(root: Path) -> Path:
    directory = root / "multi_panel" / "EUR" / "1"
    directory.mkdir(parents=True)
    for name, positions in BLOCKS.items():
        lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
        lines += [f"1\t1:{bp}_A_G\tG\tA\t0.{30 + i}\t{bp}" for i, bp in enumerate(positions)]
        (directory / f"{name}.tsv").write_text("\n".join(lines) + "\n")
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((5, 5))
        ld = matrix @ matrix.T + np.eye(5) * 0.5
        with gzip.open(directory / f"{name}.unphased.vcor1.gz", "wt") as fh:
            fh.writelines("\t".join(f"{v:.6f}" for v in row) + "\n" for row in ld)
    return root / "multi_panel"


@pytest.fixture(scope="module")
def _region_pair_seed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("observed-view-ragged-regions")
    observed = _build_source(root, REGION_SOURCES, REGION_ANALYSES)
    panel = _write_multi_block_panel(root)
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        observed, completed, panel, ancestry="EUR", cis_window_bp=CIS_WINDOW_BP, min_cor=0.0,
        release_id="comp-v1",
    )
    return observed, completed, panel


def test_retained_regions_follow_the_hand_derived_block_table(tmp_path, _region_pair_seed):
    observed, completed, panel = _region_pair_seed
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out, panel=panel)

    verdict = _verdict(result, out)
    assert result.exit_code == 0, verdict["failures"]
    blocks = {a["analysis_id"]: a["blocks"] for a in verdict["sample"]["analyses"]}
    assert blocks == EXPECTED_BLOCKS
    assert verdict["totals"]["regions"] == 4


def test_completion_used_no_block_outside_the_hand_derived_table(_region_pair_seed):
    """The completed Store's own record of its blocks agrees with the hand table, independently
    of the check: neither the 199,999 block nor the 3-observation blocks were completed."""
    _, completed, _ = _region_pair_seed
    ids = [a.analysis_id for a in read_analysis_records(completed / "analyses.tsv")]
    root = completed / "index.sqlite"
    import sqlite3

    with sqlite3.connect(root) as db:
        rows = db.execute("SELECT analysis_index, block_id FROM completion_quality").fetchall()
    used: dict[str, set[str]] = {name: set() for name in ids}
    for index, block_id in rows:
        used[ids[index]].add(block_id)
    for name, expected in EXPECTED_BLOCKS.items():
        assert used[name] <= set(expected), (name, used[name])
    assert used["cis"] == set(EXPECTED_BLOCKS["cis"])
    assert used["tl_at_threshold"] == set(EXPECTED_BLOCKS["tl_at_threshold"])

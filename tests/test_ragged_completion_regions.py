"""Ragged completion fills only inside each Analysis's regions (issue #430, ADR 0039).

An Analysis's regions are its cis region, where it has a Trait position, and its
significant trans regions. The cis boundary is the region table's `cis` row where it
has one, and the Trait position +/- `cis_window_bp` otherwise; suggestive leads and
QC-panel rows never make a region, and an Analysis the table gives no region is carried
Observed-Only. Only panel variants inside a region's `[start, end]` become imputed or
missing rows.

The panel is three blocks; every Analysis observes five of each block's panel variants it
touches, enough to fit (`impute.min_observed_points()`), so the old rules -- a whole block
around a cis window, and any block holding four observations -- would have completed
every one of them. Each block holds panel-only variants the source never observed, some
inside a region and some outside, so whether a block was completed, and how far, is read
off which of them became rows. Rows, not imputed counts: an imputed cell depends on the
elastic net converging on a fixture this small, a row on the region alone.

====== ============ ================================== ============================
block  extent       observed                           panel-only
====== ============ ================================== ============================
1      900k-1,250k  900k 1,000k 1,100k 1,150k 1,200k    950k 1,050k 1,250k
2      100k-400k    100k 150k 200k 300k 350k            250k 400k
3      100k-350k    100k 150k 200k 300k 350k            250k
====== ============ ================================== ============================
"""

from __future__ import annotations

import gzip
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from region_tables import write_region_table
from stored_rows import stored_rows
from test_ragged_completion import _write_ssf_manifest

from opengwasdb.completion.impute import min_observed_points
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.variants.axis import VariantAxis

OBSERVED = {
    "1": (900_000, 1_000_000, 1_100_000, 1_150_000, 1_200_000),
    "2": (100_000, 150_000, 200_000, 300_000, 350_000),
    "3": (100_000, 150_000, 200_000, 300_000, 350_000),
}
PANEL_ONLY = {"1": (950_000, 1_050_000, 1_250_000), "2": (250_000, 400_000), "3": (250_000,)}
BLOCKS = {"1": "900000-1250000", "2": "100000-400000", "3": "100000-350000"}

#: `cis_signals` is shaped like OGS-00007's: a cis row narrower than the block around its
#: Trait position, a significant trans region on chromosome 2 and a suggestive lead on 3.
CIS = ("1", 1_000_000, 1_200_000)
SIGNIFICANT = ("2", 100_000, 320_000)


def alid(chromosome: str, bp: int) -> str:
    return f"{chromosome}:{bp}:A:G"


def _write_block(chromosome_dir: Path, name: str, positions: list[int], seed: int) -> None:
    """One LD block: its variant table and a precomputed eigendecomposition."""
    chromosome = chromosome_dir.name
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    lines += [f"{chromosome}\t{alid(chromosome, bp)}\tG\tA\t0.3\t{bp}" for bp in positions]
    (chromosome_dir / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = len(positions)
    noise = np.random.default_rng(seed).normal(size=(n, n))
    ld = noise @ noise.T + np.eye(n) * n * 0.1
    values, vectors = np.linalg.eigh((ld + ld.T) / 2)
    order = np.argsort(values)[::-1]
    np.savez(chromosome_dir / f"{name}.ldeig.npz", values=values[order], vectors=vectors[:, order])


def _write_panel(root: Path) -> Path:
    for seed, (chromosome, name) in enumerate(BLOCKS.items()):
        directory = root / "panel" / "EUR" / chromosome
        directory.mkdir(parents=True)
        positions = sorted([*OBSERVED[chromosome], *PANEL_ONLY[chromosome]])
        _write_block(directory, name, positions, seed)
    return root / "panel"


def _write_source(
    root: Path, analyses: list[dict], coverage: str = "cis_and_signals"
) -> Path:
    """An Observed-Only Ragged Store of Association Coverage `coverage`; each Analysis
    observes `OBSERVED` on its `chromosomes`."""
    filtered = root / "filtered"
    filtered.mkdir()
    for a in analyses:
        with gzip.open(filtered / f"{a['analysis_id']}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "chromosome\tbase_pair_location\teffect_allele\tother_allele"
                "\tbeta\tstandard_error\teffect_allele_frequency\n"
            )
            for chromosome in a["chromosomes"]:
                for k, bp in enumerate(OBSERVED[chromosome]):
                    z = (1.5 + 0.7 * k) * (-1) ** k
                    fh.write(f"{chromosome}\t{bp}\tA\tG\t{z * 0.15:.6f}\t0.15\t{0.2 + 0.1 * k}\n")
    manifest = root / "manifest.tsv"
    _write_ssf_manifest(manifest, [
        {
            "analysis_index": i, "analysis_id": a["analysis_id"], "trait_id": f"T{i}",
            "trait_chr": a.get("trait_chr", ""), "trait_bp": a.get("trait_bp", ""),
            "n": 5000, "mhc": "FALSE", "filtered_file": f"{a['analysis_id']}.tsv.gz",
            "assigned_ancestry": "EUR", "source_genome_build": "GRCh38",
        }
        for i, a in enumerate(analyses)
    ])
    source = root / "obs.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, source, store_id="test", release_id="obs-v1",
        association_coverage=coverage,
    )
    return source


@dataclass(frozen=True)
class Rows:
    """One Analysis's completed rows, by ALID: the imputed and missing ones, and all."""

    all: dict[str, int]
    imputed: set[str]
    missing: set[str]
    z: dict[str, float]
    se: dict[str, float]
    eaf: dict[str, float]

    @property
    def added(self) -> set[str]:
        return {a for a in self.all if a not in self.observed_in_source}

    observed_in_source: frozenset[str] = frozenset()


def completed_rows(store: Path, analysis: int, source: Path | None = None) -> Rows:
    rows = stored_rows(store, analysis)
    with closing(VariantAxis(store)) as axis:
        records = axis.by_indices(rows.variant_index.tolist())
    alids = [records[int(v)].alid for v in rows.variant_index]
    flags = rows.flags if rows.flags is not None else np.zeros(len(alids), dtype=np.uint8)
    observed: frozenset[str] = frozenset()
    if source is not None:
        observed = frozenset(completed_rows(source, analysis).all)
    return Rows(
        all={a: i for i, a in enumerate(alids)},
        imputed={a for a, f in zip(alids, flags.tolist(), strict=True) if f == 1},
        missing={a for a, z in zip(alids, rows.z.tolist(), strict=True) if np.isnan(z)},
        z=dict(zip(alids, rows.z.tolist(), strict=True)),
        se=dict(zip(alids, rows.se.tolist(), strict=True)),
        eaf=dict(zip(alids, rows.eaf.tolist(), strict=True)),
        observed_in_source=observed,
    )


def within(a: str, *regions: tuple[str, int, int]) -> bool:
    chromosome, bp, *_ = a.split(":")
    return any(chromosome == c and start <= int(bp) <= end for c, start, end in regions)


def test_the_panel_holds_what_the_assertions_read(tmp_path):
    """Each block holds enough of an Analysis's observations to fit, and panel-only
    variants both inside and outside the regions, or nothing below proves anything."""
    panel = _write_panel(tmp_path)
    for chromosome, name in BLOCKS.items():
        text = (panel / "EUR" / chromosome / f"{name}.tsv").read_text()
        assert all(alid(chromosome, bp) in text for bp in PANEL_ONLY[chromosome])
        assert len(OBSERVED[chromosome]) >= min_observed_points()
    # The cis row and the window below each cut block 1 short of both its ends.
    assert not within(alid("1", 950_000), CIS) and not within(alid("1", 1_250_000), CIS)
    assert within(alid("1", 1_050_000), CIS)
    assert within(alid("2", 250_000), SIGNIFICANT) and not within(alid("2", 400_000), SIGNIFICANT)


# --- a cis-only Store records no region table --------------------------------

#: The Trait position and window of the cis-only Analysis: 975,000-1,125,000.
TRAIT_BP, WINDOW = 1_050_000, 75_000
GENE = {"analysis_id": "gene", "trait_chr": "1", "trait_bp": TRAIT_BP, "chromosomes": ["1"]}


@pytest.fixture(scope="module")
def cis_only(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("cis-only")
    source = _write_source(root, [GENE], coverage="cis")
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        source, completed, _write_panel(root), ancestry="EUR", cis_window_bp=WINDOW,
        min_cor=0.0, release_id="comp-v1",
    )
    return source, completed


def test_a_cis_only_store_completes_within_the_trait_position_and_window(cis_only):
    source, completed = cis_only
    window = ("1", TRAIT_BP - WINDOW, TRAIT_BP + WINDOW)
    rows = completed_rows(completed, 0, source)

    assert alid("1", 1_050_000) in rows.added, "the window's panel-only variant is completed"
    assert rows.added, "the fixture must add rows for the assertion below to mean anything"
    outside = sorted(a for a in rows.added if not within(a, window))
    assert outside == [], "rows were added outside the Trait position +/- the window"


# --- a cis-and-signals Store, shaped like OGS-00007 ---------------------------

#: Analyses of the Store completed with a region table: `cis_signals` has a Trait position
#: and `cis`, `significant_trans` and `suggestive_trans` rows; `sig_only` has no Trait
#: position and one significant region, its other rows QC-panel and suggestive ones;
#: `no_sig` has no Trait position and no significant region at all.
TABLED = [
    {"analysis_id": "cis_signals", "trait_chr": "1", "trait_bp": 1_050_000,
     "chromosomes": ["1", "2", "3"]},
    {"analysis_id": "sig_only", "chromosomes": ["1", "2", "3"]},
    {"analysis_id": "no_sig", "chromosomes": ["1", "2", "3"]},
]
TABLE = [
    ("cis_signals", "cis", *CIS),
    ("cis_signals", "significant_trans", *SIGNIFICANT),
    ("cis_signals", "suggestive_trans", "3", 200_000, 200_000),
    ("sig_only", "significant_trans", *SIGNIFICANT),
    ("sig_only", "suggestive_trans", "3", 150_000, 150_000),
    ("sig_only", "qc_panel", "1", 900_000, 1_200_000),
    ("sig_only", "qc_panel", "3", 100_000, 350_000),
    ("no_sig", "suggestive_trans", "1", 1_100_000, 1_100_000),
    ("no_sig", "qc_panel", "1", 900_000, 1_200_000),
    ("no_sig", "qc_panel", "2", 100_000, 350_000),
    ("no_sig", "qc_panel", "3", 100_000, 350_000),
]
#: Each Analysis's regions, from the table above: the only places rows may be added.
REGIONS = {0: (CIS, SIGNIFICANT), 1: (SIGNIFICANT,), 2: ()}


@pytest.fixture(scope="module")
def tabled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("tabled")
    source = _write_source(root, TABLED)
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        source, completed, _write_panel(root), ancestry="EUR", cis_window_bp=500_000,
        min_cor=0.0, release_id="comp-v1",
        regions=write_region_table(root / "sparse_regions.tsv", TABLE),
    )
    return source, completed


def test_a_cis_analysis_completes_its_cis_row_and_its_significant_trans_region(tabled):
    source, completed = tabled
    rows = completed_rows(completed, 0, source)

    assert alid("1", 1_050_000) in rows.added, "the cis row's panel-only variant is completed"
    assert alid("2", 250_000) in rows.added, "the significant trans region is completed"
    assert sorted(a for a in rows.added if not within(a, CIS, SIGNIFICANT)) == []
    # The parts of the blocks outside the boundaries, and the suggestive lead's block.
    for left_out in (alid("1", 950_000), alid("1", 1_250_000), alid("2", 400_000),
                     alid("3", 250_000)):
        assert left_out not in rows.all, left_out


def test_an_analysis_with_no_trait_position_completes_only_its_significant_region(tabled):
    source, completed = tabled
    rows = completed_rows(completed, 1, source)

    assert alid("2", 250_000) in rows.added, "the significant region is completed"
    assert sorted(a for a in rows.added if not within(a, SIGNIFICANT)) == []
    # Blocks 1 and 3 hold five of its observations each, all QC-panel or suggestive rows.
    assert not any(a.startswith(("1:", "3:")) for a in rows.added)


def test_an_analysis_with_no_significant_region_keeps_exactly_its_observed_rows(tabled):
    source, completed = tabled
    rows = completed_rows(completed, 2, source)

    assert len(rows.observed_in_source) == 15, "it observes five variants in each block"
    assert set(rows.all) == rows.observed_in_source
    assert not rows.imputed


@pytest.mark.parametrize("analysis", [0, 1, 2])
def test_every_observed_row_is_carried_exactly(tabled, analysis):
    source, completed = tabled
    before, after = completed_rows(source, analysis), completed_rows(completed, analysis)

    assert len(before.all) == 15
    for a in before.all:
        assert a not in after.imputed
        for plane in ("z", "se", "eaf"):
            assert getattr(after, plane)[a] == getattr(before, plane)[a], (a, plane)


def test_every_added_row_lies_inside_a_region_of_its_analysis(tabled):
    source, completed = tabled
    for analysis, regions in REGIONS.items():
        rows = completed_rows(completed, analysis, source)
        assert sorted(a for a in rows.added if not within(a, *regions)) == [], analysis
        assert rows.imputed | rows.missing >= rows.added


# --- refusals ---------------------------------------------------------------


@pytest.fixture(scope="module")
def tabled_source(tabled) -> Path:
    return tabled[0]


def _refused(tmp_path: Path, source: Path, match: str, **kwargs) -> None:
    with pytest.raises(ValueError, match=match):
        complete_ragged_store(
            source, tmp_path / "comp.opengwasdb", _write_panel(tmp_path), ancestry="EUR",
            min_cor=0.0, **kwargs,
        )
    assert not (tmp_path / "comp.opengwasdb").exists()


@pytest.fixture(scope="module")
def placed_signals_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A `cis_and_signals` Store whose every Analysis has a Trait position: one the old
    rule completed on its cis windows alone, its significant trans regions nowhere."""
    return _write_source(tmp_path_factory.mktemp("placed-signals"), [GENE])


def test_a_store_whose_coverage_is_not_cis_is_refused_without_a_table(
    tmp_path, placed_signals_source
):
    _refused(tmp_path, placed_signals_source, "Association Coverage is cis_and_signals")


def test_an_analysis_with_no_trait_position_the_table_omits_is_refused(
    tmp_path, tabled_source
):
    table = write_region_table(tmp_path / "t.tsv", [r for r in TABLE if r[0] != "no_sig"])
    _refused(
        tmp_path, tabled_source, "no Trait position and no row in region table", regions=table
    )


def test_a_table_in_another_assembly_is_refused(tmp_path, tabled_source):
    table = write_region_table(tmp_path / "t.tsv", TABLE, assembly="GRCh37")
    _refused(tmp_path, tabled_source, "'GRCh37', not the Store's GRCh38", regions=table)


def test_a_table_without_a_column_completion_reads_is_refused(tmp_path, tabled_source):
    table = write_region_table(tmp_path / "t.tsv", TABLE, without="source_genome_build")
    _refused(tmp_path, tabled_source, "has no source_genome_build column", regions=table)


def test_a_region_that_ends_before_it_starts_is_refused(tmp_path, tabled_source):
    table = write_region_table(
        tmp_path / "t.tsv", [*TABLE, ("sig_only", "significant_trans", "2", 320_000, 100_000)]
    )
    _refused(tmp_path, tabled_source, "start 320000 and end 100000 are no span", regions=table)


def test_a_resume_refuses_a_table_changed_since_its_checkpoints(
    tmp_path, tabled_source, monkeypatch
):
    """A checkpoint's blocks were filled inside the regions its table held; resumed
    against other regions, its fills and the new rows would disagree."""
    from opengwasdb.completion.checkpoint import checkpoint_dir_for
    from opengwasdb.layouts.ragged import complete as ragged

    table = write_region_table(tmp_path / "t.tsv", TABLE)
    real_run_block = ragged._run_block

    def crash_after_the_block(task):
        real_run_block(task)
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(ragged, "_run_block", crash_after_the_block)
    dst = tmp_path / "comp.opengwasdb"
    with pytest.raises(RuntimeError, match="simulated crash"):
        complete_ragged_store(
            tabled_source, dst, _write_panel(tmp_path), ancestry="EUR", min_cor=0.0,
            regions=table,
        )
    monkeypatch.undo()
    assert list((checkpoint_dir_for(dst) / "blocks").glob("*.npz")), "a checkpoint to resume"
    write_region_table(table, [*TABLE, ("no_sig", "significant_trans", *SIGNIFICANT)])

    with pytest.raises(ValueError, match="has changed since"):
        ragged.resume_ragged_completion(checkpoint_dir_for(dst))
    assert not dst.exists()


def test_a_region_kind_the_table_does_not_define_is_refused(tmp_path, tabled_source):
    table = write_region_table(
        tmp_path / "t.tsv", [*TABLE, ("sig_only", "significant_cis", "2", 1, 10)]
    )
    _refused(tmp_path, tabled_source, "significant_cis", regions=table)


def test_a_table_row_for_an_analysis_the_store_lacks_is_refused(tmp_path, tabled_source):
    table = write_region_table(
        tmp_path / "t.tsv", [*TABLE, ("elsewhere", "significant_trans", *SIGNIFICANT)]
    )
    _refused(tmp_path, tabled_source, "elsewhere", regions=table)

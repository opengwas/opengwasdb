"""Ragged completion fills only inside each Analysis's regions (issue #430, ADR 0039).

An Analysis's regions are its cis region, where it has a Trait position, and its
significant trans regions. The cis boundary is the region table's `cis` row where it
has one, and the Trait position +/- `cis_window_bp` otherwise; suggestive leads and
QC-panel rows never make a region, and an Analysis the table gives no region is carried
Observed-Only. Only panel variants inside a region's `[start, end]` become imputed or
missing rows. The table is the one `build-ragged-ssf --regions` checks and writes into the
Store as `regions.tsv`, which completion reads from its source and carries forward.

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

import csv
import gzip
import hashlib
import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ld_blocks import write_ld_block
from region_tables import edit_manifest, with_regions, write_region_table
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


def _write_panel(root: Path) -> Path:
    """The three blocks of the table above, each a variant table and an
    eigendecomposition, as a real panel ships them (ADR 0031)."""
    for seed, (chromosome, name) in enumerate(BLOCKS.items()):
        positions = sorted([*OBSERVED[chromosome], *PANEL_ONLY[chromosome]])
        write_ld_block(
            root / "panel" / "EUR" / chromosome, name,
            [(alid(chromosome, bp), 0.3, bp) for bp in positions], seed,
            write_matrix=False, write_npz=True,
        )
    return root / "panel"


def _write_source(
    root: Path,
    analyses: list[dict],
    coverage: str = "cis_and_signals",
    regions: Path | None = None,
    name: str = "obs.opengwasdb",
    *,
    hm_code: str | None = None,
    deletion: str | None = None,
) -> Path:
    """An Observed-Only Ragged Store of Association Coverage `coverage`, built with the
    region table at `regions`, if any; each Analysis observes `OBSERVED` on its
    `chromosomes`, and the `deletion` ALID where it is on one of them. With `hm_code`
    every row carries that harmoniser code; 10 declares a row's other allele its
    reference, so the build keeps an indel (ADR 0079)."""
    filtered = root / "filtered"
    filtered.mkdir(parents=True, exist_ok=True)
    code = "" if hm_code is None else f"\t{hm_code}"
    for a in analyses:
        with gzip.open(filtered / f"{a['analysis_id']}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "chromosome\tbase_pair_location\teffect_allele\tother_allele"
                "\tbeta\tstandard_error\teffect_allele_frequency"
                + ("" if hm_code is None else "\thm_code") + "\n"
            )
            for chromosome in a["chromosomes"]:
                rows = []
                for k, bp in enumerate(OBSERVED[chromosome]):
                    z = (1.5 + 0.7 * k) * (-1) ** k
                    rows.append(
                        (bp, f"{chromosome}\t{bp}\tA\tG\t{z * 0.15:.6f}\t0.15\t{0.2 + 0.1 * k}")
                    )
                if deletion is not None and deletion.split(":")[0] == chromosome:
                    _c, bp, reference, alt = deletion.split(":")
                    row = f"{chromosome}\t{bp}\t{alt}\t{reference}\t-0.45\t0.15\t0.25"
                    rows.append((int(bp), row))
                fh.writelines(f"{row}{code}\n" for _bp, row in sorted(rows))
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
    source = root / name
    build_ragged_from_ssf(
        manifest, filtered, source, store_id="test", release_id="obs-v1",
        association_coverage=coverage, regions=regions,
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


def test_a_cis_store_without_regions_tsv_completes_within_the_trait_position_and_window(
    cis_only,
):
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
    source = _write_source(
        root, TABLED, regions=write_region_table(root / "sparse_regions.tsv", TABLE)
    )
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        source, completed, _write_panel(root), ancestry="EUR", cis_window_bp=500_000,
        min_cor=0.0, release_id="comp-v1",
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



# --- indels: one the source observes, ones the panel declares, ones it cannot name --

#: Per chromosome, a panel indel no ALID names (its panel declares no reference, ADR
#: 0079): in block 1 outside the cis row, and in block 2 outside the significant region.
UNNAMED = {"1": 1_240_000, "2": 380_000}
#: Panel deletions the panel's indel-reference table settles, by the id the panel writes
#: (alleles sorted, its `EA` the first) and the ALID completion spells reference first:
#: one inside the cis row, one inside the significant region.
DECLARED = {"1:1060000:G:GA": "1:1060000:GA:G", "2:260000:C:CT": "2:260000:CT:C"}
#: A deletion every Analysis observes inside the significant region, by its ALID and by
#: the id the panel holds it under; the table settles the panel's row too.
OBSERVED_DELETION, OBSERVED_DELETION_PANEL_ID = "2:180000:AT:A", "2:180000:A:AT"


def _add_indel_rows(panel: Path) -> Path:
    """`_write_panel`'s blocks with the `UNNAMED`, `DECLARED` and observed-deletion
    rows, in position order, each row's `EA` its id's first allele; and the panel's
    indel-reference table, which settles every one of them but the `UNNAMED`."""
    import test_panel_indels as panel_fixture

    added: dict[str, list[str]] = {}
    for chromosome, bp in UNNAMED.items():
        added.setdefault(chromosome, []).append(f"{chromosome}:{bp}:A:AT")
    for snp in [*DECLARED, OBSERVED_DELETION_PANEL_ID]:
        added.setdefault(snp.split(":")[0], []).append(snp)
    for seed, (chromosome, snps) in enumerate(added.items()):
        tsv = panel / "EUR" / chromosome / f"{BLOCKS[chromosome]}.tsv"
        header, *rows = tsv.read_text(encoding="utf-8").splitlines()
        for snp in snps:
            _c, bp, ea, oa = snp.split(":")
            rows.append(f"{chromosome}\t{snp}\t{oa}\t{ea}\t0.3\t{bp}")
        rows.sort(key=lambda row: int(row.split("\t")[-1]))
        tsv.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        n = len(rows)
        noise = np.random.default_rng(seed).normal(size=(n, n))
        ld = noise @ noise.T + np.eye(n) * n * 0.1
        values, vectors = np.linalg.eigh((ld + ld.T) / 2)
        order = np.argsort(values)[::-1]
        np.savez(tsv.with_suffix(".ldeig.npz"), values=values[order], vectors=vectors[:, order])
    settled = {**{snp: alid for snp, alid in DECLARED.items()},
               OBSERVED_DELETION_PANEL_ID: OBSERVED_DELETION}
    table = panel / "EUR" / "indel_reference.tsv"
    table.write_text(
        "#chromosome\tposition\tallele1\tallele2\treference\toutcome\n" + "".join(
            f"{c}\t{bp}\t{a1}\t{a2}\t{alid.split(':')[2]}\tsettled_agree\n"
            for snp, alid in sorted(settled.items(), key=lambda kv: (kv[0].split(":")[0],
                                                                      int(kv[0].split(":")[1])))
            for c, bp, a1, a2 in [snp.split(":")]
        ),
        encoding="utf-8",
    )
    panel_fixture._compress_and_index(table)
    return panel


@dataclass(frozen=True)
class Completed:
    source: Path
    completed: Path
    panel: Path


def _complete_with_indels(root: Path, **kwargs) -> Completed:
    """`TABLED`'s Store from harmonised files, built with `TABLE` and every Analysis
    observing `OBSERVED_DELETION`, completed against the panel `_add_indel_rows` writes."""
    source = _write_source(
        root, TABLED, regions=write_region_table(root / "sparse_regions.tsv", TABLE),
        hm_code="10", deletion=OBSERVED_DELETION,
    )
    panel = _add_indel_rows(_write_panel(root))
    completed = root / "comp.opengwasdb"
    complete_ragged_store(
        source, completed, panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        release_id="comp-v1", **kwargs,
    )
    return Completed(source, completed, panel)


@pytest.fixture(scope="module")
def indels(tmp_path_factory: pytest.TempPathFactory) -> Completed:
    return _complete_with_indels(tmp_path_factory.mktemp("indels"))


def _completion_record(store: Path) -> dict:
    return json.loads((store / "manifest.json").read_text(encoding="utf-8"))[
        "provenance"]["completion"]


def _alids_at(store: Path, chromosome: str, bp: int) -> list[str]:
    from opengwasdb.variants.axis import iter_variant_records

    return [
        r.alid for r in iter_variant_records(store / "variants.tsv.gz")
        if (r.chromosome, r.position) == (chromosome, bp)
    ]


def test_the_indel_fixture_holds_what_the_assertions_read(indels):
    """The source keeps its deletion, spelt reference first; the panel declares three
    indels and leaves two unnamed; the declared ones lie inside the regions and the
    unnamed ones outside them."""
    for analysis in (0, 1, 2):
        assert OBSERVED_DELETION in completed_rows(indels.source, analysis).all, analysis
    record = _completion_record(indels.completed)["panel_indels"]
    assert (record["reference_from"], record["n_indels"], record["n_placed"]) == ("table", 5, 3)
    assert record["unplaced"]["no_reference"] == 2
    for alid in [*DECLARED.values(), OBSERVED_DELETION]:
        assert within(alid, *REGIONS[0]), alid
    assert within(DECLARED["2:260000:C:CT"], *REGIONS[1])
    assert not any(within(f"{c}:{bp}:A:AT", *REGIONS[0]) for c, bp in UNNAMED.items())


def test_a_declared_panel_deletion_in_a_region_is_a_row_spelt_reference_first(indels):
    """Completion names a panel deletion by its ALID, reference first, wherever it
    plans, adds and writes it (ADR 0079): a row of each Analysis whose region holds it,
    never its panel id's sorted spelling."""
    cis_signals = completed_rows(indels.completed, 0, indels.source)
    sig_only = completed_rows(indels.completed, 1, indels.source)

    assert set(DECLARED.values()) <= cis_signals.added
    assert DECLARED["2:260000:C:CT"] in sig_only.added
    assert DECLARED["1:1060000:G:GA"] not in sig_only.all, "outside its only region"
    for snp, alid in DECLARED.items():
        chromosome, bp, *_ = snp.split(":")
        assert _alids_at(indels.completed, chromosome, int(bp)) == [alid]


def test_an_observed_deletion_is_carried_exactly_and_named_once(indels):
    """The source's deletion and the panel's row for it are one variant: carried
    observed, values unchanged, with no second spelling beside it."""
    for analysis in (0, 1, 2):
        before = completed_rows(indels.source, analysis)
        after = completed_rows(indels.completed, analysis)
        assert OBSERVED_DELETION not in after.imputed
        for plane in ("z", "se", "eaf"):
            got = getattr(after, plane)[OBSERVED_DELETION]
            assert got == getattr(before, plane)[OBSERVED_DELETION], (analysis, plane)
    assert _alids_at(indels.completed, "2", 180_000) == [OBSERVED_DELETION]


def test_a_completion_with_indels_passes_the_observed_view_check(indels):
    """The check rebuilds each Analysis's region rows from the panel as completion
    named them, so the reference-first deletion rows lie inside the regions."""
    from opengwasdb.validation.observed_view_ragged import check_ragged

    verdict = check_ragged(
        indels.source, indels.completed, seed=7, n_random=3, ld_panel=indels.panel,
        ancestry="EUR",
    )

    assert verdict.record["totals"]["regions"] == 3, "both Analyses' regions were checked"
    assert verdict.passed, verdict.failures


def test_a_panel_row_no_alid_names_outside_the_regions_is_not_counted(indels):
    """An Analysis's `completion_quality` counts are the rows it is given: where every
    panel row inside its regions has an ALID, as here, its imputed and missing counts add
    up to the rows completion added it. A panel row no ALID names is never a row, and one
    outside every region of the Analysis is not counted missing either, whether or not
    the block's fit could have filled it."""
    from opengwasdb.model.analyses import read_analysis_records

    records = read_analysis_records(indels.completed / "analyses.tsv")
    for analysis in (0, 1):
        rows = completed_rows(indels.completed, analysis, indels.source)
        counted = int(records[analysis].completion_n_imputed_total) + int(
            records[analysis].completion_n_missing_total
        )
        assert rows.added, analysis
        assert counted == len(rows.added), analysis



def test_a_worker_that_places_fewer_indels_than_the_plan_stops_the_completion(
    tmp_path, monkeypatch
):
    """The completion record says how the plan's read of the panel placed its indels
    (`panel_indels`), and each block is read again where it is completed. A read that
    places fewer, as a worker reading the indel table through another process's handle
    once did, stops the completion: its fills and its record would describe two panels."""
    from dataclasses import replace

    from block_rereads import reread_with

    from opengwasdb.completion.ld_panel import PanelReadMismatchError

    def one_indel_fewer(block):
        placed = [
            k for k, a in enumerate(block.alids)
            if a is not None and len(a.split(":")[2]) != len(a.split(":")[3])
        ]
        if not placed:
            return block
        alids = list(block.alids)
        alids[placed[0]] = None
        return replace(block, alids=alids, n_indels_placed=block.n_indels_placed - 1)

    reread_with(monkeypatch, one_indel_fewer)
    with pytest.raises(PanelReadMismatchError, match=r"places \d+ indels?, where"):
        _complete_with_indels(tmp_path)


# --- the build records the table, and completion carries it ---------------------


def _table_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_the_build_writes_the_table_into_the_store_as_regions_tsv(tabled, tmp_path):
    source, _ = tabled

    assert _table_rows(source / "regions.tsv") == _table_rows(
        source.parent / "sparse_regions.tsv"
    )


def test_a_build_given_a_table_differs_from_one_without_only_by_regions_tsv(tmp_path):
    """The table changes nothing else a build writes: the file, and the manifest entry
    that declares it, are the whole difference. Both are 0.2.2, the table or not, since
    every build spells an indel reference first (spec §21.3)."""
    from store_digest import store_digest

    table = write_region_table(tmp_path / "t.tsv", TABLE)
    bare = store_digest(_write_source(tmp_path, TABLED, name="bare.opengwasdb"))
    tabled = store_digest(_write_source(tmp_path, TABLED, regions=table, name="t.opengwasdb"))

    assert tabled["files"] == sorted([*bare["files"], "regions.tsv"])
    assert "regions" not in bare["manifest.json"]
    assert (bare["manifest.json"]["format_version"], tabled["manifest.json"]["format_version"]) == (
        "0.2.2", "0.2.2"
    )
    assert {
        k: v for k, v in tabled["manifest.json"].items() if k != "regions"
    } == bare["manifest.json"]
    assert {
        k: v for k, v in tabled.items() if k not in ("files", "regions.tsv", "manifest.json")
    } == {k: v for k, v in bare.items() if k not in ("files", "manifest.json")}


def _declared(store: Path) -> dict[str, str]:
    return json.loads((store / "manifest.json").read_text(encoding="utf-8"))["regions"]


def test_the_build_declares_regions_tsv_in_the_manifest_by_path_and_sha256(tabled):
    """A reader finds a Store's region table through its manifest (spec §21.2)."""
    source, _ = tabled

    assert _declared(source) == {
        "path": "regions.tsv",
        "sha256": hashlib.sha256((source / "regions.tsv").read_bytes()).hexdigest(),
    }


def _format_version(store: Path) -> str:
    return json.loads((store / "manifest.json").read_text(encoding="utf-8"))["format_version"]


def test_a_build_is_stamped_the_lowest_format_version_its_features_use(tabled, cis_only):
    """0.2.1 adds only the optional `regions.tsv`, and 0.2.2 an indel spelt reference
    first, which every build of this package does (spec §21.3, ADR 0079): a build is
    0.2.2 whether or not it carries the table."""
    assert _format_version(tabled[0]) == "0.2.2"
    assert _format_version(cis_only[0]) == "0.2.2"


def test_a_completion_keeps_its_sources_format_version(tabled, cis_only):
    """It carries the source's `regions.tsv`, or carries none, and so its version."""
    assert _format_version(tabled[1]) == "0.2.2"
    assert _format_version(cis_only[1]) == "0.2.2"


def _as_0_9_0_wrote_it(store: Path, tmp_path: Path) -> Path:
    """A copy of `store` stamped as package 0.9.0 stamps it: 0.2.1 with its region table,
    0.2.0 without. These fixtures hold no indel, so their ALIDs are 0.9.0's too."""
    import shutil

    from opengwasdb.variants.axis import iter_variant_records

    assert not any(
        len(r.effect_allele) != len(r.other_allele)
        for r in iter_variant_records(store / "variants.tsv.gz")
    ), "the restamp is honest only for a Store without indels"
    copy = shutil.copytree(store, tmp_path / f"{store.parent.name}-{store.name}")
    version = "0.2.1" if (copy / "regions.tsv").exists() else "0.2.0"
    edit_manifest(copy, lambda m: {**m, "format_version": version})
    return copy


def test_validate_accepts_this_builds_stores_and_0_9_0s(tmp_path, tabled, cis_only):
    """This build's 0.2.2 Stores, with the table and without, and the same Stores as
    package 0.9.0 stamps them, 0.2.1 and 0.2.0, all validate: the rules this branch
    gates at 0.2.2 do not touch a 0.9.0 Store (spec §21.3)."""
    from opengwasdb.validation.validate import validate_store

    stores = [*tabled, *cis_only]
    stores += [_as_0_9_0_wrote_it(store, tmp_path) for store in (tabled[0], cis_only[0])]
    for store in stores:
        result = validate_store(store)
        assert result.ok, (store, result.errors)
    assert sorted(_format_version(store) for store in stores) == [
        "0.2.0", "0.2.1", "0.2.2", "0.2.2", "0.2.2", "0.2.2"
    ]


@pytest.mark.parametrize("stamp", ["0.2.2", "0.2.1"])
def test_the_variant_index_is_rebuilt_on_a_store_carrying_regions_tsv(tmp_path, tabled, stamp):
    """Its arrays are a 0.2.0 Store's, so the in-place index build accepts it alike, as
    this build writes it and as package 0.9.0 stamps it."""
    import shutil

    from opengwasdb.layouts.ragged.by_variant import add_variant_index

    source = shutil.copytree(tabled[0], tmp_path / "obs.opengwasdb")
    edit_manifest(source, lambda m: {**m, "format_version": stamp})

    add_variant_index(source, force=True)

    assert _format_version(source) == stamp


def test_a_completion_refuses_a_source_stamped_other_than_its_features(tmp_path, cis_only):
    """A Store stamped other than the lowest version its features use disagrees with
    itself; completion refuses it rather than re-stamp it either way. Stamped 0.2.1, it
    also claims sorted indels, which completion refuses first (ADR 0079)."""
    import shutil

    from opengwasdb.store.open import UnsupportedFormatVersion

    source = shutil.copytree(cis_only[0], tmp_path / "restamped.opengwasdb")
    edit_manifest(source, lambda m: {**m, "format_version": "0.2.1"})

    with pytest.raises(UnsupportedFormatVersion, match="0.2.1 Store sorts an indel's alleles"):
        complete_ragged_store(
            source, tmp_path / "comp.opengwasdb", _write_panel(tmp_path), ancestry="EUR",
            cis_window_bp=WINDOW, min_cor=0.0,
        )
    assert not (tmp_path / "comp.opengwasdb").exists()


def test_the_completed_manifest_declares_the_regions_tsv_it_carries(tabled):
    source, completed = tabled

    assert _declared(completed) == _declared(source)
    assert _declared(completed)["sha256"] == hashlib.sha256(
        (completed / "regions.tsv").read_bytes()
    ).hexdigest()


def test_a_quoted_column_spanning_lines_reaches_regions_tsv_unchanged(tmp_path):
    """A column completion does not read is carried as the table holds it, line breaks
    inside a quoted value included."""
    label = "first line\nsecond line"
    table = write_region_table(tmp_path / "t.tsv", TABLE, target_label=label)
    assert [r["target_label"] for r in _table_rows(table)] == [label] * len(TABLE)

    source = _write_source(tmp_path, TABLED, regions=table)

    assert _table_rows(source / "regions.tsv") == _table_rows(table)


def test_the_completed_store_carries_regions_tsv_unchanged(tabled):
    source, completed = tabled

    assert (completed / "regions.tsv").read_bytes() == (source / "regions.tsv").read_bytes()


# --- the build refuses a table it could not complete from -------------------------


def _build_refused(tmp_path: Path, rows: list, match: str, **table: str) -> None:
    regions = write_region_table(tmp_path / "t.tsv", rows, **table)
    with pytest.raises(ValueError, match=match):
        _write_source(tmp_path, TABLED, regions=regions)
    assert not (tmp_path / "obs.opengwasdb").exists()


def test_a_region_kind_the_table_does_not_define_is_refused(tmp_path):
    rows = [*TABLE, ("sig_only", "significant_cis", "2", 1, 10)]
    _build_refused(tmp_path, rows, "significant_cis")


def test_a_table_row_for_an_analysis_the_store_lacks_is_refused(tmp_path):
    rows = [*TABLE, ("elsewhere", "significant_trans", *SIGNIFICANT)]
    _build_refused(tmp_path, rows, "elsewhere")


def test_a_table_in_another_assembly_is_refused(tmp_path):
    _build_refused(tmp_path, TABLE, "'GRCh37', not the Store's GRCh38", assembly="GRCh37")


def test_a_table_without_a_column_completion_reads_is_refused(tmp_path):
    _build_refused(
        tmp_path, TABLE, "has no source_genome_build column", without="source_genome_build"
    )


def test_a_region_that_ends_before_it_starts_is_refused(tmp_path):
    rows = [*TABLE, ("sig_only", "significant_trans", "2", 320_000, 100_000)]
    _build_refused(tmp_path, rows, "start 320000 and end 100000 are no span")


# --- completion refuses a source it cannot place every region of ------------------


def _refused(tmp_path: Path, source: Path, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        complete_ragged_store(
            source, tmp_path / "comp.opengwasdb", _write_panel(tmp_path), ancestry="EUR",
            min_cor=0.0,
        )
    assert not (tmp_path / "comp.opengwasdb").exists()


def test_a_store_whose_coverage_is_not_cis_is_refused_without_regions_tsv(tmp_path):
    """A `cis_and_signals` Store whose every Analysis has a Trait position: one the old rule
    completed on its cis windows alone, its significant trans regions nowhere."""
    source = _write_source(tmp_path / "placed", [GENE])

    _refused(
        tmp_path, source,
        "Association Coverage is cis_and_signals and it holds no regions.tsv, the region "
        "table its build records: rebuild it with build-ragged-ssf --regions",
    )


def test_a_store_holding_a_regions_tsv_its_manifest_does_not_declare_is_refused(
    tmp_path, tabled
):
    """Completion finds the regions through the manifest; a file it does not declare is a
    Store that disagrees with itself, not a table to complete inside."""
    import shutil

    source = shutil.copytree(tabled[0], tmp_path / "undeclared.opengwasdb")
    edit_manifest(source, lambda m: {k: v for k, v in m.items() if k != "regions"})

    _refused(tmp_path, source, "manifest.json declares no regions entry")


def test_an_analysis_with_no_trait_position_the_table_omits_is_refused(tmp_path):
    regions = write_region_table(tmp_path / "t.tsv", [r for r in TABLE if r[0] != "no_sig"])
    source = _write_source(tmp_path, TABLED, regions=regions)

    _refused(tmp_path, source, "no Trait position and no row in region table")


def test_a_resume_refuses_a_regions_tsv_changed_since_its_checkpoints(
    tmp_path, tabled, monkeypatch
):
    """A checkpoint's blocks were filled inside the regions the source's `regions.tsv` held;
    resumed against other regions, its fills and the new rows would disagree."""
    import shutil

    from opengwasdb.completion.checkpoint import checkpoint_dir_for
    from opengwasdb.layouts.ragged import complete as ragged

    source = shutil.copytree(tabled[0], tmp_path / "obs.opengwasdb")
    real_run_block = ragged._run_block

    def crash_after_the_block(task):
        real_run_block(task)
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(ragged, "_run_block", crash_after_the_block)
    dst = tmp_path / "comp.opengwasdb"
    with pytest.raises(RuntimeError, match="simulated crash"):
        complete_ragged_store(source, dst, _write_panel(tmp_path), ancestry="EUR", min_cor=0.0)
    monkeypatch.undo()
    assert list((checkpoint_dir_for(dst) / "blocks").glob("*.npz")), "a checkpoint to resume"
    with_regions(source, [*TABLE, ("no_sig", "significant_trans", *SIGNIFICANT)])

    with pytest.raises(ValueError, match="regions.tsv has changed since"):
        ragged.resume_ragged_completion(checkpoint_dir_for(dst))
    assert not dst.exists()


# --- info and validate --------------------------------------------------------------


def test_info_shows_the_regions_a_store_records(tabled):
    """`TABLE` by hand: 1 cis, 2 significant_trans, 3 suggestive and 5 QC-panel rows;
    `cis_signals` and `sig_only` have a completion region, `no_sig` none."""
    import json

    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    source, _ = tabled
    text = CliRunner().invoke(app, ["info", str(source)])
    as_json = CliRunner().invoke(app, ["info", str(source), "--format", "json"])

    assert text.exit_code == 0, text.output
    assert (
        "regions: cis 1, significant_trans 2, suggestive_trans 3, qc_panel 5; "
        "2 Analyses have a completion region, 1 have none"
    ) in text.output
    assert json.loads(as_json.output)["regions"] == {
        "rows_by_kind": {"cis": 1, "significant_trans": 2, "suggestive_trans": 3, "qc_panel": 5},
        "analyses_with_region": 2, "analyses_without_region": 1,
    }


def _undeclared(store: Path) -> None:
    """The file without its entry, stamped as a writer that never declared it stamps."""
    edit_manifest(
        store,
        lambda m: {**{k: v for k, v in m.items() if k != "regions"}, "format_version": "0.2.0"},
    )


def _deleted(store: Path) -> None:
    (store / "regions.tsv").unlink()


def _altered(store: Path) -> None:
    """Still a valid table, of the Store's own Analyses, but not the one declared."""
    path = store / "regions.tsv"
    path.write_bytes(path.read_bytes().replace(b"test-policy", b"test-policz"))


def _stamped_0_2_0(store: Path) -> None:
    """Declares the table, but as a 0.2.0 Store, which an older reader takes as one it
    fully understands."""
    edit_manifest(store, lambda m: {**m, "format_version": "0.2.0"})


def _foreign(store: Path) -> None:
    with_regions(store, [*TABLE, ("elsewhere", "significant_trans", *SIGNIFICANT)])


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (_foreign, "names 1 Analyses the Store does not hold"),
        (_undeclared, "manifest.json declares no regions entry"),
        (_deleted, "but the Store holds no such file"),
        (_altered, "not the sha256 manifest.json declares"),
    ],
    ids=["foreign-analysis", "present-undeclared", "declared-missing", "sha256-mismatch"],
)
def test_validate_fails_regions_table_and_nothing_else(tmp_path, tabled, breakage, message):
    import shutil

    from opengwasdb.validation.validate import validate_store

    source = shutil.copytree(tabled[0], tmp_path / "obs.opengwasdb")
    assert validate_store(source).ok, "the fixture must validate before it is broken"
    breakage(source)

    result = validate_store(source)

    assert result.failed() == {"regions.table"}, result.errors
    assert any(message in e for e in result.errors), result.errors


def test_validate_fails_the_stamp_of_a_store_declaring_regions_tsv_as_0_2_0(tmp_path, tabled):
    """The table is the one feature 0.2.1 adds, so a Store declaring it is 0.2.1 (spec
    §21.3); stamped 0.2.0, it fails the stamp rule, and the table's own rule passes."""
    import shutil

    from opengwasdb.validation.validate import validate_store

    source = shutil.copytree(tabled[0], tmp_path / "obs.opengwasdb")
    assert validate_store(source).ok, "the fixture must validate before it is restamped"
    _stamped_0_2_0(source)

    result = validate_store(source)

    assert result.failed() == {"manifest.format_stamp"}, result.errors
    assert any(
        "stamped 0.2.0, but a release carrying regions.tsv is 0.2.1" in e for e in result.errors
    ), result.errors

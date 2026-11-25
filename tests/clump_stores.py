"""Store Releases on Test Panel positions, for the Clumps of their 5e-8 tier (#418).

The Test Panel (`tests/ld/data/test-panel/`, ADR 0088) carries an LD Store at `store/` and a
simulated GWAS z per row. These fixtures build Stores from those rows with the package's own
builders, so a Store's 5e-8 tier has real loci the LD Store can clump:

- **Analyses.** `a` and `c` hold every row at the panel's z; `b` holds them at 0.95 z, but for
  one causal variant's LD over `COMPLETION_BLOCK`, and leaves `UNOBSERVED_IN_B` out, which a
  completion imputes; `d` holds every row at the panel's z.
  `a` and `b` are EUR, `c` has no Assigned Ancestry and `d` is AFR, so with only an EUR LD Store
  mapped `c` and `d` are not clumped, for two different reasons.
- **Absent rows.** Two variants the panel does not hold (`OFF_PANEL`) and the rows the panel's
  filters drop, which its LD Store does not hold, all at a z that clears 5e-8.
- **Indels.** The panel's indels are kept, spelt reference first as a 0.3.0 Store spells them,
  so a deletion's ALID is not the LD Store's. `REVERSED_INDEL` adds the insertion of row
  `AMBIGUOUS_ROW`'s deletion: the two share one sorted spelling.

A test copies a Store before writing into it.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dense_sources import build_dense_from_records

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import write_analyses
from opengwasdb.readers.interface import ImputationScoreStatus
from opengwasdb.readers.rows import ReaderBatch
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.variants.normalise import canonical_alid
from tests.ld import testdata

#: The Test Panel's LD Store: HGDP+1kGP EUR at k = 10, chr21 and chr22.
LD_STORE = testdata.TEST_PANEL / "store"

#: Each Analysis's Assigned Ancestry; blank is Unassigned.
ANCESTRY = {"a": "EUR", "b": "EUR", "c": "", "d": "AFR"}

#: Variants the Test Panel does not hold, at a z that clears 5e-8: absent from its LD Store.
OFF_PANEL = (("21", 42150000, "A", "G", 9.0), ("22", 20450000, "C", "T", -6.5))

#: A deletion of the panel (`GT>G`) whose insertion `REVERSED_INDEL` the Store also holds.
AMBIGUOUS_ROW = 429
REVERSED_INDEL = ("21", 42889631, "G", "GT", 6.0)

#: Strong rows of a locus that `b` leaves out, and the LD block completion imputes them from.
UNOBSERVED_IN_B = (287, 288)
COMPLETION_BLOCK = (278, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291)
CAUSAL_ROW = 287
CAUSAL_Z = 80.0

#: The z a row the panel's filters drop is given: it clears 5e-8, so the tier holds it.
DROPPED_ROW_Z = 6.2
#: A row the panel simulated no z for, and that is not one the filters drop.
QUIET_Z = 0.3


@dataclass(frozen=True)
class Row:
    """One source association: the variant as its source spells it (REF, ALT) and its z."""

    chrom: str
    pos: int
    ref: str
    alt: str
    z: float

    @property
    def alid(self) -> str:
        """Its ALID as a 0.3.0 Store spells it: an indel reference first (ADR 0079)."""
        return canonical_alid(self.chrom, self.pos, self.ref, self.alt)


def panel_rows() -> list[Row]:
    """Every Test Panel row, the off-panel variants and the reversed indel, in genomic order."""
    v = testdata.variants()
    z = np.asarray(testdata.gwas().z, dtype=np.float64)
    rows = []
    for i in range(len(v)):
        dropped = DROPPED_ROW_Z if not v.passes_filters[i] else QUIET_Z
        value = z[i] if np.isfinite(z[i]) else dropped
        rows.append(Row(str(v.chrom[i]), int(v.pos[i]), str(v.ref[i]), str(v.alt[i]), float(value)))
    rows.extend(Row(c, p, r, a, z_) for c, p, r, a, z_ in (*OFF_PANEL, REVERSED_INDEL))
    return sorted(rows, key=lambda row: (int(row.chrom), row.pos, row.alid))


def unobserved_in_b() -> set[str]:
    v = testdata.variants()
    return {canonical_alid(str(v.chrom[i]), int(v.pos[i]), str(v.ref[i]), str(v.alt[i]))
            for i in UNOBSERVED_IN_B}


def analysis_rows() -> dict[str, list[Row]]:
    """Each Analysis's rows (module docstring)."""
    rows = panel_rows()
    skip = unobserved_in_b()
    block = _b_block_z()
    return {
        "a": rows,
        "b": [Row(r.chrom, r.pos, r.ref, r.alt, block.get(r.alid, 0.95 * r.z))
              for r in rows if r.alid not in skip],
        "c": rows,
        "d": rows,
    }


def set_ancestry(store: Path, ancestry: dict[str, str] = ANCESTRY) -> None:
    """Record each Analysis's Assigned Ancestry in `analyses.tsv`, as a registry would."""
    table = read_analyses(store / "analyses.tsv")
    for row in table.rows:
        row["assigned_ancestry"] = ancestry[row["analysis_id"]]
        row["ancestry_assignment_method"] = (
            "source_trusted_no_af" if ancestry[row["analysis_id"]] else "unassigned"
        )
    write_analyses(store / "analyses.tsv", table)


# ── Dense ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Record:
    """One Analysis's row, for `dense_sources.build_dense_from_records`."""

    analysis_id: str
    row: Row


def _batch(records: Sequence[Record]) -> ReaderBatch:
    """The reader batch of one Analysis's rows: each as its source spells it, REF and ALT."""
    rows = [record.row for record in records]
    n = len(rows)
    return ReaderBatch(
        alid=np.array([r.alid for r in rows], dtype=object),
        source_row=np.arange(n, dtype=np.int64),
        flipped=np.zeros(n, dtype=bool),
        z=np.array([r.z for r in rows], dtype=np.float64),
        se=np.full(n, 0.05),
        eaf=np.full(n, np.nan),
        imputation_score=np.full(n, np.nan),
        imputation_status=np.full(n, ImputationScoreStatus.UNDECLARED, dtype=object),
        rsid=np.array([""] * n, dtype=object),
        ref=np.array([r.ref for r in rows], dtype=object),
        alt=np.array([r.alt for r in rows], dtype=object),
    )


def build_dense(path: Path) -> Path:
    """A Dense Observed-Only Store of the four Analyses."""
    records = [Record(analysis_id, row)
               for analysis_id, rows in analysis_rows().items() for row in rows]
    build_dense_from_records(records, path, batch=_batch, store_id="clumps",
                             release_id="clumps-obs", reference_assembly="GRCh38")
    set_ancestry(path)
    return path


def _block_ld() -> tuple[list[str], np.ndarray, np.ndarray]:
    """`COMPLETION_BLOCK`'s ALIDs, its r (the Test Panel's reference r at k = 10, signed to
    each variant's A1), and each row's sign from ALT to A1 (testdata's r is signed to ALT)."""
    v = testdata.variants()
    pairs = testdata.reference_r(10)
    rows = list(COMPLETION_BLOCK)
    alids = [canonical_alid(str(v.chrom[i]), int(v.pos[i]), str(v.ref[i]), str(v.alt[i]))
             for i in rows]
    sign = np.array([1.0 if alid.split(":")[2] == v.alt[i] else -1.0
                     for alid, i in zip(alids, rows, strict=True)])
    at = {(int(i), int(j)): float(r)
          for i, j, r in zip(pairs.row_i, pairs.row_j, pairs.r, strict=True)}
    ld = np.eye(len(rows))
    for a, i in enumerate(rows):
        for b, j in enumerate(rows[a + 1 :], start=a + 1):
            ld[a, b] = ld[b, a] = at[(i, j)] * sign[a] * sign[b]
    return alids, ld, sign


def _b_block_z() -> dict[str, float]:
    """`b`'s z over `COMPLETION_BLOCK`, as its source spells it (ALT): one causal variant,
    `CAUSAL_ROW`, seen through the block's LD, so a completion can impute the rows `b`
    leaves out from the ones it observes."""
    alids, ld, sign = _block_ld()
    column = ld[:, COMPLETION_BLOCK.index(CAUSAL_ROW)]
    return {alid: float(CAUSAL_Z * r * s) for alid, r, s in zip(alids, column, sign, strict=True)}


def _completion_panel(root: Path) -> Path:
    """An LD reference panel of one block over `COMPLETION_BLOCK` (`_block_ld`)."""
    v = testdata.variants()
    alids, ld, _sign = _block_ld()
    block = root / "EUR" / "21"
    block.mkdir(parents=True)
    positions = [int(v.pos[i]) for i in COMPLETION_BLOCK]
    name = f"{positions[0]}-{positions[-1]}"
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, position in zip(alids, positions, strict=True):
        chrom, _pos, a1, a2 = alid.split(":")
        lines.append(f"{chrom}\t{alid}\t{a2}\t{a1}\t0.3\t{position}")
    (block / f"{name}.tsv").write_text("\n".join(lines) + "\n")
    with gzip.open(block / f"{name}.unphased.vcor1.gz", "wt") as fh:
        for row in ld:
            fh.write("\t".join(f"{value:.6f}" for value in row) + "\n")
    return root


def build_dense_completed(root: Path) -> Path:
    """The Dense Store, Reference-Completed: `b`'s two left-out rows are imputed."""
    source = build_dense(root / "obs.opengwasdb")
    completed = root / "completed.opengwasdb"
    complete_dense_store(
        source, completed, _completion_panel(root / "ld_panel"), ancestry="EUR", min_cor=0.0,
        release_id="clumps-comp",
    )
    return completed


# ── Ragged ───────────────────────────────────────────────────────────────────

_SSF = ("chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
        "standard_error", "hm_code")


def build_ragged(root: Path) -> Path:
    """A Ragged Observed-Only Store of the four Analyses, from harmonised GWAS-SSF files."""
    filtered = root / "filtered"
    filtered.mkdir(parents=True)
    lines = ["analysis_index\tanalysis_id\ttrait_id\tanalysis_label\tn\tfiltered_file"
             "\tsource_genome_build\tstored_effect_scale\toriginal_sd_method"]
    for index, (analysis_id, rows) in enumerate(sorted(analysis_rows().items())):
        with gzip.open(filtered / f"{analysis_id}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write("\t".join(_SSF) + "\n")
            for r in rows:
                fh.write(f"{r.chrom}\t{r.pos}\t{r.alt}\t{r.ref}\t{r.z}\t1\t10\n")
        lines.append(f"{index}\t{analysis_id}\t{analysis_id}\t{analysis_id}\t1000"
                     f"\t{analysis_id}.tsv.gz\tGRCh38\tsd\tdeclared_standardised")
    manifest = root / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = root / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="clumps", release_id="clumps-rag",
                          association_coverage="gwas")
    set_ancestry(store)
    return store


# ── Hybrid ───────────────────────────────────────────────────────────────────


def build_hybrid(root: Path) -> Path:
    """A Hybrid Observed-Only Store of the four Analyses. Its panel is every other SNV of the
    Test Panel, so a locus spans its Dense Component and its Ragged Overflow."""
    from test_hybrid_build import _make_manifest, _make_vcf

    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for analysis_id, rows in sorted(analysis_rows().items()):
        lines = [f"{r.chrom}\t{r.pos}\t.\t{r.ref}\t{r.alt}\t.\tPASS\t.\tES:SE\t{r.z}:1\n"
                 for r in rows]
        entries.append((analysis_id, _make_vcf(root, analysis_id, lines), analysis_id))
    snvs = [r for r in panel_rows() if len(r.ref) == len(r.alt) == 1]
    panel = root / "panel.txt"
    panel.write_text("".join(f"{r.alid}\n" for r in snvs[::2]), encoding="utf-8")
    store = root / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _make_manifest(root, entries), store, reference_panel=panel, store_id="clumps",
        release_id="clumps-hyb", source_assembly="GRCh38",
    )
    set_ancestry(store)
    return store


def copied(store: Path, directory: Path) -> Path:
    """A private copy of a built Store, for a test that writes into it."""
    return Path(shutil.copytree(store, directory / store.name))

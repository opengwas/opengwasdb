"""Read-only interface for the LD reference panel used in Reference Completion.

Panel layout (flat/production):
    ld_dir/{ancestry}/{chr}/{block_name}.tsv              (required)
    ld_dir/{ancestry}/{chr}/{block_name}.ldeig.npz        (eigendecomposition)
    ld_dir/{ancestry}/{chr}/{block_name}.unphased.vcor1.gz (optional LD matrix)

Completion consumes eigenvectors only, so the eigendecomposition is the primary
artifact (ADR 0031, store-format spec §13.1). A block loads when it has a variant
table plus *either* an eigendecomposition or an LD matrix to derive one from; the
matrix alone is retained for panels predating that contract.

A panel's `SNP` id sorts its two alleles, and its eigenvectors and `EAF` are for
its `EA` allele. An indel's ALID spells its reference allele first (ADR 0079),
which the id cannot say, so an indel is placed only where the panel declares its
reference: a block's own `REF` column, or else the indel-reference table beside
the panel's blocks (`ld_dir/{ancestry}/indel_reference.tsv.gz`, which
`scripts/build_panel_indel_reference.py` writes from GRCh38 evidence, and which
is read only beside a provenance its checksum matches). Every other indel names
no ALID. A variant whose ALID's `A1` is not its `EA` is re-signed for its `A1`
when its block is read (`LDBlock.flipped`); a row with no `EA`, or whose `EA`
is neither of its alleles, names none.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pysam

from opengwasdb.completion.impute import ld_pca, min_observed_points
from opengwasdb.variants import CanonicalVariant, VariantNormalisationError, orient_to_canonical
from opengwasdb.variants.normalise import identity_allele, normalise_chromosome

log = logging.getLogger(__name__)


@dataclass
class LDBlock:
    block_id: str       # "{chr}/{block_name}"
    chrom: str
    start_bp: int
    end_bp: int
    tsv_path: Path
    ld_path: Path | None      # None when the panel stores no LD matrix
    ldeig_npz_path: Path | None
    n_ld_snps: int
    snp_ids: list[str]  # panel-native ids; see canonical_panel_alid
    eaf: np.ndarray     # float64, length n_ld_snps, for the row's EA; NaN where absent
    #: Each row's canonical ALID, or None where it names none: an unparseable
    #: id, an indel the panel declares no reference for, or a row whose `EA`
    #: is not one of its alleles.
    alids: list[str | None]
    #: Each row whose ALID's A1 is not its EA, so its eigenvector row and EAF
    #: are re-signed for its A1 before they meet the Store's statistics.
    flipped: np.ndarray
    #: Where the block's indels took their reference: ``ref_column``, ``table``
    #: or ``none``; and how many indels it holds and how many it placed.
    indel_reference: str
    n_indels: int
    n_indels_placed: int
    #: Per reason, the indels (`INDEL_UNPLACED_REASONS`) and the SNVs and
    #: substitutions (`SNV_UNPLACED_REASONS`) it does not place.
    unplaced_indels: dict[str, int]
    unplaced_snvs: dict[str, int]

    def eaf_for_a1(self) -> np.ndarray:
        """Each row's frequency of its ALID's A1."""
        return np.where(self.flipped, 1.0 - self.eaf, self.eaf)

    def oriented_eigenvectors(self, eigenvectors: np.ndarray) -> np.ndarray:
        """``eigenvectors`` (one row per block position) re-signed for each row's A1:
        a flipped row's correlations with every other row change sign."""
        if not self.flipped.any():
            return eigenvectors
        oriented: np.ndarray = np.where(self.flipped[:, None], -eigenvectors, eigenvectors)
        return oriented


def _strip_chr(s: str) -> str:
    return s[3:] if s.startswith("chr") else s


def panel_id_parts(snp_id: str) -> tuple[str, str, str, str] | None:
    """``(chrom, pos, allele, allele)`` of a panel SNP id, in its own order:
    ``[chr]CHR:POS:A:B``, or ``[chr]CHR:POS_A_B`` as the UKB EUR panel writes it."""
    parts = _strip_chr(snp_id).split(":")
    if len(parts) == 4:
        chrom, pos_s, first, second = parts
        return chrom, pos_s, first, second
    if len(parts) == 2 and parts[1].count("_") == 2:
        chrom, rest = parts
        pos_s, first, second = rest.split("_")
        return chrom, pos_s, first, second
    return None


def canonical_panel_alid(snp_id: str, reference: str | None = None) -> str | None:
    """Normalise an LD-panel SNP id to a canonical store ALID (``chr:pos:a1:a2``).

    Handles both the legacy ``[chr]CHR:POS_REF_ALT`` form (underscores, as in the
    UKB EUR panel) and an already-colon-delimited ``CHR:POS:A1:A2`` form, sorting
    the alleles of an SNV or a substitution. Returns None for ids that don't
    parse — callers treat that as "not matchable" rather than an error, so a panel
    carrying a stray malformed row does not fail a whole block.

    An indel's ALID spells its reference allele first (ADR 0079), and no panel id
    says which allele that is: every panel on the host writes the pair sorted,
    whatever the genome's reference (issue #381). ``reference`` is the allele
    the panel declares the reference, from its `REF` column or its
    indel-reference table; without one, or with one that is neither allele, an
    indel names no ALID, so completion neither matches nor fills it.
    """
    parts = panel_id_parts(snp_id)
    if parts is None:
        return None
    chrom, pos_s, first, second = parts
    if len(first) != len(second):
        if reference is None or identity_allele(reference) not in {
            identity_allele(first), identity_allele(second)
        }:
            return None
        alt = second if identity_allele(reference) == identity_allele(first) else first
        first, second = reference, alt
    try:
        return orient_to_canonical(
            chrom, int(pos_s), second, first, other_is_reference=True
        ).variant.alid
    except (VariantNormalisationError, ValueError):
        return None


def panel_variant(alid: str) -> CanonicalVariant | None:
    """The canonical variant a panel ALID (`canonical_panel_alid`'s) names, or
    None where it names none. An ALID read back keeps its order: an indel's A1
    is its reference (ADR 0079), and an SNV's pair is already sorted."""
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos_s, a1, a2 = parts
    try:
        return orient_to_canonical(chrom, int(pos_s), a2, a1, other_is_reference=True).variant
    except (VariantNormalisationError, ValueError):
        return None


#: An indel's key in an indel-reference table: its chromosome, its position,
#: and its two alleles sorted, each as `identity_allele` spells it.
IndelKey = tuple[str, int, str, str]


def indel_key(chromosome: str, position: str | int, first: str, second: str) -> IndelKey | None:
    """The key a table holds an indel under, or None for alleles of one length
    or a site that names none. The table's builder keys the panel's ids and its
    evidence by it, and completion looks each panel indel up by it, so the two
    never spell one key two ways."""
    first, second = identity_allele(first.strip()), identity_allele(second.strip())
    if len(first) == len(second):
        return None
    try:
        chrom = normalise_chromosome(chromosome)
        pos = int(position)
    except (VariantNormalisationError, ValueError):
        return None
    a1, a2 = sorted((first, second))
    return chrom, pos, a1, a2


#: The indel-reference table and its provenance, beside a panel's blocks in
#: ``ld_dir/{ancestry}``: bgzipped and tabix-indexed, one row per panel indel
#: (`scripts/build_panel_indel_reference.py`).
INDEL_REFERENCE_TABLE = "indel_reference.tsv.gz"
INDEL_REFERENCE_PROVENANCE = "indel_reference.json"
#: The table's columns. ``allele1``/``allele2`` are the panel id's (sorted)
#: alleles, ``reference`` one of them or ``.``, and ``outcome`` how it was
#: settled or why not.
INDEL_REFERENCE_COLUMNS = ("chromosome", "position", "allele1", "allele2", "reference", "outcome")


#: What a table's provenance must record: the table's checksum, and the
#: counts and script revision a completion record repeats.
_PROVENANCE_KEYS = ("table_sha256", "counts", "revision")


class IndelReferenceTableError(ValueError):
    """An indel-reference table completion does not read: one with no
    provenance beside it, or one its provenance's checksum does not match."""


def indel_reference_table(ancestry_dir: Path) -> Path | None:
    """The panel's indel-reference table, or None when it has none."""
    path = Path(ancestry_dir) / INDEL_REFERENCE_TABLE
    return path if path.exists() else None


@dataclass(frozen=True)
class _IndelReference:
    """An opened indel-reference table, its checksum and its provenance."""

    tabix: pysam.TabixFile
    sha256: str
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


@lru_cache(maxsize=8)
def _open_indel_reference(path: str) -> _IndelReference:
    """The table at ``path``, once its provenance is found and its checksum
    matches the one the provenance records. A table that cannot say what
    evidence settled it, or is not the table its provenance describes, would
    place indels by a reference nothing vouches for."""
    table = Path(path)
    provenance_path = table.with_name(INDEL_REFERENCE_PROVENANCE)
    if not provenance_path.is_file():
        raise IndelReferenceTableError(
            f"{table} has no {INDEL_REFERENCE_PROVENANCE} beside it, so nothing says "
            "what evidence settled its indels; completion does not read it (ADR 0079)"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise IndelReferenceTableError(f"{provenance_path} is not a JSON object")
    missing = [key for key in _PROVENANCE_KEYS if provenance.get(key) is None]
    if missing:
        raise IndelReferenceTableError(f"{provenance_path} records no {', '.join(missing)}")
    digest = _sha256(table)
    if provenance["table_sha256"] != digest:
        raise IndelReferenceTableError(
            f"{table} has sha256 {digest}, but {provenance_path} records table_sha256 "
            f"{provenance['table_sha256']}: the table is not the one its provenance describes"
        )
    return _IndelReference(pysam.TabixFile(path), digest, provenance)


# An opened table is its process's own. A block pool forks after the parent has
# read the table, and a child reading through the parent's handle shares one file
# offset with the parent and every other child, so their fetches read each other's
# bytes: a fetch fails, splits a line, or ends early and leaves indels unplaced.
os.register_at_fork(after_in_child=_open_indel_reference.cache_clear)


def _table_references(
    table: Path, chrom: str, start_bp: int, end_bp: int
) -> dict[IndelKey, str]:
    """``key -> reference`` for the settled indels the table holds in one
    block's extent."""
    tabix = _open_indel_reference(str(table)).tabix
    if chrom not in tabix.contigs:
        return {}
    settled: dict[IndelKey, str] = {}
    for line in tabix.fetch(chrom, max(start_bp - 1, 0), end_bp):
        chromosome, pos, allele1, allele2, reference, _outcome = line.split("\t")
        key = indel_key(chromosome, pos, allele1, allele2)
        if key is not None and reference != ".":
            settled[key] = reference
    return settled


class PanelPlacement:
    """How a completion's panel blocks placed their rows, for its log and its
    record (`provenance.completion.panel_indels` and `panel_snvs_unplaced`,
    ADR 0079).

    A panel with neither a `REF` column nor an indel-reference table places no
    indel; its blocks are still read, its indels count as missing where a fill
    would land, and the record says why. Every count is summed over LD blocks:
    an indel on a block boundary counts once per block.
    """

    def __init__(self) -> None:
        self.sources: set[str] = set()
        self.tables: set[Path] = set()
        self.n_indels = 0
        self.n_placed = 0
        self.unplaced_indels = Counter({reason: 0 for reason in INDEL_UNPLACED_REASONS})
        self.unplaced_snvs = Counter({reason: 0 for reason in SNV_UNPLACED_REASONS})

    def add(self, block: LDBlock) -> None:
        self.sources.add(block.indel_reference)
        if block.indel_reference == "table":
            table = indel_reference_table(block.tsv_path.parent.parent)
            if table is not None:
                self.tables.add(table)
        self.n_indels += block.n_indels
        self.n_placed += block.n_indels_placed
        self.unplaced_indels.update(block.unplaced_indels)
        self.unplaced_snvs.update(block.unplaced_snvs)

    def record(self) -> dict[str, dict[str, object]]:
        """The completion record's two entries. ``panel_indels``: where the
        references came from, each table read (with its checksum and
        provenance), the indels read and placed, and, per reason, those not
        placed, which sum to the difference. ``panel_snvs_unplaced``: per
        reason, the SNVs and substitutions not placed."""
        source = next(iter(self.sources)) if len(self.sources) == 1 else (
            "mixed" if self.sources else "none"
        )
        return {
            "panel_indels": {
                "reference_from": source,
                "tables": [_table_record(table) for table in sorted(self.tables)],
                "n_indels": self.n_indels,
                "n_placed": self.n_placed,
                "unplaced": dict(self.unplaced_indels),
            },
            "panel_snvs_unplaced": dict(self.unplaced_snvs),
        }

    def log(self, ld_dir: str | Path, ancestry: str) -> None:
        if self.n_indels and not self.n_placed:
            log.warning(
                "LD panel %s/%s names no indel's reference (no REF column, no %s): "
                "its %d indels are neither matched nor filled",
                ld_dir, ancestry, INDEL_REFERENCE_TABLE, self.n_indels,
            )
        elif self.n_indels:
            log.info(
                "LD panel %s/%s: %d of %d indels placed by %s",
                ld_dir, ancestry, self.n_placed, self.n_indels, ", ".join(sorted(self.sources)),
            )
        unplaced_snvs = sum(self.unplaced_snvs.values())
        if unplaced_snvs:
            log.warning(
                "LD panel %s/%s: %d SNV rows are not placed (%s): they record no EA, "
                "one that is neither of their alleles, or an id no ALID can spell",
                ld_dir, ancestry, unplaced_snvs, dict(self.unplaced_snvs),
            )


def _table_record(table: Path) -> dict[str, object]:
    opened = _open_indel_reference(str(table))
    return {
        "path": str(table),
        "sha256": opened.sha256,
        "counts": opened.provenance["counts"],
        "revision": opened.provenance["revision"],
    }


def snp_position(snp_id: str) -> int:
    """Extract the base-pair position from a panel SNP id (either format).

    Returns -1 if it cannot be parsed.
    """
    parts = _strip_chr(snp_id).split(":")
    field = parts[1] if len(parts) >= 2 else parts[0]
    try:
        return int(field.split("_")[0])
    except ValueError:
        return -1


@dataclass(frozen=True)
class _BlockTable:
    """One block TSV's columns: ids, EA frequencies, extent, and, where the
    panel writes them, each row's EA and its REF."""

    snp_ids: list[str]
    eaf: np.ndarray
    min_bp: int
    max_bp: int
    effect: list[str] | None
    reference: list[str] | None


def _float_or_nan(cell: str | None) -> float:
    try:
        return float(cell or "nan")
    except ValueError:
        return float("nan")


def _int_or_zero(cell: str | None) -> int:
    try:
        return int(cell or "")
    except ValueError:
        return 0


def _optional_column(rows: list[dict[str, str]], names: set[str], name: str) -> list[str] | None:
    """One column's cells, empty where a row has none, or None when the block has no such column."""
    if name not in names:
        return None
    return [row.get(name) or "" for row in rows]


def _read_block_tsv(tsv_path: Path) -> _BlockTable:
    """Parse a block TSV."""
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        names = set(reader.fieldnames or ())
        rows = list(reader)
    bps = [_int_or_zero(row.get("BP")) for row in rows] or [0]
    return _BlockTable(
        snp_ids=[row["SNP"] for row in rows],
        eaf=np.array([_float_or_nan(row.get("EAF")) for row in rows], dtype=np.float64),
        min_bp=min(bps),
        max_bp=max(bps),
        effect=_optional_column(rows, names, "EA"),
        reference=_optional_column(rows, names, "REF"),
    )


#: Why a panel row naming a variant is not placed: an indel the panel
#: declares no reference for; a row whose `EA` is empty, or whose block has no
#: `EA` column; a row whose `EA` is neither of its alleles; and a row whose id
#: no ALID can spell (an allele outside the alphabet, say), once any
#: reference it needs is found.
INDEL_UNPLACED_REASONS = (
    "no_reference", "no_effect_allele", "effect_allele_not_an_allele", "identity_invalid",
)
SNV_UNPLACED_REASONS = ("no_effect_allele", "effect_allele_not_an_allele", "identity_invalid")


@dataclass(frozen=True)
class _BlockAlleles:
    alids: list[str | None]
    flipped: np.ndarray
    source: str
    n_indels: int
    n_placed: int
    unplaced_indels: dict[str, int]
    unplaced_snvs: dict[str, int]


@dataclass(frozen=True)
class _Row:
    """One panel row's placement: its ALID or None, whether its A1 is not its
    EA, whether it is an indel, and why a row naming a variant is not placed."""

    alid: str | None
    flipped: bool
    indel: bool
    unplaced: str | None


def _block_alleles(
    table: _BlockTable, chrom: str, indel_table: Path | None
) -> _BlockAlleles:
    """Each row's ALID and whether its A1 is not its EA (`_place_row`).

    An indel's reference is the row's `REF`, where the block has the column,
    else the panel's indel-reference table.
    """
    source = _reference_source(table, indel_table)
    settled: Mapping[IndelKey, str] = {}
    if source == "table" and indel_table is not None:
        settled = _table_references(indel_table, chrom, table.min_bp, table.max_bp)
    return _tallied([_place_row(table, i, settled) for i in range(len(table.snp_ids))], source)


def _tallied(rows: list[_Row], source: str) -> _BlockAlleles:
    """A block's rows, with its indels counted and its unplaced rows tallied by
    reason, indels apart from SNVs."""
    unplaced = {
        True: Counter({reason: 0 for reason in INDEL_UNPLACED_REASONS}),
        False: Counter({reason: 0 for reason in SNV_UNPLACED_REASONS}),
    }
    n_indels = n_placed = 0
    for row in rows:
        n_indels += row.indel
        n_placed += row.indel and row.alid is not None
        if row.unplaced is not None:
            unplaced[row.indel][row.unplaced] += 1
    return _BlockAlleles(
        alids=[row.alid for row in rows],
        flipped=np.array([row.flipped for row in rows], dtype=bool),
        source=source,
        n_indels=n_indels,
        n_placed=n_placed,
        unplaced_indels=dict(unplaced[True]),
        unplaced_snvs=dict(unplaced[False]),
    )


def _place_row(table: _BlockTable, row: int, settled: Mapping[IndelKey, str]) -> _Row:
    """One row's placement.

    A row is placed only where its `EA`, which its LD and `EAF` are coded on,
    is one of its alleles (`_oriented_row`). A row with no `EA` -- an empty
    cell, or a block with no `EA` column -- places nothing, SNV or indel,
    since nothing says which allele its LD is for.
    """
    snp_id = table.snp_ids[row]
    parts = panel_id_parts(snp_id)
    if parts is None:
        return _Row(None, False, False, None)
    indel = len(parts[2]) != len(parts[3])
    effect = _effect_allele(table, row)
    if not effect:
        return _Row(None, False, indel, "no_effect_allele")
    if not indel:
        return _oriented_row(canonical_panel_alid(snp_id), effect, indel=False)
    reference = _row_reference(table, row, parts, settled)
    if reference is None:
        return _Row(None, False, True, "no_reference")
    return _oriented_row(canonical_panel_alid(snp_id, reference), effect, indel=True)


def _effect_allele(table: _BlockTable, row: int) -> str:
    """A row's `EA` as identity spells it, ``""`` where its cell is empty or
    the block records none."""
    return "" if table.effect is None else identity_allele(table.effect[row].strip())


def _oriented_row(alid: str | None, effect: str, *, indel: bool) -> _Row:
    """A row's placement once its id is read: not placed where no ALID can
    spell it or its `EA` is neither allele, else flipped where its `EA` is its
    ALID's A2."""
    if alid is None:
        return _Row(None, False, indel, "identity_invalid")
    a1, a2 = alid.split(":")[2:]
    if effect not in (a1, a2):
        return _Row(None, False, indel, "effect_allele_not_an_allele")
    return _Row(alid, effect != a1, indel, None)


def _reference_source(table: _BlockTable, indel_table: Path | None) -> str:
    """Where a block's indels take their reference: its `REF` column, the
    panel's table, or nowhere."""
    if table.reference is not None:
        return "ref_column"
    return "none" if indel_table is None else "table"


def _row_reference(
    table: _BlockTable,
    row: int,
    parts: tuple[str, str, str, str],
    settled: Mapping[IndelKey, str],
) -> str | None:
    """One indel row's declared reference: its `REF` cell, else the table's.
    None where neither names one of the row's two alleles."""
    if table.reference is not None:
        reference: str | None = identity_allele(table.reference[row].strip())
    else:
        key = indel_key(*parts)
        reference = None if key is None else settled.get(key)
    alleles = {identity_allele(parts[2]), identity_allele(parts[3])}
    return reference if reference in alleles else None


def panel_indel_frequencies(ld_dir: str | Path, ancestry: str) -> dict[str, float]:
    """``{ALID: A1 frequency}`` of every indel the panel places (ADR 0079).

    The panel's `EAF` is for its `EA`, and a placed indel's A1 is its reference,
    so a deletion's frequency is complemented (`LDBlock.eaf_for_a1`), as its
    eigenvector row is re-signed. A frequency outside ``[0, 1]`` is none.
    """
    frequencies: dict[str, float] = {}
    for chrom in list_chromosomes(ld_dir, ancestry):
        for block in list_all_blocks(ld_dir, ancestry, chrom):
            for alid, eaf in zip(block.alids, block.eaf_for_a1().tolist(), strict=True):
                if alid is None or not 0.0 <= eaf <= 1.0:
                    continue
                _chrom, _pos, a1, a2 = alid.split(":")
                if len(a1) != len(a2):
                    frequencies[alid] = eaf
    return frequencies


def load_block(tsv_path: Path) -> LDBlock | None:
    """Load one LD block from its TSV path. Returns None if unreadable/incomplete.

    A block is loadable when it can yield eigenvectors by either route: a stored
    eigendecomposition, or an LD matrix to derive one from. Requiring the matrix
    would make eigendecomposition-only panels silently empty.

    chrom is taken from the parent directory name, matching the panel's
    ld_dir/{ancestry}/{chr}/{block_name}.tsv layout.
    """
    tsv_path = Path(tsv_path)
    ld_path = tsv_path.with_suffix(".unphased.vcor1.gz")
    npz_path = tsv_path.with_suffix(".ldeig.npz")
    has_ld = ld_path.exists()
    has_npz = npz_path.exists()
    if not (has_ld or has_npz):
        return None

    try:
        table = _read_block_tsv(tsv_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s: %s — skipping", tsv_path, exc)
        return None

    chrom = tsv_path.parent.name
    alleles = _block_alleles(table, chrom, indel_reference_table(tsv_path.parent.parent))
    return LDBlock(
        block_id=f"{chrom}/{tsv_path.stem}",
        chrom=chrom,
        start_bp=table.min_bp,
        end_bp=table.max_bp,
        tsv_path=tsv_path,
        ld_path=ld_path if has_ld else None,
        ldeig_npz_path=npz_path if has_npz else None,
        n_ld_snps=len(table.snp_ids),
        snp_ids=table.snp_ids,
        eaf=table.eaf,
        alids=alleles.alids,
        flipped=alleles.flipped,
        indel_reference=alleles.source,
        n_indels=alleles.n_indels,
        n_indels_placed=alleles.n_placed,
        unplaced_indels=alleles.unplaced_indels,
        unplaced_snvs=alleles.unplaced_snvs,
    )


def blocks_over_variants(
    ld_dir: str | Path,
    ancestry: str,
    analyses_by_alid: Mapping[str, Sequence[int]],
    chromosomes: Iterable[str],
    *,
    min_observed: int | None = None,
) -> dict[int, list[LDBlock]]:
    """Per Analysis, the LD blocks it already holds enough observations in.

    The block-enumeration path for an Analysis with no Trait position to scan a
    cis window around -- a Store Family with no single encoding gene per
    Analysis has no `trait_chr`/`trait_bp` by design, and completion was a
    silent no-op for every one of them (issue #102). Where the cis path asks
    "which blocks are near this Analysis's gene", this asks "which blocks does
    this Analysis already have enough of a region in to impute from".

    `analyses_by_alid` maps a canonical ALID to the Analyses observing it, so
    the count is over **the Analysis's observations at the block's own panel
    SNPs** -- the same set `completion.block.run_block` counts when it decides
    whether it can fit. Counting positions inside the block's base-pair extent
    instead would include off-panel variants and enumerate blocks that then
    impute nothing.

    `min_observed` defaults to `impute.min_observed_points()`: below it,
    `poly_rescale` returns NaN and imputation is rejected, so enumerating the
    block would add its panel variants to the axis as missing rows and fill
    none of them. Spec §17's "do not expand singleton suggestive associations"
    falls out of the same threshold.

    Blocks are read one chromosome at a time and only the selected ones are
    retained: a whole panel's blocks carry every SNP id and EAF they hold, and
    the Store Families this path exists for span the genome.
    """
    if min_observed is None:
        min_observed = min_observed_points()
    per_analysis: dict[int, list[LDBlock]] = defaultdict(list)
    if not analyses_by_alid:
        return {}
    for chrom in chromosomes:
        for block in list_all_blocks(ld_dir, ancestry, chrom):
            observed: Counter[int] = Counter()
            for alid in block.alids:
                if alid is None:
                    continue
                for analysis_index in analyses_by_alid.get(alid, ()):
                    observed[analysis_index] += 1
            for analysis_index, n_observed in observed.items():
                if n_observed >= min_observed:
                    per_analysis[analysis_index].append(block)
    return dict(per_analysis)


@dataclass(frozen=True)
class SharedEndpoint:
    """The position a block ends at and the next block starts at, and the
    variants both blocks hold there, which the next block owns."""

    following: str
    alids: frozenset[str]


def shared_endpoints(blocks: Iterable[LDBlock]) -> dict[str, SharedEndpoint]:
    """Each block's end that one other block starts at, keyed by the ending block.

    Blocks are half-open, ``[start, end)``: a variant at a position one block
    ends at and the next starts at belongs to the block that starts there, and
    only that block fills it; it stays in both blocks' LD context (ADR 0072).
    A block no other block follows, such as a chromosome's last, keeps its end.
    The panels completion reads put a shared endpoint in both blocks, since
    they were cut with an inclusive end. A variant two blocks hold anywhere
    else is no endpoint of theirs, and a completion resolves its two fills by
    the duplicate rule (D5).

    A block's extent is the span of its variants' positions, so an endpoint is
    a position both blocks hold a variant at.
    """
    by_chromosome: dict[str, list[LDBlock]] = defaultdict(list)
    for block in blocks:
        by_chromosome[block.chrom].append(block)
    shared: dict[str, SharedEndpoint] = {}
    for chromosome_blocks in by_chromosome.values():
        shared.update(_chromosome_endpoints(chromosome_blocks))
    return shared


def ceded_at_end(blocks: Iterable[LDBlock]) -> dict[str, frozenset[str]]:
    """Per block, the variants at its end the next block owns, for a completion
    that completes every Analysis against every block (`shared_endpoints`)."""
    return {block_id: end.alids for block_id, end in shared_endpoints(blocks).items()}


def _chromosome_endpoints(blocks: list[LDBlock]) -> dict[str, SharedEndpoint]:
    starting: dict[int, list[LDBlock]] = defaultdict(list)
    for block in blocks:
        starting[block.start_bp].append(block)
    shared: dict[str, SharedEndpoint] = {}
    for block in blocks:
        following = [b for b in starting.get(block.end_bp, ()) if b.block_id != block.block_id]
        if len(following) != 1:
            continue
        (successor,) = following
        alids = _alids_at(block, block.end_bp) & _alids_at(successor, successor.start_bp)
        if alids:
            shared[block.block_id] = SharedEndpoint(successor.block_id, frozenset(alids))
    return shared


def _alids_at(block: LDBlock, position: int) -> set[str]:
    """The canonical ALIDs a block holds at one position."""
    return {
        alid
        for snp, alid in zip(block.snp_ids, block.alids, strict=True)
        if snp_position(snp) == position and alid is not None
    }


def list_chromosomes(ld_dir: str | Path, ancestry: str) -> list[str]:
    """Return the chromosome names present in the panel for *ancestry*."""
    ancestry_dir = Path(ld_dir) / ancestry
    if not ancestry_dir.is_dir():
        return []
    return sorted(
        (p.name for p in ancestry_dir.iterdir() if p.is_dir()),
        key=lambda c: (len(c), c),
    )


class LdPanelNotFoundError(Exception):
    """``ld_dir``/``ancestry`` has no chromosome directories to read blocks from."""


def check_panel_has_chromosomes(ld_dir: str | Path, ancestry: str) -> None:
    """Raise if the panel has no chromosome directory for *ancestry*.

    `list_all_blocks` treats a missing `ld_dir/{ancestry}/{chr}`
    as "no blocks here" and returns `[]` -- correct when scanning one
    chromosome a real panel simply has nothing on, wrong when `ld_dir` or
    `ancestry` is misconfigured: every chromosome then reports the same empty
    result, and completion runs to a successful finish with the "0 imputed"
    line reading like a genuine finding rather than a broken path. That is the
    same completes-and-contains-nothing shape ADR 0039 closed for ancestry
    mismatches and gene-target-less Store Families, reached here by a
    different route -- called once, up front, so it fails before the
    LD-block scan rather than after it.
    """
    ancestry_dir = Path(ld_dir) / ancestry
    if not list_chromosomes(ld_dir, ancestry):
        hint = ""
        if Path(ld_dir).name == ancestry:
            hint = (
                f" ld_dir={str(ld_dir)!r} already ends in the ancestry directory -- "
                "ld_dir must be the panel *root* (blocks live at "
                "ld_dir/{ancestry}/{chr}/...), not the ancestry directory itself."
            )
        raise LdPanelNotFoundError(
            f"No chromosome directories under {ancestry_dir} (ld_dir={str(ld_dir)!r}, "
            f"ancestry={ancestry!r}) -- completion would scan every chromosome, find "
            "no blocks anywhere, and finish reporting 0 imputed as if that were a "
            f"genuine result.{hint}"
        )


def list_all_blocks(ld_dir: str | Path, ancestry: str, chrom: str) -> list[LDBlock]:
    """Return every LD block for one chromosome, genome-wide (no region filter).

    Used for `gwas`-coverage Dense completion, where the block set is the
    reference panel's entire block list rather than a per-analysis cis window.
    """
    panel_dir = Path(ld_dir) / ancestry / chrom
    if not panel_dir.is_dir():
        return []

    blocks: list[LDBlock] = []
    for tsv_path in sorted(panel_dir.glob("*.tsv")):
        block = load_block(tsv_path)
        if block is not None:
            blocks.append(block)
    return blocks


def load_ld_eigenvectors(
    block: LDBlock,
    thresh: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (eigenvalues, eigenvectors) for a block, truncated to *thresh* variance.

    Reads the stored eigendecomposition; falls back to deriving one from the LD
    matrix only when no decomposition is present or it cannot be read.

    A panel stores a finite number of eigenvectors. When that number is too small
    to reach *thresh*, this returns what is stored and warns — proceeding silently
    would mean imputing against less variance than the caller asked for, with the
    shortfall visible only as degraded results (spec §13.1).
    """
    if block.ldeig_npz_path is not None:
        try:
            data = np.load(str(block.ldeig_npz_path))
            vals: np.ndarray = data["values"].astype(np.float64)
            vecs: np.ndarray = data["vectors"].astype(np.float64)
            clamped = np.maximum(vals, 0)
            total = float(clamped.sum()) or 1.0
            cumvar = np.cumsum(clamped) / total
            k_wanted = int(np.searchsorted(cumvar, thresh)) + 1
            k = min(k_wanted, vecs.shape[1])
            if k_wanted > vecs.shape[1]:
                log.warning(
                    "Block %s stores %d eigenvectors but %d are needed for %.3f "
                    "variance; using %.4f — panel is under-resolved for this "
                    "truncation",
                    block.block_id, vecs.shape[1], k_wanted, thresh,
                    float(cumvar[k - 1]) if k else 0.0,
                )
            return vals[:k], vecs[:, :k]
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not load %s: %s — falling back to LD matrix",
                block.ldeig_npz_path, exc,
            )

    return _load_from_ld_matrix(block, thresh)


def _load_from_ld_matrix(block: LDBlock, thresh: float) -> tuple[np.ndarray, np.ndarray]:
    if block.ld_path is None:
        raise RuntimeError(
            f"Block {block.block_id} has neither a readable eigendecomposition nor "
            f"an LD matrix to derive one from"
        )
    import pandas as pd
    try:
        full = pd.read_csv(
            block.ld_path, header=None, delimiter="\t"
        ).values.astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Cannot load LD matrix {block.ld_path}: {exc}") from exc
    return ld_pca(full, thresh)

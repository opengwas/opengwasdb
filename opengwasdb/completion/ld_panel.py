"""Read-only interface for the LD reference panel used in Reference Completion.

Panel layout (flat/production):
    ld_dir/{ancestry}/{chr}/{block_name}.tsv              (required)
    ld_dir/{ancestry}/{chr}/{block_name}.ldeig.npz        (eigendecomposition)
    ld_dir/{ancestry}/{chr}/{block_name}.unphased.vcor1.gz (optional LD matrix)

Completion consumes eigenvectors only, so the eigendecomposition is the primary
artifact (ADR 0031, store-format spec §13.1). A block loads when it has a variant
table plus *either* an eigendecomposition or an LD matrix to derive one from; the
matrix alone is retained for panels predating that contract.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.completion.impute import ld_pca, min_observed_points
from opengwasdb.variants import VariantNormalisationError, orient_to_canonical

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
    eaf: np.ndarray     # float64, length n_ld_snps; NaN where absent


def _strip_chr(s: str) -> str:
    return s[3:] if s.startswith("chr") else s


def canonical_panel_alid(snp_id: str) -> str | None:
    """Normalise an LD-panel SNP id to a canonical store ALID (``chr:pos:a1:a2``).

    Handles both the legacy ``[chr]CHR:POS_REF_ALT`` form (underscores, as in the
    UKB EUR panel) and an already-colon-delimited ``CHR:POS:A1:A2`` form, orienting
    alleles to the canonical A1 = min(ref, alt). Returns None for ids that don't
    parse — callers treat that as "not matchable" rather than an error, so a panel
    carrying a stray malformed row does not fail a whole block.
    """
    s = _strip_chr(snp_id)
    parts = s.split(":")
    if len(parts) == 4:
        chrom, pos_s, ref, alt = parts
    elif len(parts) == 2 and parts[1].count("_") == 2:
        chrom, rest = parts
        pos_s, ref, alt = rest.split("_")
    else:
        return None
    try:
        return orient_to_canonical(chrom, int(pos_s), ref, alt).variant.alid
    except (VariantNormalisationError, ValueError):
        return None


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


def _read_block_tsv(tsv_path: Path) -> tuple[list[str], np.ndarray, int, int]:
    """Parse a block TSV and return (snp_ids, eaf, min_bp, max_bp)."""
    import csv

    snp_ids: list[str] = []
    eafs: list[float] = []
    bps: list[int] = []

    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            snp_ids.append(row["SNP"])
            try:
                eafs.append(float(row.get("EAF") or "nan"))
            except ValueError:
                eafs.append(float("nan"))
            try:
                bps.append(int(row["BP"]))
            except (KeyError, ValueError):
                bps.append(0)

    eaf_arr = np.array(eafs, dtype=np.float64)
    min_bp = int(min(bps)) if bps else 0
    max_bp = int(max(bps)) if bps else 0
    return snp_ids, eaf_arr, min_bp, max_bp


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
        snp_ids, eaf, blk_start, blk_end = _read_block_tsv(tsv_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s: %s — skipping", tsv_path, exc)
        return None

    chrom = tsv_path.parent.name
    return LDBlock(
        block_id=f"{chrom}/{tsv_path.stem}",
        chrom=chrom,
        start_bp=blk_start,
        end_bp=blk_end,
        tsv_path=tsv_path,
        ld_path=ld_path if has_ld else None,
        ldeig_npz_path=npz_path if has_npz else None,
        n_ld_snps=len(snp_ids),
        snp_ids=snp_ids,
        eaf=eaf,
    )


def find_blocks(
    ld_dir: str | Path,
    ancestry: str,
    chrom: str,
    start_bp: int,
    end_bp: int,
) -> list[LDBlock]:
    """Return all LD blocks whose genomic extent intersects [start_bp, end_bp]."""
    blocks: list[LDBlock] = []
    for block in list_all_blocks(ld_dir, ancestry, chrom):
        if block.end_bp < start_bp or block.start_bp > end_bp:
            continue
        blocks.append(block)
    return blocks


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
            for snp_id in block.snp_ids:
                alid = canonical_panel_alid(snp_id)
                if alid is None:
                    continue
                for analysis_index in analyses_by_alid.get(alid, ()):
                    observed[analysis_index] += 1
            for analysis_index, n_observed in observed.items():
                if n_observed >= min_observed:
                    per_analysis[analysis_index].append(block)
    return dict(per_analysis)


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

    `list_all_blocks`/`find_blocks` treat a missing `ld_dir/{ancestry}/{chr}`
    as "no blocks here" and return `[]` -- correct when scanning one
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

    Used for Full Coverage Dense completion, where the block set is the
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

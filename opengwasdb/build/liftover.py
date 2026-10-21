"""Batch hg19 → hg38 liftover utility for the build pipelines.

Converts a collection of (bare_chrom, pos, ref, alt) tuples in hg19
coordinates to hg38 coordinates, and to ALIDs, in a single pass using one
LiftOver object.

A chain block has an orientation, and on a reverse-strand block the source's
alleles are read on the other strand of the target. Until issue #284 this lift
ignored that: it took the target coordinate and kept the source's alleles,
so every reverse-strand SNP was stored under an hg38 position carrying
hg19-strand alleles -- a variant the resolver's `oriented_variant` (which does
complement) could never match. The projection now comes from that same helper
(`opengwasdb.readers.gwas_vcf.project_lifted_alleles`), so the two lifts agree
by construction rather than by coincidence.
"""

from __future__ import annotations

import gzip
import itertools
import logging
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengwasdb.readers.gwas_vcf import project_lifted_alleles
from opengwasdb.variants.axis import parse_canonical_alid

log = logging.getLogger(__name__)

#: The other-strand spelling of an allele, for reading a reference's variant
#: as the same variant on the opposite strand.
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

_BUILD_ALIASES: dict[str, str] = {
    "hg19": "hg19", "grch37": "hg19", "b37": "hg19", "37": "hg19",
    "hg38": "hg38", "grch38": "hg38", "b38": "hg38", "38": "hg38",
}

#: A source variant as the lift is given it: ``(chrom, pos, ref, alt)``.
SourceVariantKey = tuple[str, int, str, str]


class LiftoverFailureError(RuntimeError):
    """Raised when the liftover failure rate exceeds the configured threshold."""


class LiftedAlleleMismatchError(ValueError):
    """A lifted variant's alleles disagree with the reference at its hg38 position.

    The rule the issue asked for: the lift moved the variant, and the reference
    is the authority on what lives at the position it moved to. A disagreement
    means the lift produced a variant that does not exist -- the shape of
    failure #284 was, at every reverse-strand SNP.
    """


def normalise_build(build: str) -> str:
    canonical = _BUILD_ALIASES.get(build.lower().strip())
    if canonical is None:
        raise ValueError(
            f"Unknown genome build {build!r}. "
            "Use hg19/hg38 or aliases GRCh37/GRCh38, b37/b38, 37/38."
        )
    return canonical


@dataclass(frozen=True)
class LiftedRow:
    """One lifted source variant: its hg38 coordinate and projected alleles.

    ``effect_allele``/``other_allele`` are the source's ``ref``/``alt`` as they
    read *on the target assembly*, which on a reverse-strand chain block means
    complemented. They are kept in source order -- not sorted -- because the
    caller needs to know which of them the source called the effect allele in
    order to sign its statistics (``orient_to_canonical().flipped``).

    ``strand`` is the chain block's orientation, recorded rather than consumed:
    the allele check reads it to know which spelling of a variant the reference
    *should* carry, so a lift that stops projecting its alleles still says what
    it was supposed to do (issue #284).
    """

    chromosome: str
    position: int
    effect_allele: str
    other_allele: str
    strand: str = "+"

    @property
    def alid(self) -> str:
        a1, a2 = sorted((self.effect_allele, self.other_allele))
        return f"{self.chromosome}:{self.position}:{a1}:{a2}"


def statistics_swap(source: SourceVariantKey, row: LiftedRow) -> bool:
    """Whether the lift swapped which allele is canonical, so z and EAF must follow.

    A source reader signs a statistic against the source-assembly canonical A1
    (``min(ref, alt)``, see `stream_vcf_associations`). A reverse-strand block
    complements both alleles, and a complement reverses their alphabetical
    order -- or, for a palindromic pair, maps A1 onto A2 -- so the projected
    alleles a Dense or Hybrid build stores may call the *other* allele A1. The
    stored z is then the negation of the source's and the stored frequency is
    ``1 - eaf``, exactly as BESD orients a probe (issue #284).

    Computed from the row's projected alleles in source order, so it is exact
    for an indel too rather than assuming complement always reverses order.
    """
    ref, alt = source[2], source[3]
    source_first = min(ref, alt)
    projected_first = row.effect_allele if source_first == ref else row.other_allele
    return projected_first != min(row.effect_allele, row.other_allele)


def build_liftover_lookup(
    variants: Iterable[SourceVariantKey],
    from_build: str = "hg19",
    to_build: str = "hg38",
    failure_threshold: float = 0.01,
    chain_file: str | Path | None = None,
) -> dict[SourceVariantKey, str]:
    """Convert hg19 (bare_chrom, pos, ref, alt) tuples to hg38 ALIDs.

    Returns a dict mapping each input tuple to its hg38 ALID
    (``chr:pos:A1:A2`` where A1 = alphabetically first allele), with the
    alleles projected onto the mapping's strand (issue #284). Variants that
    fail liftover are omitted; a warning is logged.

    When the failure rate exceeds ``failure_threshold``, ``LiftoverFailureError``
    is raised before any output is returned.

    The ``LiftOver`` object is cached per process (`_liftover_engine`), so a
    batch of window-task workers each loads the chain once (issue #197).

    Parameters
    ----------
    variants:
        Iterable of ``(bare_chrom, pos, ref, alt)`` tuples.  Both bare (``1``)
        and chr-prefixed (``chr1``) chrom forms are accepted.
    from_build / to_build:
        Genome build identifiers.  Ignored when ``chain_file`` is provided.
    failure_threshold:
        Maximum allowed fraction of failures before raising.  Default 0.01 (1%).
    chain_file:
        Path to a chain file.  When provided, ``LiftOver`` is initialised from
        the file rather than by downloading by build name.
    """
    lifted, attempts, failures = lifted_rows(
        variants, from_build=from_build, to_build=to_build, chain_file=chain_file
    )
    enforce_liftover_threshold(
        attempts,
        failures,
        from_build=normalise_build(from_build),
        to_build=normalise_build(to_build),
        failure_threshold=failure_threshold,
    )
    return {key: row.alid for key, row in lifted.items()}


def lift_hg19_to_hg38(
    variants: Iterable[SourceVariantKey],
    *,
    chain_file: str | Path | None = None,
    failure_threshold: float = 0.01,
) -> dict[SourceVariantKey, LiftedRow]:
    """`lifted_rows` for the hg19→hg38 lift every builder performs, threshold applied.

    The pair -- lift, then refuse a batch whose failure rate exceeds the
    threshold -- is one operation for every caller that lifts a source union
    (the Dense and Hybrid Pass 1, the Hybrid off-reference fold and the BESD
    builder), so it lives here rather than being spelled out four times.
    """
    rows, attempts, failures = lifted_rows(
        variants, from_build="hg19", to_build="hg38", chain_file=chain_file
    )
    enforce_liftover_threshold(
        attempts,
        failures,
        from_build="hg19",
        to_build="hg38",
        failure_threshold=failure_threshold,
    )
    return rows


def enforce_liftover_threshold(
    attempts: int,
    failures: int,
    *,
    from_build: str,
    to_build: str,
    failure_threshold: float,
) -> None:
    """Log the failure rate and raise when it exceeds ``failure_threshold``.

    The one place the threshold rule lives: `build_liftover_lookup` applies it
    to a whole batch, and a builder that lifts through `lifted_rows` (for the
    per-row strand it needs) applies it here instead, rather than restating it.
    """
    if not (attempts and failures):
        return
    rate = failures / attempts
    log.warning(
        "Liftover %s→%s: %d/%d variants failed (%.1f%%)",
        from_build, to_build, failures, attempts, rate * 100,
    )
    if rate > failure_threshold:
        raise LiftoverFailureError(
            f"Liftover failure rate {rate:.1%} ({failures}/{attempts}) exceeds "
            f"threshold {failure_threshold:.1%}"
        )


#: One ``LiftOver`` per process, keyed by build pair and chain file. A worker
#: pool that lifts one window per task must not reload the chain each task.
_LIFTOVER_CACHE: dict[tuple[str, str, str | None], Any] = {}


def _liftover_engine(from_build: str, to_build: str, chain_file: str | Path | None) -> Any:
    """The cached ``LiftOver`` for this build pair (issue #197)."""
    key = (
        normalise_build(from_build),
        normalise_build(to_build),
        str(chain_file) if chain_file is not None else None,
    )
    engine = _LIFTOVER_CACHE.get(key)
    if engine is None:
        try:
            from pyliftover import LiftOver  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyliftover is required for coordinate liftover: pip install pyliftover"
            ) from exc
        engine = LiftOver(str(chain_file)) if chain_file is not None else LiftOver(key[0], key[1])
        _LIFTOVER_CACHE[key] = engine
    return engine


def lifted_rows(
    variants: Iterable[SourceVariantKey],
    *,
    from_build: str = "hg19",
    to_build: str = "hg38",
    chain_file: str | Path | None = None,
) -> tuple[dict[SourceVariantKey, LiftedRow], int, int]:
    """Lift a batch of source tuples; return ``(rows, attempts, failures)``.

    ``rows`` holds only the successful entries, each with its hg38 coordinate
    and its alleles projected onto the mapping's strand (issue #284) -- what a
    caller needs to sign statistics, which an ALID string alone cannot say.
    The failure *rate* is deliberately not enforced here: a caller that lifts
    one window per worker aggregates the counts across workers and enforces the
    threshold once, before writing any artifact (issue #197).

    Where a position maps under more than one chain block the highest-scoring
    mapping is taken, which is this lift's long-standing behaviour. The
    resolver's `oriented_variant` refuses such a row instead (ADR 0044); that
    difference belongs to the resolver's one-variant-at-a-time contract and is
    left as it was.
    """
    engine = _liftover_engine(from_build, to_build, chain_file)
    result: dict[SourceVariantKey, LiftedRow] = {}
    attempts = 0
    failures = 0
    for chrom, pos, ref, alt in variants:
        attempts += 1
        bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
        mapped = engine.convert_coordinate(f"chr{bare}", pos - 1)  # 1-based → 0-based
        if not mapped:
            failures += 1
            continue
        new_chrom_full: str = mapped[0][0]
        new_pos = int(mapped[0][1]) + 1  # 0-based → 1-based
        strand: str = mapped[0][2]
        new_bare = new_chrom_full
        if new_bare.lower().startswith("chr"):
            new_bare = new_bare[3:]
        effect, other = project_lifted_alleles(strand, ref, alt)
        result[(chrom, pos, ref, alt)] = LiftedRow(new_bare, new_pos, effect, other, strand)
    return result, attempts, failures


def liftover_batch(
    variants: Iterable[SourceVariantKey],
    *,
    from_build: str = "hg19",
    to_build: str = "hg38",
    chain_file: str | Path | None = None,
) -> tuple[dict[SourceVariantKey, str], int, int]:
    """Lift a batch of hg19 tuples; return ``(mapping, attempts, failures)``.

    ``mapping`` holds only the successful raw-tuple -> hg38 ALID entries, as
    `lifted_rows` computes them.
    """
    rows, attempts, failures = lifted_rows(
        variants, from_build=from_build, to_build=to_build, chain_file=chain_file
    )
    return {key: row.alid for key, row in rows.items()}, attempts, failures


# ── Reference alleles at the lifted position (issue #284) ────────────────────

#: A position on the reference assembly: ``(bare chrom, 1-based pos)``.
PositionKey = tuple[str, int]
#: The unordered allele pair a reference variant carries at one position.
AllelePair = frozenset[str]


def read_reference_alleles(
    path: str | Path, positions: Collection[PositionKey]
) -> dict[PositionKey, set[AllelePair]]:
    """The allele pairs a Variant Reference carries at `positions`, streamed.

    Reads any of the Variant Reference shapes the Dense builders already accept
    -- a plain ALID list, a ``*.variant-ref.tsv.gz`` artifact, or a Store
    Variant Table (``alid`` column) -- by its ``alid`` column when it has a
    header and by each line's first field otherwise, exactly as
    `opengwasdb.variants.reference` classifies them.

    Only the requested positions are retained. A nine-million-variant panel
    streamed against a build's lifted set is a dict the size of the
    intersection, not of the panel: the alternative is either gigabytes of
    source keys or no check at all.

    The reference must be on the release's Reference Assembly; nothing in an
    ALID says which assembly it is on, so a caller that hands over an hg19 list
    gets the disagreement it asked for.
    """
    wanted = set(positions)
    pairs: dict[PositionKey, set[AllelePair]] = {}
    if not wanted:
        return pairs
    reference_path = Path(path)
    with _open_text(reference_path) as handle:
        first_line = handle.readline()
        # A tab-separated first line is a header, commented or not (that is how
        # `read_variant_reference` classifies these shapes). It is consumed
        # here, once: rewinding and re-reading it would offer "alid" to the
        # parser as a variant.
        header = "\t" in first_line
        alid_column = _alid_column(first_line, reference_path) if header else None
        lines = handle if header else itertools.chain([first_line], handle)
        for line_number, line in enumerate(lines, start=2 if header else 1):
            token = _reference_alid_token(line, alid_column)
            if token is None:
                continue
            parsed = parse_canonical_alid(token)
            if parsed is None:
                raise ValueError(
                    f"reference {reference_path} line {line_number} is not a canonical "
                    f"ALID: {token!r}"
                )
            key = (parsed.chromosome, parsed.position)
            if key in wanted:
                pairs.setdefault(key, set()).add(
                    frozenset({parsed.effect_allele, parsed.other_allele})
                )
    return pairs


def _open_text(path: Path) -> Any:
    if str(path).lower().endswith((".gz", ".bgz", ".bgzf")):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _alid_column(first_line: str, path: Path) -> int:
    """Which field of a tab-separated header holds the ALID.

    A tab-separated table with no ``alid`` column is refused rather than read
    positionally, because guessing which column holds the identity is how a
    build checks itself against the wrong thing.
    """
    header = [field.lstrip("#").strip() for field in first_line.rstrip("\n").split("\t")]
    if "alid" not in header:
        raise ValueError(
            f"reference {path} is tab-separated but has no 'alid' column: {header}"
        )
    return header.index("alid")


def _reference_alid_token(line: str, alid_column: int | None) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = line.rstrip("\n").split("\t")
    if alid_column is None:
        return fields[0].strip() or None
    field = fields[alid_column].strip() if alid_column < len(fields) else ""
    return field or None


@dataclass(frozen=True)
class LiftedAlleleCheck:
    """What one build's lifted alleles were checked against, and what happened.

    `n_verified` are the lifted variants the reference names with the same
    allele pair; `n_other_variant` the positions where it names a *different*
    variant (a statement about the reference's coverage, not about the lift);
    `n_absent` the positions it says nothing about; `n_strand_contradicted` the
    lifted variants it spells on the other strand. Only the last is a defect.
    """

    n_lifted: int
    n_verified: int
    n_other_variant: int
    n_absent: int
    n_strand_contradicted: int = 0


def _complement(pair: frozenset[str]) -> frozenset[str]:
    return frozenset(allele.translate(_COMPLEMENT) for allele in pair)


def require_lifted_alleles_match_reference(
    lifted: Mapping[SourceVariantKey, LiftedRow],
    reference_path: str | Path,
) -> LiftedAlleleCheck:
    """Fail when a reference spells a lifted variant the way the other strand does.

    The validation rule of issue #284, stated so that it can only fire on
    positive evidence. A Variant Reference is a list of variants, not a genome:
    a position it names with a *different* allele pair says nothing about
    whether this lift's variant exists -- the pair may be a second allele at a
    multi-allelic site, or the site's indel. What it cannot be is silent about
    is the case a strand-blind lift produced: there the reference holds the
    other-strand spelling of the very variant the lift stored (``A:G`` where the
    lift wrote ``C:T``), which is a disagreement about the strand and nothing
    else.

    Measured over eQTLGen's 8,932,843 `.esi` rows: the pre-#284 lift
    contradicted the reference at 1,654 variants against the EUR alid panel
    (6,769 against gnomAD v4 common), and the stratum-projected lift at none,
    while the "different variant at this position" count is identical (2,856 and
    132,671) before and after -- which is why that class is counted rather than
    failed.

    Only rows a reverse-strand block produced are checked. A forward-strand
    row's alleles are the source's own, so there is no projection for a
    reference to contradict, and a complementary pair at its position is an
    ordinary second variant at a multi-allelic site. The row's recorded strand
    is what decides -- not whether the alleles happen to differ from the
    source's -- so a lift that stops projecting still fails here rather than
    sliding into the "different variant" count.
    """
    pairs = read_reference_alleles(
        reference_path, [(row.chromosome, row.position) for row in lifted.values()]
    )
    mismatches: list[str] = []
    n_verified = 0
    n_other_variant = 0
    for source, row in lifted.items():
        expected = pairs.get((row.chromosome, row.position))
        if expected is None:
            continue
        stored = frozenset({row.effect_allele, row.other_allele})
        if stored in expected:
            n_verified += 1
            continue
        if row.strand == "-" and _complement(stored) in expected:
            reference_alleles = " or ".join(sorted(":".join(sorted(pair)) for pair in expected))
            mismatches.append(
                f"{row.alid} (lifted from {source[0]}:{source[1]}:{source[2]}:{source[3]}; "
                f"the reference carries {reference_alleles} there -- this variant's other "
                f"strand)"
            )
            continue
        n_other_variant += 1
    if mismatches:
        shown = "; ".join(mismatches[:5])
        more = f" and {len(mismatches) - 5} more" if len(mismatches) > 5 else ""
        raise LiftedAlleleMismatchError(
            f"{len(mismatches)} lifted variant(s) disagree with the reference about their "
            f"strand: the reference spells each of them on the other strand from the one "
            f"the lift stored, which is what a lift that ignores its mapping's strand "
            f"produces (issue #284): {shown}{more}. Reference: {reference_path}"
        )
    return LiftedAlleleCheck(
        n_lifted=len(lifted),
        n_verified=n_verified,
        n_other_variant=n_other_variant,
        n_absent=len(lifted) - n_verified - n_other_variant,
        n_strand_contradicted=len(mismatches),
    )

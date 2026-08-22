"""Allele-flipped EAF detection at build time (issue #115, ADR 0037 §6).

`GCST003566.h.tsv.gz` — one of the ten Analyses in the
`gwas-catalog-eur-hybrid` pilot — reports `effect_allele_frequency` against
the *other* allele. Nothing in the pipeline could have caught it: until
ADR 0036 no store retained EAF at all, and now that one does, a store built
from that source would carry a systematically wrong frequency for every
variant, with nothing in the row to say so. A user filtering on MAF reads
`0.42` where the truth is `0.58` — worse than a missing value, because it
looks like an answer.

The check is a correlation, not a threshold on differences. Each Analysis's
A1-oriented EAF is correlated against a reference's over the overlapping
variants; `r < 0` fails the build. The separation is unambiguous and, crucially,
population bottlenecking does not confuse it: FinnGen's frequencies differ from
EUR reference data by up to 3000× in magnitude (ADR 0037's max log-residual of
8.07, the Finnish founder effect) yet still correlate at r = +0.995, because a
bottleneck changes magnitudes, not direction. An absolute-difference threshold
would flag that cohort; this does not.

Three things this module refuses to do quietly:

* **Interpret a correlation it has no business interpreting.** Too few
  overlapping variants, or a variant sample with almost no frequency spread
  (a near-monomorphic slice correlates meaninglessly), yields `unverified`
  — never `passed`.
* **Trust a reference it has not checked.** A panel column that is minor
  allele frequency rather than allele-oriented EAF is symmetric about 0.5 and
  would correlate a flipped Analysis at r ≈ +1. `load_eaf_reference` refuses a
  reference that reports a frequency above 0.5 for only a negligible fraction
  of its variants.
* **Guess which Analysis is wrong.** With no authoritative panel, the consensus
  of the other Analyses can prove that Analyses disagree; it cannot say which
  one is misreporting. A disagreeing multi-Analysis build therefore fails
  rather than picking a side.

Each Analysis is correlated over a sample of **its own** variants, not of the
store's variant axis. The distinction is not cosmetic: `GCST90199621` in the
metabolome pilot covers 0.8% of its store's axis and drew 89 of 20,000
axis-sampled sites against the EUR panel — too few to conclude anything —
where 20,000 of its own variants are ample. The sample is a deterministic
bottom-k by hash, so a build and a later audit of the same store agree without
either persisting the list.

`opengwasdb.build.phenotype_sd`'s shape: pure computation over the values the
builder already has in hand, no store awareness and no source-file I/O beyond
reading the reference, so it is testable against synthetic data with a known
answer. The builders call `verify_eaf_orientation` after their resolve pass and
before any statistic array is written — a build that is going to fail should
fail before it spends an hour writing zarr.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import EafOrientationMethod, EafOrientationOutcome
from opengwasdb.variants.normalise import VariantNormalisationError, orient_to_canonical

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_MIN_VARIANCE",
    "DEFAULT_SAMPLE_SITES",
    "EafOrientationError",
    "EafOrientationReport",
    "EafReference",
    "EafReferenceError",
    "OrientationEvidence",
    "MIN_FRACTION_ABOVE_HALF",
    "apply_orientation_evidence",
    "check_eaf_orientation",
    "enforce_eaf_orientation",
    "load_eaf_reference",
    "sample_column",
    "select_rows",
    "select_sites",
    "site_hash",
    "site_hashes",
    "verify_eaf_orientation",
]

# How many variants each Analysis's correlation is computed over. Per Analysis,
# not per store: an Analysis covering 0.8% of the store's variant axis -- a
# filtered cis+signals Analysis in the metabolome pilot, say -- draws 89 hits
# from a 20,000-variant sample of the *axis*, which is not enough to say
# anything, while 20,000 of its own variants is ample. The measured separation
# (|r| > 0.99 either way) is nowhere near needing more, and the bound is what
# keeps the reference load affordable on a genome-scale build.
DEFAULT_SAMPLE_SITES = 20_000

# Below this many overlapping variants the correlation is not interpreted at
# all. Not a power calculation: at r ≈ ±0.999 a few dozen variants would
# already separate the two cases. It is a guard against reading a sign off a
# handful of sites that happen to overlap for a reason — a single region, one
# chromosome arm — rather than off the Analysis as a whole.
DEFAULT_MIN_OVERLAP = 500

# Minimum variance of the frequencies on *both* sides of the correlation. A
# slice of near-monomorphic variants (all EAF ≈ 0.99) has essentially no
# spread, so its correlation is dominated by noise and its sign says nothing.
# 0.005 is sd ≈ 0.07; a genuine genome-wide sample sits around 0.12.
DEFAULT_MIN_VARIANCE = 0.005

# Fewer Analyses than this and there is no consensus to speak of: with two, a
# disagreement is symmetric and neither side is more credible than the other.
MIN_CONSENSUS_ANALYSES = 3

# The fraction of a reference's variants that must report a frequency above 0.5
# for it to be allele-oriented EAF rather than minor allele frequency. A real
# panel sits near half (the UKB EUR panel: 0.4998 over 9.85M variants); a MAF
# column sits at zero. Deliberately a fraction rather than "the maximum exceeds
# 0.5", which a single rounded row would satisfy.
MIN_FRACTION_ABOVE_HALF = 0.01

# Columns of the LD reference panel's per-block variant tables
# (`ld_dir/{ancestry}/{chr}/{block}.tsv`, see `opengwasdb.completion.ld_panel`).
_PANEL_BLOCK_COLUMNS = ("CHR", "BP", "EA", "OA", "EAF")


class EafReferenceError(ValueError):
    """Raised when a supplied EAF reference cannot be trusted as one."""


class EafOrientationError(RuntimeError):
    """Raised when a build's EAF orientation check does not pass."""


@dataclass(frozen=True)
class EafReference:
    """A1-oriented reference frequencies, restricted to the sampled sites.

    `checksum` and `n_variants` describe the *whole* reference, not the
    retained sample: they are the identity a store records so that validation
    can check the recorded evidence long after the panel itself has moved or
    gone (issue #115 — standalone validation cannot depend on an external
    panel still being available). `fraction_above_half` is likewise over every
    row read, and is what distinguishes an allele-oriented EAF column from a
    MAF one.
    """

    reference_id: str
    checksum: str
    n_variants: int
    fraction_above_half: float
    eaf: Mapping[str, float]


@dataclass(frozen=True)
class OrientationEvidence:
    """One Analysis's outcome, and the evidence it was reached from."""

    analysis_id: str
    outcome: EafOrientationOutcome
    n_overlap: int
    r: float
    note: str = ""
    # False when the Analysis stores no frequency at all. Such an Analysis is
    # `unverified` — there is no checked column here — but it is not an
    # unverifiable *frequency*, so it never blocks a build the way one is
    # entitled to.
    stores_eaf: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "outcome": self.outcome.value,
            "n_overlap": self.n_overlap,
            "r": None if not np.isfinite(self.r) else round(float(self.r), 6),
            "stores_eaf": self.stores_eaf,
            "note": self.note,
        }


@dataclass(frozen=True)
class EafOrientationReport:
    """Every Analysis's evidence, plus what it was all compared against."""

    method: EafOrientationMethod
    evidence: tuple[OrientationEvidence, ...]
    reference_id: str = ""
    reference_checksum: str = ""
    n_reference_variants: int = 0
    n_sites: int = 0
    min_overlap: int = DEFAULT_MIN_OVERLAP
    min_variance: float = DEFAULT_MIN_VARIANCE

    @property
    def failures(self) -> tuple[OrientationEvidence, ...]:
        return tuple(e for e in self.evidence if e.outcome is EafOrientationOutcome.FAILED)

    @property
    def unverified(self) -> tuple[OrientationEvidence, ...]:
        """Analyses whose stored frequencies could not be verified.

        Excludes Analyses that store no frequency: nothing about those is
        unverified, and folding them in here would bury the ones that are.
        """
        return tuple(
            e
            for e in self.evidence
            if e.outcome is EafOrientationOutcome.UNVERIFIED and e.stores_eaf
        )

    def provenance(self, *, allow_unverified: bool = False) -> dict[str, Any]:
        """The `provenance["eaf_orientation"]` sub-dict a store records.

        Everything validation needs to re-read the outcome without the
        reference: what was compared against what, over how many variants,
        with what result, and whether an operator overrode an unverified
        outcome deliberately.
        """
        return {
            "method": self.method.value,
            "reference_id": self.reference_id,
            "reference_checksum": self.reference_checksum,
            "n_reference_variants": self.n_reference_variants,
            "n_sites": self.n_sites,
            "min_overlap": self.min_overlap,
            "min_variance": self.min_variance,
            "allow_unverified": bool(allow_unverified),
            "analyses": [e.to_dict() for e in self.evidence],
        }


# ── Deterministic site selection ─────────────────────────────────────────────


def site_hash(alid: str) -> int:
    """One variant's stable sampling hash — the rule `select_sites`,
    `site_hashes` and `select_rows` all agree on."""
    return int.from_bytes(hashlib.blake2b(alid.encode("utf-8"), digest_size=8).digest(), "big")


def select_sites(alids: Iterable[str], *, k: int = DEFAULT_SAMPLE_SITES) -> list[str]:
    """The `k` variants the correlation is computed over, chosen deterministically.

    A bottom-`k`-by-hash sample: uniform over the input, identical for two runs
    over the same variant set, and — unlike "the first k" or "every nth" —
    independent of the order the caller happens to iterate in, so a build and a
    later audit of the same store agree without either persisting the list.
    Fewer than `k` inputs returns all of them.
    """
    if k <= 0:
        return []
    heap: list[tuple[int, str]] = []
    for alid in alids:
        key = (-site_hash(alid), alid)
        if len(heap) < k:
            heapq.heappush(heap, key)
        elif key > heap[0]:
            heapq.heapreplace(heap, key)
    return sorted(alid for _h, alid in heap)


def site_hashes(alids: Sequence[str]) -> np.ndarray:
    """`select_sites`' hash for every variant on a store's axis, computed once.

    A Dense or Hybrid builder holds row indices into a shared axis, not ALIDs,
    and hashing every Analysis's ALIDs separately would mean hashing the axis
    once per Analysis — 22 billion hashes on the largest store here. Hashing
    the axis once and indexing into the result is the same selection, in about
    ten seconds.
    """
    return np.fromiter(
        (site_hash(alid) for alid in alids), dtype=np.uint64, count=len(alids)
    )


def select_rows(hashes: np.ndarray, rows: np.ndarray, *, k: int) -> np.ndarray:
    """The `k` of `rows` `select_sites` would have chosen, given `site_hashes`.

    Same bottom-`k`-by-hash rule, vectorised over row indices. Returned sorted
    by row so a caller's gather stays in axis order.
    """
    if k <= 0 or rows.size == 0:
        return np.empty(0, dtype=rows.dtype)
    if rows.size <= k:
        return np.sort(rows)
    row_hashes = hashes[rows]
    smallest = np.argpartition(row_hashes, k - 1)[:k]
    return np.sort(rows[smallest])


def sample_column(
    rows: np.ndarray,
    eaf: np.ndarray,
    alids: Sequence[str],
    hashes: np.ndarray,
    *,
    k: int = DEFAULT_SAMPLE_SITES,
) -> dict[str, float]:
    """`{alid: eaf}` for up to `k` of one Analysis's own EAF-carrying variants.

    Builders hold a resolved column as parallel `(row, eaf)` arrays over a
    shared axis, so this takes that shape directly rather than making every
    caller build a dict of ten million entries to throw all but twenty thousand
    of away. Non-finite frequencies are dropped *before* the sample is drawn:
    an Analysis whose source reported no frequency at some variant has NaN
    there, NaN is not evidence, and sampling first would spend the budget on
    variants that carry none.
    """
    if rows.size == 0:
        return {}
    with_eaf = rows[np.isfinite(eaf)]
    if with_eaf.size == 0:
        return {}
    selected = select_rows(hashes, with_eaf, k=k)
    lookup = dict(zip(rows.tolist(), eaf.tolist(), strict=True))
    return {alids[row]: float(lookup[row]) for row in selected.tolist()}


# ── Reference loading ────────────────────────────────────────────────────────


def _iter_panel_directory(root: Path, ancestry: str) -> Iterator[tuple[str, float]]:
    """A1-oriented `(alid, eaf)` from an LD reference panel's block tables.

    The layout `opengwasdb.completion.ld_panel` already reads:
    `{root}/{ancestry}/{chr}/{block}.tsv`, with `CHR`/`BP`/`EA`/`OA`/`EAF`
    columns. Blocks overlap, so a variant may be read more than once; the
    values agree, and the file order is sorted, so the checksum is
    deterministic either way. Files directly under the ancestry directory
    (the panel's own block lookup table) are not block tables and are skipped.
    """
    ancestry_dir = root / ancestry
    if not ancestry_dir.is_dir():
        raise EafReferenceError(
            f"EAF reference {root} has no {ancestry!r} subdirectory; "
            f"present: {sorted(p.name for p in root.iterdir() if p.is_dir())}"
        )
    block_files = sorted(
        (p for chrom_dir in sorted(ancestry_dir.iterdir()) if chrom_dir.is_dir()
         for p in chrom_dir.glob("*.tsv")),
        key=lambda p: (p.parent.name, p.name),
    )
    if not block_files:
        raise EafReferenceError(f"EAF reference {ancestry_dir} contains no block tables")
    for path in block_files:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = [c for c in _PANEL_BLOCK_COLUMNS if c not in (reader.fieldnames or ())]
            if missing:
                raise EafReferenceError(
                    f"EAF reference block {path} is missing column(s): {', '.join(missing)}"
                )
            for row in reader:
                oriented = _orient_row(row["CHR"], row["BP"], row["EA"], row["OA"], row["EAF"])
                if oriented is not None:
                    yield oriented


def _iter_panel_table(path: Path) -> Iterator[tuple[str, float]]:
    """A1-oriented `(alid, eaf)` from a single reference table.

    Either an `alid` column with an already-A1-oriented `eaf` (the shape a
    store's own variant table exports), or explicit
    `chromosome`/`position`/`effect_allele`/`other_allele` columns whose `eaf`
    describes the effect allele and is oriented here. Column names are matched
    case-insensitively, so the LD panel's own upper-case spelling also reads.
    """
    opener: Any = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = {name.lower(): name for name in (reader.fieldnames or ())}
        eaf_col = fields.get("eaf") or fields.get("effect_allele_frequency")
        if eaf_col is None:
            raise EafReferenceError(
                f"EAF reference {path} has no 'eaf' or 'effect_allele_frequency' column"
            )
        alid_col = fields.get("alid")
        allele_cols = tuple(
            fields.get(name)
            for name in ("chromosome", "position", "effect_allele", "other_allele")
        )
        if alid_col is None and any(c is None for c in allele_cols):
            raise EafReferenceError(
                f"EAF reference {path} needs either an 'alid' column or "
                "chromosome/position/effect_allele/other_allele columns"
            )
        for row in reader:
            if alid_col is not None:
                alid = (row.get(alid_col) or "").strip()
                eaf = _parse_frequency(row[eaf_col])
                if eaf is not None and alid.count(":") == 3:
                    yield alid, eaf
                continue
            chrom, pos, effect, other = (row[str(c)] for c in allele_cols)
            oriented = _orient_row(chrom, pos, effect, other, row[eaf_col])
            if oriented is not None:
                yield oriented


def _orient_row(
    chromosome: str, position: str, effect: str, other: str, raw_eaf: str
) -> tuple[str, float] | None:
    eaf = _parse_frequency(raw_eaf)
    if eaf is None:
        return None
    try:
        orientation = orient_to_canonical(chromosome, position, effect, other)
    except VariantNormalisationError:
        return None
    return orientation.variant.alid, (1.0 - eaf if orientation.flipped else eaf)


def _parse_frequency(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= parsed <= 1.0):
        return None
    return parsed


def load_eaf_reference(
    path: str | Path,
    sites: Iterable[str],
    *,
    ancestry: str | None = None,
    min_rows_for_orientation_check: int = DEFAULT_MIN_OVERLAP,
) -> EafReference:
    """Load A1-oriented reference frequencies at `sites`, and check them.

    `path` is either an LD reference panel directory (`ancestry` required, e.g.
    `/data/opengwasdb/reference/ukb-hg38` with `EUR`) or a single table with an
    `eaf` column. The whole reference is streamed — every row contributes to
    the checksum and to the fraction-above-0.5 check — but only the sites the
    Analyses' samples actually name are retained, so a 9-million-variant panel
    costs a dict the size of the sample rather than of the panel.

    Raises `EafReferenceError` when the reference is empty, or when only a
    negligible fraction of its rows report a frequency above 0.5. That second
    case is minor allele frequency wearing an EAF column's name: it is
    symmetric about 0.5, so it would correlate a flipped Analysis at r ≈ +1 and
    certify the exact defect this check exists to find. A fraction rather than
    a maximum, so one rounded row cannot wave a MAF column through.
    """
    reference_path = Path(path)
    wanted = set(sites)
    if reference_path.is_dir():
        if not ancestry:
            raise EafReferenceError(
                f"EAF reference {reference_path} is a panel directory, which needs an ancestry "
                "to select a population from (e.g. EUR)"
            )
        rows = _iter_panel_directory(reference_path, ancestry)
        reference_id = f"{reference_path}#{ancestry}"
    else:
        if not reference_path.exists():
            raise EafReferenceError(f"EAF reference {reference_path} does not exist")
        rows = _iter_panel_table(reference_path)
        reference_id = str(reference_path)

    digest = hashlib.sha256()
    kept: dict[str, float] = {}
    n_variants = 0
    n_above_half = 0
    for alid, eaf in rows:
        n_variants += 1
        digest.update(f"{alid}\t{eaf:.6f}\n".encode())
        if eaf > 0.5:
            n_above_half += 1
        if alid in wanted:
            kept[alid] = eaf
    fraction_above_half = n_above_half / n_variants if n_variants else 0.0

    if n_variants == 0:
        raise EafReferenceError(f"EAF reference {reference_id} yielded no usable rows")
    if (
        n_variants >= min_rows_for_orientation_check
        and fraction_above_half < MIN_FRACTION_ABOVE_HALF
    ):
        raise EafReferenceError(
            f"EAF reference {reference_id} reports a frequency above 0.5 for only "
            f"{fraction_above_half:.4%} of its {n_variants} variants, so it is minor allele "
            "frequency, not allele-oriented effect allele frequency — correlating against "
            "it would certify a flipped source as correct"
        )
    log.info(
        "EAF reference %s: %d variants read, %d of %d sampled sites present, "
        "%.2f%% above 0.5",
        reference_id, n_variants, len(kept), len(wanted), 100.0 * fraction_above_half,
    )
    return EafReference(
        reference_id=reference_id,
        checksum=f"sha256:{digest.hexdigest()}",
        n_variants=n_variants,
        fraction_above_half=fraction_above_half,
        eaf=kept,
    )


# ── The check ────────────────────────────────────────────────────────────────


def _correlate(
    analysis_id: str,
    observed: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    min_overlap: int,
    min_variance: float,
) -> OrientationEvidence:
    """One Analysis against one baseline, with every gate applied in order."""
    shared = [alid for alid in observed if alid in baseline]
    n = len(shared)
    if n < min_overlap:
        return OrientationEvidence(
            analysis_id=analysis_id,
            outcome=EafOrientationOutcome.UNVERIFIED,
            n_overlap=n,
            r=float("nan"),
            note=f"only {n} variants overlap the reference, fewer than {min_overlap}",
        )
    a = np.fromiter((observed[alid] for alid in shared), dtype=np.float64, count=n)
    b = np.fromiter((baseline[alid] for alid in shared), dtype=np.float64, count=n)
    var_a, var_b = float(np.var(a)), float(np.var(b))
    if var_a < min_variance or var_b < min_variance:
        return OrientationEvidence(
            analysis_id=analysis_id,
            outcome=EafOrientationOutcome.UNVERIFIED,
            n_overlap=n,
            r=float("nan"),
            note=(
                f"frequency variance too low to read a direction from "
                f"(analysis {var_a:.5f}, reference {var_b:.5f}, minimum {min_variance})"
            ),
        )
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return OrientationEvidence(
            analysis_id=analysis_id,
            outcome=EafOrientationOutcome.UNVERIFIED,
            n_overlap=n,
            r=float("nan"),
            note="correlation is not finite",
        )
    outcome = (
        EafOrientationOutcome.FAILED if r < 0.0 else EafOrientationOutcome.PASSED
    )
    return OrientationEvidence(analysis_id=analysis_id, outcome=outcome, n_overlap=n, r=r)


def _consensus_baselines(
    observations: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Per Analysis, the median EAF the *other* Analyses report at each site.

    Leaving the Analysis itself out is what makes the comparison independent:
    an Analysis included in its own baseline pulls that baseline toward itself,
    which is precisely the direction that would hide a flip. Sites where fewer
    than two other Analyses report a frequency are dropped — a "median" of one
    is that one Analysis's opinion, not a consensus.
    """
    per_site: dict[str, dict[str, float]] = {}
    for analysis_id, observed in observations.items():
        for alid, value in observed.items():
            per_site.setdefault(alid, {})[analysis_id] = value
    baselines: dict[str, dict[str, float]] = {a: {} for a in observations}
    for alid, by_analysis in per_site.items():
        for analysis_id in observations:
            others = [v for a, v in by_analysis.items() if a != analysis_id]
            if len(others) >= 2:
                baselines[analysis_id][alid] = float(np.median(others))
    return baselines


def check_eaf_orientation(
    observations: Mapping[str, Mapping[str, float]],
    *,
    reference: EafReference | None = None,
    n_sites: int = 0,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    min_variance: float = DEFAULT_MIN_VARIANCE,
) -> EafOrientationReport:
    """Correlate each Analysis's A1-oriented EAF against the best baseline there is.

    `observations` maps analysis id to `{alid: A1-oriented eaf}` — the same
    orientation the store holds, so what is checked is what will be read back.

    With a `reference`, every Analysis is correlated against it. Without one,
    and with at least `MIN_CONSENSUS_ANALYSES` Analyses carrying EAF, each is
    correlated against the consensus of the others; that can establish that
    Analyses disagree but not which is right, which is why the caller fails such
    a build rather than dropping an Analysis. With neither, every Analysis is
    `unverified` — recorded and warned about, never silently `passed`.
    """
    with_eaf = {a: o for a, o in observations.items() if o}
    order = list(observations)

    if reference is not None:
        evidence = tuple(
            _correlate(
                analysis_id,
                with_eaf[analysis_id],
                reference.eaf,
                min_overlap=min_overlap,
                min_variance=min_variance,
            )
            if analysis_id in with_eaf
            else _no_eaf(analysis_id)
            for analysis_id in order
        )
        return EafOrientationReport(
            method=EafOrientationMethod.REFERENCE_PANEL,
            evidence=evidence,
            reference_id=reference.reference_id,
            reference_checksum=reference.checksum,
            n_reference_variants=reference.n_variants,
            n_sites=n_sites,
            min_overlap=min_overlap,
            min_variance=min_variance,
        )

    if len(with_eaf) >= MIN_CONSENSUS_ANALYSES:
        baselines = _consensus_baselines(with_eaf)
        evidence = tuple(
            _correlate(
                analysis_id,
                with_eaf[analysis_id],
                baselines[analysis_id],
                min_overlap=min_overlap,
                min_variance=min_variance,
            )
            if analysis_id in with_eaf
            else _no_eaf(analysis_id)
            for analysis_id in order
        )
        return EafOrientationReport(
            method=EafOrientationMethod.CONSENSUS,
            evidence=evidence,
            reference_id=f"consensus:{len(with_eaf)}-analyses",
            n_sites=n_sites,
            min_overlap=min_overlap,
            min_variance=min_variance,
        )

    note = (
        "no EAF reference supplied"
        if not with_eaf
        else (
            f"no EAF reference supplied, and {len(with_eaf)} analyses carry EAF -- "
            f"fewer than the {MIN_CONSENSUS_ANALYSES} a consensus needs"
        )
    )
    evidence = tuple(
        _no_eaf(analysis_id)
        if analysis_id not in with_eaf
        else OrientationEvidence(
            analysis_id=analysis_id,
            outcome=EafOrientationOutcome.UNVERIFIED,
            n_overlap=0,
            r=float("nan"),
            note=note,
        )
        for analysis_id in order
    )
    return EafOrientationReport(
        method=EafOrientationMethod.NONE,
        evidence=evidence,
        n_sites=n_sites,
        min_overlap=min_overlap,
        min_variance=min_variance,
    )


def _no_eaf(analysis_id: str) -> OrientationEvidence:
    """An Analysis whose source reported no frequency has nothing to orient.

    `unverified` rather than `passed`: there is no wrong column here, but
    neither is there a checked one, and the two must not read alike.
    """
    return OrientationEvidence(
        analysis_id=analysis_id,
        outcome=EafOrientationOutcome.UNVERIFIED,
        n_overlap=0,
        r=float("nan"),
        note="this analysis stores no EAF, so there is no orientation to check",
        stores_eaf=False,
    )


def enforce_eaf_orientation(
    report: EafOrientationReport, *, allow_unverified: bool = False
) -> None:
    """Fail the build on a flipped Analysis, and on an unverifiable one that
    the operator did not say to accept.

    A `failed` outcome always raises, whichever baseline produced it: against a
    panel it names a flipped source, and against the consensus it says the
    Analyses in this build contradict each other, which consensus cannot
    adjudicate.

    An `unverified` outcome raises only when a reference was actually supplied
    — asking for verification and silently not getting it is the failure mode
    this whole module exists to close. Set `allow_unverified` to accept it
    deliberately; the builders record that choice in provenance, so the store
    says an operator overrode the check rather than saying nothing. A build
    with no reference at all is unverified by construction: it warns and
    records, since demanding a panel from every Dense/Ragged build would be a
    different decision than this issue makes.
    """
    failures = report.failures
    if failures:
        detail = "; ".join(
            f"{e.analysis_id}: r = {e.r:+.4f} over {e.n_overlap} variants" for e in failures
        )
        against = (
            "the reference panel"
            if report.method is EafOrientationMethod.REFERENCE_PANEL
            else "the consensus of the other analyses in this build"
        )
        adjudication = (
            ""
            if report.method is EafOrientationMethod.REFERENCE_PANEL
            else " -- a consensus can establish that the analyses disagree but not which is "
            "right, so this build cannot be completed by dropping an Analysis"
        )
        raise EafOrientationError(
            f"effect allele frequency is anti-correlated with {against}, so it is reported "
            f"against the other allele ({detail}){adjudication}. Reference: "
            f"{report.reference_id or 'none'}"
        )

    unverified = report.unverified
    for evidence in unverified:
        log.warning(
            "EAF orientation unverified for analysis %s: %s", evidence.analysis_id, evidence.note
        )
    if unverified and report.method is EafOrientationMethod.REFERENCE_PANEL:
        if not allow_unverified:
            detail = "; ".join(f"{e.analysis_id}: {e.note}" for e in unverified)
            raise EafOrientationError(
                "an EAF reference was supplied but these analyses could not be verified "
                f"against it ({detail}). Pass allow_unverified to accept that deliberately "
                "-- it is recorded in the store's provenance"
            )
        log.warning(
            "Accepting %d unverified analyses because allow_unverified was set", len(unverified)
        )


def verify_eaf_orientation(
    observations: Mapping[str, Mapping[str, float]],
    *,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None,
    allow_unverified: bool = False,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    min_variance: float = DEFAULT_MIN_VARIANCE,
) -> EafOrientationReport:
    """Load the reference, run the check, and stop the build if it does not pass.

    The single entry point every builder uses, so Dense, Hybrid and Ragged
    cannot drift into checking different things or reporting them differently.
    `observations` is already sampled per Analysis (`sample_column`, or the
    caller's own selection where it holds ALIDs directly); the reference is
    loaded only at the sites those samples actually name, which is what keeps
    a nine-million-variant panel down to a dict of the size of the sample.
    """
    sites = sorted({alid for observed in observations.values() for alid in observed})
    reference = (
        None
        if eaf_reference is None
        else load_eaf_reference(eaf_reference, sites, ancestry=eaf_reference_ancestry)
    )
    report = check_eaf_orientation(
        observations,
        reference=reference,
        n_sites=len(sites),
        min_overlap=min_overlap,
        min_variance=min_variance,
    )
    enforce_eaf_orientation(report, allow_unverified=allow_unverified)
    return report


def apply_orientation_evidence(
    analyses: Sequence[Analysis], report: EafOrientationReport
) -> list[Analysis]:
    """Stamp each Analysis's `analyses.tsv` orientation columns from the report.

    Matched by `analysis_id`, not position: the report is built from whatever
    the builder had EAF for, which need not be in analysis order. An Analysis
    the report says nothing about keeps its columns blank rather than being
    given an outcome nobody measured.
    """
    by_id = {e.analysis_id: e for e in report.evidence}
    stamped: list[Analysis] = []
    for analysis in analyses:
        evidence = by_id.get(analysis.analysis_id)
        if evidence is None:
            stamped.append(analysis)
            continue
        stamped.append(
            replace(
                analysis,
                eaf_orientation=evidence.outcome.value,
                eaf_orientation_r=("" if not np.isfinite(evidence.r) else f"{evidence.r:.6f}"),
                eaf_orientation_n=str(evidence.n_overlap),
            )
        )
    return stamped

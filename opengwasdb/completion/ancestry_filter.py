"""The per-Analysis ancestry-match filter for Reference Completion (ADR 0028).

An Analysis carrying an ``assigned_ancestry`` (from Catalogue-driven ancestry
assignment) should only be imputed against a matching-ancestry LD panel;
others are carried through observed-only. Store releases with no ancestry
information anywhere impute every Analysis -- the same behaviour as before
ADR 0028 existed.

That fallback is backward compatibility, not a judgement that the store is
single-ancestry: nothing here can tell "every Analysis really is EUR" apart
from "nobody ran ancestry assignment on a multi-cohort manifest". Since
completion imputes from the panel's own LD structure and EAF
(``opengwasdb.completion.block``), the second case silently applies one
ancestry's LD/EAF to Analyses it does not describe. The fallback therefore
warns rather than only logging at info level (issue #108).

Shared by Dense, Ragged, and Hybrid completion so the filter applies
uniformly regardless of which layout's entry point is called directly,
rather than only when the caller happens to be ``complete_hybrid_store``.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: The super-population each spelling names. Exact matches on a normalised
#: label, deliberately not the substring matching `ancestry.routing` does for
#: free-text Reported Populations: there, ordering makes "North African" match
#: `african` before `north africa` and come back AFR, which is tolerable when
#: guessing at a cohort's free-text description and is not tolerable when
#: deciding which Analyses an LD panel may impute. A label this table does not
#: name is unroutable, and matches nothing.
_ANCESTRY_ALIASES: dict[str, str] = {
    "afr": "AFR", "african": "AFR",
    "amr": "AMR", "american": "AMR", "admixed american": "AMR", "ad mixed american": "AMR",
    "eas": "EAS", "east asian": "EAS",
    "eur": "EUR", "european": "EUR",
    "mid": "MID", "middle eastern": "MID",
    "naf": "NAF", "north african": "NAF",
    "sas": "SAS", "south asian": "SAS",
}


class AncestryFilterError(Exception):
    """The requested panel ancestry matches no Analysis in the store."""


def canonical_ancestry(label: str) -> str | None:
    """Map an ancestry label to a super-population code, or None.

    Two vocabularies for one concept meet here: LD panel directories are named
    for super-population codes (`EUR`), while an Analysis's `assigned_ancestry`
    may hold the word (`European`) -- and both spellings occur in one
    `analyses.tsv`, because the registry writes the code down its
    AF-assignment path and the source's own word down its trusted-label path.
    Matching them by string equality is what made a whole store complete to
    nothing, silently (issue #98).

    Returns None for a label that names no super-population. "Mixed" is
    known-but-unroutable (ADR 0028) and must not become EUR by failing to
    normalise; neither must "North African" become AFR by matching loosely.
    """
    text = " ".join((label or "").replace("_", " ").replace("-", " ").split()).lower()
    return _ANCESTRY_ALIASES.get(text)


def _matches(assigned: str, ancestry: str) -> bool:
    """Whether an Analysis's `assigned_ancestry` names the panel's ancestry.

    Exact equality first, so a panel named for something outside the
    super-population vocabulary (a cohort-specific panel, say) keeps working
    on its own terms rather than being normalised into nothing.
    """
    if assigned == ancestry:
        return True
    canonical = canonical_ancestry(assigned)
    return canonical is not None and canonical == canonical_ancestry(ancestry)


def derive_impute_analysis_ids(
    analyses_rows: Any,
    ancestry: str,
) -> set[str] | None:
    """Return the Analysis ids to impute, or ``None`` to impute all.

    ``None`` when no row carries ``assigned_ancestry`` -- the source has no
    ancestry information to filter on. Otherwise the set of
    ``analysis_id``\\ s whose ``assigned_ancestry`` matches *ancestry*.

    Raises `AncestryFilterError` when ancestry *is* known and nothing matches
    the panel. That case cannot be answered with a set: an empty one completes
    the store to zero imputed cells, and ``None`` would impute everything
    against a panel every Analysis is known not to match. Both produce a
    release stamped ``reference_completed`` that is indistinguishable
    downstream from a genuine one, which is the failure this raises to avoid
    (issue #98).
    """
    rows = list(analyses_rows)
    if not any(row.get("assigned_ancestry") for row in rows):
        log.warning(
            "No assigned_ancestry on any of %d analyses: imputing all of them against "
            "the %s panel regardless of their true ancestry. If this store spans "
            "multiple ancestries, its imputed cells will carry %s LD and allele "
            "frequencies. Run ancestry assignment (ADR 0028) to impute only "
            "matching analyses.",
            len(rows), ancestry, ancestry,
        )
        return None
    matched = {
        row["analysis_id"]
        for row in rows
        if _matches(str(row.get("assigned_ancestry") or ""), ancestry)
    }
    if not matched:
        present = sorted({str(row.get("assigned_ancestry") or "") for row in rows} - {""})
        raise AncestryFilterError(
            f"panel ancestry {ancestry!r} matches none of the {len(rows)} analyses in this "
            f"store; their assigned_ancestry values are {present}. Completing anyway would "
            "produce a release stamped reference_completed with zero imputed cells, which "
            "nothing downstream can tell apart from one where imputation was attempted and "
            "failed. Complete this store against a panel its analyses match, or correct "
            "assigned_ancestry if it is wrong. (Spellings of one ancestry are reconciled "
            f"automatically -- {ancestry!r} normalises to "
            f"{canonical_ancestry(ancestry) or 'no super-population'} -- so this is a "
            "genuine mismatch, not a vocabulary one.)"
        )
    log.info(
        "Ancestry-matched completion: %d/%d analyses match panel ancestry %s",
        len(matched), len(rows), ancestry,
    )
    return matched

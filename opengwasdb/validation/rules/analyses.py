"""Rules that read `analyses.tsv`: its structure, vocabularies and EAF evidence (ADR 0074).

`analyses.tsv` is the sole source of truth for Analytical Metadata (ADR 0030,
ADR 0034). These rules check what a *built* release can promise: one row per
`analysis_index`, no duplicate `analysis_id`, vocabulary conformance for the
controlled columns present, the Top-Hit Count columns, completion metadata that
describes its own Analysis, and the EAF orientation evidence a release must carry
for the frequencies it stores (issue #115).

Deliberately narrower than `opengwasdb.model.analyses.validate_analyses` (issue
#16): that function's `REQUIRED_COLUMNS` enforces non-blank
`sample_size_kind`/`sample_size_scope`/`sample_size`/`original_effect_scale` per
row, which is the contract a *registry* manifest must satisfy before a build even
runs. No current build path has a source for those fields (issue #22 scope note),
so a built store leaves them honestly blank.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum

from opengwasdb.model.analyses import (
    N_MISSING_SE_OUT_OF_RANGE,
    NON_NEGATIVE_INTEGER_OR_BLANK,
    RETIRED_ANALYSIS_COLUMNS,
    TOP_HIT_COUNT_COLUMNS,
)
from opengwasdb.model.enums import (
    SD_RESCALING_METHODS,
    AncestryAssignmentMethod,
    EafOrientationOutcome,
    EafScope,
    OriginalSdMethod,
    StoredEffectScale,
)
from opengwasdb.model.regions import (
    REGIONS_FILE,
    RegionTableError,
    check_region_table,
    read_store_regions,
)
from opengwasdb.store.open import records_se_out_of_range
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.encoding.plan import EafEncoding
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.validation.rule import Check, Kind, Rule, Severity, Target
from opengwasdb.validation.rules.release import REQUIRED

#: The rule every `analyses.tsv` rule runs after: the table could be read at all.
READABLE = "analyses.readable"
#: The rule every EAF-orientation rule runs after.
ORIENTATION_READABLE = "eaf_orientation.readable"


def readable(target: Target) -> Iterator[str]:
    """The table parses; a failure is this rule's, rather than every later rule's."""
    _table = target.analyses
    yield from ()


def retired_columns(target: Target) -> Iterator[str]:
    table = target.analyses
    retired_present = [c for c in RETIRED_ANALYSIS_COLUMNS if c in table.fieldnames]
    if retired_present:
        yield (
            "analyses.tsv carries retired column(s) "
            f"{', '.join(retired_present)} -- a Store Release built against a "
            "prior column list is not valid against the current schema "
            "(store-format spec §7a, ADR 0034/0035)"
        )


def _integer_rows(target: Target) -> Iterator[tuple[int | None, dict[str, str]]]:
    """Each row with its `analysis_index` as an int, or None when it is not one."""
    for row in target.analyses.rows:
        try:
            yield int(row.get("analysis_index", "")), row
        except ValueError:
            yield None, row


def analysis_index(target: Target) -> Iterator[str]:
    """One row per `analysis_index`, covering exactly 0..n-1."""
    seen: set[int] = set()
    for index, row in _integer_rows(target):
        if index is None:
            raw = row.get("analysis_index", "")
            yield f"analyses.tsv row has a non-integer analysis_index {raw!r}"
            continue
        if index in seen:
            yield f"analyses.tsv has more than one row for analysis_index {index}"
        seen.add(index)
    n = len(target.analyses.rows)
    if seen and seen != set(range(n)):
        yield f"analyses.tsv analysis_index values are not exactly the range 0..{n - 1}"


def analysis_id(target: Target) -> Iterator[str]:
    """No `analysis_id` names two rows (a row with no integer index is judged above)."""
    seen: set[str] = set()
    for index, row in _integer_rows(target):
        if index is None:
            continue
        identifier = row.get("analysis_id", "")
        if identifier in seen:
            yield f"analyses.tsv has more than one row for analysis_id {identifier!r}"
        seen.add(identifier)


def completion_declared(target: Target) -> Iterator[str]:
    """Every count is an integer, and an Analysis completed against no ancestry
    gained no imputed cells.

    The two columns are written by different steps -- `completed_against` from the
    ancestry-match filter, `completion_n_imputed_total` from the per-block quality
    rollup -- so their disagreement is what a filter applied to one and not the
    other looks like from outside (ADR 0028). It is also what `eaf_scope` is
    derived from, which is why it is checked here rather than left to the reader.
    A count that is not an integer is reported on every row, completed or not: the
    rule holding the count to the mask cannot read it, and leaves it to this one.
    """
    for row in target.analyses.rows:
        try:
            n_imputed = int(row.get("completion_n_imputed_total", "") or 0)
        except ValueError:
            yield (
                f"analyses.tsv row {row.get('analysis_id', '')!r} has a non-integer "
                f"completion_n_imputed_total {row.get('completion_n_imputed_total', '')!r}"
            )
            continue
        if row.get("completed_against", ""):
            continue
        if n_imputed:
            yield (
                f"analysis {row.get('analysis_id', '')!r} declares "
                f"completion_n_imputed_total={n_imputed} but completed_against is blank; "
                "it was not completed against any ancestry, so it holds no imputed cells "
                "and the count -- and any eaf_scope derived from it -- describes cells "
                "the release does not contain (ADR 0028, ADR 0037 §4)"
            )


_VOCABULARIES: tuple[tuple[str, type[StrEnum]], ...] = (
    ("stored_effect_scale", StoredEffectScale),
    ("original_sd_method", OriginalSdMethod),
    ("ancestry_assignment_method", AncestryAssignmentMethod),
    ("eaf_orientation", EafOrientationOutcome),
)


def vocabularies(target: Target) -> Iterator[str]:
    """Each controlled column present holds only its vocabulary's values."""
    allowed = {column: {member.value for member in enum} for column, enum in _VOCABULARIES}
    for row in target.analyses.rows:
        for column, values in allowed.items():
            value = row.get(column, "")
            if value and value not in values:
                yield (
                    f"analyses.tsv analysis {row.get('analysis_id')!r} has invalid "
                    f"{column} {value!r}"
                )


def original_sd_recorded(target: Target) -> Iterator[str]:
    """An Analysis whose `original_sd_method` rescales records the SD it was
    divided by: finite and positive (ADR 0029 methods 2-5, ADR 0084).

    A builder divides such an Analysis's SEs by `original_sd` and records it.
    A blank one is what a Ragged SSF build before issue #366 wrote while storing
    its source's SEs undivided -- OGS-00006, -00024, -00025 and -00026, 4,443
    Analyses -- and nothing else in the Store says so: z is scale-free, and the
    SEs look like SEs. A Store whose SEs were standardised by an SD it does not
    record cannot be read back in the source's units either.
    """
    rescaling = {method.value for method in SD_RESCALING_METHODS}
    for row in target.analyses.rows:
        method = row.get("original_sd_method", "")
        if method not in rescaling:
            continue
        raw = row.get("original_sd", "")
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not (math.isfinite(value) and value > 0):
            yield (
                f"analysis {row.get('analysis_id', '')!r} declares original_sd_method="
                f"{method!r}, which divides its standard errors by a phenotype SD, but "
                f"records original_sd {raw!r}: a Release that does not record a finite, "
                "positive SD cannot show its SEs were standardised, and one built by "
                "build-ragged-ssf before issue #366 holds its source's SEs (spec §7a)"
            )


def sd_scale_standardised(target: Target) -> Iterator[str]:
    """An Analysis labelled SD Units was standardised: its `original_sd_method` is a
    rescaling method or `declared_standardised` (spec §7, ADR 0084).

    `stored_effect_scale=sd` over any other method, or none, is an Analysis whose
    SEs were stored undivided under an SD Units label, which nothing else in the
    Store shows. A scale that is not `sd` claims no SD Units, and a blank scale
    with a blank method is "not standardised, scale unknown", so neither fails.
    """
    standardised = {method.value for method in SD_RESCALING_METHODS}
    standardised.add(OriginalSdMethod.DECLARED_STANDARDISED.value)
    for row in target.analyses.rows:
        if row.get("stored_effect_scale", "") != StoredEffectScale.SD.value:
            continue
        method = row.get("original_sd_method", "")
        if method and method not in {m.value for m in OriginalSdMethod}:
            continue  # an unknown spelling is `analyses.vocabularies`'s to report
        if method not in standardised:
            yield (
                f"analysis {row.get('analysis_id', '')!r} has stored_effect_scale='sd' but "
                f"original_sd_method={method!r}: SD Units need a rescaling method or "
                "declared_standardised, else its standard errors may be a source's own "
                "(spec §7, ADR 0084)"
            )


def top_hit_count_columns(target: Target) -> Iterator[str]:
    """The Top-Hit Count columns (ADR 0032) are present and hold non-negative integers.

    Every build path computes them, unlike the completion rollups (Reference-
    Completed releases only), so their presence is required unconditionally.
    """
    table = target.analyses
    missing = [c for c in TOP_HIT_COUNT_COLUMNS if c not in table.fieldnames]
    if missing:
        yield f"analyses.tsv is missing Top-Hit Count column(s): {', '.join(missing)}"
        return
    for row in table.rows:
        for column in TOP_HIT_COUNT_COLUMNS:
            value = row.get(column, "")
            if not value.isdigit():
                yield (
                    f"analyses.tsv analysis {row.get('analysis_id')!r} has invalid "
                    f"{column} {value!r} (must be a non-negative integer)"
                )


# --- n_missing_se_out_of_range (issue #338, ADR 0068) ---


def se_out_of_range_count(target: Target) -> Iterator[str]:
    """`n_missing_se_out_of_range` holds a non-negative integer on every row of a
    release at format 0.2.3 or later; below it the column may be absent or blank
    (unknown, never 0), and a value it does hold is still a non-negative integer.

    Every `analyses.tsv` a release carries is judged, a Hybrid root's included:
    the root counts both its components, the Dense Component its own.
    """
    table = target.analyses
    version = target.store.manifest.format_version
    required = records_se_out_of_range(version)
    if N_MISSING_SE_OUT_OF_RANGE not in table.fieldnames:
        if required:
            yield (
                f"analyses.tsv has no {N_MISSING_SE_OUT_OF_RANGE} column, which a release at "
                f"format_version {version} records for every Analysis (spec §7a, issue #338)"
            )
        return
    for row in table.rows:
        value = row.get(N_MISSING_SE_OUT_OF_RANGE, "")
        analysis = row.get("analysis_id", "")
        if not re.fullmatch(NON_NEGATIVE_INTEGER_OR_BLANK, value):
            yield (
                f"analysis {analysis!r} has {N_MISSING_SE_OUT_OF_RANGE} {value!r}; expected a "
                "non-negative integer"
            )
        elif not value and required:
            yield (
                f"analysis {analysis!r} has a blank {N_MISSING_SE_OUT_OF_RANGE}: a release at "
                f"format_version {version} records the count for every Analysis, and blank "
                "(unknown) belongs only to a release before 0.2.3 (spec §7a, issue #338)"
            )


def se_out_of_range_partition(target: Target) -> Iterator[str]:
    """A Hybrid root counts at least what its Dense Component counts, Analysis by
    Analysis: the root's count is both components' (issue #338)."""
    root = _counts(target.analyses.rows)
    for child in target.children:
        if not child.store.analyses_path.exists():
            continue
        dense = _counts(read_analyses(child.store.analyses_path).rows)
        for analysis, count in dense.items():
            at_root = root.get(analysis)
            if at_root is not None and count is not None and at_root < count:
                yield (
                    f"analysis {analysis!r} counts {at_root} associations left out for an "
                    f"SE above float16's range at the root but {count} in its Dense "
                    "Component; the root counts both components"
                )


def _counts(rows: Iterable[dict[str, str]]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for row in rows:
        value = row.get(N_MISSING_SE_OUT_OF_RANGE, "")
        out[row.get("analysis_id", "")] = int(value) if value.isdigit() else None
    return out


def _with_analyses(target: Target) -> bool:
    """Any release with an `analyses.tsv`, a Hybrid root's included."""
    return target.store.analyses_path.exists()


def _hybrid_root(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.HYBRID and _with_analyses(target)


# --- EAF orientation (issue #115) ---


def orientation_readable(target: Target) -> Iterator[str]:
    """The table the orientation rules read can be read; absent, they have nothing to do."""
    if not target.store.analyses_path.exists():
        return
    try:
        _table = target.analyses
    except Exception as exc:  # reported, in the words this check has always used
        yield f"cannot read analyses.tsv for the EAF orientation check: {exc}"


def _component_eaf_plans(target: Target) -> list[EafEncoding]:
    """Every `eaf` plan that describes some part of this release's arrays.

    One for a Dense or Ragged release. Two for a Hybrid one, whose nested Dense
    Component declares its own (spec §6a, §16) -- and whose `analyses.tsv`
    describes the Analyses of both.
    """
    plans = [target.store.manifest.encoding.eaf]
    plans += [child.store.manifest.encoding.eaf for child in target.children]
    return plans


def _declares_association(target: Target) -> list[str]:
    return [
        str(row.get("analysis_id", ""))
        for row in target.analyses.rows
        if row.get("eaf_scope", "") == EafScope.ASSOCIATION.value
    ]


def eaf_scope_plan(target: Target) -> Iterator[str]:
    """`eaf_scope` (per Analysis) and the plan's `eaf` (per release) agree (issue #106).

    `reference` excepted: a completed release whose Analyses reported no frequency
    of their own still holds one for every cell it imputed -- the panel's -- so an
    Analysis that gained imputed cells declaring `eaf_scope=association` states a
    fact about its arrays (ADR 0037 §4, issue #113). A Hybrid release's Analyses
    span both components, and `eaf_reference` is per component (spec §6a), so the
    question is asked of every component's plan.
    """
    declares = _declares_association(target)
    holds_frequencies = any(
        not plan.is_absent or plan.reference for plan in _component_eaf_plans(target)
    )
    if not holds_frequencies and declares:
        yield (
            f"{len(declares)} analysis/analyses declare "
            f"eaf_scope=association (first {declares[0]!r}) but no component "
            "of this release declares an eaf plane or reference EAF; the store's metadata "
            "and its arrays disagree about whether it holds frequencies "
            "(ADR 0036, ADR 0037)"
        )
    if target.store.manifest.encoding.eaf.kind in ("float32", "int8_residual") and not declares:
        yield (
            "the manifest declares an eaf plane but no Analysis declares "
            "eaf_scope=association; a release stores frequencies for the Analyses that "
            "say they have them, or for none"
        )


def _checked_rows(target: Target) -> Iterator[tuple[str, str, dict[str, str]]]:
    """(analysis_id, recorded outcome, row) for each Analysis owing orientation evidence.

    When no component stores frequencies *of its own*, every
    `eaf_scope=association` describes the panel's reference EAF and nothing else
    (issue #113): there is no cohort column to check, so no evidence is owed.
    """
    stores_own = any(not plan.is_absent for plan in _component_eaf_plans(target))
    for row in target.analyses.rows:
        if row.get("eaf_scope", "") != EafScope.ASSOCIATION.value:
            continue
        outcome = row.get("eaf_orientation", "")
        if outcome or stores_own:
            yield row.get("analysis_id", ""), outcome, row


def _recorded_evidence(target: Target) -> dict[str, dict[str, str]]:
    recorded = target.store.manifest.provenance.get("eaf_orientation") or {}
    return {str(entry.get("analysis_id", "")): entry for entry in recorded.get("analyses", [])}


def orientation_evidence(target: Target) -> Iterator[str]:
    """Every Analysis storing EAF carries orientation evidence, recorded twice alike.

    Checks the *recorded* evidence, not a reference panel: a Store Release must be
    interpretable long after its panel has moved. A blank outcome means the
    frequencies were never checked against a reference, so a column reported
    against the other allele would be indistinguishable from a correct one; a
    `failed` one says it is reported against the other allele. `analyses.tsv` and
    the manifest's provenance must agree on what was checked.
    """
    by_id = _recorded_evidence(target)
    for analysis_id, outcome, row in _checked_rows(target):
        if not outcome:
            yield (
                f"analysis {analysis_id!r} stores per-association EAF but carries no "
                "eaf_orientation evidence -- its frequencies have never been checked "
                "against a reference, so a column reported against the other allele "
                "would be indistinguishable from a correct one (issue #115). Rebuild "
                "the store, or audit it with a reference panel"
            )
            continue
        if outcome == EafOrientationOutcome.FAILED.value:
            yield (
                f"analysis {analysis_id!r} recorded eaf_orientation=failed "
                f"(r = {row.get('eaf_orientation_r', '?')}): its stored EAF is reported "
                "against the other allele and must not be read as a frequency of the "
                "stored effect allele"
            )
        entry = by_id.get(analysis_id)
        if entry is None:
            yield (
                f"analysis {analysis_id!r} records eaf_orientation={outcome!r} in "
                "analyses.tsv, but manifest.json's provenance has no orientation "
                "evidence for it -- the two must agree on what was checked"
            )
        elif str(entry.get("outcome", "")) != outcome:
            yield (
                f"analysis {analysis_id!r} records eaf_orientation={outcome!r} in "
                f"analyses.tsv but {entry.get('outcome')!r} in manifest.json provenance"
            )


def orientation_unverified(target: Target) -> Iterator[str]:
    """`unverified` is honestly recorded, and still not evidence of correctness."""
    by_id = _recorded_evidence(target)
    for analysis_id, outcome, _row in _checked_rows(target):
        if outcome == EafOrientationOutcome.UNVERIFIED.value:
            entry = by_id.get(analysis_id, {})
            yield (
                f"analysis {analysis_id!r} stores EAF whose orientation is unverified"
                + (f": {entry['note']}" if entry.get("note") else "")
            )


def regions_table(target: Target) -> Iterator[str]:
    """The region table is declared in the manifest exactly when the release holds
    `regions.tsv`, with the SHA-256 of its bytes; it parses; and it names only Analyses
    the release holds (spec §17, issue #430). A table that does not would refuse the
    release's completion. The version a release carrying it is stamped is
    `manifest.format_stamp`'s."""
    try:
        table = read_store_regions(target.store.path)
        if table is not None:
            check_region_table(table, (row["analysis_id"] for row in target.analyses.rows))
    except RegionTableError as exc:
        yield f"{REGIONS_FILE}: {exc}"


def _with_regions(target: Target) -> bool:
    """A release that holds `regions.tsv` or whose manifest declares one."""
    return _validated_table(target) and (
        (target.store.path / REGIONS_FILE).exists() or target.store.manifest.regions is not None
    )


def _validated_table(target: Target) -> bool:
    """Dense and Ragged releases and a Hybrid's Dense Component, but not its outer one."""
    return target.layout is not PrimaryStorageLayout.HYBRID


def _top_level_with_table(target: Target) -> bool:
    return target.parent is None and target.store.analyses_path.exists()


def _table_rule(name: str, run: Callable[[Target], Iterable[str]]) -> Check:
    return Check(
        name=name, kinds=Kind.RELEASE, when=_validated_table, after=(READABLE,), run=run
    )


def _orientation_rule(
    name: str, run: Callable[[Target], Iterable[str]], severity: Severity = Severity.ERROR
) -> Check:
    return Check(
        name=name,
        kinds=Kind.RELEASE,
        when=_top_level_with_table,
        after=(ORIENTATION_READABLE,),
        severity=severity,
        run=run,
    )


#: The rules that read `analyses.tsv` as a table of Analyses.
TABLE_RULES: tuple[Rule, ...] = (
    Check(
        name=READABLE, kinds=Kind.RELEASE, when=_validated_table, after=(REQUIRED,), run=readable
    ),
    _table_rule("analyses.retired_columns", retired_columns),
    _table_rule("analyses.analysis_index", analysis_index),
    _table_rule("analyses.analysis_id", analysis_id),
    _table_rule("analyses.completion_declared", completion_declared),
    _table_rule("analyses.vocabularies", vocabularies),
    _table_rule("analyses.original_sd_recorded", original_sd_recorded),
    _table_rule("analyses.sd_scale_standardised", sd_scale_standardised),
    _table_rule("analyses.top_hit_count_columns", top_hit_count_columns),
    Check(
        name="analyses.se_out_of_range_count", kinds=Kind.RELEASE, when=_with_analyses,
        after=(READABLE,), run=se_out_of_range_count,
    ),
    Check(
        name="analyses.se_out_of_range_partition", kinds=Kind.RELEASE, when=_hybrid_root,
        after=("analyses.se_out_of_range_count",), run=se_out_of_range_partition,
    ),
    Check(
        name="regions.table", kinds=Kind.RELEASE, when=_with_regions, after=(READABLE,),
        run=regions_table,
    ),
)

#: The rules that judge the EAF a release stores against the evidence it records.
ORIENTATION_RULES: tuple[Rule, ...] = (
    Check(
        name=ORIENTATION_READABLE,
        kinds=Kind.RELEASE,
        when=_top_level_with_table,
        run=orientation_readable,
    ),
    _orientation_rule("eaf_scope.plan", eaf_scope_plan),
    _orientation_rule("eaf_orientation.evidence", orientation_evidence),
    _orientation_rule("eaf_orientation.unverified", orientation_unverified, Severity.WARNING),
)

"""Focused seam tests for `_collect_analyses` (dense Observed build).

`_collect_analyses` gives each Analysis the manifest row declared of it, in the
manifest's order, with the frequency scope its kept records show. The seam locks
two contracts the dense source build relies on: blank optional metadata stays
blank, and one EAF-bearing record stamps the whole Analysis association/unverified.
"""

from __future__ import annotations

from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.layouts.dense.build import (
    AnalysisSource,
    NormalisedAssociation,
    _collect_analyses,
)
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import EafOrientationOutcome, EafScope
from opengwasdb.readers.interface import StreamTally
from opengwasdb.variants import CanonicalVariant

_VARIANT = CanonicalVariant(chromosome="1", position=1000, effect_allele="A", other_allele="G")


def _record(analysis_id: str, *, eaf: float | None) -> NormalisedAssociation:
    return NormalisedAssociation(analysis_id=analysis_id, variant=_VARIANT, z=2.0, se=0.5, eaf=eaf)


def _source(analysis_id: str, **declared: str) -> AnalysisSource:
    return AnalysisSource(
        analysis=Analysis(analysis_id=analysis_id, **declared),
        rules=AnalysisRules(analysis_id),
        se_divisor=1.0,
        batches=[],
        tally=StreamTally(),
    )


def test_later_eaf_record_stamps_whole_analysis_as_association_unverified() -> None:
    """The EAF scope is per-Analysis, not per-record: a first record that
    carries no EAF cannot hide that a later record of the same Analysis does."""
    [analysis] = _collect_analyses(
        [_record("a1", eaf=None), _record("a1", eaf=0.25)], [_source("a1")]
    )

    assert analysis.analysis_id == "a1"
    assert analysis.eaf_scope == EafScope.ASSOCIATION.value
    assert analysis.eaf_orientation == EafOrientationOutcome.UNVERIFIED.value


def test_declared_metadata_is_carried_and_blank_stays_blank() -> None:
    """Attribution Metadata is what the manifest row declares; unresolved columns
    are written blank, never fabricated."""
    populated = _source(
        "a2", analysis_label="Disease", trait_ontology_id="EFO:0001073", license="CC0",
        first_author="J. Smith", stored_effect_scale="log_or",
    )
    by_id = {
        a.analysis_id: a
        for a in _collect_analyses(
            [_record("a2", eaf=0.1), _record("a3", eaf=None)], [populated, _source("a3")]
        )
    }

    assert by_id["a2"].analysis_label == "Disease"
    assert by_id["a2"].trait_ontology_id == "EFO:0001073"
    assert by_id["a2"].first_author == "J. Smith"
    assert by_id["a2"].stored_effect_scale == "log_or"
    assert by_id["a3"].analysis_label == ""
    assert by_id["a3"].first_author == ""
    assert by_id["a3"].eaf_scope == EafScope.ABSENT.value
    assert by_id["a3"].eaf_orientation == ""


def test_analyses_follow_the_manifest_order_and_one_without_a_kept_record_is_absent() -> None:
    result = _collect_analyses(
        [_record("b1", eaf=None), _record("b2", eaf=None)],
        [_source("b2"), _source("gone"), _source("b1")],
    )

    assert [a.analysis_id for a in result] == ["b2", "b1"]

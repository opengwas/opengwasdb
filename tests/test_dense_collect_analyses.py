"""Focused seam tests for `_collect_analyses` (dense Observed build).

`_collect_analyses` groups one Analysis per id from normalised association
records. The seam locks three contracts the dense source build relies on:
blank optional metadata stays blank, one EAF-bearing record stamps the whole
Analysis association/unverified, and a mixed `stored_effect_scale` across an
Analysis's records is refused.
"""

from __future__ import annotations

import pytest

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.layouts.dense.build import _collect_analyses
from opengwasdb.model.enums import EafOrientationOutcome, EafScope, StoredEffectScale
from opengwasdb.variants import CanonicalVariant

_VARIANT = CanonicalVariant(chromosome="1", position=1000, effect_allele="A", other_allele="G")


def _record(
    analysis_id: str, *, scale: StoredEffectScale, eaf: float | None, **meta
) -> NormalisedAssociation:
    return NormalisedAssociation(
        analysis_id=analysis_id,
        variant=_VARIANT,
        z=2.0,
        se=0.5,
        stored_effect_scale=scale,
        eaf=eaf,
        **meta,
    )


def test_later_eaf_record_stamps_whole_analysis_as_association_unverified() -> None:
    """The EAF scope is per-Analysis, not per-record: a first record that
    carries no EAF cannot hide that a later record of the same Analysis does."""
    first = _record("a1", scale=StoredEffectScale.SD, eaf=None, first_author="")
    later = _record(
        "a1",
        scale=StoredEffectScale.SD,
        eaf=0.25,
        analysis_label="Height",
        first_author="J. Smith",
    )
    [analysis] = _collect_analyses([first, later])

    assert analysis.analysis_id == "a1"
    assert analysis.eaf_scope == EafScope.ASSOCIATION.value
    assert analysis.eaf_orientation == EafOrientationOutcome.UNVERIFIED.value
    assert analysis.stored_effect_scale == StoredEffectScale.SD.value


def test_populated_optional_metadata_is_carried_and_blank_stays_blank() -> None:
    """Attribution Metadata resolves from the source; unresolved columns are
    written blank, never fabricated."""
    populated = _record(
        "a2",
        scale=StoredEffectScale.LOG_OR,
        eaf=0.1,
        analysis_label="Disease",
        trait_ontology_id="EFO:0001073",
        trait_ontology_label="body height",
        license="CC0",
        publication_doi="10.1000/xyz",
        publication_pmid="12345678",
        consortium="GIANT",
        first_author="J. Smith",
    )
    blank = _record("a3", scale=StoredEffectScale.SD, eaf=None)
    by_id = {a.analysis_id: a for a in _collect_analyses([blank, populated])}

    assert by_id["a2"].analysis_label == "Disease"
    assert by_id["a2"].trait_ontology_id == "EFO:0001073"
    assert by_id["a2"].publication_doi == "10.1000/xyz"
    assert by_id["a2"].first_author == "J. Smith"
    assert by_id["a3"].analysis_label == ""
    assert by_id["a3"].first_author == ""
    assert by_id["a3"].eaf_scope == EafScope.ABSENT.value
    assert by_id["a3"].eaf_orientation == ""


def test_mixed_stored_effect_scale_within_an_analysis_is_refused() -> None:
    """One Analysis is one effect scale; a source mixing scales says so loudly."""
    with pytest.raises(ValueError, match=r"analysis a1 has mixed stored_effect_scale values"):
        _collect_analyses(
            [
                _record("a1", scale=StoredEffectScale.SD, eaf=None),
                _record("a1", scale=StoredEffectScale.LOG_OR, eaf=None),
            ]
        )


def test_first_record_conflict_keeps_first_and_output_is_sorted() -> None:
    """Insertion order decides which record's metadata stands for a repeated
    Analysis; output order is by analysis_id."""
    first = _record("b2", scale=StoredEffectScale.SD, eaf=None, analysis_label="Second id")
    second = _record("b1", scale=StoredEffectScale.SD, eaf=None, analysis_label="First id")
    result = _collect_analyses([first, second])

    assert [a.analysis_id for a in result] == ["b1", "b2"]

"""The Reference-Completion ancestry-match filter (ADR 0028, issue #108).

`derive_impute_analysis_ids` decides which Analyses an LD panel may impute.
Its no-ancestry-anywhere fallback -- impute everything -- is deliberate
backward compatibility, but it is also the one path where a genuinely
mixed-ancestry store gets imputed against a single panel's LD/EAF with no
signal to the operator, so it must announce itself.
"""

from __future__ import annotations

import logging

import pytest

from opengwasdb.completion.ancestry_filter import (
    AncestryFilterError,
    canonical_ancestry,
    derive_impute_analysis_ids,
)


def _rows(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"analysis_id": aid, "assigned_ancestry": anc} for aid, anc in pairs]


def test_matching_subset_is_selected_when_ancestry_is_known():
    got = derive_impute_analysis_ids(
        _rows(("a1", "EUR"), ("a2", "EAS"), ("a3", "EUR")), "EUR"
    )
    assert got == {"a1", "a3"}


def test_no_ancestry_anywhere_still_imputes_everything():
    # Backward compatibility (pre-ADR-0028 releases): None means "impute all".
    assert derive_impute_analysis_ids(_rows(("a1", ""), ("a2", "")), "EUR") is None


def test_no_ancestry_anywhere_warns_that_one_panel_is_applied_to_all(caplog):
    """The fallback must be loud: silently imputing every Analysis against one
    panel is indistinguishable, in the logs, from correctly imputing a
    genuinely single-ancestry store (issue #108)."""
    with caplog.at_level(logging.WARNING, logger="opengwasdb.completion.ancestry_filter"):
        derive_impute_analysis_ids(_rows(("a1", ""), ("a2", ""), ("a3", "")), "EUR")

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "no-ancestry fallback imputed every Analysis without warning"
    message = warnings[0].getMessage()
    assert "3" in message  # how many Analyses are affected
    assert "EUR" in message  # which panel they are all being imputed against


def test_known_ancestry_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="opengwasdb.completion.ancestry_filter"):
        derive_impute_analysis_ids(_rows(("a1", "EUR"), ("a2", "EAS")), "EUR")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_ancestry_known_but_none_match_fails_rather_than_completing_nothing():
    """Issue #98. Distinct from the fallback: ancestry *is* known and nothing
    matches, so nothing would be imputed -- and a Reference-Completed release
    holding zero imputed cells is indistinguishable, to everything downstream,
    from one where imputation was attempted and failed. Returning `None` here
    would be worse still (impute everything against a panel every Analysis is
    known not to match), so the answer is neither: refuse."""
    with pytest.raises(AncestryFilterError) as excinfo:
        derive_impute_analysis_ids(_rows(("a1", "EAS"), ("a2", "AFR")), "EUR")

    message = str(excinfo.value)
    assert "EUR" in message  # the panel that was asked for
    assert "EAS" in message and "AFR" in message  # what the store actually holds


def test_a_different_vocabulary_for_the_same_ancestry_still_matches():
    """Issue #98, the case that produced a silently empty store: the registry
    writes `assigned_ancestry="European"` while an LD panel directory is named
    `EUR`. They are the same ancestry spelled two ways, and both spellings
    occur in one `analyses.tsv` (opengwasdb-stores writes the code down the
    `af_assigned` path and the word down `source_trusted_no_af`)."""
    got = derive_impute_analysis_ids(
        _rows(("a1", "European"), ("a2", "East Asian"), ("a3", "EUR")), "EUR"
    )
    assert got == {"a1", "a3"}


def test_matching_is_symmetric_in_which_side_spells_it_out():
    """The panel directory is the side spelled out, the metadata the side
    abbreviated -- the mirror of the case above, and equally real."""
    assert derive_impute_analysis_ids(_rows(("a1", "EUR"), ("a2", "AFR")), "European") == {"a1"}


def test_a_panel_outside_the_superpopulation_vocabulary_matches_on_its_own_terms():
    """Normalisation must not become a requirement to be normalisable: a
    cohort-specific panel (`FIN`) names an ancestry the super-population
    vocabulary has no code for, and matching it is exact-equality's job."""
    assert derive_impute_analysis_ids(
        _rows(("a1", "FIN"), ("a2", "EUR")), "FIN"
    ) == {"a1"}


def test_two_labels_that_normalise_to_nothing_do_not_match_each_other():
    """The failure mode the exact-equality branch must not open: `None ==
    None` would make every unmappable label match every other one."""
    with pytest.raises(AncestryFilterError):
        derive_impute_analysis_ids(_rows(("a1", "Mixed"),), "FIN")


def test_an_unmappable_ancestry_label_matches_nothing_rather_than_everything():
    """"Mixed" is known-but-unroutable (ADR 0028): it is not EUR, and it must
    not become EUR by failing to normalise."""
    with pytest.raises(AncestryFilterError):
        derive_impute_analysis_ids(_rows(("a1", "Mixed"), ("a2", "admixed")), "EUR")


def test_the_refusal_names_a_way_forward():
    """And one every caller has: `complete_hybrid_store` and the CLI take no
    `impute_analysis_ids`, so naming that parameter would be advice most
    callers -- including the command #98 was reported against -- cannot take."""
    with pytest.raises(AncestryFilterError, match="against a panel its analyses match"):
        derive_impute_analysis_ids(_rows(("a1", "EAS"),), "EUR")


def test_the_refusal_says_whether_the_panel_name_was_understood():
    """A mismatch after normalisation and a panel named something the
    vocabulary has no code for are different problems with the same symptom."""
    with pytest.raises(AncestryFilterError, match="normalises to EUR"):
        derive_impute_analysis_ids(_rows(("a1", "EAS"),), "European")
    with pytest.raises(AncestryFilterError, match="no super-population"):
        derive_impute_analysis_ids(_rows(("a1", "EAS"),), "FIN")


def test_north_african_is_not_african():
    """`ancestry.routing`'s substring matching answers AFR here, because
    `african` precedes `north africa` in its ordered list. Guessing at a
    cohort's free-text description that way is tolerable; deciding which
    Analyses an LD panel may impute that way is not -- it would impute a NAF
    store against the AFR panel and refuse it against its own."""
    assert canonical_ancestry("North African") == "NAF"
    assert canonical_ancestry("African") == "AFR"
    with pytest.raises(AncestryFilterError):
        derive_impute_analysis_ids(_rows(("a1", "North African"),), "AFR")

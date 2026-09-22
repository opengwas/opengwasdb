"""The effect-source resolution rule (issue #213).

GWAS-SSF permits an Analysis to report its effect as either `beta` or
`odds_ratio`, and `beta = log(odds_ratio)`. Which column an Analysis resolved
to is a fact about the file, not an assumption a reader may make silently --
so it is resolved once, here, and reported to the caller.

The rules these tests pin down, each of which has a silent-wrong-answer
failure mode if it is not enforced:

* `beta` and `odds_ratio` both present -> `beta` wins, by an explicit rule.
* A candidate effect column named twice in the header -> `ValueError`, never
  last-wins. A real harmonised file (`GCST006329`) carries `beta ` and `beta`.
* Surrounding whitespace is not part of a column's name, so `beta ` and `beta`
  collide; the matched cell is still looked up by its verbatim spelling.
* `beta` accepts the enumerated spellings `("beta", "BETA")` (issue #214);
  mixed case is *not* folded, and a header carrying both spellings is ambiguous.
* Neither column -> `None`, never a fabricated source.
"""

from __future__ import annotations

import dataclasses

import pytest

from opengwasdb.readers import EffectSource, EffectSourceKind, resolve_effect_source


def test_beta_column_resolves_to_beta():
    source = resolve_effect_source(["chromosome", "beta", "standard_error"])

    assert source == EffectSource(
        column_name="beta",
        kind=EffectSourceKind.BETA,
        is_derived=False,
        assumes_standardised=False,
    )


def test_odds_ratio_column_resolves_to_a_derived_odds_ratio_source():
    """`beta = log(odds_ratio)`, so the beta is derived, not read verbatim."""
    source = resolve_effect_source(["chromosome", "odds_ratio", "standard_error"])

    assert source == EffectSource(
        column_name="odds_ratio",
        kind=EffectSourceKind.ODDS_RATIO,
        is_derived=True,
        assumes_standardised=False,
    )


def test_beta_wins_over_odds_ratio_when_both_are_present():
    source = resolve_effect_source(["chromosome", "odds_ratio", "beta"])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA
    assert source.column_name == "beta"


def test_neither_effect_column_resolves_to_none():
    assert resolve_effect_source(["chromosome", "base_pair_location", "standard_error"]) is None


def test_a_bytes_header_is_accepted():
    source = resolve_effect_source([b"chromosome", b"odds_ratio", b"standard_error"])

    assert source is not None
    assert source.kind is EffectSourceKind.ODDS_RATIO
    assert source.column_name == "odds_ratio"


def test_a_whitespace_padded_spelling_resolves_to_the_exact_header_name():
    """The column is looked up by its verbatim header cell, padding included."""
    source = resolve_effect_source(["chromosome", "beta ", "standard_error"])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA
    assert source.column_name == "beta "


def test_a_padded_and_unpadded_beta_are_a_duplicate():
    """The `GCST006329` shape: `beta ` holds the values and `beta` is all `NA`.

    Matching exactly would make these two different columns and let a last-wins
    lookup read the empty one; whitespace is not part of a column's name.
    """
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        resolve_effect_source(["chromosome", "beta ", "standard_error", "beta"])


def test_a_padded_and_unpadded_odds_ratio_are_a_duplicate():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'odds_ratio' in header"):
        resolve_effect_source(["odds_ratio ", "odds_ratio"])


def test_a_bytes_header_with_a_padded_duplicate_is_rejected():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        resolve_effect_source([b"beta ", b"beta"])


def test_duplicate_beta_is_rejected():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        resolve_effect_source(["beta", "beta", "odds_ratio"])


def test_duplicate_odds_ratio_is_rejected():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'odds_ratio' in header"):
        resolve_effect_source(["odds_ratio", "odds_ratio"])


def test_duplicate_lower_precedence_column_is_rejected_even_when_beta_is_present():
    """Precedence does not excuse a malformed header: the duplicate is still read.

    A last-wins resolution would silently take one of two `odds_ratio` columns
    even though `beta` happens to be the one used today; the duplicate is a
    header no reader can honestly interpret.
    """
    with pytest.raises(ValueError, match=r"Duplicate effect column 'odds_ratio' in header"):
        resolve_effect_source(["beta", "odds_ratio", "odds_ratio"])


def test_a_duplicate_non_effect_column_is_not_this_rule():
    """`standard_error` is not a candidate effect column; this rule stays silent.

    The duplicate-projected-column rule lives in the projection, not here; this
    test fixes the boundary so the two cannot quietly become one.
    """
    source = resolve_effect_source(["beta", "standard_error", "standard_error"])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA


def test_effect_source_kind_values_are_the_column_spellings():
    assert EffectSourceKind.BETA.value == "beta"
    assert EffectSourceKind.ODDS_RATIO.value == "odds_ratio"


def test_effect_source_is_frozen():
    source = resolve_effect_source(["beta"])
    assert source is not None

    # The attribute name is held in a variable so this needs neither a type
    # suppression nor a constant `setattr`; a frozen dataclass raises from its
    # own `__setattr__` all the same.
    attribute = "column_name"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(source, attribute, "odds_ratio")


# --- `BETA`, an enumerated spelling of `beta` (issue #214) ---


def test_upper_case_beta_resolves_to_the_beta_kind():
    """`GCST90044776` spells the column `BETA`; it is the same effect source."""
    source = resolve_effect_source(["chromosome", "BETA", "standard_error"])

    assert source == EffectSource(
        column_name="BETA",
        kind=EffectSourceKind.BETA,
        is_derived=False,
        assumes_standardised=False,
    )


def test_a_padded_upper_case_beta_resolves_to_the_exact_header_name():
    source = resolve_effect_source(["BETA "])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA
    assert source.column_name == "BETA "


def test_upper_case_beta_wins_over_odds_ratio():
    source = resolve_effect_source(["BETA", "odds_ratio"])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA
    assert source.column_name == "BETA"


@pytest.mark.parametrize("spelling", ["Beta", "bEtA", "BEta"])
def test_mixed_case_beta_is_not_an_accepted_spelling(spelling):
    """The accepted set is enumerated, not case-folded: only `beta` and `BETA`."""
    assert resolve_effect_source(["chromosome", spelling, "standard_error"]) is None


def test_a_file_carrying_both_beta_and_upper_case_beta_is_ambiguous():
    """Two spellings of one column is a header no reader can choose from."""
    with pytest.raises(
        ValueError, match=r"Ambiguous effect column: header carries both 'beta' and 'BETA'"
    ):
        resolve_effect_source(["chromosome", "beta", "standard_error", "BETA"])


def test_a_padded_beta_and_upper_case_beta_are_ambiguous():
    with pytest.raises(
        ValueError, match=r"Ambiguous effect column: header carries both 'beta' and 'BETA'"
    ):
        resolve_effect_source(["beta ", "BETA"])


def test_two_upper_case_betas_are_a_duplicate():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'BETA' in header"):
        resolve_effect_source(["BETA", "BETA"])


def test_a_padded_and_unpadded_upper_case_beta_are_a_duplicate():
    with pytest.raises(ValueError, match=r"Duplicate effect column 'BETA' in header"):
        resolve_effect_source(["BETA ", "BETA"])


def test_a_duplicate_is_reported_before_ambiguity():
    """A header with two `beta`s and one `BETA` is a duplicate, not only ambiguous."""
    with pytest.raises(ValueError, match=r"Duplicate effect column 'beta' in header"):
        resolve_effect_source(["beta", "beta", "BETA"])


def test_a_bytes_header_with_upper_case_beta_is_accepted():
    source = resolve_effect_source([b"chromosome", b"BETA", b"standard_error"])

    assert source is not None
    assert source.kind is EffectSourceKind.BETA
    assert source.column_name == "BETA"

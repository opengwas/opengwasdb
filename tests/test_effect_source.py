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

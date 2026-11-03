"""The one rule that signs a statistic by an allele swap (`readers.orientation`).

Every projection a Source Reader presents reads its z and frequency through
it, so these tests drive the projections themselves over in-memory rows: a
flipped row's z is negated and its frequency complemented, and an unflipped
row's are untouched, in the association stream, the batch form, the site
extraction and the metrics block.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.orientation import a1_frequencies, is_palindromic, swap, swap_one
from opengwasdb.readers.tabular import (
    SourceRow,
    extract_at_sites,
    metrics_chunks_from_rows,
    stream_associations,
)


def _row(*, flipped: bool, beta: float | None = 0.4, se: float | None = 0.2,
         af: float | None = 0.3) -> SourceRow:
    # The effect allele is `alt` and A1 is A, so the row whose alt is G is flipped.
    ref, alt = ("G", "A") if not flipped else ("A", "G")
    return SourceRow(
        chromosome="1", position=100, ref=ref, alt=alt, alid="1:100:A:G",
        flipped=flipped, af_alt=af, beta=beta, se=se,
    )


def test_swap_one_negates_z_and_complements_frequency_only_when_flipped() -> None:
    assert swap_one(False, 2.0, 0.25) == (2.0, 0.25)
    assert swap_one(True, 2.0, 0.25) == (-2.0, 0.75)
    assert swap_one(True, 2.0, None) == (-2.0, None)
    z, eaf = swap_one(True, math.inf, None)
    assert z == -math.inf and eaf is None


def test_swap_is_swap_one_over_arrays() -> None:
    flipped = np.array([False, True, True])
    z, eaf = swap(flipped, np.array([1.0, 2.0, -3.0]), np.array([0.1, 0.2, np.nan]))
    np.testing.assert_array_equal(z, [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(eaf[:2], [0.1, 0.8])
    assert np.isnan(eaf[2])
    np.testing.assert_array_equal(a1_frequencies(flipped[:2], np.array([0.1, 0.2])), [0.1, 0.8])


@pytest.mark.parametrize("flipped", [False, True])
def test_the_association_stream_orients_through_the_rule(flipped: bool) -> None:
    (association,) = stream_associations([_row(flipped=flipped)], StoredEffectScale.SD)
    assert association.z == pytest.approx(-2.0 if flipped else 2.0)
    assert association.eaf == pytest.approx(0.7 if flipped else 0.3)
    # The source's own labelling is kept, whichever allele is stored.
    assert (association.ref, association.alt) == (("A", "G") if flipped else ("G", "A"))


@pytest.mark.parametrize("flipped", [False, True])
def test_site_extraction_and_metric_blocks_orient_the_frequency(flipped: bool) -> None:
    rows = [_row(flipped=flipped)]
    assert extract_at_sites(rows, {"1:100:A:G"})["1:100:A:G"].af == pytest.approx(
        0.7 if flipped else 0.3
    )
    (chunk,) = metrics_chunks_from_rows(rows)
    np.testing.assert_allclose(chunk.a1_frequency(), [0.7 if flipped else 0.3])


def test_a_row_without_an_effect_keeps_a_nan_z_on_either_allele() -> None:
    for flipped in (False, True):
        (association,) = stream_associations(
            [_row(flipped=flipped, beta=None)], StoredEffectScale.SD
        )
        assert math.isnan(association.z)


def test_palindromes_are_judged_case_blind() -> None:
    assert is_palindromic("a", "T") and is_palindromic("C", "g")
    assert not is_palindromic("A", "G")

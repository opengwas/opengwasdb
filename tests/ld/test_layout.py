"""The ragged walk the store's arrays share (spec §4, §5): `spans` and its inverse, `spans_at`,
against a loop that builds every span element by element."""

from __future__ import annotations

import numpy as np
import pytest

from ogld.store.layout import spans, spans_at


def walked(first: list[int], length: list[int]) -> tuple[list[int], list[int]]:
    """Each element of the spans, in order: the span it belongs to and its value."""
    span, value = [], []
    for i, (start, count) in enumerate(zip(first, length, strict=True)):
        for k in range(count):
            span.append(i)
            value.append(start + k)
    return span, value


@pytest.mark.parametrize("seed", range(6))
def test_spans_at_inverts_spans_element_by_element(seed: int) -> None:
    rng = np.random.default_rng(seed)
    length = rng.integers(0, 6, size=40)
    length[[0, 7, 8, -1]] = 0  # empty spans first, together and last
    first = rng.integers(-5, 50, size=40)
    span, value = walked(first.tolist(), length.tolist())
    assert len(value) > 50, "the spans must hold many elements for this to mean anything"
    np.testing.assert_array_equal(spans(first, length), value)
    at = rng.permutation(len(value))
    found_span, found_value = spans_at(first, length, at)
    np.testing.assert_array_equal(found_span, np.asarray(span)[at])
    np.testing.assert_array_equal(found_value, np.asarray(value)[at])

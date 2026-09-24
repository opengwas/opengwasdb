"""Shared assertions for serial-versus-parallel band-write comparisons.

Epic #217's band-write tickets (issue #220 and its siblings) each build a
serial release and a parallel one and assert they are identical. The
comparison is the same shape in the Dense and Hybrid suites, so it lives here
rather than as copies the duplication gate would rightly flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The arrays the Dense Component band write owns. A residual `eaf` build also
#: carries the baseline and exception tables; a `float32` or `absent` one does
#: not, so each array is compared only where the build produced it.
BAND_WRITE_ARRAYS = (
    "z",
    "se",
    "eaf",
    "z_overflow_index",
    "z_overflow_value",
    "eaf_exception_index",
    "eaf_exception_value",
    "eaf_baseline",
)


def assert_same_band_arrays(before: Any, after: Any) -> None:
    """Every band-written array, byte for byte (NaN equal to NaN)."""
    for name in BAND_WRITE_ARRAYS:
        if name not in before:
            assert name not in after, name
            continue
        np.testing.assert_array_equal(before[name][:], after[name][:], err_msg=name)


def assert_same_top_hits(before: Any, after: Any, names: tuple[str, ...]) -> None:
    """Every top-hit tier's arrays, for the fields the two builds both wrote."""
    for key in ("p_5e_04", "p_5e_06", "p_5e_08"):
        for name in names:
            if name not in before[f"top_hits/{key}"]:
                assert name not in after[f"top_hits/{key}"], f"top_hits/{key}/{name}"
                continue
            np.testing.assert_array_equal(
                before[f"top_hits/{key}"][name][:],
                after[f"top_hits/{key}"][name][:],
                err_msg=f"top_hits/{key}/{name}",
            )

"""Shared assertions for serial-versus-parallel band-write comparisons.

Epic #217's band-write tickets (issue #220 and its siblings) each build a
serial release and a parallel one and assert they are identical. The
comparison is the same shape in the Dense and Hybrid suites, so it lives here
rather than as copies the duplication gate would rightly flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengwasdb.store.arrays import ArrayRole, create_array

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


def replace_exception_table(root: Any, index_name: str, value_name: str) -> None:
    """Drop one exception table's first cell, leaving the arrays shard-valid.

    Used to plant the "the table lost a cell" defect in a built store.  The
    arrays are rewritten through the seam, so they are sharded as a 0.2.0
    release's must be and the error under test is the lost cell, not a format
    rule the corruption accidentally tripped (#247).  The tables are written
    uncompressed (`compressor=None`).
    """
    kept_index = np.asarray(root[index_name][:])[1:]
    kept_value = np.asarray(root[value_name][:])[1:]
    for name, data, dtype in (
        (index_name, kept_index, "int64"),
        (value_name, kept_value, "float32"),
    ):
        del root[name]
        create_array(
            root,
            name,
            ArrayRole.EXCEPTION_TABLE,
            data=np.asarray(data, dtype=dtype),
            compressor=None,
        )


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

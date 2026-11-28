"""Shared assertions that two Stores built two ways hold the same data.

Epic #217's band-write tickets (issue #220 and its siblings) each build a
serial release and a parallel one and assert they are identical. The
comparison is the same shape in the Dense and Hybrid suites, so it lives here
rather than as copies the duplication gate would rightly flag. So does the
whole-Store Hybrid comparison (`assert_hybrid_stores_match`) the single-pass,
checkpoint, rsid-conflict and plain-list suites judge a build by.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.query import query_store
from opengwasdb.store.arrays import create_array
from opengwasdb.store.open import open_store
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.validation import validate_store

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
    write_exception_table(root, index_name, value_name, kept_index, kept_value)


def write_exception_table(
    root: Any, index_name: str, value_name: str, index: Any, value: Any
) -> None:
    """Replace an exception table's two arrays, uncompressed, through the array seam."""
    for name, data, dtype in ((index_name, index, "int64"), (value_name, value, "float32")):
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


def assert_hybrid_stores_match(reference: Path, candidate: Path) -> None:
    """Every stored data artifact matches: variant tables, dense matrix,
    overflow CSR, top hits and analyses.tsv (manifest timestamps excluded)."""
    assert validate_store(candidate).ok, validate_store(candidate).errors

    for name in ("variants.tsv.gz",):
        with gzip.open(reference / name, "rt", encoding="utf-8") as handle:
            left = handle.read()
        with gzip.open(candidate / name, "rt", encoding="utf-8") as handle:
            right = handle.read()
        assert left == right, name

    np.testing.assert_array_equal(
        np.load(reference / "dense" / "dense_to_shared.npy"),
        np.load(candidate / "dense" / "dense_to_shared.npy"),
    )
    ref_store = open_store(reference)
    cand_store = open_store(candidate)
    ref_dense = ref_store.dense_component().arrays(mode="r")
    cand_dense = cand_store.dense_component().arrays(mode="r")
    for name in ("z", "se"):
        np.testing.assert_array_equal(ref_dense[name][:], cand_dense[name][:])
    ref_ragged = ref_store.arrays(mode="r")["ragged"]
    cand_ragged = cand_store.arrays(mode="r")["ragged"]
    for name in ("offsets", "variant_index", "z", "se"):
        np.testing.assert_array_equal(ref_ragged[name][:], cand_ragged[name][:])

    with query_store(reference) as left_q, query_store(candidate) as right_q:
        for threshold in (5e-4, 5e-6, 5e-8):
            left_hits = left_q.top_hits(threshold=threshold)
            right_hits = right_q.top_hits(threshold=threshold)
            for field in ("variant_index", "analysis_index", "z", "se"):
                np.testing.assert_array_equal(left_hits[field], right_hits[field])

    assert (reference / "analyses.tsv").read_text() == (candidate / "analyses.tsv").read_text()

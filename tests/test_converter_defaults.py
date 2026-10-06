"""The converter's default shapes are the shapes #246 decided (ADR 0058).

A default that drifts silently is the failure this test exists for: a release
converted with a shape nobody decided is indistinguishable from one converted
with the decided shape, and the benchmark that justified the decision no longer
describes the tool. The test pins ADR 0058's decision in three places -- the
seam's constants, `convert_dense_release`'s signature defaults, and the layout
those defaults produce for a full-size Dense plane -- so changing any of them is
a visible, reviewed act.
"""

from __future__ import annotations

import inspect

from opengwasdb.store.arrays import (
    DENSE_CHUNK_SHAPE,
    DENSE_SHARD_SHAPE,
    TOP_HIT_SHARD_CHUNKS,
    ArrayRole,
    chunk_layout,
    shard_layout,
)
from opengwasdb.store.convert import convert_dense_release

#: OGS-00009's Dense plane -- the release ADR 0058's decision was measured on.
OGS00009_PLANE = (9_847_701, 2_024)

#: The decided Dense Analysis-axis inner chunk (ADR 0058).
DECIDED_ANALYSIS_CHUNK = 64


def test_the_seam_and_converter_defaults_are_the_decided_shapes():
    parameters = inspect.signature(convert_dense_release).parameters
    assert DENSE_SHARD_SHAPE == (100_000, 1_024)
    assert parameters["dense_analysis_chunk"].default == DECIDED_ANALYSIS_CHUNK
    assert parameters["dense_shard"].default == DENSE_SHARD_SHAPE
    assert TOP_HIT_SHARD_CHUNKS == 64
    assert parameters["top_hit_shard_chunks"].default == TOP_HIT_SHARD_CHUNKS


def test_the_defaults_produce_the_decided_layout_for_a_full_size_plane():
    """The converter's own planning path, with its defaults, on OGS-00009's plane."""
    inner = chunk_layout(
        ArrayRole.DENSE_STATISTIC_PLANE,
        OGS00009_PLANE,
        hint=(DENSE_CHUNK_SHAPE[0], DECIDED_ANALYSIS_CHUNK),
    )
    shard = shard_layout(
        ArrayRole.DENSE_STATISTIC_PLANE,
        OGS00009_PLANE,
        inner_chunk=inner,
        dense_shard=DENSE_SHARD_SHAPE,
    )
    assert inner == (1_000, DECIDED_ANALYSIS_CHUNK)
    assert shard == (100_000, 1_024)

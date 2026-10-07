"""Dense layout defaults."""

from opengwasdb.store.arrays import (
    DENSE_CHUNK_SHAPE,
    SHARDED_COMPRESSOR_RECORD,
    SHARDED_STORE_ZARR_FORMAT,
    ArrayRole,
    chunk_layout,
    shard_layout,
)

#: The maximum Dense grid chunk hint; the seam's policy clips it to the array
#: dimensions (ADR 0021).  One definition, imported from the seam so the
#: converter (#245) and the Dense builders agree.  Format 0.2.0's decided inner
#: chunk is the seam's `DENSE_CHUNK_SHAPE` -- `(1000, 64)` -- so the builders'
#: default and the converter's `--dense-analysis-chunk 64` are the same shape.
DEFAULT_CHUNK_SHAPE = DENSE_CHUNK_SHAPE
#: The one compressor configuration, sourced from the array-creation seam so the
#: bytes a plane is stored with and the bytes this blob publishes cannot drift.
#: The v3 sharded spelling since #247: a 0.2.0 release stores every array inside
#: the `sharding_indexed` codec, and the published record must name that codec.
DEFAULT_COMPRESSOR = SHARDED_COMPRESSOR_RECORD
DEFAULT_DTYPE = "float16"
TOP_HIT_THRESHOLDS = (5e-8, 5e-6, 5e-4)


def dense_layout_records(
    shape: tuple[int, int], *, hint: tuple[int, int] = DEFAULT_CHUNK_SHAPE
) -> dict[str, object]:
    """The layout a Dense plane of `shape` is written with, as the three
    recordings publish it (manifest `provenance.dense`, the `index.sqlite`
    `dense` blob, the `data.zarr` root attrs).

    The **inner chunk** comes from the seam's role policy (the builder's
    `chunk_shape` hint, clipped to the plane), and the **shard** from the shard
    policy, so a builder cannot record a layout its arrays do not have (issue
    #245).  The compressor is the v3 sharded record and `zarr_format` is 3,
    because that is the format the arrays are in.
    """
    inner = chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, tuple(shape), hint=hint)
    shard = shard_layout(ArrayRole.DENSE_STATISTIC_PLANE, tuple(shape), inner_chunk=inner)
    return {
        "chunk_shape": list(inner),
        "shard_shape": list(shard),
        "compressor": DEFAULT_COMPRESSOR,
        "zarr_format": SHARDED_STORE_ZARR_FORMAT,
    }


def dense_index_metadata(
    shape: tuple[int, int], *, hint: tuple[int, int] = DEFAULT_CHUNK_SHAPE, **extra: object
) -> dict[str, object]:
    """The `dense` blob every Dense-shaped builder writes into `index.sqlite`.

    Deliberately carries no `se_dtype`. Two of the three builders write this
    before the SE encoding has been measured, and `manifest.json`'s `encoding`
    block is the authoritative declaration in any case (spec §6a) — a duplicate
    that cannot be right is worse than no duplicate (issue #118).

    The chunk, shard and compressor come from `dense_layout_records`, so the
    blob cannot describe a layout the arrays do not have (issue #245, #247).
    """
    return {**dense_layout_records(shape, hint=hint), **extra}


def dense_provenance_block(
    shape: tuple[int, int],
    *,
    hint: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    se_dtype: str | None = None,
) -> dict[str, object]:
    """The `manifest.json` `provenance.dense` block the Dense writers share.

    One helper so the Dense VCF builder, the in-memory builder and the Hybrid
    Dense Component cannot record different layouts for the same arrays: the
    chunk, shard and compressor all come from `dense_layout_records`, and
    `se_dtype` is added only when the caller has measured it.
    """
    block: dict[str, object] = {
        "statistic_arrays": ["z", "se"],
        **dense_layout_records(shape, hint=hint),
    }
    if se_dtype is not None:
        block["se_dtype"] = se_dtype
    return block

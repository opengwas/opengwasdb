"""Dense layout defaults."""

from opengwasdb.store.arrays import COMPRESSOR_RECORD

DEFAULT_CHUNK_SHAPE = (1000, 1000)
#: The one compressor configuration, sourced from the array-creation seam so the
#: bytes a plane is stored with and the bytes this blob publishes cannot drift.
DEFAULT_COMPRESSOR = COMPRESSOR_RECORD
DEFAULT_DTYPE = "float16"
TOP_HIT_THRESHOLDS = (5e-8, 5e-6, 5e-4)


def dense_index_metadata(chunk_shape: tuple[int, int], **extra: object) -> dict[str, object]:
    """The `dense` blob every Dense-shaped builder writes into `index.sqlite`.

    Deliberately carries no `se_dtype`. Two of the three builders write this
    before the SE encoding has been measured, and `manifest.json`'s `encoding`
    block is the authoritative declaration in any case (spec §6a) — a duplicate
    that cannot be right is worse than no duplicate (issue #118).
    """
    return {"chunk_shape": list(chunk_shape), "compressor": DEFAULT_COMPRESSOR, **extra}

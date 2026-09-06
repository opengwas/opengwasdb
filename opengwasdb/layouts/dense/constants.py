"""Dense layout defaults."""

DEFAULT_CHUNK_SHAPE = (1000, 1000)
DEFAULT_COMPRESSOR = {
    "library": "numcodecs.Blosc",
    "cname": "zstd",
    "clevel": 3,
    "shuffle": "bitshuffle",
}
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

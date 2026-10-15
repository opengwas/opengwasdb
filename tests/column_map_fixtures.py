"""Comparison helpers the Column Map tests share (issue #281).

Each reduces a reader's or a store's output to bytes, so "the map reads what
the dedicated reader reads" is an equality of values, not of approximations.
"""

from __future__ import annotations

import numpy as np

from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.tabular import MetricsChunk

_CHUNK_ARRAYS = (
    "alid", "flipped", "palindromic", "af_alt", "beta", "se", "imputation_score",
    "imputation_score_status", "effect_from_odds_ratio_fallback", "se_from_ci",
    "se_from_p_value",
)


def array_bytes(array: np.ndarray) -> bytes:
    """An array's dtype and values, so `0.0` and `-0.0` differ; an object array
    by value, not by pointer."""
    body = repr(array.tolist()).encode() if array.dtype == object else array.tobytes()
    return array.dtype.str.encode() + body


def chunk_bytes(chunks: list[MetricsChunk]) -> list[bytes]:
    """Every array of every blocked-metrics chunk, named and as bytes."""
    return [
        name.encode() + b":" + array_bytes(getattr(chunk, name))
        for chunk in chunks
        for name in _CHUNK_ARRAYS
    ]


def ssf_streams(reader: GwasSsfReader | ColumnMapReader) -> dict[str, object]:
    """Every stream a GWAS-SSF-shaped reader offers, comparable by equality."""
    metrics = list(reader.stream_metrics())
    alids = [row.alid for row in metrics]
    return {
        "associations": repr(list(reader.stream_associations())),
        "variants": repr(list(reader.stream_variants())),
        "metrics": repr(metrics),
        "chunks": chunk_bytes(list(reader.stream_metric_chunks())),
        "sites": repr(sorted(reader.extract_at_sites(alids).items())),
        "effect_kind": None if reader.effect_source is None else reader.effect_source.kind,
    }

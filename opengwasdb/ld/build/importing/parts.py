"""Native Hail 0.2.64 BlockMatrix decoding (ADR 0101).

Wire definitions: StreamBlockInputBuffer/LZ4InputBlockBuffer in InputBuffers.scala
and RichDenseMatrixDouble.scala at hail-is/hail tag 0.2.64. Unallocated parts
are not measured zeros.
"""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt

from ogquery.ld.format import InputError


def optional_module(name: str) -> Any:
    """Load an import-only dependency without making readers depend on the extra."""
    try:
        return importlib.import_module(name)
    except ImportError as missing:
        raise InputError(f"{name} is required: install opengwasdb[ld-import]") from missing


def _block(stream: BinaryIO, lz4: Any) -> bytes | None:
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise ValueError("truncated Hail LZ4 header")
    compressed, raw = struct.unpack("<ii", header)
    if not 4 <= compressed <= 40_000 or not 0 < raw <= 32_768:
        raise ValueError("invalid Hail LZ4 lengths")
    payload = stream.read(compressed - 4)
    if len(payload) != compressed - 4:
        raise ValueError("truncated Hail LZ4 payload")
    result = lz4.decompress(payload, uncompressed_size=raw)
    if len(result) != raw:
        raise ValueError("wrong Hail decompressed length")
    return cast(bytes, result)


def _decoded(path: Path) -> bytearray:
    lz4 = optional_module("lz4.block")
    decoded = bytearray()
    with path.open("rb") as stream:
        while (block := _block(stream, lz4)) is not None:
            decoded.extend(block)
            # A hostile part must not grow memory beyond Hail's largest dense block.
            if len(decoded) > 9 + 8 * 4096 * 4096:
                raise ValueError("Hail matrix exceeds maximum block size")
    return decoded


def _matrix(decoded: bytearray) -> npt.NDArray[np.float64]:
    if len(decoded) < 9:
        raise ValueError("missing Hail matrix header")
    rows, cols, row_major = struct.unpack_from("<iiB", decoded)
    if row_major not in (0, 1) or not 0 < rows <= 4096 or not 0 < cols <= 4096:
        raise ValueError("invalid Hail matrix shape/order")
    if len(decoded) != 9 + 8 * rows * cols:
        raise ValueError("wrong Hail matrix byte count")
    return np.frombuffer(decoded, dtype="<f8", offset=9).reshape(
        (rows, cols), order="C" if row_major else "F"
    )


def read_part(path: Path) -> npt.NDArray[np.float64]:
    """Decode one dense part, refusing truncated/corrupt bytes by file name."""
    try:
        return _matrix(_decoded(path))
    except InputError:
        raise
    except (ValueError, RuntimeError, OSError) as corrupt:
        raise ValueError(f"{path}: {corrupt}") from corrupt

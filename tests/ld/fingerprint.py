"""An answer's fingerprint, for the tests that hold a query's result to a recorded one."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def answer_fingerprint(columns: dict[str, Any]) -> str:
    """Every column of an answer, read as bytes: an answer's fingerprint, never a subset of it."""
    hasher = hashlib.sha256()
    for name, value in columns.items():
        if isinstance(value, tuple):
            hasher.update(name.encode())
            for item in value:
                hasher.update(str(item).encode())
                hasher.update(b"\0")
            continue
        array = np.ascontiguousarray(value)
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()

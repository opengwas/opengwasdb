"""Pure helpers behind #253's interleaved A/B runner.

The runner itself needs the heavy-job lock (it queries a real store), so what is
tested here is the part that can be: the fingerprint that names the code a side
actually ran, and the result digest whose whole job is to notice a value, dtype,
shape or NaN-position difference between the two sides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks._artifact import tree_fingerprint
from benchmarks._query_ab import digest as _digest


def _tree(root: Path, **modules: str) -> Path:
    package = root / "opengwasdb"
    package.mkdir(parents=True)
    for name, text in modules.items():
        (package / f"{name}.py").write_text(text, encoding="utf-8")
    return root


def test_tree_fingerprint_names_the_code_not_the_checkout(tmp_path: Path) -> None:
    """A fingerprint must follow a source edit, and ignore non-source files."""
    root = _tree(tmp_path / "tree", __init__="x = 1\n", mod="y = 2\n")
    first = tree_fingerprint(root)
    assert first == tree_fingerprint(root), "an unchanged tree must fingerprint the same"
    (root / "opengwasdb" / "mod.py").write_text("y = 3\n", encoding="utf-8")
    assert tree_fingerprint(root) != first, "a source edit must change the fingerprint"
    edited = tree_fingerprint(root)
    (root / "opengwasdb" / "README.txt").write_text("not code\n", encoding="utf-8")
    assert tree_fingerprint(root) == edited, "only the Python sources are the code"


def test_digest_sees_values_dtype_shape_and_nan_positions() -> None:
    """Two results differ in the digest if they differ in any of those."""
    base = {"z": np.array([1.0, np.nan, 3.0], dtype="float32")}
    assert _digest(base) == _digest({"z": base["z"].copy()})
    assert _digest(base) != _digest({"z": np.array([1.0, 3.0, np.nan], dtype="float32")})
    assert _digest(base) != _digest({"z": np.array([1.0, 2.0, 3.0], dtype="float32")})
    assert _digest(base) != _digest({"z": np.array([1.0, np.nan, 3.0], dtype="float64")})
    assert _digest(base) != _digest({"z": np.array([1.0, np.nan], dtype="float32")})
    assert _digest(base) != _digest({"se": base["z"].copy()})

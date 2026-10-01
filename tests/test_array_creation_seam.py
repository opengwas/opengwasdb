"""#243: every Store array is created through one seam, and roles fix layout.

The static scan in this module is the enforcement arm of the refactor: a new
direct ``create_dataset``/``create_group``/``Blosc(...)`` anywhere outside
``opengwasdb/store/arrays.py`` fails here rather than waiting for a review to
notice it.  The scanner is tested against synthetic sources first, because a
scan that matches nothing is worse than no scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

from opengwasdb.layouts.dense.constants import DEFAULT_COMPRESSOR
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    ArrayRole,
    chunk_layout,
    compressor,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "opengwasdb"
SEAM = PACKAGE_ROOT / "store" / "arrays.py"

#: Call attribute names that create a Zarr array or group, and so must appear
#: only in the seam.
_FORBIDDEN_ATTRS = frozenset({"create_dataset", "create_array", "create_group", "require_group"})
#: ``zarr.<name>`` module functions that create an array.
_FORBIDDEN_ZARR_FUNCS = frozenset({"create", "zeros", "open_array", "array"})
#: Object names whose construction is the seam's to own.
_FORBIDDEN_CONSTRUCTORS = frozenset({"Blosc"})


def _violations(source: str, filename: str = "<source>") -> list[str]:
    """The forbidden creation calls in one source string, as `file:line: token`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        token: str | None = None
        if isinstance(func, ast.Attribute):
            if func.attr in _FORBIDDEN_ATTRS:
                token = f"{func.attr}("
            elif (
                func.attr in _FORBIDDEN_ZARR_FUNCS
                and isinstance(func.value, ast.Name)
                and func.value.id == "zarr"
            ):
                token = f"zarr.{func.attr}("
        elif isinstance(func, ast.Name) and func.id in _FORBIDDEN_CONSTRUCTORS:
            token = f"{func.id}("
        if token is not None:
            found.append(f"{filename}:{node.lineno}: {token}")
    return found


def test_scanner_detects_each_forbidden_call() -> None:
    """The scanner must catch every pattern it is meant to forbid."""
    source = "\n".join(
        [
            "import zarr",
            "from numcodecs import Blosc",
            "g.create_dataset('a', data=x)",
            "g.create_array('b', data=x)",
            "g.create_group('c')",
            "g.require_group('d')",
            "zarr.create('e')",
            "zarr.zeros((1,))",
            "zarr.open_array('f')",
            "zarr.array([1])",
            "Blosc(cname='zstd')",
        ]
    )
    tokens = {v.split(": ")[1] for v in _violations(source)}
    assert tokens == {
        "create_dataset(",
        "create_array(",
        "create_group(",
        "require_group(",
        "zarr.create(",
        "zarr.zeros(",
        "zarr.open_array(",
        "zarr.array(",
        "Blosc(",
    }


def test_scanner_ignores_calls_that_are_not_creation() -> None:
    """A mention in a string or a read is not a creation."""
    source = "\n".join(
        [
            "g['z']",
            "text = \"create_dataset('a')\"",
            "g.create_dataset_meta('a')",
            "other.zeros((1,))",
        ]
    )
    assert _violations(source) == []


def test_no_module_outside_the_seam_creates_arrays() -> None:
    """Every array/group creation and Blosc construction lives in the seam."""
    scanned = 0
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == SEAM or "__pycache__" in path.parts:
            continue
        scanned += 1
        offenders.extend(
            _violations(path.read_text(encoding="utf-8"), str(path.relative_to(PACKAGE_ROOT)))
        )
    # Assert the scan is meaningful before trusting a clean result: the package
    # has far more than a handful of modules, and the seam itself must match the
    # scanner (it is the one place these calls are allowed).
    assert scanned > 50, f"only {scanned} modules scanned; the glob is wrong"
    assert _violations(SEAM.read_text(encoding="utf-8")), "the scanner missed the seam's own calls"
    assert offenders == [], "direct Zarr/Blosc creation outside the seam:\n" + "\n".join(offenders)


#: One shape and chunk hint per role, for the coverage test below.  Kept as a
#: dict keyed by role so adding an `ArrayRole` without a case fails loudly.
_ROLE_CASES: dict[ArrayRole, tuple[tuple[int, ...], object]] = {
    ArrayRole.DENSE_STATISTIC_PLANE: ((10_000, 100), (1000, 1000)),
    ArrayRole.DENSE_IMPUTED_MASK: ((10_000, 100), (1000, 1000)),
    ArrayRole.DENSE_ON_PANEL: ((10_000,), (1000, 1000)),
    ArrayRole.ASSOCIATION_SEQUENCE: ((51_000,), None),
    ArrayRole.ASSOCIATION_OFFSETS: ((51,), None),
    ArrayRole.PER_VARIANT: ((51_000,), None),
    ArrayRole.TOP_HIT_INDEX: ((51_000,), 16_384),
    ArrayRole.TOP_HIT_ANALYSIS_OFFSETS: ((51,), None),
    ArrayRole.EXCEPTION_TABLE: ((51_000,), 200_000),
    ArrayRole.SE_COEFFICIENTS: ((51, 2), None),
    ArrayRole.RHO_ARRAY: ((51_000,), None),
}


def test_every_role_maps_to_a_chunk_layout() -> None:
    """Every role has a policy, and no policy returns an empty or non-positive chunk."""
    assert set(_ROLE_CASES) == set(ArrayRole)
    for role, (shape, hint) in _ROLE_CASES.items():
        chunks = chunk_layout(role, shape, hint=hint)
        assert len(chunks) == len(shape), f"{role} chunks {chunks} for shape {shape}"
        assert all(isinstance(size, int) and size > 0 for size in chunks), f"{role}: {chunks}"


def test_role_policies_clip_only_where_the_layout_clips() -> None:
    """ADR 0021's clip applies to the Dense grid, not to the flat CSR planes."""
    assert chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, (3, 2), hint=(1000, 1000)) == (3, 2)
    assert chunk_layout(ArrayRole.DENSE_ON_PANEL, (3,), hint=(1000, 1000)) == (3,)
    assert chunk_layout(ArrayRole.TOP_HIT_INDEX, (10,), hint=16_384) == (10,)
    assert chunk_layout(ArrayRole.EXCEPTION_TABLE, (10,), hint=200_000) == (10,)
    # The Ragged sequence keeps its declared chunk even for a short component:
    # its chunk size is fixed, not a function of the component's length.
    assert chunk_layout(ArrayRole.ASSOCIATION_SEQUENCE, (4,)) == (200_000,)
    assert chunk_layout(ArrayRole.ASSOCIATION_OFFSETS, (3,)) == (10_000,)


def test_compressor_record_is_the_published_blob() -> None:
    """The bytes written and the compressor the manifest publishes share one record."""
    assert DEFAULT_COMPRESSOR == COMPRESSOR_RECORD
    config = compressor().get_config()
    assert config["cname"] == COMPRESSOR_RECORD["cname"]
    assert config["clevel"] == COMPRESSOR_RECORD["clevel"]
    # numcodecs encodes bitshuffle as 2.
    assert config["shuffle"] == 2

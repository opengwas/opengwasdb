"""#243: every Store array is created through one seam, and roles fix layout.

The static scan in this module is the enforcement arm of the refactor: a new
direct ``create_dataset``/``create_group``/creation-capable ``open_group``/
``Blosc(...)`` anywhere outside ``opengwasdb/store/arrays.py`` fails here rather
than waiting for a review to notice it.  It resolves imports, so the ordinary
aliases (``import zarr as zr``, ``from zarr import array``,
``from numcodecs import Blosc as Codec``) cannot slip past it.  The scanner is
tested against synthetic sources first, because a scan that matches nothing is
worse than no scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.encoding import StoreCodec, StoreEncoding, write_eaf_baseline
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, ZEncoding
from opengwasdb.encoding.planes import write_eaf_csr, write_se_csr
from opengwasdb.layouts.dense.constants import DEFAULT_COMPRESSOR
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    ArrayRole,
    chunk_layout,
    component_variant_chunk,
    compressor,
    open_group_for_write,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "opengwasdb"
SEAM = PACKAGE_ROOT / "store" / "arrays.py"

#: The seam's own module path, whose creation helpers other modules import.
_SEAM_MODULE = "opengwasdb.store.arrays"

#: The seam's own creation/open API: the only names other modules may use.
_SEAM_API = frozenset(
    {"create_array", "create_group", "require_group", "open_group", "open_group_for_write"}
)

#: Creation methods that only a Zarr array/group has.  Any call to one of these
#: on a receiver that is not the seam or numpy is a violation, whatever the
#: receiver is named.
_ZARR_ONLY_METHODS = frozenset(
    {
        "create_dataset",
        "require_dataset",
        "create_array",
        "require_array",
        "create_group",
        "require_group",
        "open_group",
        "open_array",
    }
)
#: Creation methods shared with numpy (``np.zeros``, ``np.array``, ...).  These
#: are applied only to receivers that look like an instance -- a lowercase
#: name, or an attribute chain rooted in one -- so this package's own
#: ``Table.empty()``/``cls.empty()`` classmethod calls are not mistaken for
#: Zarr creators.  A receiver resolvable to numpy is never flagged.
_SHARED_CREATION_METHODS = frozenset(
    {
        "create",
        "zeros",
        "ones",
        "empty",
        "full",
        "array",
        "open_like",
        "zeros_like",
        "ones_like",
        "empty_like",
        "full_like",
        "save",
        "save_array",
        "save_group",
    }
)
#: ``numcodecs`` names whose construction is the seam's to own, matched against
#: any ``numcodecs.*`` submodule too (``numcodecs.blosc.Blosc``).
_FORBIDDEN_NUMCODECS = frozenset({"Blosc"})


def _is_zarr(module: str) -> bool:
    return module == "zarr" or module.startswith("zarr.")


def _is_numcodecs(module: str) -> bool:
    return module == "numcodecs" or module.startswith("numcodecs.")


def _is_numpy(module: str) -> bool:
    return module == "numpy" or module.startswith("numpy.")


def _root_name(node: ast.expr) -> str | None:
    """The leftmost identifier of an expression, e.g. ``assets.group`` -> ``assets``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _looks_like_instance(node: ast.expr) -> bool:
    """Whether a receiver looks like an object, not a class or type.

    ``cls``/``self`` and Capitalised-or-``_Capitalised`` names are treated as
    types, so this package's ``ZOverflowTable.empty()`` and ``cls.empty()`` are
    not judged by the shared-creation-method rule.
    """
    name = _root_name(node)
    if name is None or name in {"cls", "self"}:
        return False
    return not name.lstrip("_")[:1].isupper()


def _import_aliases(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve local names to their source: modules and ``from``-imported names."""
    modules: dict[str, str] = {}
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, names


def _module_of(node: ast.expr, modules: dict[str, str], names: dict[str, str]) -> str | None:
    """The dotted module an expression refers to, when it is a resolvable import.

    ``from pkg import mod as alias`` maps ``alias`` to ``pkg.mod`` itself, so a
    base name resolves to that canonical path; ``import zarr as zr`` maps to
    ``zarr``.
    """
    if isinstance(node, ast.Name):
        if node.id in modules:
            return modules[node.id]
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _module_of(node.value, modules, names)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _call_violation(
    func: ast.expr, modules: dict[str, str], names: dict[str, str]
) -> str | None:
    """The forbidden-creation token for one called function, or `None`.

    Two rules, in order:

    1. any call whose callee resolves into the ``zarr`` or ``numcodecs``
       packages -- any attribute path, alias or submodule -- is a violation.
       Type annotations and ``isinstance``/``cast`` arguments are not calls, so
       ``zarr.Array`` and ``isinstance(x, zarr.Array)`` are untouched.
    2. Zarr-only creation methods (``group.create_dataset`` and friends) are a
       violation on any non-seam, non-numpy receiver; the methods Zarr shares
       with numpy (``zeros``, ``array``, ``*_like``, ...) are a violation on a
       receiver that looks like an instance rather than a numpy alias or a
       class.
    """
    if isinstance(func, ast.Attribute):
        attr = func.attr
        base = _module_of(func.value, modules, names)
        if base == _SEAM_MODULE and attr in _SEAM_API:
            return None  # the seam's own API, e.g. `store_arrays.create_array(...)`
        if base is not None and (_is_zarr(base) or _is_numcodecs(base)):
            return f"{base}.{attr}("
        if base is not None and _is_numpy(base):
            return None
        if attr in _ZARR_ONLY_METHODS:
            return f"{attr}("
        if attr in _SHARED_CREATION_METHODS and _looks_like_instance(func.value):
            return f"{attr}("
        return None
    if isinstance(func, ast.Name):
        name = func.id
        canonical = names.get(name)
        if canonical is not None:
            module, _, attr = canonical.rpartition(".")
            if module == _SEAM_MODULE and attr in _SEAM_API:
                return None
            if _is_zarr(module) or _is_numcodecs(module):
                return f"{module}.{attr}("
            return None
        if name in _FORBIDDEN_NUMCODECS:
            return f"{name}("
    return None


def _violations(source: str, filename: str = "<source>") -> list[str]:
    """The forbidden creation calls in one source string, as `file:line: token`."""
    tree = ast.parse(source, filename=filename)
    modules, names = _import_aliases(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        token = _call_violation(node.func, modules, names)
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
            "g.open_group('e')",
            "zarr.create('f')",
            "zarr.zeros((1,))",
            "zarr.open_array('g')",
            "zarr.array([1])",
            "zarr.open_group('h', mode='w')",
            "zarr.group('i')",
            "zarr.open('j')",
            "Blosc(cname='zstd')",
        ]
    )
    tokens = {v.split(": ")[1] for v in _violations(source)}
    assert tokens == {
        "create_dataset(",
        "create_array(",
        "create_group(",
        "require_group(",
        "open_group(",
        "zarr.create(",
        "zarr.zeros(",
        "zarr.open_array(",
        "zarr.array(",
        "zarr.open_group(",
        "zarr.group(",
        "zarr.open(",
        "numcodecs.Blosc(",
    }


def test_scanner_detects_aliased_imports() -> None:
    """Ordinary import aliases must not be a way round the guard."""
    cases = {
        "import zarr as zr\nzr.array([1])": "zarr.array(",
        "from zarr import array\narray([1])": "zarr.array(",
        "from zarr import open_group as og\nog('x', mode='w')": "zarr.open_group(",
        "from zarr import zeros as mk\nmk((1,))": "zarr.zeros(",
        "from zarr import array as arr\narr([1])": "zarr.array(",
        "import zarr\nzarr.open_group('x')": "zarr.open_group(",
        "import numcodecs as nc\nnc.Blosc()": "numcodecs.Blosc(",
        "from numcodecs import Blosc as Codec\nCodec()": "numcodecs.Blosc(",
        # Submodule paths: the constructors live under them too, and the token
        # names the full resolved path so there is no ambiguity.
        "import zarr.creation as zc\nzc.array([1])": "zarr.creation.array(",
        "from zarr import creation as zc\nzc.array([1])": "zarr.creation.array(",
        "import zarr\nzarr.creation.array([1])": "zarr.creation.array(",
        "from zarr.api import create as c\nc('x')": "zarr.api.create(",
        "from zarr.creation import zeros as z\nz((1,))": "zarr.creation.zeros(",
        "import numcodecs.blosc as nb\nnb.Blosc()": "numcodecs.blosc.Blosc(",
        "from numcodecs.blosc import Blosc as Codec\nCodec()": "numcodecs.blosc.Blosc(",
        # The public group creators are creations too.
        "import zarr\nzarr.group('x')": "zarr.group(",
        "import zarr\nzarr.open('x')": "zarr.open(",
        "from zarr import group as g\ng('x')": "zarr.group(",
        "from zarr import open as o\no('x')": "zarr.open(",
    }
    for source, token in cases.items():
        found = _violations(source)
        assert found, f"scanner missed {source!r}"
        assert any(token in violation for violation in found), (source, found)


def test_scanner_allows_the_seams_own_api() -> None:
    """Names imported from the seam are the sanctioned way to create."""
    source = "\n".join(
        [
            "from opengwasdb.store.arrays import (",
            "    create_array, create_group, open_group, open_group_for_write, require_group,",
            ")",
            "from opengwasdb.store import arrays as store_arrays",
            "create_array(g, 'a', role)",
            "create_group(g, 'b')",
            "require_group(g, 'c')",
            "open_group('p')",
            "open_group_for_write('q', 'a')",
            "store_arrays.create_array(g, 'd', role)",
            "store_arrays.open_group_for_write('r', 'w')",
            "from opengwasdb.store.arrays import chunk_layout",
            "from opengwasdb.store.arrays import compressor",
            "chunk_layout(role, shape, component_chunk=1000)",
            "compressor()",
        ]
    )
    assert _violations(source) == []


def test_scanner_ignores_calls_that_are_not_creation() -> None:
    """A mention in a string, a read, or a numpy call is not a creation."""
    source = "\n".join(
        [
            "import numpy as np",
            "g['z']",
            "text = \"create_dataset('a')\"",
            "g.create_dataset_meta('a')",
            "np.zeros((1,))",
            "staged.arrays(mode='w')",
        ]
    )
    assert _violations(source) == []


def test_scanner_detects_the_remaining_zarr_creators() -> None:
    """The creators the round-2 scan missed are caught by the rooted-call rule."""
    cases = {
        "import zarr\nzarr.zeros_like(x)": "zarr.zeros_like(",
        "import zarr\nzarr.ones_like(x)": "zarr.ones_like(",
        "import zarr\nzarr.save('p', x)": "zarr.save(",
        "import zarr\nzarr.save_array('p', x)": "zarr.save_array(",
        "import zarr\nzarr.save_group('p')": "zarr.save_group(",
        "import zarr\nzarr.storage.DirectoryStore('p')": "zarr.storage.DirectoryStore(",
        # Group methods on a receiver the scanner cannot type.
        "group.create_dataset('a')": "create_dataset(",
        "root.require_dataset('a', shape=(1,))": "require_dataset(",
        "group.create_array('a')": "create_array(",
        "group.require_array('a')": "require_array(",
        "group.create_group('a')": "create_group(",
        "group.require_group('a')": "require_group(",
        "group.create('a')": "create(",
        "group.zeros((1,))": "zeros(",
        "group.ones((1,))": "ones(",
        "group.empty(0)": "empty(",
        "group.full((1,), 0)": "full(",
        "group.array([1])": "array(",
        "group.zeros_like(x)": "zeros_like(",
        "group.empty_like(x)": "empty_like(",
        "group.full_like(x, 1)": "full_like(",
        "group.save('p', x)": "save(",
        "group.open_group('a')": "open_group(",
    }
    for source, token in cases.items():
        found = _violations(source)
        assert found, f"scanner missed {source!r}"
        assert any(token in violation for violation in found), (source, found)


def test_scanner_allows_numpy_creators_and_own_classmethods() -> None:
    """The instance rule must not swallow numpy, nor this package's own helpers."""
    source = "\n".join(
        [
            "import numpy as np",
            "from numpy import zeros, array, save",
            "np.zeros((1,))",
            "np.ones((1,))",
            "np.empty(0)",
            "np.full((1,), 0)",
            "np.array([1])",
            "np.zeros_like(x)",
            "np.ones_like(x)",
            "np.empty_like(x)",
            "np.full_like(x, 1)",
            "np.save('p', x)",
            "zeros((1,))",
            "array([1])",
            "save('p', x)",
            # This package's own classmethods are named like Zarr creators.
            "ZOverflowTable.empty()",
            "_ComponentCost.empty(3)",
            "_MeasureAccumulators.zeros(2)",
            "cls.empty()",
            "self.empty()",
        ]
    )
    assert _violations(source) == []


def test_no_module_outside_the_seam_creates_arrays() -> None:
    """Every array/group creation, writable open and Blosc construction is in the seam."""
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


#: One (shape, component_chunk) per role, for the default-layout coverage test
#: below.  `component_chunk` is the variant-axis chunk of the plane a
#: `PER_VARIANT` array serves; it is `None` for every other role.  Kept as a
#: dict keyed by role so adding an `ArrayRole` without a case fails loudly.
_ROLE_CASES: dict[ArrayRole, tuple[tuple[int, ...], int | None]] = {
    ArrayRole.DENSE_STATISTIC_PLANE: ((10_000, 100), None),
    ArrayRole.DENSE_IMPUTED_MASK: ((10_000, 100), None),
    ArrayRole.DENSE_ON_PANEL: ((10_000,), None),
    ArrayRole.ASSOCIATION_SEQUENCE: ((51_000,), None),
    ArrayRole.ASSOCIATION_OFFSETS: ((51,), None),
    ArrayRole.PER_VARIANT: ((51_000,), 1000),
    ArrayRole.TOP_HIT_INDEX: ((51_000,), None),
    ArrayRole.TOP_HIT_ANALYSIS_OFFSETS: ((51,), None),
    ArrayRole.EXCEPTION_TABLE: ((51_000,), None),
    ArrayRole.SE_COEFFICIENTS: ((51, 2), None),
    ArrayRole.RHO_ARRAY: ((51_000,), None),
}


def test_every_role_has_a_default_layout_from_role_shape_and_component_chunk() -> None:
    """`chunk_layout(role, shape, component_chunk=...)` is total.

    The converter has role, shape and the component plane's chunk; it needs no
    builder group and no other input.
    """
    assert set(_ROLE_CASES) == set(ArrayRole)
    for role, (shape, component_chunk) in _ROLE_CASES.items():
        chunks = chunk_layout(role, shape, component_chunk=component_chunk)
        assert len(chunks) == len(shape), f"{role} chunks {chunks} for shape {shape}"
        assert all(isinstance(size, int) and size > 0 for size in chunks), f"{role}: {chunks}"


def test_per_variant_layout_follows_the_explicit_component_chunk() -> None:
    """`PER_VARIANT` is a function of the caller's component chunk, not the group."""
    assert chunk_layout(ArrayRole.PER_VARIANT, (10_000,), component_chunk=3) == (3,)
    assert chunk_layout(ArrayRole.PER_VARIANT, (10_000,), component_chunk=None) == (10_000,)
    # Never coarser than the plane (spec §6), and never coarser than the cap.
    assert chunk_layout(ArrayRole.PER_VARIANT, (1_000_000,), component_chunk=500_000) == (200_000,)
    # An explicit hint still wins over the component chunk, clipped to length.
    assert chunk_layout(ArrayRole.PER_VARIANT, (10,), hint=7, component_chunk=3) == (7,)


def test_converter_layout_matches_the_writers_for_dense_and_ragged(tmp_path: Path) -> None:
    """A converter's `chunk_layout` and a writer's sibling-derived chunk agree.

    Dense `z` is a 2-D variant x Analysis grid and Ragged `z` is a flat CSR
    sequence, so the two components exercise different sibling chunks.  The
    writer reads the component chunk and passes it in; the converter supplies
    the same value directly.  They must produce the same array layout (spec §6:
    a per-variant array is no coarser than the plane it serves).
    """
    baseline = np.linspace(0.1, 0.9, 10, dtype=np.float32)

    dense = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    dense.create_dataset("z", shape=(10, 4), chunks=(3, 4), dtype="int16")
    write_eaf_baseline(dense, baseline)
    dense_component = component_variant_chunk(dense)
    assert dense_component == dense["z"].chunks[0]
    assert tuple(dense["eaf_baseline"].chunks) == chunk_layout(
        ArrayRole.PER_VARIANT, (10,), component_chunk=dense_component
    )
    assert tuple(dense["eaf_baseline"].chunks) == (3,)

    ragged = zarr.open_group(str(tmp_path / "ragged.zarr"), mode="w")
    ragged.create_dataset("z", shape=(10,), chunks=(7,), dtype="int16")
    write_eaf_baseline(ragged, baseline)
    ragged_component = component_variant_chunk(ragged)
    assert ragged_component == ragged["z"].chunks[0]
    assert tuple(ragged["eaf_baseline"].chunks) == chunk_layout(
        ArrayRole.PER_VARIANT, (10,), component_chunk=ragged_component
    )
    assert tuple(ragged["eaf_baseline"].chunks) == (7,)


def test_role_defaults_clip_only_where_the_layout_clips() -> None:
    """ADR 0021's clip applies to the Dense grid, not to the flat CSR planes."""
    assert chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, (3, 2)) == (3, 2)
    assert chunk_layout(ArrayRole.DENSE_ON_PANEL, (3,)) == (3,)
    assert chunk_layout(ArrayRole.TOP_HIT_INDEX, (10,)) == (10,)
    assert chunk_layout(ArrayRole.EXCEPTION_TABLE, (10,)) == (10,)
    # The Ragged sequence keeps its declared chunk even for a short component:
    # its chunk size is fixed, not a function of the component's length.
    assert chunk_layout(ArrayRole.ASSOCIATION_SEQUENCE, (4,)) == (200_000,)
    assert chunk_layout(ArrayRole.ASSOCIATION_OFFSETS, (3,)) == (10_000,)


def test_explicit_overrides_still_win() -> None:
    """A caller's explicit chunks/chunk hint is preserved, as it was before."""
    assert chunk_layout(ArrayRole.ASSOCIATION_SEQUENCE, (4,), hint=(1,)) == (1,)
    assert chunk_layout(ArrayRole.ASSOCIATION_OFFSETS, (3,), hint=(7,)) == (7,)
    assert chunk_layout(ArrayRole.TOP_HIT_INDEX, (10,), hint=4) == (4,)
    assert chunk_layout(ArrayRole.EXCEPTION_TABLE, (10,), hint=3) == (3,)
    assert chunk_layout(ArrayRole.DENSE_STATISTIC_PLANE, (3, 2), hint=(1000, 1000)) == (3, 2)


def test_csr_writers_honour_an_explicit_chunks_override(tmp_path: Path) -> None:
    """`write_se_csr`/`write_eaf_csr` must forward `chunks=` to the array."""
    group = zarr.open_group(str(tmp_path / "ragged.zarr"), mode="w")
    plan = StoreEncoding(
        z=ZEncoding("float16"), se=SeEncoding("float16"), eaf=EafEncoding("float32")
    )
    codec = StoreCodec(plan)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    eaf = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    analysis_index = np.array([0, 0, 1], dtype=np.int64)
    write_se_csr(group, codec, values, eaf, analysis_index, None, chunks=(1,))
    write_eaf_csr(group, codec, np.arange(3, dtype=np.int32), eaf, baseline=None, chunks=(1,))
    assert tuple(group["se"].chunks) == (1,)
    assert tuple(group["eaf"].chunks) == (1,)


def test_legacy_per_variant_chunk_size_still_accepts_a_group(tmp_path: Path) -> None:
    """The exported `encoding.per_variant_chunk_size(group, length)` contract holds.

    Round 2 changed this signature; pre-existing callers pass a group and must
    keep working.  The explicit-chunk helper has a distinct name.
    """
    from opengwasdb.encoding import component_chunk_size, per_variant_chunk_size

    group = zarr.open_group(str(tmp_path / "group.zarr"), mode="w")
    group.create_dataset("z", shape=(10,), chunks=(7,), dtype="int16")
    assert per_variant_chunk_size(group, 10) == 7
    assert per_variant_chunk_size(group, 10) == component_chunk_size(7, 10)
    assert per_variant_chunk_size(group, 10) == chunk_layout(
        ArrayRole.PER_VARIANT, (10,), component_chunk=component_variant_chunk(group)
    )[0]


def test_open_group_for_write_rejects_a_read_mode(tmp_path: Path) -> None:
    """The write opener fails loudly rather than silently opening for read."""
    with pytest.raises(ValueError, match="open_group_for_write"):
        open_group_for_write(tmp_path / "store.zarr", "r")


def test_compressor_record_is_the_published_blob() -> None:
    """The bytes written and the compressor the manifest publishes share one record."""
    assert DEFAULT_COMPRESSOR == COMPRESSOR_RECORD
    config = compressor().get_config()
    assert config["cname"] == COMPRESSOR_RECORD["cname"]
    assert config["clevel"] == COMPRESSOR_RECORD["clevel"]
    # numcodecs encodes bitshuffle as 2.
    assert config["shuffle"] == 2


def test_store_module_introspection_lists_lazy_exports() -> None:
    """The lazy `open` re-exports are discoverable and cached, not hidden."""
    import opengwasdb.store as store

    assert "open_store" in dir(store)
    assert "open_store" in store.__dir__()
    value = store.open_store
    assert callable(value)
    assert store.__dict__["open_store"] is value

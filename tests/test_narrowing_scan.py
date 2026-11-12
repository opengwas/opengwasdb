"""A statistic is narrowed to `float16` in the codec and nowhere else (#350, #355).

Each narrowing outside `opengwasdb/encoding/` was a place a Store's SEs could be
rounded, or turned to `+inf`, by a rule the codec did not know about: the Dense
build rounded "exact" SE exceptions to `float16` before the codec saw them
(#355), and the SE byte measurement narrowed by a cast of its own. This scans
the package's source for both shapes of it: the casts that narrow a value, and a
call to the codec's own `se_float16` from outside the package that owns the
format's rounding. Callers ask the codec instead (`StoreCodec.float16_se`).
"""

from __future__ import annotations

import ast
from pathlib import Path

import opengwasdb

_PACKAGE = Path(opengwasdb.__file__).parent
_CODEC = _PACKAGE / "encoding" / "codec.py"
_ENCODING = _PACKAGE / "encoding"
#: Files that narrow something other than an association statistic.
_OTHER_STATISTICS = {
    # The Rho Matrix is stored as float16; it is not an association plane.
    _PACKAGE / "layouts" / "dense" / "rho.py",
}
_FLOAT16 = {"float16", "f2", "<f2", "half"}
#: Calls that produce values of the dtype they are given, and the position
#: of their dtype when it is passed positionally.
_ALLOCATING = {
    "array": 1, "asarray": 1, "full": 2, "empty": 1, "zeros": 1, "ones": 1, "fromiter": 1,
}


def _names_float16(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in _FLOAT16
    return isinstance(node, ast.Attribute) and node.attr in _FLOAT16


def _narrows(call: ast.Call) -> bool:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name == "astype" and call.args:
        return _names_float16(call.args[0])
    if name in _FLOAT16:  # np.float16(value)
        return True
    if name in _ALLOCATING:
        position = _ALLOCATING[name]
        positional = len(call.args) > position and _names_float16(call.args[position])
        return positional or any(
            kw.arg == "dtype" and _names_float16(kw.value) for kw in call.keywords
        )
    return False


def _se_float16_call(node: ast.Call) -> bool:
    """Whether this calls the codec's own `se_float16` rounding."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name == "se_float16"


def _cast_lines(path: Path) -> list[int]:
    """The lines of one file that narrow a value by a cast or a `float16` allocation."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _narrows(node)
    ]


def _se_helper_lines(path: Path) -> list[int]:
    """The lines of one file that call `se_float16`, which only `encoding/` may do itself.

    The helper is the codec's rounding rule, so a caller outside the package that
    owns the format asks `StoreCodec.float16_se` for the value a `float16` plane
    holds instead. It was the Dense band loader's direct call that rounded every
    SE before the plan was chosen (issue #355).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _se_float16_call(node)
    ]


def _narrowing_sites() -> list[str]:
    sites = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        label = path.relative_to(_PACKAGE.parent)
        if path not in _OTHER_STATISTICS and path != _CODEC:
            sites.extend(f"{label}:{line}" for line in _cast_lines(path))
        if _ENCODING not in path.parents:
            sites.extend(f"{label}:{line}" for line in _se_helper_lines(path))
    return sites


def test_the_scan_finds_a_narrowing_where_one_is_written(tmp_path) -> None:
    sources = {
        "np.asarray(v).astype(np.float16)": True,
        "v.astype('float16')": True,
        "np.full(3, np.nan, dtype=np.float16)": True,
        "np.empty(0, np.float16)": True,
        "np.float16(v)": True,
        "v.astype(np.float32)": False,
        "create_array(group, 'se', role, dtype='float16')": False,
    }
    for source, narrows in sources.items():
        call = ast.parse(source, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _narrows(call) is narrows, source
    assert _se_float16_call(ast.parse("se_float16(v)", mode="eval").body)
    assert not _se_float16_call(ast.parse("codec.float16_se(v)", mode="eval").body)


def test_the_scan_flags_the_codec_helper_outside_the_encoding_package(tmp_path) -> None:
    """The pre-#355 band loader's shape: `se_float16` called from a layout module."""
    pre_fix = tmp_path / "build_vcf.py"
    pre_fix.write_text(
        "def _load_se_column(spill, n, c):\n"
        "    column = _load_sparse_column(spill, n, 'se', c)\n"
        "    return se_float16(column)\n",
        encoding="utf-8",
    )
    assert _se_helper_lines(pre_fix) == [3]


def test_no_statistic_is_narrowed_to_float16_outside_the_codec() -> None:
    assert _narrowing_sites() == []

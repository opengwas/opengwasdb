"""A statistic is narrowed to `float16` in the codec and nowhere else (#350).

Each narrowing outside `opengwasdb/encoding/codec.py` was a place a Store's
SEs could be rounded, or turned to `+inf`, by a rule the codec did not know
about: the Dense build rounded "exact" SE exceptions to `float16` before the
codec saw them (#355), and the SE byte measurement narrowed by a cast of its
own. This scans the package's source for the casts that narrow values.
"""

from __future__ import annotations

import ast
from pathlib import Path

import opengwasdb

_PACKAGE = Path(opengwasdb.__file__).parent
_CODEC = _PACKAGE / "encoding" / "codec.py"
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


def _narrowing_sites() -> list[str]:
    sites = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path == _CODEC or path in _OTHER_STATISTICS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sites.extend(
            f"{path.relative_to(_PACKAGE.parent)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _narrows(node)
        )
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


def test_no_statistic_is_narrowed_to_float16_outside_the_codec() -> None:
    assert _narrowing_sites() == []

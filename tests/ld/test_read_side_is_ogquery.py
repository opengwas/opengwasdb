"""`opengwasdb.ld` reads LD Stores through ogquery, not through a copy (opengwas/ogquery#2).

The LD read path moved to `ogquery.ld` (ogquery ADR 0002). opengwasdb keeps building, writing,
validating and the Genotype Store reader, and re-exports the read side where its own modules
name it. Three properties keep that true:

* every read name `opengwasdb.ld` and its mixed modules export is the very object `ogquery.ld`
  has, so an `isinstance` across the two import paths cannot say no and one process decodes with
  one reader;
* no function or class body under `opengwasdb/ld/` (nor the shared primitives in
  `opengwasdb/variants/`) is also one of ogquery's, so nothing here can re-implement a read and
  drift from ogquery's answers;
* no test patches a re-exported read name on an opengwasdb module: ogquery's code looks the name
  up in its own module, so such a patch changes nothing and the test passes on a path it never
  reached. Patch the ogquery module instead.
"""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import ogquery.ld
import pytest

import opengwasdb.ld

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "opengwasdb" / "ld"
OGQUERY = Path(ogquery.ld.__file__).resolve().parent.parent

#: What `opengwasdb.ld` keeps of its own: building, validating, the version, and the `open`
#: that dispatches a Genotype Store to its reader.
OWN_TOP = {"build", "BuildInputs", "validate", "Finding", "__version__", "open"}

#: Each opengwasdb module that re-exports an ogquery module's read side, and the names it keeps
#: as its own though ogquery has the same name: `REncoding`, opengwasdb's subclass carrying the
#: build members, and `open`.
MIXED = {
    "opengwasdb.ld.format": {"REncoding"},
    "opengwasdb.ld.format.encoding": {"REncoding"},
    "opengwasdb.ld.format.geometry": set(),
    "opengwasdb.ld.format.provenance": set(),
    "opengwasdb.ld.query": {"open"},
    "opengwasdb.ld.query.ldstore": {"open"},
    "opengwasdb.ld.store": set(),
    "opengwasdb.ld.store.frames": {"REncoding"},
    "opengwasdb.ld.store.layout": {"REncoding"},
    "opengwasdb.ld.store.tables": set(),
}


def _ogquery_of(name: str) -> types.ModuleType:
    return importlib.import_module(name.replace("opengwasdb.ld", "ogquery.ld", 1))


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n, v in vars(module).items() if not n.startswith("_")]
    return [n for n in names if not isinstance(getattr(module, n), types.ModuleType)]


def test_the_top_level_read_names_are_ogquery_s() -> None:
    read = [name for name in opengwasdb.ld.__all__ if name not in OWN_TOP]
    assert len(read) == 10, read  # the read surface is not empty, nor a stray few
    assert [n for n in read if getattr(opengwasdb.ld, n) is not getattr(ogquery.ld, n)] == []


@pytest.mark.parametrize("name", sorted(MIXED))
def test_a_module_s_read_names_are_ogquery_s_objects(name: str) -> None:
    ours = importlib.import_module(name)
    theirs = _ogquery_of(name)
    shared = [n for n in _public(theirs) if hasattr(ours, n) and n not in MIXED[name]]
    assert shared, f"{name} re-exports nothing of {theirs.__name__}"
    assert [n for n in shared if getattr(ours, n) is not getattr(theirs, n)] == []


def test_opengwasdb_s_encoding_extends_ogquery_s() -> None:
    from ogquery.ld.format import REncoding as ReadEncoding

    from opengwasdb.ld.format import REncoding

    assert issubclass(REncoding, ReadEncoding) and REncoding is not ReadEncoding
    record = {
        "r_dtype": "int16",
        "r_scale": 511,
        "r_reserved_code": -32768,
        "rounding": "half_to_even",
    }
    assert type(REncoding.of(record)) is REncoding, "of() gives the build's class, which quantises"
    assert not hasattr(ReadEncoding(511), "quantise"), "ogquery's reader never quantises"


#: Bodies opengwasdb keeps though ogquery has the same one, each with the reason it cannot call
#: ogquery's: a name ogquery keeps private is not a seam opengwasdb may depend on.
EXEMPT = {
    # `isinstance(value, int) and not isinstance(value, bool)`: ogquery's are
    # `ogquery.ld.format.encoding._whole` and `ogquery.ld.store.layout._is_row`, both private.
    "store/validation.py:_is_int": "ogquery's copies are private (`_whole`, `_is_row`)",
    # `len(self.indptr) - 1`: `Partners`, a writer's table, counts its rows as ogquery's
    # `PartnerFrames.rows` counts its own; two classes over their own CSR, not one function twice.
    "store/tables.py:__len__": "Partners' own row count, as PartnerFrames.rows counts its own",
}


def _forwards(body: list[ast.stmt]) -> bool:
    """Whether `body` is trivial forwarding or an accessor: one statement that returns, or calls,
    an attribute or an argument -- `return self._variant_table.rows_of(alids)`,
    `return self._opened.location`, `self._handles.close()`. Two readers composing one
    `VariantTable` forward to it alike; that is the seam, not a copy (opengwas/ogld#94)."""
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Return):
        value = statement.value
    elif isinstance(statement, ast.Expr):
        value = statement.value
    else:
        return False

    def plain(node: ast.expr | None) -> bool:
        return isinstance(node, ast.Name | ast.Attribute | ast.Constant) or node is None

    if plain(value):
        return True
    if isinstance(value, ast.Call):
        arguments = [*value.args, *(k.value for k in value.keywords)]
        return isinstance(value.func, ast.Name | ast.Attribute) and all(map(plain, arguments))
    return False


def _bodies(path: Path) -> dict[str, str]:
    """Each function and class in `path`, at any depth, by `file:name`, as a docstring-free AST
    dump, less trivial forwarding (`_forwards`) and the named exemptions."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            name = f"{path.parent.name}/{path.name}:{node.name}"
            if body and not _forwards(body) and name not in EXEMPT:
                found[name] = ast.dump(ast.Module(body=body, type_ignores=[]))
    return found


def test_the_forwarding_rule_exempts_forwarding_and_nothing_else() -> None:
    def body(source: str) -> list[ast.stmt]:
        return ast.parse(source).body

    assert _forwards(body("return self._variant_table.rows_of(alids)"))
    assert _forwards(body("return self._opened.location"))
    assert _forwards(body("self._handles.close()"))
    assert not _forwards(body("return isinstance(value, int) and not isinstance(value, bool)"))
    assert not _forwards(body("return (code - 1) / self.scale"))
    assert not _forwards(body("return f(g(x))"))
    assert not _forwards(body("x = 1\nreturn x"))


def test_each_exemption_is_still_a_copy_of_a_private_name() -> None:
    """An exemption outlives its reason silently: each must still be a body ogquery has."""
    theirs = set()
    for path in sorted((OGQUERY / "ld").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef):
                theirs.add(ast.dump(ast.Module(body=_undocumented(node), type_ignores=[])))
    for name in EXEMPT:
        file, _, function = name.partition(":")
        path = PACKAGE / file
        (node,) = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == function
        ]
        assert ast.dump(ast.Module(body=_undocumented(node), type_ignores=[])) in theirs, name


def _undocumented(node: ast.FunctionDef) -> list[ast.stmt]:
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def test_no_body_under_opengwasdb_ld_is_also_ogquery_s() -> None:
    """The ticket's `rg` for a read function that exists twice, as an AST comparison."""
    theirs: dict[str, str] = {}
    for path in sorted((OGQUERY / "ld").rglob("*.py")) + [
        OGQUERY / "identity.py",
        OGQUERY / "alid.py",
    ]:
        theirs.update({dump: name for name, dump in _bodies(path).items()})
    assert len(theirs) > 200, "ogquery's read side must be scanned for this to mean anything"
    ours = {}
    sources = sorted(PACKAGE.rglob("*.py")) + [
        ROOT / "opengwasdb" / "variants" / "normalise.py",
        ROOT / "opengwasdb" / "variants" / "sorted_spelling.py",
    ]
    for path in sources:
        ours.update(_bodies(path))
    assert len(ours) > 300
    copies = sorted(
        f"{name} = ogquery's {theirs[dump]}" for name, dump in ours.items() if dump in theirs
    )
    assert copies == []


# ------------------------------------------------------------ a patch must reach the code


def _imports(nodes: list[ast.stmt]) -> dict[str, str]:
    """The `opengwasdb.ld` names `nodes` import (not descending into nested functions)."""
    modules: dict[str, str] = {}
    stack = list(nodes)
    while stack:
        node = stack.pop(0)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                name = alias.asname or alias.name
                modules[name] = full
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
        stack.extend(child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt))
    return modules


def _patches(scope: list[ast.stmt], modules: dict[str, str]) -> list[tuple[int, str, str]]:
    """(line, module, attribute) of each patch in `scope`, its targets resolved by `modules`."""
    found = []
    for statement in scope:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            found += _patches(statement.body, {**modules, **_imports(statement.body)})
            continue
        if isinstance(statement, ast.ClassDef):
            found += _patches(statement.body, modules)
            continue
        for node in ast.walk(statement):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("setattr", "object", "delattr") or not node.args:
                continue
            target = node.args[0]
            module = _module_named(target, modules)
            if module is not None and len(node.args) > 1:
                name = node.args[1]
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    found.append((node.lineno, module, name.value))
            elif isinstance(target, ast.Constant) and isinstance(target.value, str):
                if target.value.startswith("opengwasdb.ld."):
                    module, _, attribute = target.value.rpartition(".")
                    found.append((node.lineno, module, attribute))
    return found


def _module_named(target: ast.expr, modules: dict[str, str]) -> str | None:
    """The dotted module `target` names: an imported alias (`layout`), or a dotted path after
    `import opengwasdb.ld.store.layout` (`opengwasdb.ld.store.layout`), or an alias's attribute
    (`store.layout`); None for anything else."""
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name):
        return None
    head = modules.get(target.id, target.id)
    if head != "opengwasdb" and not head.startswith("opengwasdb.ld"):
        return None
    return ".".join([head, *reversed(parts)])


def _patched_module_names(path: Path) -> list[tuple[int, str, str]]:
    """(line, module, attribute) of each `monkeypatch.setattr`/`patch.object` whose target is a
    module imported from `opengwasdb.ld` where the call is (a function's own imports first), or
    one named by its dotted string."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = _patches(tree.body, _imports(tree.body))
    return [entry for entry in found if entry[1].startswith("opengwasdb.ld")]


def re_exported_patches(paths: list[Path]) -> list[str]:
    """Each patch of a name an opengwasdb module only re-exports from ogquery."""
    wrong = []
    for path in paths:
        for line, module_name, attribute in _patched_module_names(path):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue  # not a module: a class or a function, patched where it is defined
            try:
                theirs = _ogquery_of(module_name)
            except ImportError:
                continue
            value = getattr(module, attribute, None)
            if value is not None and value is getattr(theirs, attribute, object()):
                wrong.append(f"{path.relative_to(ROOT)}:{line}: {module_name}.{attribute}")
    return wrong


def test_the_patch_check_finds_a_patch_on_a_re_export(tmp_path: Path) -> None:
    source = tmp_path / "test_example.py"
    source.write_text(
        "import opengwasdb.ld.store.layout\n"
        "from opengwasdb.ld import store\n"
        "from opengwasdb.ld.store import layout\n"
        "def test(monkeypatch):\n"
        "    monkeypatch.setattr(layout, 'alid_companion', None)\n"
        "    monkeypatch.setattr(layout, 'new_r_array', None)\n"
        "    monkeypatch.setattr('opengwasdb.ld.store.frames.FrameCache', None)\n"
        "    monkeypatch.setattr(opengwasdb.ld.store.layout, 'never_data', None)\n"
        "    monkeypatch.setattr(store.layout, 'file_sha256', None)\n",
        encoding="utf-8",
    )
    found = [entry.rsplit(": ", 1)[1] for entry in _check_paths([source], tmp_path)]
    # `new_r_array` is opengwasdb's own, so patching it there reaches the code that calls it.
    assert found == [
        "opengwasdb.ld.store.layout.alid_companion",
        "opengwasdb.ld.store.frames.FrameCache",
        "opengwasdb.ld.store.layout.never_data",
        "opengwasdb.ld.store.layout.file_sha256",
    ]


def _check_paths(paths: list[Path], root: Path) -> list[str]:
    global ROOT
    saved, ROOT = ROOT, root
    try:
        return re_exported_patches(paths)
    finally:
        ROOT = saved


def test_no_test_patches_a_read_name_where_ogquery_does_not_look() -> None:
    paths = sorted((ROOT / "tests").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    assert len(paths) > 200
    assert re_exported_patches(paths) == []


def test_build_stays_the_function_once_its_submodule_is_imported() -> None:
    """`opengwasdb.ld`'s names are lazy (`opengwasdb.lazy_exports`), and importing the submodule
    `opengwasdb.ld.build` binds it on the package under the function's name; the package keeps
    the function, as the eager initialiser did."""
    submodule = importlib.import_module("opengwasdb.ld.build")

    assert isinstance(submodule, types.ModuleType)
    assert opengwasdb.ld.build is submodule.build
    assert callable(opengwasdb.ld.build) and not isinstance(opengwasdb.ld.build, types.ModuleType)

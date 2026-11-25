"""Nothing but the Store Reader derives beta or p from a stored z and SE (ADR 0104).

`opengwasdb.store_reader.stats.association_statistics` is the one derivation.
This guard parses every module of `opengwasdb/` and `benchmarks/` and flags each
expression that looks like another one:

- a product whose factors include a z-like and an se-like operand -- a name, an
  attribute or a string subscript such as `z`, `z_exp`, `rec.z` or
  `result["se"]`, at any depth of a `*` chain or across lines;
- `np.multiply` of the same;
- a call of `erfc`, `p_value_from_z`, `norm.sf`, `logsf` or `log_ndtr`;
- `10 ** x` or `np.power(10, x)` of a `log10_p`-like `x`.

`ogquery/stats.py` (re-exported as `store_reader/stats.py`, which holds no code) is the
derivation itself. Every other site must be in `ALLOWED`, by module and enclosing
function, with its reason; an entry that no longer matches anything fails too, so the
list cannot go stale.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import ogquery.stats
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("opengwasdb", "benchmarks")
EXEMPT = "opengwasdb/store_reader/stats.py"

_SOURCE = "a source's own columns, read before anything is stored"
_IVW = "the p of an IVW estimate's z: a statistic of an estimator, not of a stored association"

#: (module, enclosing function) -> why that site is not a second derivation.
ALLOWED: dict[tuple[str, str], str] = {
    ("opengwasdb/readers/effect_source.py", "derive_z_score_effect"): _SOURCE,
    ("opengwasdb/readers/gwas_vcf.py", "_row"): _SOURCE,
    ("opengwasdb/readers/tabular.py", "_z_score_arrays"): _SOURCE,
    ("opengwasdb/build/phenotype_sd_pipeline.py", "_gather_evidence"): (
        "a source reader's associations, judged to set the SD before the build stores them"
    ),
    ("opengwasdb/top_hits.py", "recomputed_abs_z_and_p"): (
        "what a 0.2 tier stored beside z, recomputed to check it (validator, rewrite) or "
        "to write it back on a 0.2 rebuild: the p of the stored z, not a second derivation"
    ),
    ("opengwasdb/build/effect_consistency.py", "log10_two_sided_p"): (
        "a build's check of the values it is writing, before any query"
    ),
    ("opengwasdb/ld/build/diagnostic.py", "sampling_rate"): (
        "a chi-square tail of LD sampling; no association involved"
    ),
    ("benchmarks/_hybrid_report.py", "ivw"): _IVW,
    ("benchmarks/benchmark_finngen_dense.py", "run_mr"): _IVW,
    ("benchmarks/benchmark_hybrid_ieua.py", "_ivw"): _IVW,
    ("benchmarks/benchmark_ogs00010_completed.py", "mr_condition"): _IVW,
    ("benchmarks/benchmark_ukbb_dense.py", "run_mr"): _IVW,
}

_Z = re.compile(r"^(z|abs_z|zs|z\d+|z_\w+|\w+_z)$", re.IGNORECASE)
_SE = re.compile(r"^(se|ses|se\d+|se_\w+|\w+_se|sebeta)$", re.IGNORECASE)
_P_CALLS = {"erfc", "p_value_from_z", "p_values_from_z", "logsf", "log_ndtr"}


def _operand_name(node: ast.expr) -> str:
    """The name an operand is known by: `z`, `rec.z`'s `z`, `r["z"]`'s `z`, `float(z)`'s `z`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        return _operand_name(node.value)
    if isinstance(node, ast.Call) and node.args and _called(node) in {"float", "asarray"}:
        return _operand_name(node.args[0])
    return ""


def _called(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _factors(node.left) + _factors(node.right)
    return [node]


def _z_and_se(operands: list[ast.expr]) -> bool:
    names = [_operand_name(operand) for operand in operands]
    return any(_Z.match(n) for n in names) and any(_SE.match(n) for n in names)


def _is_log10_p(node: ast.expr) -> bool:
    return "log10_p" in _operand_name(node)


def _derivation(node: ast.AST) -> str | None:
    """What kind of derivation `node` is, or None."""
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult) and _z_and_se(_factors(node)):
            return "z * se"
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.left, ast.Constant)
            and node.left.value in (10, 10.0)
            and _is_log10_p(node.right)
        ):
            return "10 ** log10_p"
    if isinstance(node, ast.Call):
        name = _called(node)
        if name in _P_CALLS:
            return name
        func = node.func
        if name == "sf" and isinstance(func, ast.Attribute) and _operand_name(func.value) == "norm":
            return "norm.sf"
        if name == "multiply" and len(node.args) == 2 and _z_and_se(list(node.args)):
            return "np.multiply(z, se)"
        if name == "power" and len(node.args) == 2 and _is_log10_p(node.args[1]):
            return "np.power(10, log10_p)"
    return None


def derivations(source: str) -> Iterator[tuple[int, str, str]]:
    """(line, enclosing function, kind) for each derivation in `source`."""
    tree = ast.parse(source)

    def walk(node: ast.AST, function: str) -> Iterator[tuple[int, str, str]]:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else function
            )
            kind = _derivation(child)
            if kind is not None:
                yield getattr(child, "lineno", 0), function, kind
            yield from walk(child, inner)

    yield from walk(tree, "<module>")


PLANTED = '''
import numpy as np
from scipy.stats import norm

def beta_from_result(result):
    return result["z"] * result["se"]

def beta_from_record(rec):
    return rec.z * rec.se

def beta_by_numpy(z, se):
    return np.multiply(z, se)

def signed(sign, d):
    return (sign
            * d["z_exp"]
            * d["se_exp"])

def p_of(z, math):
    return math.erfc(abs(z) / 2 ** 0.5), p_value_from_z(z), norm.sf(z), norm.logsf(z)

def p_over_an_array(stats, z):
    return stats.p_values_from_z(z)

def p_back(row):
    return 10.0 ** row["log10_p"], np.power(10, log10_p)

def not_a_derivation(z, n, se):
    return z * n, se * 2, z ** 2
'''


def test_the_guard_sees_each_form_it_claims_to():
    found = {(function, kind) for _, function, kind in derivations(PLANTED)}
    assert found == {
        ("beta_from_result", "z * se"),
        ("beta_from_record", "z * se"),
        ("beta_by_numpy", "np.multiply(z, se)"),
        ("signed", "z * se"),
        ("p_of", "erfc"),
        ("p_of", "p_value_from_z"),
        ("p_of", "norm.sf"),
        ("p_of", "logsf"),
        ("p_over_an_array", "p_values_from_z"),
        ("p_back", "10 ** log10_p"),
        ("p_back", "np.power(10, log10_p)"),
    }


def _scanned() -> list[Path]:
    return sorted(path for top in SCANNED for path in (ROOT / top).rglob("*.py"))


def test_nothing_else_derives_beta_or_p_from_a_stored_z_and_se():
    files = _scanned()
    assert len(files) > 300, len(files)
    unexplained, matched = [], set()
    for path in files:
        module = path.relative_to(ROOT).as_posix()
        if module == EXEMPT:
            continue
        for line, function, kind in derivations(path.read_text(encoding="utf-8")):
            if (module, function) in ALLOWED:
                matched.add((module, function))
            else:
                unexplained.append(f"{module}:{line} in {function}: {kind}")
    assert unexplained == []
    assert set(ALLOWED) - matched == set(), "allow-list entries that match nothing"


@pytest.mark.parametrize("function", ["association_statistics", "p_value_from_z"])
def test_the_exempt_module_is_where_the_derivation_lives(function):
    """The exemption covers the derivation the guard would otherwise flag."""
    # The derivation is ogquery's now (#475); `EXEMPT` is the re-export, which holds none.
    source = Path(ogquery.stats.__file__).read_text(encoding="utf-8")
    kinds = {kind for _, where, kind in derivations(source) if where == function}
    assert kinds, function


def test_a_caller_of_p_values_from_z_outside_the_allow_list_is_caught():
    """`p_values_from_z` is the derivation over an array (#471): its one caller outside
    `stats.py` is listed, and any other is reported, as this one would be unlisted."""
    module = "opengwasdb/top_hits.py"
    found = derivations((ROOT / module).read_text(encoding="utf-8"))
    calls = {(function, kind) for _, function, kind in found if kind == "p_values_from_z"}
    assert calls == {("recomputed_abs_z_and_p", "p_values_from_z")}
    unlisted = {(module, function) for function, _ in calls} - set(ALLOWED)
    assert unlisted == set()
    planted = "def elsewhere(stats, z):\n    return stats.p_values_from_z(z)\n"
    assert [(f, k) for _, f, k in derivations(planted)] == [("elsewhere", "p_values_from_z")]

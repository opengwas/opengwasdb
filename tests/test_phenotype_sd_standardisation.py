"""One module standardises a Store's SEs by its Analyses' phenotype SDs (ADR 0084).

Issue #366 found the divide restated at five builder call sites and absent from
the two Ragged builders, so four published Ragged SSF Store Releases hold their
sources' SEs. These tests hold the seam: `PhenotypeSd` is the only divide,
`storable` applies it to what a writer stores, a writer must say which scale
its SEs are in, and no source builder may say "already applied".
`tests/test_sd_standardisation_seam.py` builds one fixture through every builder.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import opengwasdb
from opengwasdb.encoding.association_codec import CellValues, storable
from opengwasdb.encoding.sd_units import PhenotypeSd
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

_PACKAGE = Path(opengwasdb.__file__).parent
_OWNER = _PACKAGE / "encoding" / "sd_units.py"
#: opengwasdb.ld computes LD from genotypes and has no phenotype SD; the scan's
#: name match finds a genotype SD there (`ld/build/pc_check.py`), so it does not
#: scan it (#455).
_LD_ENGINE = _PACKAGE / "ld"
#: A divisor whose spelling names a standard deviation.
_SD_NAMED = re.compile(r"divisor|original_sd|phenotype_sd|\bsd\b|_sd\b", re.IGNORECASE)


def _divisions_by_an_sd(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            right = node.value
        else:
            continue
        if _SD_NAMED.search(ast.get_source_segment(source, right) or ""):
            found.append(f"{path.relative_to(_PACKAGE.parent)}:{node.lineno}")
    return found


def test_nothing_outside_the_owner_divides_by_a_phenotype_sd() -> None:
    """A builder that restates the divide is how #366 happened: two builders
    never had it. The owner must be found by the scan, or the scan sees nothing."""
    assert _divisions_by_an_sd(_OWNER), "the scan must find the owner's own divide"
    elsewhere = [
        hit
        for path in sorted(_PACKAGE.rglob("*.py"))
        if path != _OWNER and _LD_ENGINE not in path.parents
        for hit in _divisions_by_an_sd(path)
    ]
    assert elsewhere == [], f"SE divided by a phenotype SD outside {_OWNER.name}: {elsewhere}"


def test_storable_divides_each_analysis_by_its_own_sd_and_leaves_z_alone() -> None:
    z = np.array([[2.0, -3.0, 1.0]], dtype=np.float32)
    se = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    kept = storable(
        CellValues(z, se), np.arange(3)[None, :], 3, where="test",
        phenotype_sd=PhenotypeSd.dividing_by(2.0, 1.0, 0.5),
    )

    np.testing.assert_array_equal(np.asarray(kept.cells.se), [[0.25, 0.5, 1.0]])
    np.testing.assert_array_equal(np.asarray(kept.cells.z), z)
    np.testing.assert_array_equal(se, [[0.5, 0.5, 0.5]]), "the batch handed over is not edited"


def test_applied_sds_divide_nothing_and_return_the_cells_handed_over() -> None:
    """A completion's cells, and a Pass 2 spill, are already in SD Units."""
    cells = CellValues(np.array([2.0], np.float32), np.array([0.5], np.float32))

    kept = storable(cells, 0, 1, where="test", phenotype_sd=PhenotypeSd.APPLIED)

    assert kept.cells is cells


def test_the_divide_is_float32_by_float32() -> None:
    """The arithmetic every builder used before the divide moved (byte identity)."""
    se = np.array([0.0213, 0.1200874], dtype=np.float32)

    got = PhenotypeSd.dividing_by(0.9840085).standardised(se, 0, 1)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, se / np.float32(0.9840085))


def test_an_se_standardised_beyond_float16s_range_is_left_out() -> None:
    """The out-of-range rule (#338) judges the SE the Store would hold, as it
    would one stored there."""
    kept = storable(
        CellValues(np.array([1.0], np.float32), np.array([40_000.0], np.float32)), 0, 1,
        where="test", phenotype_sd=PhenotypeSd.dividing_by(0.5),
    )

    assert not kept.stored[0]
    assert kept.se_out_of_range.tolist() == [1]


@pytest.mark.parametrize("sd", [0.0, -1.0, float("inf"), float("nan")])
def test_a_phenotype_sd_that_is_no_divisor_is_refused(sd: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PhenotypeSd.dividing_by(sd)


def test_a_writer_that_does_not_say_which_scale_its_ses_are_in_does_not_run() -> None:
    """No default: a default is how the Ragged builders stored unstandardised SEs."""
    for entry in (storable, RaggedCSRWriter.add_analysis):
        parameter = inspect.signature(entry).parameters["phenotype_sd"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, entry
        assert parameter.default is inspect.Parameter.empty, entry


# ── Issue #366: no silent no-op, no default, no bypass ───────────────────────

#: The only modules that may name `PhenotypeSd.APPLIED`, each a writer of cells a
#: Store -- or a Hybrid Pass 2 spill -- already holds in SD Units. The scan below
#: searches the whole package, so a module that needs `APPLIED` is added here on
#: purpose, with its reason; a source builder never is, whatever its filename.
_MAY_SAY_APPLIED: dict[str, str] = {
    # Defines `APPLIED`.
    "encoding/sd_units.py": "the owner",
    # `encode` takes what a Store holds: a grid `storable` already standardised,
    # or a completion's cells.
    "encoding/association_codec.py": "AssociationCodec.encode re-judges stored cells",
    # Reference-completion fills are imputed from a Store's SEs, already in SD Units.
    "completion/fill_shards.py": "a completion's fills",
    # Writes the Ragged Overflow and judges crossovers, both read from the source Store.
    "layouts/hybrid/complete.py": "Hybrid completion: stored cells",
    # Writes the source Store's cells and its fills.
    "layouts/ragged/complete.py": "Ragged completion: stored cells",
    # Reads the `.ovf.npz` spills Pass 2's `storable` already standardised; no source row.
    "layouts/hybrid/overflow_spills.py": "the Hybrid Overflow, from standardised spills",
}
#: A parameter or field that carries the SD an Analysis's SEs are divided by: a
#: divisor by name, or a `PhenotypeSd` by type. (`original_sd` columns are the
#: recorded value, blank where none was declared, and are judged by the parser.)
_DIVISOR = re.compile(r"divisor", re.IGNORECASE)


def _is_divisor(name: str, annotation: ast.expr | None) -> bool:
    return bool(_DIVISOR.search(name)) or (
        annotation is not None
        and re.search(r"\bPhenotypeSd\b", ast.unparse(annotation)) is not None
    )


def test_no_divisor_is_no_phenotype_sd() -> None:
    """`dividing_by()` used to divide nothing, silently: SE 0.5 stored as 0.5
    with no SD supplied. No rescaling is an explicit 1.0 the parser supplies."""
    with pytest.raises(ValueError, match="at least one phenotype SD"):
        PhenotypeSd.dividing_by()


@pytest.mark.parametrize("divisors", [(2.0,), (2.0, 1.0, 1.0, 1.0)])
def test_storable_refuses_a_phenotype_sd_per_analysis_count_that_is_not_the_batchs(
    divisors: tuple[float, ...],
) -> None:
    cells = CellValues(np.ones((1, 3), np.float32), np.full((1, 3), 0.5, np.float32))

    with pytest.raises(ValueError, match=rf"{len(divisors)} phenotype SD\(s\) for 3 Analyses"):
        storable(
            cells, np.arange(3)[None, :], 3, where="test",
            phenotype_sd=PhenotypeSd.dividing_by(*divisors),
        )


def test_every_allowed_module_exists_and_names_applied() -> None:
    """An entry that names no module, or a module that no longer needs `APPLIED`,
    is a stale exception: it would let the next writer there pass unseen."""
    stale = {
        module: reason
        for module, reason in _MAY_SAY_APPLIED.items()
        if not (_PACKAGE / module).exists() or not _references(_PACKAGE / module, "APPLIED")
    }
    assert stale == {}, stale


def _references(path: Path, attribute: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr == attribute)
        or (isinstance(node, ast.Name) and node.id == attribute)
    ]


def _naming_applied(package: Path) -> dict[str, list[int]]:
    """Every module of `package` that names `APPLIED`, by AST, and where."""
    return {
        relative: lines
        for path in sorted(package.rglob("*.py"))
        if (lines := _references(path, "APPLIED"))
        and (relative := path.relative_to(package).as_posix())
    }


def test_no_module_but_a_stored_cell_writer_can_say_its_ses_are_already_standardised() -> None:
    """`APPLIED` divides by nothing. A source builder that passed it would store
    its source's SEs, which is what the Ragged builders did before #366. Searched
    package-wide, not by filename: a builder copied to `layouts/ragged/source_ssf.py`,
    or a helper in `source_sd.py`, is caught like `build_ssf.py` (issue #366)."""
    found = {
        module: lines
        for module, lines in _naming_applied(_PACKAGE).items()
        if module not in _MAY_SAY_APPLIED
    }
    assert found == {}, (
        f"modules naming PhenotypeSd.APPLIED that are not stored-cell writers: {found}; a "
        "source builder passes `dividing_by`, and a new stored-cell writer is added to "
        "_MAY_SAY_APPLIED with its reason"
    )


def test_nothing_outside_the_owner_constructs_a_phenotype_sd_directly() -> None:
    """`PhenotypeSd(None)` is `APPLIED` under another spelling."""
    found = {
        str(path.relative_to(_PACKAGE.parent)): lines
        for path in sorted(_PACKAGE.rglob("*.py"))
        if path != _OWNER
        and (
            lines := [
                node.lineno
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "PhenotypeSd"
            ]
        )
    }
    assert found == {}, found


def _defaulted_sd_parameters(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            positional = [*arguments.posonlyargs, *arguments.args]
            defaulted = positional[len(positional) - len(arguments.defaults):]
            defaulted += [
                arg for arg, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
                if d is not None
            ]
            found += [
                f"{node.name}({arg.arg}=…)" for arg in defaulted
                if _is_divisor(arg.arg, arg.annotation)
            ]
        elif isinstance(node, ast.ClassDef):
            found += [
                f"{node.name}.{item.target.id}"
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.value is not None
                and _is_divisor(item.target.id, item.annotation)
                and "ClassVar" not in ast.unparse(item.annotation)
            ]
    return [f"{path.relative_to(_PACKAGE.parent)}: {name}" for name in found]


def test_no_phenotype_sd_has_a_default() -> None:
    """A default SD is how a builder stores its source's SEs without saying so
    (CLAUDE.md: never substitute a default for missing data)."""
    found = [
        hit for path in sorted(_PACKAGE.rglob("*.py")) for hit in _defaulted_sd_parameters(path)
    ]
    assert found == [], found

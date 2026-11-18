"""One module standardises a Store's SEs by its Analyses' phenotype SDs (ADR 0084).

Issue #366 found the divide restated at five builder call sites and absent from
the two Ragged builders, so four published Ragged SSF Store Releases hold their
sources' SEs. These tests hold the seam: `PhenotypeSd` is the only divide,
`storable` the only caller, and a writer must say which scale its SEs are in.
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
from opengwasdb.encoding.phenotype_sd import PhenotypeSd
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

_PACKAGE = Path(opengwasdb.__file__).parent
_OWNER = _PACKAGE / "encoding" / "phenotype_sd.py"
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
        if path != _OWNER
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

    got = PhenotypeSd.dividing_by(0.9840085).standardised(se, 0)

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

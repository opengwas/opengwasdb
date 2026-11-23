"""beta and pval come from one function, at known z and SE (ADR 0104).

`association_statistics` is the one place the Store Reader turns a stored z and
SE into `beta`, `pval` and `log10_p`. Every query method's record and every
resolved row carry its output (`tests/test_store_reader_digests.py` checks that
on every layout); this module checks the function itself, value by value, and
that nothing else in the package derives beta from a stored z and SE.
"""

from __future__ import annotations

import math
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest
from test_store_reader_stats import exact_log10_p

from opengwasdb.store_reader.associations import Associations
from opengwasdb.store_reader.stats import association_statistics

NAN = math.nan
INF = math.inf

#: (z, se) as a float32 plane decodes them, and what each must give.
CASES = {
    "ordinary": (2.5, 0.125),
    "negative z": (-3.75, 0.0625),
    "z = 0": (0.0, 0.5),
    "missing z": (NAN, 0.5),
    "missing se": (1.5, NAN),
    "missing both": (NAN, NAN),
    "infinite se": (1.5, INF),
    "underflow": (47.8, 0.01),
}


@pytest.fixture(scope="module")
def derived() -> dict[str, tuple[float, float, float]]:
    z = np.array([z for z, _ in CASES.values()], dtype=np.float32)
    se = np.array([se for _, se in CASES.values()], dtype=np.float32)
    stats = association_statistics(z, se)
    return {
        name: (float(stats.beta[i]), float(stats.pval[i]), float(stats.log10_p[i]))
        for i, name in enumerate(CASES)
    }


def test_the_arrays_are_float64_whatever_the_stored_dtype():
    for dtype in (np.float16, np.float32, np.float64):
        stats = association_statistics(np.ones(3, dtype=dtype), np.ones(3, dtype=dtype))
        assert stats.beta.dtype == np.float64, dtype
        assert stats.pval.dtype == np.float64, dtype
        assert stats.log10_p.dtype == np.float64, dtype


def test_an_ordinary_association():
    beta, pval, log10_p = association_statistics(
        np.array([2.5], np.float32), np.array([0.125], np.float32)
    )
    assert beta[0] == 0.3125
    assert pval[0] == math.erfc(2.5 / math.sqrt(2.0))
    assert pval[0] == pytest.approx(10 ** exact_log10_p(2.5), rel=1e-14)
    assert log10_p[0] == pytest.approx(math.log10(pval[0]), rel=1e-15)


def test_a_negative_z_gives_a_negative_beta_and_the_same_p(derived):
    beta, pval, log10_p = derived["negative z"]
    assert beta == -3.75 * 0.0625
    assert pval == math.erfc(3.75 / math.sqrt(2.0))
    assert log10_p == pytest.approx(math.log10(pval), rel=1e-15)


def test_z_zero_is_beta_zero_and_p_one(derived):
    assert derived["z = 0"] == (0.0, 1.0, 0.0)


@pytest.mark.parametrize("name", ["missing z", "missing se", "missing both"])
def test_a_missing_association_is_nan_throughout(derived, name):
    assert all(math.isnan(value) for value in derived[name]), derived[name]


def test_an_infinite_se_is_missing_never_infinite(derived):
    """ADR 0068 rules an SE above float16's range an error: no beta, no p."""
    beta, pval, log10_p = derived["infinite se"]
    assert math.isnan(beta) and math.isnan(pval) and math.isnan(log10_p)


def test_a_p_below_float64_is_zero_and_log10_p_is_exact(derived):
    """The FADS1/FADS2 window's |z| = 47.8 (issue #104): p is about 1e-499."""
    beta, pval, log10_p = derived["underflow"]
    assert beta == pytest.approx(0.478, rel=1e-6)
    assert pval == 0.0
    assert math.isfinite(log10_p)
    exact = exact_log10_p(float(np.float32(47.8)))
    assert log10_p == pytest.approx(exact, rel=1e-15)
    assert -500 < log10_p < -490


def test_the_record_carries_the_function_s_output():
    z = np.array([2.5, NAN, 47.8, 1.5, -1.0], np.float32)
    se = np.array([0.125, NAN, 0.01, INF, 0.25], np.float32)
    record = Associations.from_decoded(np.arange(5), 0, z, se)
    stats = association_statistics(z, se)
    for key in ("beta", "pval"):
        assert key in record and key in list(record)
        assert np.array_equal(record[key], getattr(stats, key), equal_nan=True), key
    taken = record.take(np.array([4, 0]))
    assert np.array_equal(taken["beta"], stats.beta[[4, 0]])
    assert np.array_equal(taken["pval"], stats.pval[[4, 0]])
    empty = Associations.empty()
    assert empty["beta"].dtype == np.float64 and len(empty["pval"]) == 0


#: The source-side derivations: each turns a *source's* columns into a z and SE
#: before anything is stored, so none reads a stored z and SE.
_SOURCE_SIDE = {
    "opengwasdb/readers/effect_source.py",
    "opengwasdb/readers/tabular.py",
    "opengwasdb/readers/gwas_vcf.py",
}

_PRODUCT = re.compile(
    r"\bz\w*(\[[^]]*\])?\s*\*\s*\w*\bse\w*|\bse\w*(\[[^]]*\])?\s*\*\s*\w*\bz\w*"
    r"|\bbeta\s*=\s*.*\bz\b"
)
_POWER = re.compile(r"10(\.0)?\s*\*\*\s*\w*log10_p")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """The code of `path`, line by line: comments and string text left out, an
    f-string's expressions kept (Python 3.12 tokenizes them as code)."""
    kept = (tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_MIDDLE)
    lines: dict[int, list[str]] = {}
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type not in kept:
                lines.setdefault(token.start[0], []).append(token.string)
    return [(number, " ".join(parts)) for number, parts in sorted(lines.items())]


def test_nothing_else_in_the_package_derives_beta_or_p_from_a_stored_z_and_se():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "opengwasdb").rglob("*.py"))
    assert len(files) > 200
    found = []
    seen_source_side = set()
    for path in files:
        name = path.relative_to(root).as_posix()
        for number, line in _code_lines(path):
            if _PRODUCT.search(line):
                if name in _SOURCE_SIDE:
                    seen_source_side.add(name)
                elif name != "opengwasdb/store_reader/stats.py":
                    found.append(f"{name}:{number}: {line.strip()}")
            if _POWER.search(line):
                found.append(f"{name}:{number}: {line.strip()}")
    assert found == []
    # The pattern finds the source-side products it is told to pass, so it
    # would find one anywhere else.
    assert seen_source_side == _SOURCE_SIDE

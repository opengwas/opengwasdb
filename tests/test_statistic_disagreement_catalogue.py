"""Every way a source's p-value and its beta/se are known to disagree, and what the engine does.

A Store keeps z (= beta/se) and SE, never the source's p-value: the p a query
returns is the two-sided normal p of the stored z (spec §6). The source's p is
read once, by the resolver, as a witness against beta/se (ADR 0069). So where
a source's p is not the Wald p of its own beta/se, the Store answers with a p
the source never published -- and the witness either sees it or it does not.

`docs/statistic-disagreement.md` lists each circumstance measured on real
sources. Each test here builds a small GWAS-SSF source from that circumstance's
*mechanism* (how such a file comes to be written, never by running the check)
and pins the verdict the document claims: refused, compared and stored, or
skipped. Each is resolved under the p-value test its Source Collection declares
(#371, ADR 0085): MR-MEGA is refused by that declaration, a joint test skips the
check. A circumstance nothing sees is pinned as stored, on purpose: the test
then says exactly what a later rule changes.

Circumstances already pinned elsewhere are named in the document and not
repeated: the wrong effect scale (#345) and a z carrying no signal (#370) in
`test_resolve_effect_consistency.py`, an SE derived from p there too.
"""

from __future__ import annotations

import gzip
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from admission_helpers import ragged_ssf_store
from scipy import stats
from test_resolve import _alid, _panel_frequencies, _write_panel
from test_resolve_effect_consistency import _BASE, _resolve, _write

from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.effect_consistency import ConsistencySkip, log10_two_sided_p
from opengwasdb.model.p_value_test import PValueTest
from opengwasdb.query import query_store

_DOC = Path(__file__).resolve().parents[1] / "docs" / "statistic-disagreement.md"

#: 400 null associations and 100 with signal (|z| 3 to 12): p spans ~33 orders
#: of magnitude, and every source keeps well above the check's 50-row minimum.
_N_NULL, _N_SIGNAL = 400, 100
_SE = 0.05


@pytest.fixture
def panel(tmp_path: Path) -> AncestryReference:
    """The ancestry reference the resolver needs; this check never reads it."""
    return _write_panel(tmp_path, [_alid(index) for index in range(200)], _panel_frequencies())


def _true_z(rng: np.random.Generator) -> np.ndarray:
    signal = rng.uniform(3.0, 12.0, _N_SIGNAL) * rng.choice([-1.0, 1.0], _N_SIGNAL)
    return np.concatenate([rng.standard_normal(_N_NULL), signal])


def _source(tmp_path: Path, beta: np.ndarray, se: np.ndarray, p: np.ndarray) -> Path:
    return _write(
        tmp_path / "source.tsv",
        [*_BASE, "beta", "standard_error", "p_value"],
        {
            "beta": [f"{v:.10g}" for v in beta],
            "standard_error": [f"{v:.10g}" for v in se],
            "p_value": [f"{v:.6g}" for v in p],
        },
    )


def _two_sided(z: np.ndarray) -> np.ndarray:
    return np.asarray(2.0 * stats.norm.sf(np.abs(z)), dtype=np.float64)


# --- the mechanisms ------------------------------------------------------------


def _mr_mega(tmp_path: Path) -> Path:
    """MR-MEGA (GCST90833021, GCST90014299-304): each coefficient's SE is scaled
    by the residual mean square RSS/NDF, a chi-square on NDF (cohorts - 2) degrees
    of freedom over NDF, while p is a separate 2-df chi-square of the association
    and the ancestry axis together. NDF varies with how many cohorts carry the
    variant; few cohorts, small NDF, the widest scatter."""
    rng = np.random.default_rng(371)
    z = _true_z(rng)
    ndf = rng.choice([2, 5, 10, 20], z.size)
    scale = np.sqrt(rng.chisquare(ndf) / ndf)
    chi2 = z**2 + rng.standard_normal(z.size) ** 2
    return _source(tmp_path, z * _SE, _SE * scale, stats.chi2.sf(chi2, 2))


def _joint_two_df(tmp_path: Path) -> Path:
    """A "(2df test)" Analysis (GCST90570xxx, GCST90809xxx): the stored coefficient
    is the interaction's, and p is the joint test of interaction and main effect.
    Twenty loci carry a main effect only: the source calls them genome-wide
    significant, and their stored interaction z cannot."""
    rng = np.random.default_rng(372)
    z_interaction = _true_z(rng)
    z_main = rng.standard_normal(z_interaction.size)
    main_only = rng.choice(_N_NULL, 20, replace=False)
    z_main[main_only] = rng.uniform(7.0, 10.0, main_only.size)
    p = stats.chi2.sf(z_interaction**2 + z_main**2, 2)
    return _source(tmp_path, z_interaction * _SE, np.full(z_interaction.size, _SE), p)


def _p_is_one_sided(tmp_path: Path) -> Path:
    """GCST90446466: `p_value` holds the upper-tail one-sided p of the source's z,
    P(Z > z): p/2 where z is positive, 1 - p/2 where it is negative, beside a
    sound beta/se."""
    rng = np.random.default_rng(373)
    z = _true_z(rng)
    return _source(tmp_path, z * _SE, np.full(z.size, _SE), stats.norm.sf(z))


def _p_from_another_model(tmp_path: Path) -> Path:
    """GCST90002238: beta/se from the raw trait, p from the inverse-normal-transformed
    trait -- a second, closely correlated test of the same association."""
    rng = np.random.default_rng(374)
    z = _true_z(rng)
    z_other = 0.95 * z + math.sqrt(1 - 0.95**2) * rng.standard_normal(z.size)
    return _source(tmp_path, z * _SE, np.full(z.size, _SE), _two_sided(z_other))


def _low_frequency_rows_inflated(tmp_path: Path) -> Path:
    """GCST90572747, GCST90243957: in a small case-control study the Wald SE of a
    low-frequency variant is too small beside the score-test p, for a few rows."""
    rng = np.random.default_rng(375)
    z = _true_z(rng)
    se = np.full(z.size, _SE)
    few = rng.choice(_N_NULL, 25, replace=False)  # 5% of rows, all truly null
    z[few] = rng.uniform(2.0, 3.0, few.size) * rng.choice([-1.0, 1.0], few.size)
    se[few] = _SE / 3.0  # stored |z| 6 to 9: genome-wide significant, wrongly
    return _source(tmp_path, z * _SE, se, _two_sided(z))


def _se_printed_to_two_decimals(tmp_path: Path) -> Path:
    """GCST90244669: the SE is written with two decimal places, so 0.014 reads as
    0.01 and beta/se overstates |z| by up to half again."""
    rng = np.random.default_rng(376)
    z = _true_z(rng)
    se = rng.uniform(0.010, 0.0149, z.size)
    return _source(tmp_path, z * se, np.round(se, 2), _two_sided(z))


def _sample_size_weighted_z(tmp_path: Path) -> Path:
    """GCST004732: METAL's sample-size-weighted p beside inverse-variance beta/se."""
    rng = np.random.default_rng(377)
    z = _true_z(rng)
    p = _two_sided(z * rng.uniform(0.85, 1.15, z.size))
    return _source(tmp_path, z * _SE, np.full(z.size, _SE), p)


def _p_at_a_floor(tmp_path: Path) -> Path:
    """Many sources: p written no smaller than 1e-300 beside a z far beyond it."""
    rng = np.random.default_rng(378)
    z = _true_z(rng)
    z[-20:] = np.sign(z[-20:]) * rng.uniform(40.0, 60.0, 20)
    return _source(tmp_path, z * _SE, np.full(z.size, _SE), np.maximum(_two_sided(z), 1e-300))


@dataclass(frozen=True)
class Circumstance:
    """One row of docs/statistic-disagreement.md, as the resolver must judge it.

    The medians and hit counts are the fixture's own, taken once from the file
    it writes; `false_hits` and `lost_hits` are what the Store would get wrong
    about significance, which no median shows.
    """

    write: Callable[[Path], Path]
    declared: PValueTest
    #: "p_value_test": refused by the declaration, unread; a rule's name: refused
    #: by the check; None: stored (compared, or skipped for a joint test).
    refused_by: str | None
    median_abs_delta: float
    median_delta: float
    false_hits: int  # stored |z| >= 5.45 where the source's p > 1e-4
    lost_hits: int  # stored |z| < 3.72 where the source's p <= 5e-8


_WALD, _JOINT, _MR_MEGA = PValueTest.WALD_1DF, PValueTest.JOINT_2DF, PValueTest.MR_MEGA

CIRCUMSTANCES: dict[str, Circumstance] = {
    "mr_mega": Circumstance(_mr_mega, _MR_MEGA, "p_value_test", 0.2574, -0.0788, 5, 1),
    "joint_two_df": Circumstance(_joint_two_df, _JOINT, None, 0.2426, -0.1351, 0, 20),
    "p_is_one_sided": Circumstance(
        _p_is_one_sided, _WALD, "median_delta", 0.3010, 0.3010, 31, 0
    ),
    "p_from_another_model": Circumstance(
        _p_from_another_model, _WALD, None, 0.1442, -0.0462, 0, 0
    ),
    "low_frequency_rows_inflated": Circumstance(
        _low_frequency_rows_inflated, _WALD, None, 0.0, 0.0, 25, 0
    ),
    "se_printed_to_two_decimals": Circumstance(
        _se_printed_to_two_decimals, _WALD, None, 0.1216, -0.1216, 0, 0
    ),
    "sample_size_weighted_z": Circumstance(
        _sample_size_weighted_z, _WALD, None, 0.0396, 0.0015, 0, 0
    ),
    "p_at_a_floor": Circumstance(_p_at_a_floor, _WALD, None, 0.0, 0.0, 0, 0),
}

#: How closely a recorded median must match the fixture's literal.
_TOLERANCE = 5e-4


def _from_file(path: Path) -> dict[str, float | int]:
    """What the written file holds, measured without the resolver: every row's
    beta/se against its p."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    beta, se, p = (
        np.array([float(row[header.index(name)]) for row in rows])
        for name in ("beta", "standard_error", "p_value")
    )
    z = beta / se
    delta = log10_two_sided_p(z) - np.log10(p)
    return {
        "rows": len(rows),
        "log10_p_span": float(np.log10(p).max() - np.log10(p).min()),
        "median_abs_delta": float(np.median(np.abs(delta))),
        "median_delta": float(np.median(delta)),
        "false_hits": int(((np.abs(z) >= 5.45) & (p > 1e-4)).sum()),
        "lost_hits": int(((np.abs(z) < 3.72) & (p <= 5e-8)).sum()),
    }


def _written(tmp_path: Path, case: Circumstance) -> Path:
    """The circumstance's source, asserted to be what its docstring says it is."""
    path = case.write(tmp_path)
    measured_in_file = _from_file(path)
    # The fixture reaches the check whole and is what its docstring says it is.
    assert measured_in_file["rows"] == _N_NULL + _N_SIGNAL
    assert measured_in_file["log10_p_span"] > 20
    assert measured_in_file["median_abs_delta"] == pytest.approx(
        case.median_abs_delta, abs=_TOLERANCE
    )
    assert measured_in_file["median_delta"] == pytest.approx(case.median_delta, abs=_TOLERANCE)
    assert measured_in_file["false_hits"] == case.false_hits
    assert measured_in_file["lost_hits"] == case.lost_hits
    return path


@pytest.mark.parametrize("name", list(CIRCUMSTANCES))
def test_each_circumstance_reaches_the_verdict_the_document_claims(
    tmp_path: Path, panel: AncestryReference, name: str
) -> None:
    case = CIRCUMSTANCES[name]
    path = _written(tmp_path, case)

    result = _resolve(path, panel, p_value_test=case.declared)

    if case.refused_by == "p_value_test":
        assert result.error.startswith(f"p_value_test is {case.declared}")
        assert result.diagnostics.rows_read == 0
        return
    measured = result.diagnostics.effect_consistency
    assert measured is not None
    if case.declared is _JOINT:
        assert measured.skip_reason is ConsistencySkip.JOINT_TEST_P_VALUE
        assert not result.error
        return
    _assert_compared(result, case)


def _assert_compared(result, case: Circumstance) -> None:
    measured = result.diagnostics.effect_consistency
    assert measured is not None and measured.skip_reason is None
    assert measured.n_rows_compared == _N_NULL + _N_SIGNAL
    assert measured.median_abs_delta == pytest.approx(case.median_abs_delta, abs=_TOLERANCE)
    assert measured.median_delta == pytest.approx(case.median_delta, abs=_TOLERANCE)
    assert measured.refused_by == case.refused_by
    assert bool(result.error) == (case.refused_by is not None)


@pytest.mark.parametrize("name", ["mr_mega", "joint_two_df"])
def test_a_source_declared_a_wald_test_when_it_is_not_is_stored(
    tmp_path: Path, panel: AncestryReference, name: str
) -> None:
    """The declaration is trusted, and nothing checks it: an MR-MEGA or joint-test
    source declared `wald_1df` is compared, passes, and is stored as it was
    before #371."""
    case = CIRCUMSTANCES[name]
    path = _written(tmp_path, case)

    result = _resolve(path, panel, p_value_test=_WALD)

    _assert_compared(result, Circumstance(
        case.write, _WALD, None, case.median_abs_delta, case.median_delta,
        case.false_hits, case.lost_hits,
    ))


def test_the_document_lists_exactly_these_circumstances() -> None:
    """Each pinned circumstance has its row in the document, and the document
    names no test here that does not exist."""
    named = set(re.findall(r"`catalogue: ([a-z0-9_]+)`", _DOC.read_text(encoding="utf-8")))
    assert named == set(CIRCUMSTANCES)


def test_a_store_returns_the_p_of_its_stored_z_never_the_sources(tmp_path: Path) -> None:
    """The shortcut itself. A joint-test association whose source says p = 1e-15
    beside an interaction z of 0.5 is stored as z 0.5, and a query answers with
    z 0.5's own two-sided p, 0.617, not the source's: the source's p never
    reaches the Store."""
    rows = [
        {"chromosome": "1", "base_pair_location": 100_000, "effect_allele": "A",
         "other_allele": "G", "beta": 0.05, "standard_error": 0.1,
         "effect_allele_frequency": 0.3, "p_value": "1e-15"},
    ]
    store, _ = ragged_ssf_store(tmp_path, rows)
    with gzip.open(tmp_path / "filtered" / "trait_a.tsv.gz", "rt", encoding="utf-8") as fh:
        header, row = (line.rstrip("\n").split("\t") for line in fh)
    assert row[header.index("p_value")] == "1e-15", "the source must carry the contradicting p"

    with query_store(store) as query:
        resolved = list(query.resolve(query.analysis("trait_a")))

    assert len(resolved) == 1
    assert resolved[0]["z"] == pytest.approx(0.5, rel=1e-3)
    assert resolved[0]["log10_p"] == pytest.approx(math.log10(0.6170750774519738), abs=1e-3)

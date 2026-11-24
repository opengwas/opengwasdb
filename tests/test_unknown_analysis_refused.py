"""Every query that names an Analysis the release does not hold raises (#474).

The owner's ruling (24 Nov 2026): an unknown `analysis_id` raises `UnknownAnalysisError`
on every shape that takes one, on every layout, as `top_hits` already did (#483). Before
it, `analysis()`, `lookup()` and the Rho reads answered such an Analysis with an empty
result, which reads exactly as "no association" (spec §19).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opengwasdb.query import query_store
from opengwasdb.store_reader.analyses import UnknownAnalysisError

UNKNOWN = "no-such-analysis"

#: Each shape that names an Analysis, asked of a facade `q` whose first Analysis is
#: `known` and first variant `alid`.
SHAPES: dict[str, Callable[[Any, str, str], Any]] = {
    "analysis": lambda q, known, alid: q.analysis(UNKNOWN),
    "analysis/observed_only": lambda q, known, alid: q.analysis(UNKNOWN, observed_only=True),
    "lookup": lambda q, known, alid: q.lookup([alid], [UNKNOWN]),
    "lookup/among_known": lambda q, known, alid: q.lookup([alid], [known, UNKNOWN]),
    "lookup/no_variant_found": lambda q, known, alid: q.lookup(["1:1:A:C"], [UNKNOWN]),
    "top_hits": lambda q, known, alid: q.top_hits(analysis_id=UNKNOWN, threshold=5e-8),
}
#: The Rho reads, which Ragged does not offer.
RHO_SHAPES: dict[str, Callable[[Any, str, str], Any]] = {
    "rho": lambda q, known, alid: q.rho(known, UNKNOWN),
    "rho_row": lambda q, known, alid: q.rho_row(UNKNOWN),
    "rho_matrix": lambda q, known, alid: q.rho_matrix([known, UNKNOWN]),
}


def _asked(release: Path, ask: Callable[[Any, str, str], Any]) -> None:
    with query_store(release) as q:
        known = q.analyses_table()[0]["analysis_id"]
        alid = q.variants_table()[0]["alid"]
        assert q.analysis(known)["z"].size, "the known Analysis must answer, or proves nothing"
        ask(q, known, alid)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize(
    "name",
    [f"{layout}_{state}" for layout in ("dense", "ragged", "hybrid")
     for state in ("observed", "completed")],
)
def test_an_unknown_analysis_is_refused_by_name(digest_releases, name: str, shape: str):
    with pytest.raises(UnknownAnalysisError, match=UNKNOWN):
        _asked(digest_releases[name], SHAPES[shape])


@pytest.mark.parametrize("shape", sorted(RHO_SHAPES))
@pytest.mark.parametrize("name", ["dense_observed", "hybrid_completed"])
def test_the_rho_reads_refuse_an_unknown_analysis_with_or_without_a_rho_matrix(
    digest_releases, name: str, shape: str
):
    with pytest.raises(UnknownAnalysisError, match=UNKNOWN):
        _asked(digest_releases[name], RHO_SHAPES[shape])


@pytest.mark.parametrize("name", ["dense_observed", "hybrid_observed"])
def test_a_valid_indexed_subset_refuses_an_unknown_analysis(name: str):
    """The contract bundle's releases carry Indexed Variant Subset `contract`."""
    release = Path(__file__).parent / "store_reader_contract" / "releases" / name
    with query_store(release) as q:
        known = q.analyses_table()[0]["analysis_id"]
        assert q.analysis(known, indexed_subset="contract")["z"].size
        with pytest.raises(UnknownAnalysisError, match=UNKNOWN):
            q.analysis(UNKNOWN, indexed_subset="contract")


@pytest.mark.parametrize(
    "command",
    [["query-analysis", "{release}", UNKNOWN], ["query-lookup", "{release}", "1:1:A:C", UNKNOWN]],
)
def test_the_cli_names_an_unknown_analysis_and_exits_1(digest_releases, command: list[str]):
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    release = str(digest_releases["dense_observed"])
    result = CliRunner().invoke(app, [part.format(release=release) for part in command])
    assert result.exit_code == 1, result.output
    assert f"unknown analysis_id {UNKNOWN!r}" in result.output

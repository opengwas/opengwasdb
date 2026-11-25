"""What `top_hits` answers on each layout, whichever tiers a release stores (#483).

The owner's decision (24 Nov): a Dense release answers at a stored tier only and never
scans its grid (a scan on OGS-00016 is hours); a Hybrid answers at a stored tier only,
because its Dense Component cannot scan; a Ragged release scans an unstored threshold,
which is cheap on sparse data. A stored tier without per-Analysis offsets answers one
Analysis by filtering its rows. An unknown `analysis_id` raises on every layout.

The oracle is a brute-force filter of each Analysis's own associations, computed once
per release.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from opengwasdb.query import query_store
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store_reader.analyses import UnknownAnalysisError
from opengwasdb.store_reader.stats import z_critical
from opengwasdb.store_reader.top_hits import UnstoredThresholdError, threshold_key

RELEASES = [
    "dense_observed",
    "dense_completed",
    "ragged_observed",
    "ragged_completed",
    "hybrid_observed",
    "hybrid_completed",
]
RAGGED = [name for name in RELEASES if name.startswith("ragged")]
NOT_RAGGED = [name for name in RELEASES if not name.startswith("ragged")]
STORED = 5e-4
UNSTORED = (1e-2, 0.2)
#: Where each layout's tiers are: a Hybrid's Dense Component's and its Overflow's.
TIER_STORES = {
    "dense": ["data.zarr"],
    "ragged": ["data.zarr"],
    "hybrid": ["dense/data.zarr", "data.zarr"],
}


def _key(result) -> list[tuple]:
    return list(
        zip(
            result.analysis_index.tolist(),
            result.variant_index.tolist(),
            result.z.tolist(),
            result.status.tolist(),
            strict=True,
        )
    )


@pytest.fixture(scope="module")
def oracle(digest_releases):
    """Every association of every Analysis, once per release and `observed_only`."""
    everything = {}
    for name, release in digest_releases.items():
        with query_store(release) as q:
            ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
            for observed in (False, True):
                everything[name, observed] = {
                    aid: _key(q.analysis(aid, observed_only=observed)) for aid in ids
                }
    return everything


def _expected(oracle, name, threshold, *, analysis_id=None, observed_only=False, limit=None):
    """The rows of the Analysis (or all) with |z| above the cutoff, in genomic order."""
    cutoff = z_critical(threshold)
    per_analysis = oracle[name, observed_only]
    ids = [analysis_id] if analysis_id else list(per_analysis)
    rows = [r for aid in ids for r in per_analysis[aid] if abs(r[2]) >= cutoff]
    return sorted(rows, key=lambda r: (r[0], r[1]))[:limit]


def _ids(oracle, name) -> list[str]:
    return list(oracle[name, False])


def _copy(digest_releases, tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(digest_releases[name], tmp_path / name))


def _tier_roots(release: Path, name: str):
    """Each of the release's tier-holding groups, opened for writing."""
    for sub in TIER_STORES[name.split("_")[0]]:
        yield store_arrays.open_group_for_write(release / sub, "a")


def _asked(q, oracle, name, threshold):
    """`top_hits` at `threshold` against the oracle: all, each Analysis, `limit`."""
    for observed in (False, True):
        want = _expected(oracle, name, threshold, observed_only=observed)
        assert _key(q.top_hits(threshold=threshold, observed_only=observed)) == want
        for limit in (1, 3):
            got = q.top_hits(threshold=threshold, limit=limit, observed_only=observed)
            assert _key(got) == want[:limit]
        for aid in _ids(oracle, name):
            one = _expected(oracle, name, threshold, analysis_id=aid, observed_only=observed)
            got = q.top_hits(analysis_id=aid, threshold=threshold, observed_only=observed)
            assert _key(got) == one


@pytest.mark.parametrize("name", RELEASES)
def test_a_stored_tier_equals_the_brute_force_answer(digest_releases, oracle, name):
    assert _expected(oracle, name, STORED), "a fixture asserted meaningful"
    with query_store(digest_releases[name]) as q:
        _asked(q, oracle, name, STORED)


@pytest.mark.parametrize("name", RAGGED)
@pytest.mark.parametrize("threshold", UNSTORED)
def test_a_ragged_release_scans_an_unstored_threshold(digest_releases, oracle, name, threshold):
    assert _expected(oracle, name, threshold), "a fixture asserted meaningful"
    with query_store(digest_releases[name]) as q:
        _asked(q, oracle, name, threshold)


@pytest.mark.parametrize("name", NOT_RAGGED)
@pytest.mark.parametrize("threshold", UNSTORED)
def test_dense_and_hybrid_refuse_an_unstored_threshold_naming_the_tiers(
    digest_releases, name, threshold
):
    with query_store(digest_releases[name]) as q:
        for kwargs in ({}, {"limit": 3}, {"observed_only": True}):
            with pytest.raises(UnstoredThresholdError) as refused:
                q.top_hits(threshold=threshold, **kwargs)
            message = str(refused.value)
            assert f"{threshold:g}" in message
            assert "5e-08, 5e-06, 0.0005" in message
        aid = next(iter(q.analyses_table().values()))["analysis_id"]
        with pytest.raises(UnstoredThresholdError):
            q.top_hits(analysis_id=aid, threshold=threshold)


def _refuse_grid_scans(monkeypatch, grid) -> None:
    """Make `windows`, `z` and `read` raise: what a scan would call. `inspect` stays, for
    it is #253's point read at the hit cells."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a Dense top_hits scanned the grid")

    for method in ("windows", "z", "read"):
        monkeypatch.setattr(grid, method, refuse)


@pytest.mark.parametrize("name", NOT_RAGGED)
def test_a_dense_release_never_scans_its_grid(digest_releases, monkeypatch, name):
    """No grid scan is made for any answer a Dense or Hybrid release gives or refuses."""
    with query_store(digest_releases[name]) as q:
        dense = q if name.startswith("dense") else q._dense
        assert not hasattr(dense, "_top_hits_by_scan")
        _refuse_grid_scans(monkeypatch, dense._grid)
        aid = next(iter(q.analyses_table().values()))["analysis_id"]
        assert len(q.top_hits(threshold=STORED).z)
        assert len(q.top_hits(analysis_id=aid, threshold=STORED).z)
        with pytest.raises(UnstoredThresholdError):
            q.top_hits(threshold=0.2)
        with pytest.raises(UnstoredThresholdError):
            q.top_hits(analysis_id=aid, threshold=0.2)


@pytest.mark.parametrize("name", RELEASES)
def test_a_dropped_tier_is_unstored_on_dense_and_hybrid_and_scanned_on_ragged(
    digest_releases, oracle, tmp_path, name
):
    release = _copy(digest_releases, tmp_path, name)
    for root in _tier_roots(release, name):
        assert threshold_key(STORED) in root["top_hits"], "the tier to drop is there"
        del root["top_hits"][threshold_key(STORED)]
    with query_store(release) as q:
        if name in RAGGED:
            _asked(q, oracle, name, STORED)
        else:
            with pytest.raises(UnstoredThresholdError, match="5e-08, 5e-06"):
                q.top_hits(threshold=STORED)


@pytest.mark.parametrize("name", RELEASES)
def test_a_tier_without_offsets_answers_one_analysis_from_its_rows(
    digest_releases, oracle, tmp_path, monkeypatch, name
):
    release = _copy(digest_releases, tmp_path, name)
    dropped = 0
    for root in _tier_roots(release, name):
        for key in list(root["top_hits"]):
            if "analysis_offsets" in root["top_hits"][key]:
                del root["top_hits"][key]["analysis_offsets"]
                dropped += 1
    assert dropped, "the offsets to drop are there"
    assert any(_expected(oracle, name, STORED, analysis_id=a) for a in _ids(oracle, name))
    with query_store(release) as q:
        if name not in RAGGED:
            _refuse_grid_scans(monkeypatch, (q if name.startswith("dense") else q._dense)._grid)
        _asked(q, oracle, name, STORED)


@pytest.mark.parametrize("name", RELEASES)
@pytest.mark.parametrize("threshold", [STORED, 1e-2])
def test_an_unknown_analysis_raises_naming_it(digest_releases, name, threshold):
    with (
        query_store(digest_releases[name]) as q,
        pytest.raises(UnknownAnalysisError, match="no-such-analysis"),
    ):
        q.top_hits(analysis_id="no-such-analysis", threshold=threshold)


def test_the_cli_refuses_an_unstored_threshold_on_a_dense_release(digest_releases):
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    ran = CliRunner().invoke(
        app, ["query-top-hits", str(digest_releases["dense_observed"]), "--threshold", "0.2"]
    )
    assert ran.exit_code == 1
    assert "not a stored top-hit tier" in ran.output
    assert "5e-08, 5e-06, 0.0005" in ran.output

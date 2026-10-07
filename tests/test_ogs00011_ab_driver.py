"""The OGS-00011 A/B driver must refuse vacuous or unequal evidence (#252).

`benchmarks/ogs00011_ab.py` publishes the committed A/B and identity artifacts.
Round 2 of #252's review found that it would happily write an artifact for a run
whose reads were all empty or whose two sides disagreed, so these tests feed it
the records that must fail: an empty result for a probe known to return rows, a
timed-out probe, a row count on a shape whitelisted as empty, an unknown shape,
a before/after count or hash mismatch, and an identity run with no rows.
"""

from __future__ import annotations

import pytest

from benchmarks import ogs00011_ab

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _record(count: int, *, sha: str = _HASH_A, timed_out: bool = False) -> dict:
    return {"result_count": count, "sha256": sha, "timed_out": timed_out, "limit_s": 300.0}


def test_an_empty_result_for_a_non_empty_probe_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_shape("bulk", _record(0))


def test_a_timeout_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_shape("bulk", _record(5_000_000, timed_out=True))


def test_rows_on_a_whitelisted_empty_shape_fail() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_shape("random_lookup_10_variants_100_analyses", _record(3))


def test_an_unknown_shape_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_shape("not_a_shape", _record(1))


def test_a_count_mismatch_between_the_sides_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_pair("phewas", _record(3_152), _record(3_153))


def test_a_hash_mismatch_between_the_sides_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_pair("phewas", _record(3_152), _record(3_152, sha=_HASH_B))


def test_an_identity_run_with_no_rows_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_identity_side(
            "before", {"queries": {"x": {"rows": 0, "sha256": _HASH_A}}}
        )


def test_an_identity_query_that_differs_fails() -> None:
    with pytest.raises(SystemExit):
        ogs00011_ab._check_identity_pair(
            "x", {"rows": 1, "sha256": _HASH_A}, {"rows": 2, "sha256": _HASH_A}
        )

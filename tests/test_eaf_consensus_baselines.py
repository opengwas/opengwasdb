"""The consensus baseline is one sort per site, and says what the old loop said (issue #224).

`_consensus_baselines` used to rebuild a leave-one-out median from scratch for
every (Analysis, site) pair, and for every Analysis whether or not it reported
the site. On OGS-00011 (3,262 Analyses) that held 300 GiB on one core for more
than seven hours, almost all of it at pairs `_correlate` never reads. It is now
columnar over the observations and derives every leave-one-out median from one
sort. These tests pin the three things such a rewrite could silently get wrong:
the medians themselves, the "at least two *other* Analyses" boundary, and the
exclusion of each Analysis from its own baseline — plus the memory bound that
was the point of the change.

`_old_baselines` is the pre-#224 implementation verbatim. It is the oracle, not
a reimplementation: every equivalence below is against exactly what shipped.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Mapping

import numpy as np
import pytest

from opengwasdb.build.eaf_orientation import (
    EafReference,
    _consensus_baselines,
    check_eaf_orientation,
    correlate_frequencies,
)
from opengwasdb.model.enums import EafOrientationMethod, EafOrientationOutcome


def _old_baselines(
    observations: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """`_consensus_baselines` as it was before issue #224, unchanged."""
    per_site: dict[str, dict[str, float]] = {}
    for analysis_id, observed in observations.items():
        for alid, value in observed.items():
            per_site.setdefault(alid, {})[analysis_id] = value
    baselines: dict[str, dict[str, float]] = {a: {} for a in observations}
    for alid, by_analysis in per_site.items():
        for analysis_id in observations:
            others = [v for a, v in by_analysis.items() if a != analysis_id]
            if len(others) >= 2:
                baselines[analysis_id][alid] = float(np.median(others))
    return baselines


# Low enough that the fixtures below reach every outcome, including `failed`.
_GATES = {"min_overlap": 100, "min_variance": 0.005}


def _old_evidence(
    observed: Mapping[str, float], baseline: Mapping[str, float]
) -> tuple[str, float, int, str]:
    """The pre-#224 `_correlate`, as `(outcome, r, n_overlap, note)`."""
    shared = [alid for alid in observed if alid in baseline]
    n = len(shared)
    c = correlate_frequencies(
        np.fromiter((observed[a] for a in shared), dtype=np.float64, count=n),
        np.fromiter((baseline[a] for a in shared), dtype=np.float64, count=n),
        min_overlap=int(_GATES["min_overlap"]),
        min_variance=_GATES["min_variance"],
    )
    return c.outcome.value, c.r, n, c.note


def _new_baselines(observations: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """The new baselines, keyed back to `{analysis: {alid: median}}` for comparison."""
    out: dict[str, dict[str, float]] = {}
    for analysis_id, baseline in _consensus_baselines(observations).items():
        alids = list(observations[analysis_id])
        assert len(alids) == baseline.median.size == baseline.has_baseline.size
        out[analysis_id] = {
            alid: float(m)
            for alid, m, has in zip(
                alids, baseline.median.tolist(), baseline.has_baseline.tolist(), strict=True
            )
            if has
        }
    return out


def _same_float(a: float, b: float) -> bool:
    """Bit-identical, with NaN equal to NaN."""
    return (np.isnan(a) and np.isnan(b)) or np.float64(a).tobytes() == np.float64(b).tobytes()


def _mixed_observations(
    seed: int, *, n_analyses: int = 12, n_sites: int = 400, nan_rate: float = 0.0
) -> dict[str, dict[str, float]]:
    """Sites reported by anywhere from one Analysis to all of them, so every
    reporter count — odd and even, below and above the boundary — occurs, with
    frequencies rounded to force ties and, optionally, NaNs at scattered sites."""
    rng = np.random.default_rng(seed)
    alids = [f"1:{1000 + 7 * i}:A:G" for i in range(n_sites)]
    truth = rng.beta(0.6, 0.6, size=n_sites)
    n_reporters = rng.integers(1, n_analyses + 1, size=n_sites)
    reports = np.zeros((n_analyses, n_sites), dtype=bool)
    for i, k in enumerate(n_reporters.tolist()):
        reports[rng.choice(n_analyses, size=k, replace=False), i] = True
    observations: dict[str, dict[str, float]] = {}
    for a in range(n_analyses):
        noise = rng.normal(0.0, 0.03, size=n_sites)
        values = np.round(np.clip(truth + noise, 0.001, 0.999), 2)
        order = rng.permutation(n_sites)  # each Analysis iterates its sites in its own order
        observations[f"A{a:02d}"] = {
            alids[i]: (float("nan") if rng.random() < nan_rate else float(values[i]))
            for i in order.tolist()
            if reports[a, i]
        }
    return observations


def _reporter_counts(observations: Mapping[str, Mapping[str, float]]) -> set[int]:
    counts: dict[str, int] = {}
    for observed in observations.values():
        for alid in observed:
            counts[alid] = counts.get(alid, 0) + 1
    return set(counts.values())


# ── Equivalence ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_reachable_baseline_is_bit_identical_to_the_per_analysis_median(seed):
    observations = _mixed_observations(seed, nan_rate=0.01)
    # Meaningful fixture: every reporter count from 1 to 12 occurs, ties occur
    # (values rounded to 0.01), and NaNs occur.
    assert _reporter_counts(observations) == set(range(1, 13))
    values = [v for o in observations.values() for v in o.values()]
    assert any(np.isnan(v) for v in values)
    assert len(set(values)) < len(values) // 10

    old = _old_baselines(observations)
    new = _new_baselines(observations)

    for analysis_id, observed in observations.items():
        # Only the pairs `_correlate` could ever read — the Analysis's own sites.
        reachable = {alid: m for alid, m in old[analysis_id].items() if alid in observed}
        assert new[analysis_id].keys() == reachable.keys(), analysis_id
        for alid, m in reachable.items():
            assert _same_float(new[analysis_id][alid], m), (analysis_id, alid)
    n_compared = sum(len(b) for b in new.values())
    assert n_compared > 1000


def _report_fields(report):
    return [(e.analysis_id, e.outcome.value, e.r, e.n_overlap, e.note) for e in report.evidence]


def test_consensus_report_is_identical_to_the_old_path():
    """Same outcome, r, n_overlap and note per Analysis, in the same order, with
    a flipped Analysis, a thin one and one with no EAF in the mix."""
    observations: dict[str, dict[str, float]] = dict(
        _mixed_observations(7, n_analyses=10, n_sites=3000)
    )
    observations["A03"] = {a: 1.0 - v for a, v in observations["A03"].items()}
    observations["thin"] = dict(list(observations["A05"].items())[:50])
    observations["no_eaf"] = {}
    report = check_eaf_orientation(
        observations,
        min_overlap=int(_GATES["min_overlap"]),
        min_variance=_GATES["min_variance"],
    )
    assert report.method is EafOrientationMethod.CONSENSUS

    with_eaf = {a: o for a, o in observations.items() if o}
    old = _old_baselines(with_eaf)
    expected = [
        (a, *_old_evidence(with_eaf[a], old[a])) if a in with_eaf else None
        for a in observations
    ]
    got = _report_fields(report)
    # Meaningful: the old path produces every outcome here, and finite r values.
    assert {e[1] for e in expected if e} == {"passed", "failed", "unverified"}
    assert [g[0] for g in got] == list(observations)
    for g, e in zip(got, expected, strict=True):
        if e is None:
            assert g[1] == "unverified" and "stores no EAF" in g[4]
            continue
        assert g[:2] == e[:2] and g[3:] == e[3:], g
        assert _same_float(g[2], e[2]), (g, e)


def test_reference_panel_report_is_identical_to_the_old_path():
    observations = _mixed_observations(8, n_analyses=5, n_sites=3000)
    rng = np.random.default_rng(9)
    panel = {
        alid: float(rng.uniform(0.01, 0.99))
        for alid in {a for o in observations.values() for a in o}
    }
    for alid in list(panel)[::3]:
        del panel[alid]  # the panel does not cover every observed site
    reference = EafReference(
        reference_id="panel", checksum="x", n_variants=len(panel), fraction_above_half=0.5,
        eaf=panel,
    )
    report = check_eaf_orientation(
        observations,
        reference=reference,
        min_overlap=int(_GATES["min_overlap"]),
        min_variance=_GATES["min_variance"],
    )

    assert report.method is EafOrientationMethod.REFERENCE_PANEL
    for got, (analysis_id, observed) in zip(
        _report_fields(report), observations.items(), strict=True
    ):
        expected = (analysis_id, *_old_evidence(observed, panel))
        assert got[:2] == expected[:2] and got[3:] == expected[3:]
        assert _same_float(got[2], expected[2])


# ── The boundary: at least two *other* Analyses ─────────────────────────────


def test_a_baseline_needs_two_other_analyses_so_three_reporters():
    """At a site reported by exactly two Analyses each has one other — no
    baseline. At one reported by exactly three each has two — a baseline, the
    mean of the other two. Analyses that did not report a site get nothing."""
    observations = {
        "a": {"two": 0.10, "three": 0.20},
        "b": {"two": 0.30, "three": 0.40},
        "c": {"three": 0.70, "only_c": 0.5},
        "d": {"only_d": 0.5},
    }
    baselines = _new_baselines(observations)

    assert baselines == {
        "a": {"three": (0.40 + 0.70) / 2},
        "b": {"three": (0.20 + 0.70) / 2},
        "c": {"three": (0.20 + 0.40) / 2},
        "d": {},
    }
    # And the old rule agrees at the pairs it could ever read.
    old = _old_baselines(observations)
    assert {a: {k: v for k, v in old[a].items() if k in observations[a]} for a in old} == (
        baselines
    )


# ── Each Analysis is excluded from its own baseline ─────────────────────────


def test_an_analysis_is_left_out_of_its_own_baseline():
    """Four Analyses at every site: `self` and `o1` flipped, `o2` and `o3`
    correct. Leaving `self` out, its baseline is the median of (flipped,
    correct, correct) — the correct frequency — and it reads `failed`. Were it
    included, the median of four would be the mean of one flipped and one
    correct value, ~0.5 everywhere: no spread, `unverified`, and the flip goes
    through. That is the failure the exclusion exists to prevent."""
    rng = np.random.default_rng(3)
    n = 2000
    alids = [f"2:{10 + i}:C:T" for i in range(n)]
    truth = rng.uniform(0.05, 0.95, size=n)

    def noisy(v: np.ndarray) -> np.ndarray:
        return np.clip(v + rng.normal(0.0, 0.01, size=n), 0.0, 1.0)

    mine = noisy(1.0 - truth)
    others = {"o1": noisy(1.0 - truth), "o2": noisy(truth), "o3": noisy(truth)}
    observations = {
        "self": dict(zip(alids, mine.tolist(), strict=True)),
        **{name: dict(zip(alids, v.tolist(), strict=True)) for name, v in others.items()},
    }

    stacked = np.vstack(list(others.values()))
    excluded = np.median(stacked, axis=0)
    included = np.median(np.vstack([stacked, mine]), axis=0)
    r_excluded = float(np.corrcoef(mine, excluded)[0, 1])
    # Meaningful: including `self` changes the verdict, not just the digits.
    with_self = correlate_frequencies(mine, included)
    without_self = correlate_frequencies(mine, excluded)
    assert with_self.outcome is EafOrientationOutcome.UNVERIFIED, with_self
    assert without_self.outcome is EafOrientationOutcome.FAILED, without_self

    baseline = _consensus_baselines(observations)["self"]
    assert baseline.has_baseline.all()
    np.testing.assert_array_equal(baseline.median, excluded)

    report = check_eaf_orientation(observations)
    evidence = {e.analysis_id: e for e in report.evidence}["self"]
    assert evidence.outcome is EafOrientationOutcome.FAILED, evidence.note
    assert evidence.n_overlap == n
    assert evidence.r == pytest.approx(r_excluded, abs=1e-12)


# ── Memory: bounded by the observations, not Analyses x sites ───────────────


def _groups_of_three(n_analyses: int, per_analysis: int) -> dict[str, dict[str, float]]:
    """Every site reported by exactly three Analyses, so every observation has
    a baseline, and the site union grows with the Analysis count."""
    assert n_analyses % 3 == 0
    rng = np.random.default_rng(n_analyses)
    observations: dict[str, dict[str, float]] = {}
    for a in range(n_analyses):
        group = a // 3
        observations[f"A{a:04d}"] = {
            f"{group}:{s}:A:G": float(v) for s, v in enumerate(rng.random(per_analysis))
        }
    return observations


def _peak_bytes(fn, observations) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        fn(observations)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_peak_memory_scales_with_observations_not_analyses_times_sites():
    """Quadrupling the Analyses quadruples the observations and the site union,
    so Analyses x sites grows sixteen-fold. The step's peak must follow the
    observations (about 4x), and stay within a fixed number of bytes per
    observation, rather than following Analyses x sites."""
    per_analysis = 60
    small, large = _groups_of_three(90, per_analysis), _groups_of_three(360, per_analysis)
    n_large = 360 * per_analysis
    sites_large = len({alid for o in large.values() for alid in o})
    assert sites_large == 120 * per_analysis  # fixture: 120 groups of three
    assert 360 * sites_large / n_large == 120  # Analyses x sites is 120x the observations

    peak_small = _peak_bytes(_consensus_baselines, small)
    peak_large = _peak_bytes(_consensus_baselines, large)

    per_observation = peak_large / n_large
    assert per_observation < 400, f"{per_observation:.0f} bytes per observation"
    assert peak_large / peak_small < 6, (peak_small, peak_large)

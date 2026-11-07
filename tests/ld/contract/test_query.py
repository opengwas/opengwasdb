"""ogld#6's contract: proxies and clumping on the Test Store, with the refusals of spec §9.

Expected proxies come from the Test Panel's reference r, quantised and judged on r̂² as the
spec says. Expected clumps are plink2's leads at k = 0 (lane 2d's `clump_leads`), against a
k = 0 store the walking skeleton's `scripts/make_test_store.py` writes from `reference_r(0)`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import support
from script_modules import load_script
from support import alids, contract, panel_data, passing_rows, reference

import ogld
from ogld.format import RefusalReason

OGLD_6 = contract("ogld#6")

#: The p-value threshold for an index variant in lane 2d's plink2 --clump run (--clump-p1).
#: The 5e-8 tier, as in ogld#6; tests/data/test-panel/README.md is the authority.
P1 = 5e-8


def expected_proxies(query: int, r2: float, window: float) -> dict[int, float]:
    """Each partner of Test Panel row `query` with r̂² ≥ r2 within `window` bp: row → r."""
    ref = reference(support.K)
    r_hat = ref.code.astype(np.float64) / 32767
    keep = (r_hat * r_hat >= r2) & (ref.distance <= window)
    out = {}
    for i, j, r in zip(ref.row_i[keep], ref.row_j[keep], r_hat[keep], strict=True):
        if i == query:
            out[int(j)] = float(r)
        elif j == query:
            out[int(i)] = float(r)
    return out


def busiest_query(r2: float, window: float) -> int:
    """The passing row with the most proxies, so the test has something to compare."""
    rows = passing_rows()
    counts = [len(expected_proxies(int(q), r2, window)) for q in rows[:: max(1, len(rows) // 200)]]
    best = int(rows[:: max(1, len(rows) // 200)][int(np.argmax(counts))])
    assert max(counts) > 1, "the Test Panel must hold a cluster with proxies at this r²"
    return best


@OGLD_6
def test_proxies_default_to_r2_0_8_within_10_mb() -> None:
    store = ogld.open(support.TEST_STORE)
    query = busiest_query(0.8, 10_000_000)
    found = store.proxies(alids([query]))
    assert (found.r2_min, found.window_bp, found.t_far) == (0.8, 10_000_000, support.T_FAR)
    expected = expected_proxies(query, 0.8, 10_000_000)
    assert sorted(found.proxy) == sorted(alids(list(expected)))
    assert (found.r2 >= 0.8).all()
    assert (found.distance <= 10_000_000).all()


@OGLD_6
def test_proxies_return_signed_r() -> None:
    store = ogld.open(support.TEST_STORE)
    rows = passing_rows()
    sample = rows[:: max(1, len(rows) // 100)]
    found = store.proxies(alids(sample), r2=0.5, kb=3000)
    expected = {}
    for q in sample:
        for p, r in expected_proxies(int(q), 0.5, 3_000_000).items():
            expected[(alids([q])[0], alids([p])[0])] = r
    assert any(r < 0 for r in expected.values()), "the fixture must hold negative r"
    assert any(r > 0 for r in expected.values()), "the fixture must hold positive r"
    got = {(q, p): r for q, p, r in zip(found.query, found.proxy, found.r, strict=True)}
    assert got == expected
    np.testing.assert_array_equal(found.r2, found.r * found.r)


@OGLD_6
def test_proxies_are_ordered_and_never_the_query_itself() -> None:
    store = ogld.open(support.TEST_STORE)
    query = busiest_query(0.5, 3_000_000)
    found = store.proxies(alids([query]), r2=0.5, kb=3000)
    assert alids([query])[0] not in set(found.proxy)
    keys = list(zip(-found.r2, found.distance, found.proxy, strict=True))
    assert keys == sorted(keys), "rows go by r² descending, then distance, then ALID"


@OGLD_6
def test_proxies_below_t_far_within_3_mb_read_the_near_band() -> None:
    # r2 = 0.01 < t_far is answered within 3 Mb: every pair there is stored (spec §9).
    store = ogld.open(support.TEST_STORE)
    query = busiest_query(0.01, 3_000_000)
    found = store.proxies(alids([query]), r2=0.01, kb=3000)
    expected = expected_proxies(query, 0.01, 3_000_000)
    assert len(expected) > len(expected_proxies(query, support.T_FAR, 3_000_000)), (
        "the fixture must hold pairs below t_far, or this reads nothing the index lacks"
    )
    assert sorted(found.proxy) == sorted(alids(list(expected)))


@OGLD_6
def test_proxies_list_absent_queries() -> None:
    store = ogld.open(support.TEST_STORE)
    found = store.proxies(["not-an-alid", *alids(passing_rows()[:1])])
    assert found.absent == ("not-an-alid",)


@OGLD_6
@pytest.mark.parametrize(
    ("r2", "kb", "reason"),
    [
        (0.01, 5_000, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.8, 10_001, RefusalReason.WINDOW_OVER_FAR),
        (1.5, 1_000, RefusalReason.INVALID),
        (0.5, -1, RefusalReason.INVALID),
    ],
)
def test_each_proxies_refusal(r2: float, kb: float, reason: RefusalReason) -> None:
    store = ogld.open(support.TEST_STORE)
    with pytest.raises(ogld.Refusal) as refused:
        store.proxies(alids(passing_rows()[:1]), r2=r2, kb=kb)
    assert refused.value.reason is reason


# ------------------------------------------------------------------ clump


def p_value(z: np.ndarray) -> np.ndarray:
    return np.array([math.erfc(abs(x) / math.sqrt(2)) for x in z])


def candidates(p_max: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The passing rows at p ≤ p_max: ALIDs, |z| and rows."""
    z = panel_data().gwas().z
    rows = passing_rows()
    keep = rows[p_value(z[rows]) <= p_max]
    assert len(keep) > 2, f"the simulated GWAS must have hits at p ≤ {p_max}"
    return alids(keep), np.abs(z[keep]), keep


def k0_store(tmp_path: Path) -> ogld.LDStore:
    load_script("make_test_store").write_test_store(0, tmp_path / "k0")
    return ogld.open(tmp_path / "k0")


@OGLD_6
@pytest.mark.parametrize("r2", [0.01, 0.001])
def test_clump_leads_equal_plink2_s_at_k0(r2: float, tmp_path: Path) -> None:
    ogld.open(support.TEST_STORE)
    store = k0_store(tmp_path)
    names, score, _ = candidates(P1)
    leads = set(store.clump(names, score, r2=r2, kb=3000).leads())
    plink2 = set(alids(panel_data().clump_leads(r2)))
    assert plink2, "plink2 must have found leads for this to mean anything"
    assert leads == plink2, (
        f"only ours: {sorted(leads - plink2)}; only plink2's: {sorted(plink2 - leads)}"
    )


@OGLD_6
def test_every_member_reaches_its_lead_and_every_lead_leads_itself(tmp_path: Path) -> None:
    ogld.open(support.TEST_STORE)
    store = k0_store(tmp_path)
    names, score, _ = candidates(1e-4)
    clumps = store.clump(names, score, r2=0.01, kb=3000)
    assert list(clumps.alid) == names
    is_lead = clumps.lead == clumps.alid
    assert is_lead.any()
    assert (~is_lead).any(), "the fixture must clump some variants into others"
    assert (clumps.r[is_lead] == 1.0).all()
    assert (clumps.r[~is_lead] ** 2 >= 0.01).all()
    lead_score = dict(zip(clumps.alid, clumps.score, strict=True))
    assert all(lead_score[lead] >= s for lead, s in zip(clumps.lead, clumps.score, strict=True))


@OGLD_6
def test_a_stricter_threshold_gives_a_prefix_of_the_leads(tmp_path: Path) -> None:
    # ogld#6: the leads at p ≤ 5e-8 are the leads at p ≤ 1e-4 with p ≤ 5e-8, so the engine
    # can store one clumping per configuration.
    ogld.open(support.TEST_STORE)
    store = k0_store(tmp_path)
    loose_names, loose_score, _ = candidates(1e-4)
    strict_names, strict_score, _ = candidates(P1)
    loose = store.clump(loose_names, loose_score, r2=0.01, kb=3000).leads()
    strict = store.clump(strict_names, strict_score, r2=0.01, kb=3000).leads()
    assert list(strict) == [lead for lead in loose if lead in set(strict_names)]


@OGLD_6
@pytest.mark.parametrize(
    ("r2", "kb", "reason"),
    [
        (0.001, 10_000, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.01, 3_000.001, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.1, 10_000.001, RefusalReason.WINDOW_OVER_FAR),
        (-0.1, 1_000, RefusalReason.INVALID),
    ],
)
def test_each_clump_refusal(r2: float, kb: float, reason: RefusalReason) -> None:
    store = ogld.open(support.TEST_STORE)
    names = alids(passing_rows()[:3])
    with pytest.raises(ogld.Refusal) as refused:
        store.clump(names, np.array([6.0, 5.0, 4.0]), r2=r2, kb=kb)
    assert refused.value.reason is reason


@OGLD_6
@pytest.mark.parametrize("score", [[6.0, math.nan, 4.0], [6.0, -1.0, 4.0], [6.0, 5.0]])
def test_a_score_that_is_not_one_per_alid_finite_and_non_negative_is_invalid(
    score: list[float],
) -> None:
    store = ogld.open(support.TEST_STORE)
    with pytest.raises(ogld.Refusal) as refused:
        store.clump(alids(passing_rows()[:3]), np.array(score), r2=0.1, kb=1000)
    assert refused.value.reason is RefusalReason.INVALID

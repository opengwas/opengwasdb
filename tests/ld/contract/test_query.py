"""ogld#6's contract: proxies and clumping on the Test Store, with the refusals of ld-store §9.

Expected proxies come from the Test Panel's reference r, quantised and judged on r̂² as the
spec says. Expected clumps are plink2's leads at k = 0 (lane 2d's `clump_leads`), made with
`--clump-unphased` and the settings `clump_params()` gives (p1 = p2 = 5e-8, 3,000 kb), against a
k = 0 store the walking skeleton's `scripts/ld/make_test_store.py` writes from `reference_r(0)`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import support
from ld_script_modules import load_script
from support import alids, panel_data, passing_rows, reference

import opengwasdb.ld

#: plink2's unphased leads at r² 0.001 on the Test Panel (tests/ld/data/test-panel/README.md) are
#: held there, but a 0.2.0 store refuses r² 0.001 within 3 Mb: its t_partner is 0.01, so the
#: comparison that used them is answered at r² 0.01 only (ld-store §9; the refusal is tested below).

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


_K0: list[opengwasdb.ld.LDStore] = []


def k0_store(tmp_path_factory: pytest.TempPathFactory) -> opengwasdb.ld.LDStore:
    """A k = 0 store written from reference_r(0), once per worker."""
    if not _K0:
        out = tmp_path_factory.mktemp("k0") / "store"
        script = load_script("make_test_store")
        made = script.write_test_store(0, out, support.R_SCALE, 0.01)
        _K0.append(opengwasdb.ld.open(made))
    return _K0[0]


@pytest.mark.parametrize("r2", [0.01])
def test_clump_leads_equal_plink2_s_at_k0(
    r2: float, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Leads equal plink2's. plink2 v2.0.0-a.6.9LM joins a variant at r² > --clump-r2 and distance
    < --clump-kb × 1,000 (checked on a constructed pair at exactly r² 0.25, and at exactly 3 Mb),
    and so does `clump` (owner, 10 November 2026); `proxies` keeps ≥ and ≤. No candidate pair of
    the Test Panel sits on either edge, so the leads are the same under both; the edges are
    tested in ogquery's tests/ld/contract/test_query_edges.py (opengwas/ogquery#2)."""
    opengwasdb.ld.open(support.TEST_STORE)
    store = k0_store(tmp_path_factory)
    params = panel_data().clump_params()
    assert params.p1 == params.p2, "with p2 = p1, plink2's leads depend only on the p1 set"
    names, score, _ = candidates(params.p1)
    leads = set(store.clump(names, score, r2=r2, kb=params.kb).leads())
    plink2 = set(alids(panel_data().clump_leads(r2)))
    assert plink2, "plink2 must have found leads for this to mean anything"
    assert leads == plink2, (
        f"only ours: {sorted(leads - plink2)}; only plink2's: {sorted(plink2 - leads)}"
    )


@pytest.mark.parametrize("r2", [0.01])
def test_proxies_are_plink2_s_r_outside_the_band_of_the_store_s_step(
    r2: float, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # k = 0's reference r is plink2's --r-unphased (tests/ld/test_testdata.py). The proxies within
    # 3 Mb of the comparison's candidates, judged on plink2's r², differ from the store's only by
    # pairs within 2|r| × half a step of r2.
    opengwasdb.ld.open(support.TEST_STORE)
    store = k0_store(tmp_path_factory)
    ref = reference(0)
    *_, rows = candidates(panel_data().clump_params().p1)
    names, _ = support.oriented()
    passing = set(passing_rows().tolist())
    band = np.abs(ref.r**2 - r2) <= 2 * np.abs(ref.r) * support.HALF_STEP + support.SLACK
    seen, moved = 0, 0
    for row in rows:
        found = store.proxies(alids([row]), r2=r2, kb=2_999)
        mine = ((ref.row_i == row) | (ref.row_j == row)) & (ref.distance <= 2_999_000)
        other = np.where(ref.row_i == row, ref.row_j, ref.row_i)
        theirs = {int(x) for x in other[mine & (ref.r**2 >= r2)] if int(x) in passing}
        got = {int(np.flatnonzero(names == p)[0]) for p in found.proxy}
        inside = {int(x) for x in other[mine & band]}
        assert theirs ^ got <= inside, (row, sorted((theirs ^ got) - inside))
        seen, moved = seen + len(got), moved + len(theirs ^ got)
    assert seen > 100, "the candidates must have many proxies for this to mean anything"
    assert moved < seen / 20, "the band is a sliver of the answer"


def test_every_member_reaches_its_lead_and_every_lead_leads_itself(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    opengwasdb.ld.open(support.TEST_STORE)
    store = k0_store(tmp_path_factory)
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


def test_a_stricter_threshold_gives_a_prefix_of_the_leads(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # ogld#6: the leads at p ≤ p1 (5e-8) are the leads at p ≤ 1e-4 with p ≤ p1, so the engine
    # can store one clumping per configuration.
    opengwasdb.ld.open(support.TEST_STORE)
    store = k0_store(tmp_path_factory)
    loose_names, loose_score, _ = candidates(1e-4)
    strict_names, strict_score, _ = candidates(panel_data().clump_params().p1)
    assert len(loose_names) > len(strict_names), "the looser set must add variants to test"
    loose = store.clump(loose_names, loose_score, r2=0.01, kb=3000).leads()
    strict = store.clump(strict_names, strict_score, r2=0.01, kb=3000).leads()
    assert list(strict) == [lead for lead in loose if lead in set(strict_names)]

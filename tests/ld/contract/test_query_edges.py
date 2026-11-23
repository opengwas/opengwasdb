"""ogld#6 beyond the contract's twenty: the edges of plink2's clumping, the Bound, the order
of `leads()`, and the Far Pairs `proxies` reads, each against an independent computation.

plink2 v2.0.0-a.6.9LM compares r² with `>` and the window with `<` (checked on a constructed
pair whose r² is exactly 0.25, and on partners exactly 3,000,000 bp and one bp nearer). The
owner ruled on 10 November 2026 that `clump` follows it and `proxies` keeps ≥ and ≤ (spec §9).
A stored r̂² equals r2 exactly only at 0 or 1, so the r² edge is tested on a perfect-LD pair.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import support
from support import alids, panel_data, passing_rows, reference

import opengwasdb.ld
from opengwasdb.ld.query import clumping
from opengwasdb.ld.query.results import Clumps
from opengwasdb.ld.store import StoreReader


@cache
def passing_pairs() -> support.Pairs:
    """The reference pairs whose rows both pass the build's filters, i.e. are in the store."""
    ref = reference(support.K)
    kept = np.zeros(len(panel_data().variants().pos), dtype=bool)
    kept[passing_rows()] = True
    return support.Pairs(ref.row_i, ref.row_j, ref.distance, ref.r).where(
        kept[ref.row_i] & kept[ref.row_j], "pairs between stored variants"
    )


def clump_two(first: int, second: int, *, r2: float, kb: float) -> Clumps:
    store = opengwasdb.ld.open(support.TEST_STORE)
    return store.clump(alids([first, second]), np.array([2.0, 1.0]), r2=r2, kb=kb)


def test_a_pair_exactly_w_apart_does_not_join_and_one_bp_nearer_does() -> None:
    pairs = passing_pairs()
    code = np.abs(pairs.code.astype(np.int64))
    above = (code / support.R_SCALE) ** 2 > 0.01
    exact = (pairs.distance <= 3_000_000) & (pairs.distance % 1000 == 0) & above
    at = pairs.where(exact, "a stored pair a whole number of kb apart with r̂² > 0.01")
    i, j, d = int(at.row_i[0]), int(at.row_j[0]), int(at.distance[0])
    assert d / 1000 * 1000 == d, "kb × 1,000 must give the distance back exactly"
    assert len(set(clump_two(i, j, r2=0.01, kb=(d + 1) / 1000).leads())) == 1, "one bp wider joins"
    assert len(set(clump_two(i, j, r2=0.01, kb=d / 1000).leads())) == 2, "exactly W does not (<)"
    store = opengwasdb.ld.open(support.TEST_STORE)
    found = store.proxies(alids([i]), r2=0.01, kb=d / 1000)
    assert alids([j])[0] in set(found.proxy), "proxies keeps d ≤ W"


def test_a_pair_whose_r_hat_squared_equals_r2_does_not_join_but_is_a_proxy() -> None:
    pairs = passing_pairs()
    perfect = pairs.where(
        (np.abs(pairs.code.astype(np.int64)) == support.R_SCALE) & (pairs.distance <= 3_000_000),
        "a stored pair in perfect LD",
    )
    i, j = int(perfect.row_i[0]), int(perfect.row_j[0])
    assert len(set(clump_two(i, j, r2=0.99, kb=3000).leads())) == 1, "just below 1 joins"
    assert len(set(clump_two(i, j, r2=1.0, kb=3000).leads())) == 2, "r̂² = r2 does not (>)"
    store = opengwasdb.ld.open(support.TEST_STORE)
    assert alids([j])[0] in set(store.proxies(alids([i]), r2=1.0, kb=3000).proxy), "proxies: ≥"


def stored_r(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The reference r among `rows`, quantised and read back; NaN where the reference holds none."""
    r = support.reference_matrix(support.K, tuple(rows.tolist()))
    held = ~np.isnan(r)
    out = np.full(r.shape, np.nan)
    out[held] = support.stored_at(r[held])
    return out


def greedy(rows: npt.NDArray[np.int64], score: npt.NDArray[np.float64], r2: float, kb: float):
    """An independent greedy clumping from the reference r, quantised: row → (lead row, r)."""
    r = stored_r(rows)
    pos = panel_data().variants().pos.astype(np.int64)[rows]
    chrom = np.asarray(panel_data().variants().chrom)[rows]
    lead = np.full(len(rows), -1)
    for i in sorted(range(len(rows)), key=lambda k: (-score[k], rows[k])):
        if lead[i] >= 0:
            continue
        reach = (chrom == chrom[i]) & (np.abs(pos - pos[i]) < kb * 1000) & (r[i] * r[i] > r2)
        reach[i] = True
        lead[reach & (lead < 0)] = i
    return lead


def test_beyond_3_mb_only_the_index_joins_and_a_variant_keeps_its_strongest_lead() -> None:
    # r2 = 0.1 ≥ t_far with a 10 Mb window: a pair 3–10 Mb apart that is not in the Proxy
    # Index is the Bound and NaN, and never joins. A mutant reading NaN as 1 gives 250 leads.
    rows = passing_rows()
    z = np.abs(panel_data().gwas().z)[rows]
    ours = opengwasdb.ld.open(support.TEST_STORE).clump(alids(rows), z, r2=0.1, kb=10_000)
    lead = greedy(rows, z, 0.1, 10_000)
    assert list(ours.lead) == alids(rows[lead])
    # 439 clumps at × 32767; at the Test Store's × 511 five pairs whose true r² lies 3–7×10⁻⁵
    # below 0.1, within 2|r| × half a step of it, read as r̂² 0.1005 and join (ogld#16).
    assert len(set(lead.tolist())) == 438, "the fixture's clump count, from the reference"
    pos = panel_data().variants().pos.astype(np.int64)
    # A variant reached by two leads keeps the stronger one: the guard `lead < 0` is needed.
    r = stored_r(rows)
    leads = np.unique(lead)
    reach = (np.nan_to_num(r[:, leads]) ** 2 > 0.1) & (
        np.abs(pos[rows][:, None] - pos[rows][leads][None, :]) < 10e6
    )
    assert (reach.sum(axis=1) > 1).any(), "some variant must be reached by two leads"


def test_leads_go_strongest_first_then_by_genomic_order_then_alid() -> None:
    names = ["10:5:A:C", "2:50:A:C", "1:500:A:C", "1:100:A:G", "1:100:A:C", "1:900:A:C"]
    chrom = np.array(["10", "2", "1", "1", "1", "1"])
    position = np.array([5, 50, 500, 100, 100, 900], dtype=np.int64)
    score = np.array([5.0, 5.0, 3.0, 5.0, 5.0, 1.0])
    lead = np.array([names[0], names[1], names[2], names[3], names[4], names[4]])
    clumps = Clumps(
        alid=np.array(names), chrom=chrom, position=position, score=score, lead=lead,
        r=np.ones(6), absent=(), r2=0.1, window_bp=1000.0,
    )  # fmt: skip
    expected = ["1:100:A:C", "1:100:A:G", "2:50:A:C", "10:5:A:C", "1:500:A:C"]
    assert list(clumps.leads()) == expected


def test_clump_reads_one_lead_row_at_a_time_and_never_a_square(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lead reads its own Partner Rows row, one frame at most, and never a square among the
    hits: a 100 kb window must not read an n × n matrix (partner-rows §8.2)."""
    rows = passing_rows()
    z = np.abs(panel_data().gwas().z)[rows]
    reader = StoreReader.open(support.TEST_STORE)
    read = reader.partner_row
    entries: list[int] = []

    def counted(chrom: str, local_row: int) -> tuple[npt.NDArray[np.int32], Any]:
        partners, codes = read(chrom, local_row)
        entries.append(len(partners))
        return partners, codes

    monkeypatch.setattr(reader, "partner_row", counted)
    narrow = clumping.clump(reader, alids(rows), z, r2=0.01, kb=100)
    assert entries, "the reads must be counted for this to mean anything"
    assert sum(entries) < len(rows) ** 2 // 4, "a 100 kb window must not read an n × n matrix"
    assert list(narrow.lead) == alids(rows[greedy(rows, z, 0.01, 100)])


def test_a_far_pair_beyond_near_bp_is_a_proxy_from_both_sides() -> None:
    pairs = passing_pairs()
    code = np.abs(pairs.code.astype(np.int64))
    far = pairs.where(
        (pairs.distance > 3_000_000) & (pairs.distance <= 10_000_000) & (code >= support.FLOOR),
        "a Far Pair: beyond 3 Mb with r̂² ≥ t_far",
    )
    i, j = int(far.row_i[0]), int(far.row_j[0])
    r_hat = float(support.stored_at(far.r[:1])[0])
    store = opengwasdb.ld.open(support.TEST_STORE)
    for query, proxy in ((i, j), (j, i)):
        found = store.proxies(alids([query]), r2=support.T_FAR, kb=10_000)
        got = dict(zip(found.proxy, found.r, strict=True))
        assert got[alids([proxy])[0]] == r_hat

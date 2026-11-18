"""What `proxies` and `clump` share: judging a request against the store (spec §9; ogld#6)."""

from __future__ import annotations

from ogld.format import judge_partner_request
from ogld.store import StoreReader


def window_of(reader: StoreReader, *, r2: float, kb: float) -> float:
    """The request's window in bp, W = kb × 1,000, or the refusal it is given instead.

    It is compared with the distances and the thresholds the store recorded, never a rounded
    display. From 0.2.0 both queries read the Partner Rows and nothing else, so a store that holds
    none is refused first, naming the missing component and `ogld derive-partners`, rather than
    answered from the Near Band (partner-rows §5.1, §8.1). Then the request is judged with the
    store's own t_partner as well as its t_far: within `near_bp` a threshold below t_partner is the
    refusal `threshold_below_t_partner`, decided before any frame is read.
    """
    t_partner = reader.require_partner_rows()
    p = reader.provenance
    decision = judge_partner_request(
        r2=r2,
        kb=kb,
        t_far=p["t_far"],
        near_bp=p["near_bp"],
        far_bp=p["far_bp"],
        t_partner=t_partner,
    )
    decision.raise_if_refused()
    return decision.window_bp

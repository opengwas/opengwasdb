"""What `proxies` and `clump` share: judging a request against the store (spec §9; ogld#6)."""

from __future__ import annotations

from ogld.format import judge_request
from ogld.store import StoreReader


def window_of(reader: StoreReader, *, r2: float, kb: float) -> float:
    """The request's window in bp, W = kb × 1,000, or the Refusal `judge_request` decides.

    It is compared with the distances and t_far the store recorded, never a rounded display.
    """
    p = reader.provenance
    decision = judge_request(
        r2=r2, kb=kb, t_far=p["t_far"], near_bp=p["near_bp"], far_bp=p["far_bp"]
    )
    decision.raise_if_refused()
    return decision.window_bp

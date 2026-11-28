"""The tuning every Reference Completion takes (#327)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionOptions:
    """The tuning every Reference Completion takes: how far imputed values must correlate
    with the panel and how much of its variance to keep (``min_cor``, ``thresh``), the
    release id to stamp, the block pool's size, and whether to replace a Store already at
    the destination."""

    min_cor: float = 0.7
    thresh: float = 0.9
    release_id: str | None = None
    n_workers: int = 1
    overwrite: bool = False


#: The defaults, for a completion asked for nothing else. Frozen, so safe to share.
DEFAULT_OPTIONS = CompletionOptions()

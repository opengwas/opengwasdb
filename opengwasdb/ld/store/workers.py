"""The component pool's size and its cap, one rule for both readers (partner-rows §9.2)."""

from __future__ import annotations

from typing import Final

from ogld.format import InputError

#: The most component threads one request runs. Four is where the prototype was fastest (#65:
#: 1.051 s at G = 8 and 1.067 s at G = 16); eight is slower, 5.753 s and 5.109 s, because a
#: component's per-lead Python work holds the GIL while only the reads and the decode release it.
COMPONENT_WORKERS_MAX: Final = 4


class ComponentWorkers:
    """How many threads one `clump` runs its components on, and the cap that bounds it.

    The pool's size is the reader's, so no query signature carries it. The LD Store reader
    starts at the cap; the Genotype Store reader starts at one, since a request's own decode and
    BLAS bound are already set. Either may be set through the private seam that `clump`'s tests
    use, which refuses anything but a whole count of at least one.
    """

    _query_workers: int = COMPONENT_WORKERS_MAX

    def _set_query_workers(self, requested: int) -> None:
        """Set the threads one `clump` may run its components on, holding the cap."""
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise InputError(
                f"the component thread count is a whole number of at least 1: {requested!r}"
            )
        self._query_workers = min(requested, COMPONENT_WORKERS_MAX)

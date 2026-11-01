"""Build pipeline orchestration."""

from opengwasdb.build.source import (
    NormalisedAssociation,
    SourceRowError,
    read_normalised_associations,
)

# `build_dense_observed_from_sources` is deliberately not re-exported here. The
# forwarder that used to sit on this line took `*args, **kwargs` and returned
# `Any`, which hid the real signature from every caller and from mypy while
# adding nothing: it imported its own sibling on each call. Nothing imported it.
# The one implementation is `opengwasdb.build.observed`'s, with its real
# signature, and that is the only path to it (#58).

__all__ = [
    "NormalisedAssociation",
    "SourceRowError",
    "read_normalised_associations",
]

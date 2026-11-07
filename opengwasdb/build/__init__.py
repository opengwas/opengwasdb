"""Build pipeline orchestration."""

# `build_dense_observed_from_manifest` is deliberately not re-exported here. A
# forwarder taking `*args, **kwargs` and returning `Any` once sat on this line and
# hid the real signature from every caller and from mypy while adding nothing.
# The one implementation is `opengwasdb.build.observed`'s, with its real
# signature, and that is the only path to it (#58).

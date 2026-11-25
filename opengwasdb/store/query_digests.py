"""The Store Reader's query digests, opened the way opengwasdb opens a release (#475).

The digests live in `ogquery.digests`; this module re-exports its `__all__` and the few
names opengwasdb imports that `__all__` leaves out, and overrides two functions. `ogquery`
opens a release with the Store Reader alone, so a 0.2.x release is refused for want of a
Store Variant Table reader; opengwasdb has one. `release_answers` and `release_digests`
below default `query=` to `opengwasdb.query.query_store`, which plugs it in
(`variants.legacy_table`), and this is the only place that knows the default: the rewrite
proof (`store.rewrite`) and the tests that digest a 0.2.3 release call these.
"""

from pathlib import Path
from typing import Any

from ogquery import digests as _digests
from ogquery.digests import *
from ogquery.digests import RANGE_CELLS_ASKED as RANGE_CELLS_ASKED
from ogquery.digests import RANGE_ROWS_ASKED as RANGE_ROWS_ASKED
from ogquery.digests import QueryOpener
from ogquery.digests import adapter_name as adapter_name
from ogquery.digests import expects_refusal as expects_refusal
from ogquery.digests import probed_alids as probed_alids


def _query_store(path: Path) -> Any:
    from opengwasdb.query import query_store

    return query_store(path)


def release_answers(
    path: str | Path, *, query: QueryOpener | None = None, readable: bool = True
) -> dict[str, Any]:
    """`ogquery.digests.release_answers`, opening the release with `query_store` unless
    `query` says otherwise."""
    return _digests.release_answers(path, query=query or _query_store, readable=readable)


def release_digests(path: str | Path, *, query: QueryOpener | None = None) -> dict[str, Any]:
    """`ogquery.digests.release_digests`, opening the release with `query_store` unless
    `query` says otherwise."""
    return _digests.release_digests(path, query=query or _query_store)

"""The Store Reader's query digests, opened the way opengwasdb opens a release (#475).

The digests live in `ogquery.digests`, which re-exports here name for name. One thing
differs. `ogquery` opens a release with the Store Reader alone, so a 0.2.x release is
refused for want of a Store Variant Table reader; opengwasdb has one. `release_answers`
and `release_digests` below default `query=` to `opengwasdb.query.query_store`, which
plugs it in (`variants.legacy_table`), and this is the only place that knows the default:
the rewrite proof (`store.rewrite`) and the tests that digest a 0.2.3 release call these.
"""

from pathlib import Path
from typing import Any

from ogquery import digests as _digests
from ogquery.digests import (
    CROSS_MACHINE_RTOL as CROSS_MACHINE_RTOL,
)
from ogquery.digests import (
    RANGE_CELLS_ASKED as RANGE_CELLS_ASKED,
)
from ogquery.digests import (
    RANGE_QUESTIONS as RANGE_QUESTIONS,
)
from ogquery.digests import (
    RANGE_ROWS_ASKED as RANGE_ROWS_ASKED,
)
from ogquery.digests import (
    RESULT_ARRAY_NAMES as RESULT_ARRAY_NAMES,
)
from ogquery.digests import (
    SECTIONS as SECTIONS,
)
from ogquery.digests import (
    VALUES as VALUES,
)
from ogquery.digests import (
    IncompleteResultError as IncompleteResultError,
)
from ogquery.digests import (
    QueryOpener as QueryOpener,
)
from ogquery.digests import (
    Question as Question,
)
from ogquery.digests import (
    adapter_name as adapter_name,
)
from ogquery.digests import (
    differences as differences,
)
from ogquery.digests import (
    digest_array as digest_array,
)
from ogquery.digests import (
    expects_refusal as expects_refusal,
)
from ogquery.digests import (
    narrowed_range_question as narrowed_range_question,
)
from ogquery.digests import (
    probed_alids as probed_alids,
)
from ogquery.digests import (
    query_shapes as query_shapes,
)
from ogquery.digests import (
    questions as questions,
)
from ogquery.digests import (
    range_questions as range_questions,
)
from ogquery.digests import (
    readable_answer as readable_answer,
)
from ogquery.digests import (
    readable_differences as readable_differences,
)
from ogquery.digests import (
    readable_value as readable_value,
)
from ogquery.digests import (
    refusal_questions as refusal_questions,
)
from ogquery.digests import (
    result_digests as result_digests,
)
from ogquery.digests import (
    unposed_range_questions as unposed_range_questions,
)
from ogquery.digests import (
    values_agree as values_agree,
)


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

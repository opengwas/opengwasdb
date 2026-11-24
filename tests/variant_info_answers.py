"""What a release answers about its variants, with variant info, for comparing two forms.

`answers` asks `variants_table()` and the resolved rows, with variant info, of every
Analysis, a lookup of every variant and the widest indexed top hits; each value is kept
as `repr` writes it (`reprs`), which is what the rewrite's proof digests. A 0.3.0 release
and its 0.2.3 form must give the same answers (#469's rsIDs, #470's `source_alid`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opengwasdb.query import query_store


def reprs(rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> list[dict[str, str]]:
    """Each row's values as `repr` writes them: what the proof digests, so an int and a
    numpy int, or None and ``"."``, differ."""
    return [{name: repr(value) for name, value in row.items()} for row in rows]


def answers(release: Path) -> dict[str, Any]:
    """`variants_table()`, and the resolved rows of every Analysis, a lookup of every
    variant, and the widest indexed top hits, with variant info."""
    with query_store(release) as q:
        table = q.variants_table()
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        alids = [table[key]["alid"] for key in sorted(table)]
        results = {f"analysis/{a}": q.analysis(a) for a in ids}
        results["lookup"] = q.lookup(alids, ids)
        # A tier every layout stores: a Dense or Hybrid release refuses any other
        # threshold (#483), which `query_digests` compares as a refusal.
        results["top_hits"] = q.top_hits(threshold=5e-4)
        resolved = {
            name: reprs(q.resolve(result, include_variant_info=True))
            for name, result in results.items()
        }
    return {"variants_table": {k: reprs([row])[0] for k, row in table.items()}, **resolved}

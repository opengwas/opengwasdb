"""What the query facade answers on six fixture Store Releases, as digests (#464).

The Store Reader (#464) moved every read the facade makes into
`opengwasdb.store_reader`. Moving code must not change an answer, so this module
builds a Dense, a Ragged and a Hybrid release, each Observed-Only and
Reference-Completed, asks each one every question the facade answers, and
digests every array each answer holds (`benchmark_store_comparison.result_digests`).

`tests/golden/store_reader_digests.json` holds the digests the code at
`5d600a1d` -- before the move -- gave. They were written by running this module
from a `git archive 5d600a1d` of the repository, never by the code it checks:

    git archive 5d600a1d | tar -x -C <scratch>
    cp tests/store_reader_digests.py <scratch>/tests/
    cd <scratch> && PYTHONPATH=tests:. python tests/store_reader_digests.py <out> 5d600a1d

and `<out>` is the committed file, byte for byte, header included.

When every record gained `beta` and `pval` (ADR 0104), the pins were regenerated
the same way from `git archive 826b36bb`, the commit that added them, with that
commit named in the header. Two runs were byte-identical. With `beta` and `pval`
dropped, every one of the 352 shapes' digests and every resolved digest equalled
the `5d600a1d` pins. The 10,913 `log10_p` values differed by at most 9.7e-16
relative. Only the two arrays were added: 704 result digests and 52 resolved ones.

It touches only names that are public on both sides of the move
(`opengwasdb.query.query_store`, the builders the fixtures already call), so the
same file runs against either.

The resolved rows' `log10_p` is the one value not digested: the move replaced
scipy's `norm.logsf` with `math.erfc`, which agrees with it to 1e-12
relative error rather than bit for bit. It is kept as numbers and compared with
that tolerance; every other resolved field is digested.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from benchmarks.benchmark_store_comparison import digest_array, result_digests
from opengwasdb.query import query_store

GOLDEN = Path(__file__).resolve().parent / "golden" / "store_reader_digests.json"

#: The thresholds asked of `top_hits`: the three a Top-Hit Index holds, and two
#: it does not, which a Ragged release answers by scanning (`z_critical`).
THRESHOLDS = (5e-8, 5e-6, 5e-4, 1e-2, 0.2)

#: How many variants the PheWAS and lookup shapes name, spread over the axis.
N_VARIANTS_ASKED = 7


class _Factory:
    """The one method `RaggedResidualScenario` asks of pytest's factory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def mktemp(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True)
        return path


def build_releases(root: Path) -> dict[str, Path]:
    """The six fixture releases, by name: each layout Observed-Only and Reference-Completed."""
    import residual_stores
    from test_hybrid_completion import _residual_hybrid_crossover_source
    from test_ragged_residual_completion import RaggedResidualScenario

    def fresh(name: str) -> Path:
        path = root / name
        path.mkdir(parents=True)
        return path

    ragged = RaggedResidualScenario(cast(pytest.TempPathFactory, _Factory(root / "ragged")))
    hybrid_observed, *_ = _residual_hybrid_crossover_source(fresh("hybrid_observed"))
    dense_completed, *_ = residual_stores.dense_completed(fresh("dense_completed"))
    return {
        "dense_observed": residual_stores.dense_observed_missing(fresh("dense_observed")),
        "dense_completed": dense_completed,
        "ragged_observed": ragged.source,
        "ragged_completed": ragged.completed,
        "hybrid_observed": hybrid_observed,
        "hybrid_completed": residual_stores.hybrid_completed(fresh("hybrid_completed")),
    }


def _spread(values: list[Any], count: int) -> list[Any]:
    """`count` items spread evenly over `values`, first and last included."""
    if len(values) <= count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(i)] for i in positions]


def _table_digests(table: dict[int, dict[str, Any]]) -> dict[str, str]:
    """A `variants_table()` or `analyses_table()` as one digest per column."""
    keys = sorted(table)
    columns = sorted({name for row in table.values() for name in row})
    out = {"__keys__": digest_array(np.asarray(keys, dtype="int64"))}
    for name in columns:
        values = np.empty(len(keys), dtype=object)
        values[:] = [repr(table[key].get(name)) for key in keys]
        out[name] = digest_array(values)
    return out


def _shapes(q: Any) -> dict[str, Callable[[], Any]]:
    """Every association-returning question the adapter answers, by a stable name."""
    variants = q.variants_table()
    analyses = q.analyses_table()
    ids = [row["analysis_id"] for _, row in sorted(analyses.items())]
    records = [variants[key] for key in sorted(variants)]
    asked = _spread(records, N_VARIANTS_ASKED)
    alids = [record["alid"] for record in asked]
    rsids = [record["rsid"] for record in asked if record["rsid"]]
    first, last = records[0], records[-1]
    shapes: dict[str, Callable[[], Any]] = {}
    for observed_only in (False, True):
        flag = "observed_only" if observed_only else "all"
        for analysis_id in ids:
            shapes[f"analysis/{analysis_id}/{flag}"] = (
                lambda a=analysis_id, o=observed_only: q.analysis(a, observed_only=o)
            )
        for identifier in [*alids, *rsids, "1:1:A:C", "rs0"]:
            shapes[f"phewas/{identifier}/{flag}"] = (
                lambda i=identifier, o=observed_only: q.phewas(i, observed_only=o)
            )
        shapes[f"range_phewas/{flag}"] = lambda o=observed_only: q.range_phewas(
            first["chromosome"], first["position"], last["position"], observed_only=o
        )
        shapes[f"lookup/{flag}"] = lambda o=observed_only: q.lookup(
            [*alids, *rsids], ids, observed_only=o
        )
        for threshold in THRESHOLDS:
            shapes[f"top_hits/{threshold}/{flag}"] = (
                lambda t=threshold, o=observed_only: q.top_hits(threshold=t, observed_only=o)
            )
            for analysis_id in ids:
                shapes[f"top_hits/{threshold}/{analysis_id}/{flag}"] = (
                    lambda t=threshold, a=analysis_id, o=observed_only: q.top_hits(
                        analysis_id=a, threshold=t, observed_only=o
                    )
                )
        shapes[f"top_hits/limit/{flag}"] = lambda o=observed_only: q.top_hits(
            threshold=0.2, limit=3, observed_only=o
        )
        if hasattr(q, "range_by_analysis"):
            shapes[f"range_by_analysis/{flag}"] = lambda o=observed_only: q.range_by_analysis(
                first["chromosome"], 1, 10**9, observed_only=o
            )
    return shapes


def _resolved(q: Any, result: Any) -> tuple[dict[str, str], list[float]]:
    """A result's resolved rows: every field digested but `log10_p`, which is returned."""
    rows = list(q.resolve(result, include_variant_info=True))
    names = sorted({name for row in rows for name in row} - {"log10_p"})
    digests = {}
    for name in names:
        values = np.empty(len(rows), dtype=object)
        values[:] = [repr(row.get(name)) for row in rows]
        digests[name] = digest_array(values)
    return digests, [float(row["log10_p"]) for row in rows]


def release_digests(path: Path) -> dict[str, Any]:
    """Every answer one release gives, digested; `log10_p` per resolved shape."""
    with query_store(path) as q:
        out: dict[str, Any] = {
            "adapter": type(q).__name__,
            "variants_table": _table_digests(q.variants_table()),
            "analyses_table": _table_digests(q.analyses_table()),
            "shapes": {},
            "resolved": {},
            "log10_p": {},
        }
        for name, shape in _shapes(q).items():
            result = shape()
            out["shapes"][name] = result_digests(dict(result))
            if name.startswith(("lookup/", "top_hits/0.2/all", "range_phewas/")):
                out["resolved"][name], out["log10_p"][name] = _resolved(q, result)
    return out


def digests(root: Path) -> dict[str, Any]:
    """Every fixture release's digests, by release name."""
    return {name: release_digests(path) for name, path in build_releases(root).items()}


def main(argv: list[str]) -> None:
    """``store_reader_digests.py OUT COMMIT``: write the digests, headed by the
    commit whose code wrote them -- exactly the file that is committed."""
    import tempfile

    out, commit = Path(argv[1]), argv[2]
    with tempfile.TemporaryDirectory() as scratch:
        found = digests(Path(scratch))
    found["__written_by__"] = {
        "commit": commit,
        "how": (
            f"tests/store_reader_digests.py run from a git archive of {commit}; "
            "see its docstring"
        ),
    }
    out.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)

"""What the query facade answers on six fixture Store Releases, as digests (#464).

The Store Reader (#464) moved every read the facade makes into
`opengwasdb.store_reader`. Moving code must not change an answer, so this module
builds a Dense, a Ragged and a Hybrid release, each Observed-Only and
Reference-Completed, asks each one every question the facade answers, and
digests every array each answer holds (`benchmark_store_comparison.result_digests`).

The questions and the digests are `opengwasdb.store.query_digests` since #467, which
moved them here unchanged, so the rewrite to 0.3.0 proves itself with the same ones.

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

When `variants_table()` and resolved rows with variant info gained `source_alid`
(#470), the pins were regenerated the same way from `git archive 30cb20cc`, #470's
merge of #483, whose pins it replaces. Two runs were byte-identical. With
`source_alid` dropped, the `variants_table()` of each of the 6 releases, every one of
the 352 shapes' digests, the `analyses_table()` and all 26 resolved shapes equalled
#483's pins, and the 10,913 `log10_p` values were equal exactly. Only the column was
added: to 6 variants tables and to the 26 resolved shapes.

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
from pathlib import Path
from typing import Any, cast

import pytest

# The questions and their digests are production code since #467: the rewrite
# (`opengwasdb.store.rewrite`) proves itself with them before it publishes.
from opengwasdb.store.query_digests import release_digests as release_digests

GOLDEN = Path(__file__).resolve().parent / "golden" / "store_reader_digests.json"


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

"""Read the Store Reader contract bundle through the Store Reader alone (#474).

The bundle (`README.md` beside this file) is a set of small format 0.3.0 Store Releases
and, for each, what the Store Reader must answer: `expected/<release>.json`. This module
opens a release with the Store Reader -- `open_release(path)` and `query_release`, never
opengwasdb's `query_store`, and no Variant Table reader: a 0.3.0 release's table refuses
every read, so a read that reached `variants.tsv.gz` fails the contract -- and compares
its answers with that file. It imports the Store Reader and the questions
(`opengwasdb.store.query_digests`) and nothing else of opengwasdb or of its tests, so the
directory moves to ogquery whole: there the imports below are the only lines that change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opengwasdb.store.query_digests import (
    CROSS_MACHINE_PVAL_RTOL,
    VALUES,
    differences,
    readable_differences,
    readable_value,
    release_answers,
)
from opengwasdb.store_reader.facade import query_release
from opengwasdb.store_reader.release import open_release

BUNDLE = Path(__file__).resolve().parent
RELEASES = BUNDLE / "releases"
EXPECTED = BUNDLE / "expected"


def open_query(path: Path) -> Any:
    """The query facade over the release at `path`, opened by the Store Reader alone."""
    return query_release(open_release(path))


def version() -> dict[str, str]:
    """`VERSION`: the bundle's version and the `format_version` its releases are."""
    lines = (BUNDLE / "VERSION").read_text(encoding="utf-8").splitlines()
    return dict(line.split(" ", 1) for line in lines if line.strip())


def release_names() -> list[str]:
    """Every release the bundle holds, by its directory name."""
    return sorted(path.name for path in RELEASES.iterdir() if path.is_dir())


def expected(name: str) -> dict[str, Any]:
    """What release `name` must answer."""
    return json.loads((EXPECTED / f"{name}.json").read_text(encoding="utf-8"))


def answers(path: Path) -> dict[str, Any]:
    """What the Store Reader answers on the release at `path`: its digests and its
    readable values, in the form `expected/<release>.json` holds them. A `pval` or a
    resolved row's `log10_p` is NaN where its SE is infinite, and JSON holds no NaN, so
    each is spelt as `readable_value` spells it; `differences` reads ``"NaN"`` back as NaN."""
    found = release_answers(path, query=open_query)
    digests = found["digests"]
    for kind in VALUES:
        digests[kind] = {
            shape: [readable_value(v) for v in values] for shape, values in digests[kind].items()
        }
    return found


def mismatches(path: Path, want: dict[str, Any]) -> list[str]:
    """Every place the release at `path` answers other than `want`; none if it agrees."""
    return disagreements(answers(path), want)


def disagreements(found: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """Every place `found` (`answers`) differs from `want`. `want` was written on another
    machine, so `pval` is compared to a few ulps (`CROSS_MACHINE_PVAL_RTOL`); every other
    value exactly."""
    out = differences(want["digests"], found["digests"], pval_rtol=CROSS_MACHINE_PVAL_RTOL)
    if found["digests"]["unposed_range_questions"] != want["digests"]["unposed_range_questions"]:
        out.append("unposed_range_questions")
    return out + readable_differences(
        want["readable"], found["readable"], pval_rtol=CROSS_MACHINE_PVAL_RTOL
    )

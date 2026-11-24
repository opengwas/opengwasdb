"""Read the Store Reader contract bundle through the Store Reader alone (#474).

The bundle (`README.md` beside this file) is a set of small format 0.3.0 Store Releases
and, for each, what the Store Reader must answer: `expected/<release>.json`. This module
opens a release with the Store Reader -- `open_release` and `query_release`, never
opengwasdb's `query_store` -- and compares its answers with that file. It imports the
Store Reader and the questions (`opengwasdb.store.query_digests`) and nothing else of
opengwasdb or of its tests, so the directory moves to ogquery whole: there the two
imports below are the only lines that change.

A 0.3.0 release is read without its Variant Table (ADR 0105 §7, §8), so the release is
opened with a Variant Table reader that fails any read made of it (`NoVariantTable`):
a 0.3.0 read that reached `variants.tsv.gz` fails the contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opengwasdb.store.query_digests import (
    differences,
    readable_value,
    release_digests,
    release_readable,
)
from opengwasdb.store_reader.facade import query_release
from opengwasdb.store_reader.release import open_release

BUNDLE = Path(__file__).resolve().parent
RELEASES = BUNDLE / "releases"
EXPECTED = BUNDLE / "expected"


class VariantTableRead(AssertionError):
    """A read of a 0.3.0 release reached its Variant Table, which no 0.3.0 read may."""


class NoVariantTable:
    """A `LegacyTableOpener` whose table refuses every read (module docstring)."""

    def __call__(self, store_path: Path, *, aliases: bool) -> NoVariantTable:
        self.store_path = store_path
        return self

    def close(self) -> None:
        """Nothing was opened to close."""

    def __getattr__(self, name: str) -> Any:
        raise VariantTableRead(
            f"a 0.3.0 read of {self.__dict__.get('store_path', '?')} asked its Variant Table "
            f"for {name!r}; from 0.3.0 the sidecars answer every read"
        )


def open_query(path: Path) -> Any:
    """The query facade over the release at `path`, opened by the Store Reader alone."""
    return query_release(open_release(path, legacy=NoVariantTable()))


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
    readable values, in the form `expected/<release>.json` holds them. A resolved row's
    `log10_p` is NaN where its SE is infinite, and JSON holds no NaN, so it is spelt
    as `readable_value` spells it; `differences` reads ``"NaN"`` back as NaN."""
    digests = release_digests(path, query=open_query)
    digests["log10_p"] = {
        shape: [readable_value(v) for v in values] for shape, values in digests["log10_p"].items()
    }
    return {"digests": digests, "readable": release_readable(path, query=open_query)}


def mismatches(path: Path, want: dict[str, Any]) -> list[str]:
    """Every place the release at `path` answers other than `want`; none if it agrees."""
    found = answers(path)
    out = differences(want["digests"], found["digests"])
    if found["digests"]["unposed_range_questions"] != want["digests"]["unposed_range_questions"]:
        out.append("unposed_range_questions")
    out += [
        f"readable/{section}/{name}"
        for section in sorted(set(want["readable"]) | set(found["readable"]))
        for name in _differing(want["readable"].get(section), found["readable"].get(section))
    ]
    return out


def _differing(ours: Any, theirs: Any) -> list[str]:
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return [k for k in sorted(set(ours) | set(theirs)) if ours.get(k) != theirs.get(k)]
    return [] if ours == theirs else ["(whole section)"]

"""Rewrite a 0.2.3 Store Release into a new 0.3.0 one (`ogdb rewrite-release`, #467).

Format 0.3.0 changes how a release stores what it answers, never what it answers
(epic #450, ADR 0105). An existing 0.2.3 release reaches it by this command, not by a
rebuild: it derives a **new** Staged Release (ADR 0038) and leaves its source as it
was. The rewrite:

1. refuses a destination that exists, is the source or lies inside it, and a source
   that is not 0.2.3 -- including one already 0.3.0 -- before a byte is written;
2. stages a copy of the source (`cp -a --reflink=auto`, so an unchanged array costs
   no space where the filesystem can share it);
3. runs its **steps** on the staged copy, in order. Each is one named change the
   format makes; #468-#472 each add theirs to `STEPS`;
4. stamps every manifest 0.3.0 with a fresh `release_id` and `created_at`, keeps
   `store_id`, and records a `format_rewrite` provenance block naming the source
   `release_id` and the steps (ADR 0038 §5, as the 0.2.0 converter does);
5. asks the source and the staged release the same deterministic set of questions
   (`opengwasdb.store.query_digests`) and refuses on any difference;
6. validates the staged release, and publishes it by rename only if it is clean.

Any failure discards the staging directory; nothing is published and the source is
untouched. The staging, the destination refusals and the validation gate are the
0.2.0 converter's (`opengwasdb.store.convert`).

Only a 0.2.3 source is rewritten. 0.3.0 keeps every feature 0.2.3 added on the 0.2
series: an indel's ALID spelt reference first (0.2.2, ADR 0079) and the count of
associations left out for an SE above 65,504 (0.2.3, ADR 0068). An earlier release
lacks one or both and no mechanical step can supply them, so it is rebuilt.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengwasdb.overview import write_overview_html
from opengwasdb.store.convert import (
    copy_path,
    installed_commit,
    manifest_paths,
    refuse_destination,
    require_valid_staged_release,
)
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.store.query_digests import differences, release_digests
from opengwasdb.store_reader.alid_sidecar import AlidSidecar, write_alid_sidecar
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.format import (
    CURRENT_FORMAT_VERSION,
    LAST_0_2_FORMAT_VERSION,
    UnsupportedFormatVersion,
    carries_alid_sidecar,
    check_format_version,
    format_version_for,
)
from opengwasdb.store_reader.variant_axis import (
    variant_alid_bytes_path,
    variant_alid_rows_path,
    variant_offsets_path,
    variant_table_path,
)
from opengwasdb.variants.legacy_table import iter_variant_records

#: The one `format_version` this rewrite reads: every other is refused (module docstring).
SOURCE_FORMAT_VERSION = LAST_0_2_FORMAT_VERSION

#: The `format_version` it writes: 0.2.3's features with every step's change.
TARGET_FORMAT_VERSION = CURRENT_FORMAT_VERSION

#: How many differences a refused proof prints; it counts them all.
_SHOWN_DIFFERENCES = 20


class RewriteError(Exception):
    """A refusal. An `Exception`, so `OpenGWASDBStore.staging` discards the staging directory."""


class RewriteVerificationError(RewriteError):
    """The staged release answers some question differently from its source.

    `differences` names every one (`query_digests.differences`); the message shows the first.
    """

    def __init__(self, message: str, differences: list[str]) -> None:
        super().__init__(message)
        self.differences = differences


@dataclass(frozen=True)
class RewriteContext:
    """What a step works on: the source, read only, and the staged copy it changes.

    `components` are the release's directories that are Store Releases of their own,
    relative to its root: ``"."``, and ``"dense"`` for a Hybrid's nested Dense
    Component (spec §16), whose sidecars a step rewrites as it does the root's.
    """

    source: Path
    staged: Path
    components: tuple[str, ...]

    def staged_components(self) -> Iterator[Path]:
        for component in self.components:
            yield self.staged / component


@dataclass(frozen=True)
class RewriteStep:
    """One named change the rewrite makes to the staged copy."""

    name: str
    summary: str
    run: Callable[[RewriteContext], None]


def _write_alid_sidecar(context: RewriteContext) -> None:
    """Write each component's ALID sidecar from its Variant Table; drop the `|S32` pair.

    The ALIDs come from the table, not from the fixed-width index, which leaves out
    every ALID over 32 bytes (#127). The table must hold rows 0..n-1 in order, one per
    entry of `variant_offsets.npy`; any other table is refused, never indexed. The
    sidecar is written by its one writer, the codec's `write_alid_sidecar` (a build
    reaches it through `write_variant_axis`, which writes the whole axis; this step
    rewrites only the ALID index). Before the step ends, every ALID is searched for
    and must name its own row: the proof's questions reach only some of them.
    """
    for directory in context.staged_components():
        alids = list(_alids_in_row_order(directory))
        write_alid_sidecar(directory, alids)
        _require_every_alid_resolves(directory, alids)
        variant_alid_bytes_path(directory).unlink()
        variant_alid_rows_path(directory).unlink()


def _require_every_alid_resolves(directory: Path, alids: list[str]) -> None:
    """Refuse a sidecar in which some ALID does not find its own row."""
    import numpy as np

    sidecar = AlidSidecar.open(directory)
    if sidecar is None:
        raise RewriteError(f"{directory}: the ALID sidecar was not written")
    rows = sidecar.search(alids)
    wrong = np.flatnonzero(rows != np.arange(len(alids)))
    if wrong.size:
        first = int(wrong[0])
        raise RewriteError(
            f"{directory}: {wrong.size} ALID(s) of {len(alids)} do not resolve to their own "
            f"row in the written sidecar; first, {alids[first]!r} at row {first} resolves to "
            f"{int(rows[first])}"
        )


def _alids_in_row_order(directory: Path) -> Iterator[str]:
    import numpy as np

    n_rows = len(np.load(variant_offsets_path(directory), mmap_mode="r"))
    row = -1
    for row, record in enumerate(iter_variant_records(variant_table_path(directory))):
        if record.variant_index != row:
            raise RewriteError(
                f"{variant_table_path(directory)}: row {row} has variant_index "
                f"{record.variant_index}; a table out of order is not rewritten"
            )
        yield record.alid
    if row + 1 != n_rows:
        raise RewriteError(
            f"{variant_table_path(directory)} has {row + 1} rows but variant_offsets.npy "
            f"has {n_rows}; a table that disagrees with its offsets is not rewritten"
        )


#: The rewrite's steps, in the order they run. #468-#472 each add one.
STEPS: tuple[RewriteStep, ...] = (
    RewriteStep(
        "alid_sidecar",
        "write the variable-length ALID sidecar from the Variant Table and drop the "
        "fixed-width variant_alid_bytes.npy/variant_alid_rows.npy (ADR 0105)",
        _write_alid_sidecar,
    ),
)


def rewrite_release(
    source: str | Path,
    destination: str | Path,
    *,
    steps: tuple[RewriteStep, ...] = STEPS,
) -> Path:
    """Derive a 0.3.0 release at `destination` from the 0.2.3 one at `source`."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    refuse_destination(source, destination, error=RewriteError, named="DESTINATION")
    components = _refuse_unrewritable(source)
    started = time.perf_counter()
    reference = release_digests(source)
    _log_phase(f"asked the source {len(reference['shapes'])} questions", started)
    with OpenGWASDBStore.staging(destination) as staged:
        context = RewriteContext(source=source, staged=staged.path, components=components)
        started = time.perf_counter()
        _copy_release(source, staged.path)
        _log_phase("copied the source release", started)
        for step in steps:
            started = time.perf_counter()
            step.run(context)
            _log_phase(f"step {step.name}: {step.summary}", started)
        _stamp(context, steps, release_id=str(uuid.uuid4()), now=datetime.now(UTC).isoformat())
        _refresh_overviews(context)
        started = time.perf_counter()
        _require_same_answers(reference, release_digests(staged.path), destination)
        _log_phase("asked the staged release the same questions: every answer matches", started)
        started = time.perf_counter()
        require_valid_staged_release(
            staged.path, destination, error=RewriteError, what="rewritten"
        )
        _log_phase("validated the staged release", started)
    print(f"Published {destination} as {TARGET_FORMAT_VERSION}", flush=True)
    return destination


def _refuse_unrewritable(source: Path) -> tuple[str, ...]:
    """The release's components, after refusing a source this rewrite does not take.

    Read from the manifests alone, before anything is staged. A Hybrid is two Store
    Releases, and both must be 0.2.3: a half-rewritten one is not a source.
    """
    if not (source / "manifest.json").is_file():
        raise RewriteError(f"{source}: no manifest.json; this is not a Store Release")
    top = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if top.get("primary_layout") == "hybrid" and not (source / "dense" / "manifest.json").exists():
        raise RewriteError(
            f"{source}: primary_layout is 'hybrid' but there is no nested Dense Component "
            "manifest at dense/manifest.json (spec §16)"
        )
    for path in manifest_paths(source):
        _refuse_version(path, str(json.loads(path.read_text(encoding="utf-8"))["format_version"]))
    return (".", "dense") if (source / "dense" / "manifest.json").exists() else (".",)


def _refuse_version(manifest: Path, version: str) -> None:
    try:
        check_format_version(version, source=str(manifest))
    except UnsupportedFormatVersion as exc:
        raise RewriteError(str(exc)) from exc
    if carries_alid_sidecar(version):
        raise RewriteError(
            f"{manifest}: format_version is already {version}; there is nothing to rewrite"
        )
    if version != SOURCE_FORMAT_VERSION:
        raise RewriteError(
            f"{manifest}: format_version is {version}, and only a {SOURCE_FORMAT_VERSION} "
            f"release is rewritten to {TARGET_FORMAT_VERSION}. {TARGET_FORMAT_VERSION} keeps "
            "0.2.3's features: an indel's ALID spelt reference first (0.2.2, ADR 0079) and "
            "the count of associations left out for an SE above 65,504 (0.2.3, ADR 0068). "
            f"A {version} release lacks at least one, and no rewrite can add it: rebuild it "
            "from source (spec §21.4)."
        )


def _copy_release(source: Path, staged: Path) -> None:
    """Copy every top-level entry; a reflink where the filesystem can share the blocks."""
    staged.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        copy_path(entry, staged / entry.name)


def _stamp(
    context: RewriteContext, steps: tuple[RewriteStep, ...], *, release_id: str, now: str
) -> None:
    """Stamp every manifest as one new 0.3.0 release, recording what the rewrite did.

    Every manifest gets the same fresh `release_id`, so a Hybrid's nested Dense
    Component still names the release that nests it, as the 0.2.0 converter does.
    """
    commit = installed_commit()
    for path in manifest_paths(context.staged):
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        source_release_id = str(data["release_id"])
        data["release_id"] = release_id
        data["created_at"] = now
        data["format_version"] = format_version_for(
            carries_regions="regions" in data, reference_first_indels=True
        )
        data["provenance"] = {
            **data.get("provenance", {}),
            "format_rewrite": {
                "source_release_id": source_release_id,
                "from_format_version": SOURCE_FORMAT_VERSION,
                "to_format_version": data["format_version"],
                "steps": [step.name for step in steps],
                "tool": "ogdb rewrite-release",
                "at": now,
                "opengwasdb_git_hash": commit,
                "note": (
                    "Derived by ogdb rewrite-release: each step rewrote how the source "
                    "stores what it answers, and the source and this release answered "
                    "the same questions identically before it was published. The "
                    "source release was not modified."
                ),
            },
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_overviews(context: RewriteContext) -> None:
    """Regenerate each `overview.html` the copy carried: its header names the release."""
    for directory in context.staged_components():
        if (directory / "overview.html").exists():
            write_overview_html(directory, read_analyses(directory / "analyses.tsv"))


def _require_same_answers(
    reference: dict[str, Any], candidate: dict[str, Any], destination: Path
) -> None:
    found = differences(reference, candidate)
    if found:
        shown = "\n  ".join(found[:_SHOWN_DIFFERENCES])
        raise RewriteVerificationError(
            f"the rewritten release answers {len(found)} question(s) differently from its "
            f"source, so it is not published to {destination}; the staging directory was "
            f"removed and the source is unchanged. First differences:\n  {shown}",
            found,
        )


def _log_phase(label: str, started: float) -> None:
    print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)


__all__ = [
    "STEPS",
    "RewriteContext",
    "RewriteError",
    "RewriteStep",
    "RewriteVerificationError",
    "rewrite_release",
]

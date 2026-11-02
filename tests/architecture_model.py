"""The engine's architecture model and the checks over it (issue #358).

`docs/architecture/components.yaml` says which component owns which code; this
module loads it and checks it against the files git tracks. The functions here
take the model and the paths as arguments, so a test can hand them a small
fixture: an architecture model is a data file, and checking it must not need a
repository to exist.

The split is deliberate. `claims()` and `component_for_path()` are the ownership
lookup, and opengwas/opengwasdb#359 reuses them to place imports on components;
`check_model()` is the CI gate, and returns what is wrong rather than raising, so
one run reports every problem instead of the first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MODEL_PATH = "docs/architecture/components.yaml"

#: Where the engine's source lives, and what counts as source. Only `.py` files
#: under here are claimed: the map is about code, and a data file inside a
#: claimed directory is already covered by the directory claim.
SOURCE_ROOT = "opengwasdb/"
SOURCE_SUFFIX = ".py"


@dataclass(frozen=True)
class Claim:
    """One component's claim on one path: a directory (trailing `/`) or a file."""

    prefix: str
    component: str

    @property
    def is_directory(self) -> bool:
        return self.prefix.endswith("/")

    def covers(self, path: str) -> bool:
        """Whether this claim owns `path`. A directory claim owns what is under it."""
        if self.is_directory:
            return path.startswith(self.prefix)
        return path == self.prefix


def claims(model: Mapping[str, Any]) -> tuple[Claim, ...]:
    """Every claim in the model, longest prefix first, so the first match wins."""
    found = [
        Claim(prefix=str(prefix), component=str(component["id"]))
        for component in model["components"]
        for prefix in component["paths"]
    ]
    return tuple(sorted(found, key=lambda claim: (-len(claim.prefix), claim.prefix)))


def component_for_path(claim_list: Sequence[Claim], path: str) -> str | None:
    """The component that owns `path`, or `None` when nothing claims it.

    The longest claimed prefix wins, which is what makes a file claim inside
    another component's directory claim belong to the file's claimer: the
    directory is shared, and the file is not.
    """
    for claim in claim_list:
        if claim.covers(path):
            return claim.component
    return None


def is_source_file(path: str) -> bool:
    return path.startswith(SOURCE_ROOT) and path.endswith(SOURCE_SUFFIX)


def check_model(model: Mapping[str, Any], tracked_paths: Iterable[str]) -> list[str]:
    """Everything wrong with the model, as messages naming the file and the fix."""
    claim_list = claims(model)
    tracked = list(tracked_paths)
    return (
        _shared_prefixes(claim_list)
        + _undeclared_layers(model)
        + _stale_claims(claim_list, tracked)
        + _unclaimed_source_files(claim_list, tracked)
    )


def _shared_prefixes(claim_list: Sequence[Claim]) -> list[str]:
    """A prefix two components claim would make ownership a guess, so it is an error."""
    owners: dict[str, list[str]] = {}
    for claim in claim_list:
        owners.setdefault(claim.prefix, []).append(claim.component)
    return [
        f"{prefix}: claimed by {' and '.join(repr(owner) for owner in sorted(components))};"
        f" give each component its own paths in {MODEL_PATH}"
        for prefix, components in sorted(owners.items())
        if len(components) > 1
    ]


def _undeclared_layers(model: Mapping[str, Any]) -> list[str]:
    declared = {entry["layer"] for entry in model["layers"]}
    return [
        f"component {component['id']!r} declares layer {component['layer']}, which the model's"
        f" layers do not declare; fix one of them in {MODEL_PATH}"
        for component in model["components"]
        if component["layer"] not in declared
    ]


def _stale_claims(claim_list: Sequence[Claim], tracked: Sequence[str]) -> list[str]:
    return [
        f"{claim.prefix}: claimed by component {claim.component!r}, but no tracked file matches;"
        f" fix or remove the claim in {MODEL_PATH}"
        for claim in sorted(claim_list, key=lambda claim: claim.prefix)
        if not any(claim.covers(path) for path in tracked)
    ]


def _unclaimed_source_files(claim_list: Sequence[Claim], tracked: Sequence[str]) -> list[str]:
    return [
        f"{path}: no component claims this file; add a path for it to a component in {MODEL_PATH}"
        for path in sorted(tracked)
        if is_source_file(path) and component_for_path(claim_list, path) is None
    ]

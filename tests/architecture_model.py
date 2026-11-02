"""The engine's architecture model and the checks over it (issues #358, #359).

`docs/architecture/components.yaml` says which component owns which code; this
module loads it and checks it against the files git tracks. The functions here
take the model and the paths as arguments, so a test can hand them a small
fixture: an architecture model is a data file, and checking it must not need a
repository to exist.

The split is deliberate. `claims()` and `component_for_path()` are the ownership
lookup, and opengwas/opengwasdb#359 reuses them to place imports on components;
`check_model()` is the CI gate, and returns what is wrong rather than raising, so
one run reports every problem instead of the first.

#359 adds the other half: `import_graph()` turns module sources into the imports
that cross a component boundary, and `check_layering()` holds those that point up
a layer to `docs/architecture/layering-baseline.yaml`. That baseline only
shrinks, so the check reports both an upward import it does not list -- new debt --
and a listed import that has gone -- debt already paid.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_PATH = "docs/architecture/components.yaml"
BASELINE_PATH = "docs/architecture/layering-baseline.yaml"

#: What every component must declare. A missing one is not a defaultable field:
#: a component without `paths` claims nothing and would silently own no code.
REQUIRED_COMPONENT_KEYS = ("id", "name", "container", "layer", "stages", "summary", "paths")

#: Where the engine's source lives, and what counts as source. Only `.py` files
#: under here are claimed: the map is about code, and a data file inside a
#: claimed directory is already covered by the directory claim.
SOURCE_ROOT = "opengwasdb/"
SOURCE_SUFFIX = ".py"

#: The file that names a package rather than a module in it.
SUBPACKAGE_MODULE = ".__init__"


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


def parse_tracked_paths(raw: bytes) -> list[str]:
    """The paths in `git ls-files -z` output: NUL separated, never split on whitespace.

    A path may contain a space, and splitting on whitespace would cut
    `opengwasdb/build/new helper.py` into two paths that are not engine source, so
    the check would pass over a file nobody claims. `-z` also means git does not
    quote or escape an unusual name, so what comes back is the path itself.
    """
    return [path for path in raw.decode("utf-8").split("\0") if path]


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


@dataclass(frozen=True)
class ImportEdge:
    """One import from one engine component into another, as the source wrote it.

    `name` is the dotted target the import names -- `opengwasdb.encoding` for
    `from opengwasdb.encoding import StoreEncoding`, and the module plus the name
    for `from opengwasdb.encoding.codec import read`. Keeping the name as written
    is what makes the baseline a list of imports rather than a count: a second
    name from an already-listed module is a new entry, not a hidden one.
    """

    file: str
    name: str
    source: str
    destination: str


def module_name(path: str) -> str:
    """The dotted module a source file defines: `a/b/__init__.py` is `a.b`."""
    dotted = path[: -len(SOURCE_SUFFIX)].replace("/", ".")
    if dotted.endswith(SUBPACKAGE_MODULE):
        return dotted[: -len(SUBPACKAGE_MODULE)]
    return dotted


def package_of(path: str) -> str:
    """The package a module's relative imports are resolved against."""
    module = module_name(path)
    return module if path.endswith(SUBPACKAGE_MODULE) else module.rpartition(".")[0]


def imports_in_source(path: str, source: str) -> set[str]:
    """Every dotted name `source` imports, de-duplicated within the file.

    Every `import` and `from ... import` node counts, wherever it sits -- inside a
    function, under `if TYPE_CHECKING:`, behind a guard. They all couple the file
    to the imported module, which is what an edge in this graph means (decision 8
    on opengwas/opengwasdb-stores#416).
    """
    targets: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(path, node)
            targets.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return targets


def _import_base(path: str, node: ast.ImportFrom) -> str:
    """The module a `from ... import` names, with its relative dots resolved."""
    module = node.module or ""
    if not node.level:
        return module
    parts = package_of(path).split(".")
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, module] if module else parts)


def component_for_module(owners: Mapping[str, str], target: str) -> str | None:
    """The component that owns `target`, by the longest module of it that exists.

    `opengwasdb.encoding.StoreEncoding` is a name inside `opengwasdb.encoding`, so
    the module is the target minus its last part; `opengwasdb.layouts.dense`
    imported as a module is itself. The walk is what tells the two apart without
    guessing which trailing parts are modules.
    """
    parts = target.split(".")
    while parts:
        owner = owners.get(".".join(parts))
        if owner is not None:
            return owner
        parts.pop()
    return None


def import_graph(model: Mapping[str, Any], sources: Mapping[str, str]) -> tuple[ImportEdge, ...]:
    """Every import that crosses a component boundary, one entry per (file, name).

    Takes the sources as `path -> text` rather than reading the tree, so a test can
    hand it four lines of fixture. A file no component claims is skipped: `#358`'s
    check reports it as unclaimed, and there is no component to draw an edge from.
    """
    claim_list = claims(model)
    owners = {
        module_name(path): component
        for path in sources
        if (component := component_for_path(claim_list, path)) is not None
    }
    edges = [
        ImportEdge(file=path, name=name, source=source, destination=destination)
        for path in sorted(sources)
        if (source := owners.get(module_name(path))) is not None
        for name in sorted(imports_in_source(path, sources[path]))
        if (destination := component_for_module(owners, name)) not in (None, source)
    ]
    return tuple(edges)


def layers_of(model: Mapping[str, Any]) -> dict[str, int]:
    """Each component's layer, by id, as the model declares it."""
    return {str(component["id"]): int(component["layer"]) for component in model["components"]}


def upward_imports(
    model: Mapping[str, Any], graph: Iterable[ImportEdge]
) -> tuple[ImportEdge, ...]:
    """The imports that point at a component on a higher layer than the importer's own.

    Same layer is allowed, and so is downward: the declared order says a component
    may build on its own layer or below, and only an import against that order is
    debt. The graph arrives sorted by (file, name), so this is too.
    """
    layers = layers_of(model)
    return tuple(edge for edge in graph if layers[edge.destination] > layers[edge.source])


def check_layering(
    model: Mapping[str, Any],
    graph: Iterable[ImportEdge],
    baseline: Iterable[tuple[str, str]],
) -> list[str]:
    """What is wrong with the layering, as messages naming the file and the fix.

    Two failures, and they are opposite halves of one ratchet. An upward import the
    baseline does not list is new debt, and the fix is to move the import or the
    claim rather than to list it. A listed import that is no longer upward -- moved,
    deleted, or made legitimate by a claim move -- is debt that has been paid, and
    the entry has to go, or the list stops describing the code.
    """
    layers = layers_of(model)
    upward = upward_imports(model, graph)
    listed = {(edge.file, edge.name) for edge in upward}
    allowed = set(baseline)
    return [
        f"{edge.file}: imports {edge.name} from component {edge.source!r}"
        f" (layer {layers[edge.source]}) into {edge.destination!r}"
        f" (layer {layers[edge.destination]}), which is higher; move the import or the claim,"
        f" because {BASELINE_PATH} only shrinks"
        for edge in upward
        if (edge.file, edge.name) not in allowed
    ] + [
        f"{file}: imports {name}, listed in {BASELINE_PATH}, is no longer an upward import;"
        f" delete the entry"
        for file, name in sorted(allowed - listed)
    ]


class BaselineError(Exception):
    """A baseline file that cannot be read as one. Raised, not returned.

    A baseline nobody can read is not a list of problems to report alongside the
    rest: it is the list the check compares against, and an unreadable one would
    make every import look new, or -- worse -- every import look allowed.
    """


def load_baseline(path: Path) -> tuple[tuple[str, str], ...]:
    """The baseline's (importing file, imported name) pairs, sorted.

    Sorted so a test can compare it to a literal and a person can diff two
    versions of it. Duplicates are an error rather than a quiet `set`: an entry
    added twice is one a reviewer would believe was checked once.
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("allowed"), list):
        raise BaselineError(f"{path}: expected a mapping with an 'allowed' list")

    pairs: list[tuple[str, str]] = []
    for index, entry in enumerate(document["allowed"]):
        if not isinstance(entry, dict) or set(entry) != {"file", "name"}:
            raise BaselineError(
                f"{path}: entry #{index} must be a mapping with only 'file' and 'name'"
            )
        file, name = entry["file"], entry["name"]
        if not isinstance(file, str) or not isinstance(name, str):
            raise BaselineError(f"{path}: entry #{index} must have a string 'file' and 'name'")
        if not is_source_file(file):
            raise BaselineError(
                f"{path}: entry #{index} names {file!r}, which is not an engine source file;"
                f" a baseline entry is a path under {SOURCE_ROOT} as `git ls-files` prints it"
            )
        if (file, name) in pairs:
            raise BaselineError(f"{path}: {file} imports {name} is listed twice")
        pairs.append((file, name))
    return tuple(sorted(pairs))


def engine_sources(root: Path) -> dict[str, str]:
    """The engine's tracked source under `root`, as `path -> text`.

    `git ls-files -z` rather than a walk of the tree: the graph is about the code
    the repository holds, so a file git does not track is not part of it, and a
    path with a space stays one path. A file no component claims is included --
    `import_graph()` skips it, and `check_model()` reports it.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", SOURCE_ROOT],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        path: (root / path).read_text(encoding="utf-8")
        for path in parse_tracked_paths(listed.stdout)
        if is_source_file(path)
    }


class ModelError(Exception):
    """A model file that cannot be read as a model. Raised, not returned.

    A structural fault is not a list of problems to report alongside the rest: a
    component with no `paths` means every check downstream is meaningless, and a
    model nobody can load should stop the run where the fault is.
    """


def load_model(path: Path) -> dict[str, Any]:
    """Read a model file and hold it to the shape #418 and #359 rely on.

    Names the component in the message: the person fixing it is looking at a YAML
    file of fifteen of them, and "a component is missing a key" would send them
    hunting.
    """
    model = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ModelError(f"{path}: expected a mapping, found {type(model).__name__}")
    for key in ("layers", "components"):
        if not isinstance(model.get(key), list):
            raise ModelError(f"{path}: expected a {key!r} list")

    seen: set[str] = set()
    for index, component in enumerate(model["components"]):
        label = _label(component, index)
        if not isinstance(component, dict):
            raise ModelError(f"component {label} is not a mapping in {MODEL_PATH}")
        for key in REQUIRED_COMPONENT_KEYS:
            if key not in component:
                raise ModelError(f"component {label} has no {key!r} key in {MODEL_PATH}")
        if component["id"] in seen:
            raise ModelError(
                f"component {component['id']!r} is declared twice; ids must be unique in"
                f" {MODEL_PATH}"
            )
        seen.add(component["id"])
    return model


def _label(component: object, index: int) -> str:
    """How to name a component in a message, preferring what the reader can search for."""
    if isinstance(component, dict):
        if "id" in component:
            return repr(component["id"])
        if "name" in component:
            return f"#{index} ({component['name']!r})"
    return f"#{index}"

"""The spec's validation rules, by name, on write and on demand (spec §12; ogld#4).

Each rule yields where and how one opened hierarchy and its provenance break it. `validate`
runs all 27, in the spec's order, on a store at a path or URL; `StoreWriter.finish` runs them
on the store it wrote, before the provenance that makes it readable goes in. A rule never
trusts what it does not check: when what it needs is missing or malformed it reports that it
cannot be checked, and when an array will not decode it reports that it cannot be read, each
as a finding, rather than pass or stop.

Every rule reads in bounded windows, through Zarr directly: a chromosome's positions and
`indptr` arrays whole, as a reader keeps them (spec §10); a Near Band `_WINDOW` codes, a Proxy
Index and the Far Pairs an eighth as many entries, and the variant columns `_ROWS` rows at a
time. Eleven rules are generators over the whole store. The twelve that read the band, the
index and the Far Pairs element by element (`_SCANNED`) share one pass over each chromosome
instead: each block of rows is read once, on the threads the caller gives, and handed to all
twelve.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, TypeGuard

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format.encoding import FAR_BP, NEAR_BP, T_FAR_Q, REncoding, t_far
from ogld.format.geometry import (
    blocks,
    near_address,
    near_pair_of,
    near_row_lengths,
    owners,
    windows,
    within,
)
from ogld.format.provenance import READABLE_FORMATS, VALIDATION_RULES
from ogld.format.semantics import InputError
from ogld.identity import IdentityError, alid_locus, chromosome_rank, orient
from ogld.store import layout
from ogld.store.layout import chromosomes_listed, never_data, shown_fill
from ogld.store.location import open_group

#: Elements of a Near Band or a Proxy Index a rule reads at once: 8 Mi codes are 16 MiB.
_WINDOW: Final = 1 << 23
#: A block of rows holds at most `_WINDOW >> _ENTRY_SHIFT` Proxy Index entries, 1 Mi, and as many
#: Far Pairs, which in a valid store are never more than its rows' index entries. The rules
#: work on an entry through about 100 bytes of int64 arrays and on a band code through about 6,
#: so a block's working arrays stay near 150 MiB however dense the index is (spec §12).
_ENTRY_SHIFT: Final = 3
#: Rows of the variant columns a rule reads at once.
_ROWS: Final = 1 << 20
#: Offending elements a finding names; it counts the rest.
_NAMED: Final = 5


@dataclass(frozen=True)
class Finding:
    """One way a store breaks one rule: the rule's name, where, and what is wrong."""

    rule: str
    where: str
    message: str


class ValidationFailed(Exception):
    """A store that breaks a rule. `findings` holds every failing rule's findings."""

    def __init__(self, findings: dict[str, list[Finding]]) -> None:
        failing = [found[0] for found in findings.values() if found]
        named = "; ".join(f"{f.rule} at {f.where}: {f.message}" for f in failing)
        super().__init__(f"the LD Store breaks {len(failing)} rule(s) of spec §12: {named}")
        self.findings = findings


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Every rule of `ogld.format.VALIDATION_RULES` against the store at `location`.

    The result has one key per rule, in the spec's order, whose list is empty when the
    store passes it. Each rule reads in bounded windows and never loads a whole Near
    Band (spec §12; ogld#4). A location with no Zarr group is an `InputError`.
    """
    return validate_at(location, threads=1)


def validate_at(location: str | os.PathLike[str], *, threads: int) -> dict[str, list[Finding]]:
    """`validate`, reading and checking blocks on `threads` threads (`ogld validate --threads`)."""
    root = open_group(location)
    return validate_group(root, root.attrs.get("ld_store"), threads=threads)


def validate_group(
    root: zarr.Group,
    provenance: object,
    rules: Sequence[str] = VALIDATION_RULES,
    *,
    threads: int = 1,
) -> dict[str, list[Finding]]:
    """The `rules` against the hierarchy `root`, judged with `provenance` as its `ld_store`.
    With `threads` above 1, the pass over the Near Band and the Proxy Index reads and checks
    that many blocks at once, and holds no more (spec §12)."""
    if threads < 1:
        raise ValueError(f"validation needs at least one thread, not {threads}")
    store = _Store(root, provenance, threads)
    found = {rule: _run(rule, store) for rule in rules if rule in _RULES}
    found.update(_scan(store, [rule for rule in rules if rule in _SCANNED]))
    return {rule: found[rule] for rule in rules}


class _Unchecked(Exception):
    """What a rule needs is missing or malformed, so it cannot say the store passes it."""

    def __init__(self, where: str, why: str) -> None:
        super().__init__(why)
        self.where = where
        self.why = why


class _Unreadable(_Unchecked):
    """An array that will not read or decode as its metadata declares."""


_Problems = Iterator[tuple[str, str]]
_RULES: dict[str, Callable[[_Store], _Problems]] = {}


def _rule(name: str) -> Callable[[Callable[[_Store], _Problems]], Callable[[_Store], _Problems]]:
    def register(check: Callable[[_Store], _Problems]) -> Callable[[_Store], _Problems]:
        _RULES[name] = check
        return check

    return register


def _run(rule: str, store: _Store) -> list[Finding]:
    """The rule's findings; one it cannot finish says why, so it never passes unchecked."""
    found: list[Finding] = []
    try:
        for where, message in _RULES[rule](store):
            found.append(Finding(rule, where, message))
    except _Unchecked as unchecked:
        found.append(_finding(rule, unchecked))
    return found


def _finding(rule: str, stopped: _Unchecked) -> Finding:
    """The finding of a rule that could not finish: an array it cannot read, or a check it
    cannot make."""
    cannot = "cannot be read" if isinstance(stopped, _Unreadable) else "cannot be checked"
    return Finding(rule, stopped.where, f"{cannot}: {stopped.why}")


#: What reading a corrupt array raises: a codec's RuntimeError (zstd's "Data corruption
#: detected"), or a ValueError, TypeError or OSError from zarr or numpy. Only the read is
#: guarded, so a fault in a rule itself still propagates.
_READ_ERRORS: Final = (RuntimeError, ValueError, TypeError, IndexError, KeyError, OSError)

_MISSING: Final = object()


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _lookup(provenance: object, key: str) -> Any:
    """The value at dotted `key`, or `_MISSING`."""
    node: Any = provenance
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


class _Store:
    """The hierarchy under validation and its provenance; what rules share is read once."""

    def __init__(self, root: zarr.Group, provenance: object, threads: int = 1) -> None:
        self.root, self.threads = root, threads
        # A round trip through JSON, so a check on write judges exactly what is stored.
        self.provenance = json.loads(json.dumps(provenance)) if provenance is not None else None
        self._kept: dict[str, npt.NDArray[np.int64]] = {}
        self._arrays: dict[str, zarr.Array[Any] | None] = {}

    def value(self, key: str) -> Any:
        """The provenance value at dotted `key`, or `_Unchecked`."""
        if not isinstance(self.provenance, dict):
            raise _Unchecked("ld_store", "the root group has no ld_store provenance (spec §7)")
        found = _lookup(self.provenance, key)
        if found is _MISSING:
            raise _Unchecked(key, f"provenance has no {key} (provenance.complete)")
        return found

    def encoding(self) -> REncoding:
        """The encoding provenance records, or `_Unchecked` when this package does not implement
        it: a rule that decodes codes or recomputes the floor cannot judge them at a scale it
        does not know (provenance.encoding)."""
        try:
            return REncoding.of(self.value("encoding"))
        except InputError as unimplemented:
            raise _Unchecked("encoding", f"{unimplemented} (provenance.encoding)") from None

    def integer(self, key: str) -> int:
        value = self.value(key)
        if not _is_int(value):
            raise _Unchecked(key, f"provenance {key} is {value!r}, not an integer")
        return int(value)

    def node(self, path: str) -> zarr.Array[Any] | zarr.Group | None:
        """The node at `path`, or None; one whose metadata will not parse is `_Unreadable`."""
        try:
            return self.root.get(path)
        except _READ_ERRORS as error:
            raise _Unreadable(path, f"{type(error).__name__}: {error}") from error

    def find(self, path: str) -> zarr.Array[Any] | None:
        """The array at `path`, or None; each is opened once, as opening reads its metadata."""
        if path not in self._arrays:
            member = self.node(path)
            self._arrays[path] = member if isinstance(member, zarr.Array) else None
        return self._arrays[path]

    def array(self, path: str) -> zarr.Array[Any]:
        found = self.find(path)
        if found is None:
            raise _Unchecked(path, f"{path} is missing (envelope.arrays)")
        return found

    def read(self, path: str, lo: int | None = None, hi: int | None = None) -> npt.NDArray[Any]:
        """Elements [lo, hi) of the array at `path`; one that will not decode is `_Unreadable`."""
        array = self.array(path)
        try:
            return np.asarray(array[lo:hi])
        except _READ_ERRORS as error:
            raise _Unreadable(path, f"{type(error).__name__}: {error}") from error

    def pick(self, path: str, coords: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """The elements of the array at `path` at scattered `coords`, in that order."""
        distinct, back = np.unique(coords, return_inverse=True)
        try:
            found = np.asarray(self.array(path).get_coordinate_selection(distinct))
        except _READ_ERRORS as error:
            raise _Unreadable(path, f"{type(error).__name__}: {error}") from error
        return found[back]

    def chromosomes(self) -> list[tuple[str, int, int]]:
        """The `chromosomes` ranges, if each is a canonical name and rows in order."""
        listed = self.node("variants")
        record = listed.attrs.get("chromosomes") if isinstance(listed, zarr.Group) else None
        try:
            entries = chromosomes_listed(record)
            canonical = all(alid_locus(f"{name}:1:A:C") is not None for name, _, _ in entries)
        except ValueError:
            canonical = False
        if not canonical:
            raise _Unchecked(
                "variants", "the chromosomes ranges are malformed (variants.chromosomes)"
            )
        return entries

    def positions(self, name: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """One chromosome's positions, kept as a reader keeps them (spec §10)."""
        key = f"position/{name}"
        if key not in self._kept:
            self._kept[key] = self.read("variants/position", start, stop).astype(np.int64)
        return self._kept[key]

    def indptr(self, path: str, rows: int) -> npt.NDArray[np.int64]:
        """An `indptr` array that delimits `rows` rows; one that cannot is `_Unchecked`."""
        if path not in self._kept:
            values = self.read(path).astype(np.int64)
            for _, problem in _indptr_problems(path, values, rows):
                raise _Unchecked(path, f"{path} {problem} (near.rows, index.rows, far.rows)")
            self._kept[path] = values
        return self._kept[path]


def _indptr_problems(
    path: str, indptr: npt.NDArray[np.int64], rows: int, end: int | None = None
) -> _Problems:
    """How `indptr` fails to delimit `rows` rows, ending at `end` elements when it is given."""
    if len(indptr) != rows + 1 or indptr[0] != 0:
        yield path, f"has {len(indptr)} entries from {indptr[:1].tolist()}, not {rows + 1} from 0"
    elif (np.diff(indptr) < 0).any():
        yield path, "decreases, so it delimits no rows"
    elif end is not None and indptr[-1] != end:
        yield path, f"ends at {indptr[-1]:,}, but its rows' arrays hold {end:,} elements"


def _flag(where: str, bad: npt.NDArray[np.bool_], first: int, what: str) -> _Problems:
    """One finding for every element `bad` marks, numbered from `first`, if there is any."""
    at = np.flatnonzero(bad) + first
    if len(at):
        more = f" and {len(at) - _NAMED:,} more" if len(at) > _NAMED else ""
        yield where, f"{what} at {', '.join(str(i) for i in at[:_NAMED])}{more}"


# ------------------------------------------------------------------ envelope


@dataclass(frozen=True)
class _Component:
    """A sparse component of each chromosome, `partner` and `r` delimited by an `indptr` (spec
    §5): its group, the names the pass gives its two parts, and the name it gives the row that
    holds each entry. A rule names the component it checks; none is assumed."""

    group: str
    partner: str
    code: str
    owner: str

    def path(self, chrom: str, array: str) -> str:
        return f"{self.group}/{chrom}/{array}"


_INDEX: Final = _Component("index", "partner", "code", "owner")
_FAR: Final = _Component("far", "far_partner", "far_code", "far_owner")
#: The Proxy Index and the Far Pairs, in the order §1 lists them.
_COMPONENTS: Final = (_INDEX, _FAR)

_VARIANT_DTYPES: Final = {
    "position": "int32",
    "alid": "string",
    "a1": "string",
    "a2": "string",
    "a1_frequency": "float32",
}

#: The ALID search index's two arrays (spec §1, §6; ADR 0010). The keys are bytes, not the
#: variable-length strings `variants/alid` holds, so a search compares them as they lie.
_ALID_INDEX_DTYPES: Final = {
    "alid": "uint8",
    "row": "int32",
}


def _shape(store: _Store, path: str, dtype: str, length: int | None) -> _Problems:
    array = store.find(path)
    if array is None:
        yield path, "missing (spec §1)"
        return
    found = array.metadata.to_dict()["data_type"]
    if found != dtype:
        yield path, f"its dtype is {found}, not {dtype} (spec §1)"
    if length is not None and array.shape != (length,):
        yield path, f"its shape is {array.shape}, not ({length},) (spec §2, §4, §5)"


def _last(store: _Store, path: str) -> int | None:
    array = store.find(path)
    if array is None or not array.shape or not array.shape[0]:
        return None
    return int(store.read(path, array.shape[0] - 1)[0])


def _arrays(store: _Store) -> Iterator[str]:
    """The path of every array spec §1 names, for the store's chromosomes."""
    yield from (f"variants/{name}" for name in _VARIANT_DTYPES)
    yield from (f"{layout.ALID_INDEX}/{name}" for name in _ALID_INDEX_DTYPES)
    for chrom, _, _ in store.chromosomes():
        yield from (f"near/{chrom}/{name}" for name in ("indptr", "r"))
        for k in _COMPONENTS:
            yield from (k.path(chrom, name) for name in ("indptr", "partner", "r"))


def _counted(store: _Store, rows: int) -> _Problems:
    """`n_variants` against the variant table, when provenance records it as a number."""
    recorded = _lookup(store.provenance, "n_variants")
    if _is_int(recorded) and recorded != rows:
        yield "n_variants", f"provenance records {recorded:,}, but variants/ holds {rows:,} rows"


def _index_shapes(store: _Store) -> _Problems:
    """The ALID index's two arrays (spec §6, §1): its keys are 32 bytes an entry, and it has as
    many entries as its `row` array holds. `alid_index.count` holds their lengths to the Variant
    Set's."""
    held = store.find(layout.ALID_INDEX_ROW)
    keys = held.shape[0] * layout.ALID_INDEX_WIDTH if held is not None else None
    yield from _shape(store, f"{layout.ALID_INDEX}/alid", _ALID_INDEX_DTYPES["alid"], keys)
    yield from _shape(store, f"{layout.ALID_INDEX}/row", _ALID_INDEX_DTYPES["row"], None)


@_rule("envelope.arrays")
def _envelope_arrays(store: _Store) -> _Problems:
    if store.root.metadata.zarr_format != 3:
        yield "/", f"the hierarchy is Zarr v{store.root.metadata.zarr_format}, not v3 (spec §1)"
        return
    ranges = store.chromosomes()
    size = ranges[-1][2] if ranges else 0
    for name, dtype in _VARIANT_DTYPES.items():
        yield from _shape(store, f"variants/{name}", dtype, size)
    yield from _index_shapes(store)
    yield from _counted(store, size)
    # The container's dtype: the same for every admitted encoding (spec §1, §3; ADR 0004).
    r_dtype = REncoding.dtype.name
    for chrom, start, stop in ranges:
        for group in ("near", *(k.group for k in _COMPONENTS)):
            yield from _shape(store, f"{group}/{chrom}/indptr", "int64", stop - start + 1)
        yield from _shape(store, f"near/{chrom}/r", r_dtype, _last(store, f"near/{chrom}/indptr"))
        for k in _COMPONENTS:
            entries = _last(store, k.path(chrom, "indptr"))
            yield from _shape(store, k.path(chrom, "partner"), "int32", entries)
            yield from _shape(store, k.path(chrom, "r"), r_dtype, entries)


@_rule("envelope.fill")
def _envelope_fill(store: _Store) -> _Problems:
    """Every array's fill value is one no valid store holds there, so a chunk it lost can
    never read as data: the reserved code in an r array, a negative indptr, and so on."""
    for path in _arrays(store):
        fill = store.array(path).fill_value
        if not never_data(path, fill).all():
            yield (
                path,
                (
                    f"its fill value {shown_fill(fill)} is one valid data can hold there: a "
                    "chunk it lost would read as data (spec §6)"
                ),
            )


# ------------------------------------------------------------------ provenance

#: Every key spec §7 names, required ones and the keys inside them, with their types.
#: "?" allows null; "?import" allows it only for an `ld_matrix` source; "?unmeasured" makes it
#: null exactly when `pc_diagnostic.pairs.n_pairs` is 0, when no cross-chromosome pair was
#: there to measure (ruling of 10 November 2026). A key inside one that is null or absent is
#: not checked: the enclosing key's own row reports it.
_SCHEMA: Final = {
    "format_version": "str",
    "encoding.r_dtype": "str",
    "encoding.r_scale": "int",
    "encoding.r_reserved_code": "int",
    "encoding.rounding": "str",
    "source.kind": "kind",
    "source.description": "str",
    "source.sha256": "hashes",
    "panel.id": "str",
    "panel.ancestry": "str",
    "panel.n": "int",
    "panel.samples_sha256": "str?import",
    "reference_assembly": "str",
    "variant_filters": "object",
    "variant_filters.filter": "strings",
    "variant_filters.mac_min": "int",
    "variant_filters.info_min": "number?",
    "variant_filters.autosomes_only": "bool",
    "variants_dropped": "object",
    "variants_dropped.zero_variance": "int",
    "variants_dropped.duplicate_reversed_indel": "int",
    "n_variants": "int",
    "near_bp": "int",
    "far_bp": "int",
    "adjustment.covariates": "strings",
    "adjustment.k": "int",
    "adjustment.k_target": "int?",
    "adjustment.k_min": "int?",
    "adjustment.pc_file": "object?",
    "adjustment.pc_file.sha256": "str",
    "adjustment.pc_file.source": "str",
    "adjustment.pc_check": "object?",
    "adjustment.pc_check.window_bp": "int",
    "adjustment.pc_check.step_bp": "int",
    "adjustment.pc_check.windows": "int",
    "adjustment.pc_check.min_share": "number",
    "adjustment.pc_check.max_ratio": "number",
    "adjustment.pc_check.scan": "list",
    "adjustment.pc_check.dropped": "strings",
    "adjustment.pc_check.null": "object",
    "adjustment.pc_check.null.kind": "str",
    "adjustment.pc_check.null.permutations": "int",
    "adjustment.pc_check.null.seed": "int",
    "adjustment.pc_check.null.quantile": "number",
    "t_far": "number",
    "t_far_q": "number",
    "far_code_floor": "int",
    "t_partner": "number",
    "partner_code_floor": "int",
    "pc_diagnostic": "object?import",
    "pc_diagnostic.pairs": "object",
    "pc_diagnostic.pairs.source": "str",
    "pc_diagnostic.pairs.seed": "int",
    "pc_diagnostic.pairs.n_pairs": "int",
    "pc_diagnostic.rate_r2_ge_0_01_unadjusted": "number?unmeasured",
    "pc_diagnostic.rate_r2_ge_0_01_adjusted": "number?unmeasured",
    "pc_diagnostic.rate_r2_ge_0_01_sampling": "number",
    "pc_diagnostic.near_100kb_change": "object",
    "counts.near_pairs": "int",
    "counts.far_candidates": "int",
    "counts.far_pairs": "int",
    "counts.partner_entries": "int",
    "counts.index_entries": "int",
    "counts.alids_left_out": "int",
    "software": "object",
    "software.ogld": "str",
    "software.numpy": "str",
    "software.zarr": "str",
    "build.created_at": "str",
    "build.wall_seconds": "number",
    "build.peak_memory_bytes": "int",
    "build.threads": "int",
}

#: Each entry of the PC Check's `scan`, one per PC of the PC File (spec §7).
_SCAN_ENTRY: Final = {
    "pc": "str",
    "window": "str",
    "window_share": "number",
    "rest_share": "number",
    "ratio": "number",
    "local": "bool",
    "window_null_q99": "number",
    "rest_null_q99": "number",
}

#: The quantiles `pc_diagnostic.near_100kb_change` holds, by name (spec §7).
_QUANTILES: Final = ("0", "0.5", "0.9", "0.99", "1")

_TYPES: Final[dict[str, Callable[[object], bool]]] = {
    "str": lambda v: isinstance(v, str),
    "int": _is_int,
    "number": _is_number,
    "bool": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "list": lambda v: isinstance(v, list),
    "strings": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "hashes": lambda v: isinstance(v, dict) and all(isinstance(x, str) for x in v.values()),
    "kind": lambda v: v in ("genotypes", "ld_matrix"),
}


def _within_null(provenance: dict[str, Any], key: str) -> bool:
    """True when an enclosing key of the schema is null or absent: its own row reports it."""
    parts = key.split(".")
    enclosing = (".".join(parts[:i]) for i in range(1, len(parts)))
    return any(not isinstance(_lookup(provenance, e), dict) for e in enclosing if e in _SCHEMA)


@dataclass(frozen=True)
class _Context:
    """What decides whether a key may be null: the source kind, and the pairs measured."""

    source: object = None
    n_pairs: object = None


def _null_problem(nullable: str, only: str, context: _Context) -> str | None:
    if not nullable:
        return "null, which spec §7 allows nowhere for this key"
    if only == "import" and context.source != "ld_matrix":
        return "null, which spec §7 allows only for an ld_matrix source"
    if only == "unmeasured" and context.n_pairs != 0:
        return "null, which spec §7 allows only when pc_diagnostic.pairs.n_pairs is 0"
    return None


def _key_problem(kind: str, value: object, context: _Context) -> str | None:
    base, nullable, only = kind.partition("?")
    if value is _MISSING:
        return "missing (spec §7)"
    if value is None:
        return _null_problem(nullable, only, context)
    if not _TYPES[base](value):
        return f"{value!r:.80} is not of type {base} (spec §7)"
    if only == "unmeasured" and context.n_pairs == 0:
        return (
            f"{value!r}, but pc_diagnostic.pairs.n_pairs is 0, so nothing was measured: it is "
            "null (spec §7)"
        )
    return None


def _scan_problems(provenance: dict[str, Any]) -> _Problems:
    """The fields §7 names in each entry of the PC Check's `scan`."""
    scan = _lookup(provenance, "adjustment.pc_check.scan")
    for i, entry in enumerate(scan if isinstance(scan, list) else []):
        fields = entry if isinstance(entry, dict) else {}
        for key, kind in _SCAN_ENTRY.items():
            problem = _key_problem(kind, fields.get(key, _MISSING), _Context())
            if problem is not None:
                yield f"adjustment.pc_check.scan[{i}].{key}", problem


def _quantile_problems(provenance: dict[str, Any]) -> _Problems:
    """The five quantiles §7 names in `pc_diagnostic.near_100kb_change`."""
    change = _lookup(provenance, "pc_diagnostic.near_100kb_change")
    for q in _QUANTILES if isinstance(change, dict) else ():
        problem = _key_problem("number", change.get(q, _MISSING), _Context())
        if problem is not None:
            yield f"pc_diagnostic.near_100kb_change[{q}]", problem


@_rule("provenance.complete")
def _provenance_complete(store: _Store) -> _Problems:
    provenance = store.provenance
    if not isinstance(provenance, dict):
        yield "ld_store", "the root group has no ld_store provenance object (spec §7)"
        return
    context = _Context(
        _lookup(provenance, "source.kind"), _lookup(provenance, "pc_diagnostic.pairs.n_pairs")
    )
    for key, kind in _SCHEMA.items():
        if not _within_null(provenance, key):
            problem = _key_problem(kind, _lookup(provenance, key), context)
            if problem is not None:
                yield key, problem
    yield from _scan_problems(provenance)
    yield from _quantile_problems(provenance)


@_rule("provenance.encoding")
def _provenance_encoding(store: _Store) -> _Problems:
    """The format version is the one this package reads, and the encoding one it implements,
    `REncoding.of` (spec §3, §13). `ogld info` and every reader refuse a store with any other."""
    version = store.value("format_version")
    if version not in READABLE_FORMATS:
        yield (
            "format_version",
            f"{version!r}: this ogld reads {' and '.join(READABLE_FORMATS)} only (spec §13)",
        )
    try:
        REncoding.of(store.value("encoding"))
    except InputError as unimplemented:
        yield "encoding", str(unimplemented)


@_rule("provenance.distances")
def _provenance_distances(store: _Store) -> _Problems:
    near, far = store.integer("near_bp"), store.integer("far_bp")
    if near != NEAR_BP:
        yield "near_bp", f"{near:,}, not {NEAR_BP:,} (ADR 0001)"
    if not near < far <= FAR_BP:
        yield "far_bp", f"{far:,} is not above near_bp {near:,} and at most {FAR_BP:,}"


def _k_with_pc_file(adjustment: dict[str, Any], k: int) -> _Problems:
    """A PC File was read: the build used between k_min and k_target PCs."""
    k_min, k_target = adjustment.get("k_min"), adjustment.get("k_target")
    if not (_is_int(k_min) and _is_int(k_target) and k_min <= k <= k_target):
        yield (
            "adjustment",
            (
                f"a PC File was read, so k_min ≤ k ≤ k_target: k_min {k_min}, k {k}, "
                f"k_target {k_target} (spec §7)"
            ),
        )


def _k_without_pc_file(adjustment: dict[str, Any], k: int, source: object) -> _Problems:
    """No PC File was read: k_min applies to nothing, and a build is unadjusted."""
    if adjustment.get("k_min") is not None:
        yield (
            "adjustment.k_min",
            (
                f"{adjustment['k_min']}, but no PC File was read, so k_min applies to nothing and "
                "is null (spec §7)"
            ),
        )
    if source == "genotypes" and (k, adjustment.get("k_target")) != (0, 0):
        yield (
            "adjustment",
            (
                f"a build that read no PC File is unadjusted, k = k_target = 0: k {k}, "
                f"k_target {adjustment.get('k_target')}"
            ),
        )


def _k_records(store: _Store) -> _Problems:
    adjustment, k = store.value("adjustment"), store.integer("adjustment.k")
    covariates, source = store.value("adjustment.covariates"), store.value("source.kind")
    if not isinstance(covariates, list) or not isinstance(adjustment, dict):
        raise _Unchecked("adjustment", "adjustment.covariates is not a list (provenance.complete)")
    if k != len(covariates):
        yield "adjustment.k", f"k is {k}, but {len(covariates)} covariates are named"
    if store.value("adjustment.pc_file") is not None:
        yield from _k_with_pc_file(adjustment, k)
    else:
        yield from _k_without_pc_file(adjustment, k, source)
    if source == "ld_matrix" and adjustment.get("k_target") is not None:
        yield "adjustment.k_target", "an ld_matrix source has no target, so k_target is null"


def _t_far_records(store: _Store) -> _Problems:
    n, k = store.integer("panel.n"), store.integer("adjustment.k")
    recorded, q = store.value("t_far"), store.value("t_far_q")
    if q != T_FAR_Q:
        yield "t_far_q", f"{q!r}, not {T_FAR_Q} (ADR 0001)"
    try:
        expected = t_far(n, k)
    except ValueError as impossible:
        yield "t_far", str(impossible)
        return
    if recorded != expected:
        yield (
            "t_far",
            f"{recorded!r}, but max(0.001, q / (n − k − 1)) is {expected!r} at n {n}, k {k}",
        )
    floor, encoding = store.integer("far_code_floor"), store.encoding()
    if floor != encoding.far_code_floor(expected):
        yield (
            "far_code_floor",
            f"{floor}, but t_far {expected!r} at × {encoding.scale} gives "
            f"{encoding.far_code_floor(expected)}",
        )


@_rule("provenance.threshold")
def _provenance_threshold(store: _Store) -> _Problems:
    yield from _k_records(store)
    yield from _t_far_records(store)


# ------------------------------------------------------------------ variants


@_rule("variants.chromosomes")
def _variants_chromosomes(store: _Store) -> _Problems:
    ranges = store.chromosomes()
    expected, previous = 0, 0
    for name, start, stop in ranges:
        if chromosome_rank(name) <= previous:
            yield "variants", f"chromosome {name} is out of chromosome order (spec §2)"
        if start != expected:
            yield "variants", f"chromosome {name} starts at row {start:,}, not {expected:,}"
        expected, previous = stop, chromosome_rank(name)
    size = store.array("variants/position").shape[0]
    if expected != size:
        yield "variants", f"the ranges cover {expected:,} rows of the {size:,} in variants/"


def _strings(store: _Store, name: str, lo: int, hi: int) -> npt.NDArray[np.str_]:
    return np.array(store.read(f"variants/{name}", lo, hi).tolist(), dtype=np.str_)


@_rule("variants.order")
def _variants_order(store: _Store) -> _Problems:
    for chrom, start, stop in store.chromosomes():
        carry_pos = np.empty(0, dtype=np.int64)
        carry_alid = np.empty(0, dtype=np.str_)
        for lo, hi in windows(start, stop, _ROWS):
            pos = np.concatenate([carry_pos, store.read("variants/position", lo, hi)])
            alid = np.concatenate([carry_alid, _strings(store, "alid", lo, hi)])
            after = (pos[1:] > pos[:-1]) | ((pos[1:] == pos[:-1]) & (alid[1:] > alid[:-1]))
            first = lo - len(carry_pos) + 1
            yield from _flag(f"variants rows (chromosome {chrom})", ~after, first, "out of order")
            carry_pos, carry_alid = pos[-1:], alid[-1:]


def _engine_alid(chrom: str, position: int, a1: str, a2: str) -> tuple[str, str, str] | None:
    try:
        o = orient(chrom, position, a1, a2)
    except IdentityError:
        return None
    return o.alid, o.a1, o.a2


@_rule("variants.identity")
def _variants_identity(store: _Store) -> _Problems:
    for chrom, start, stop in store.chromosomes():
        for lo, hi in windows(start, stop, _ROWS):
            pos = store.read("variants/position", lo, hi)
            alid, a1, a2 = (_strings(store, name, lo, hi) for name in ("alid", "a1", "a2"))
            rows = zip(pos, alid, a1, a2, strict=True)
            bad = np.array(
                [_engine_alid(chrom, int(p), x, y) != (s, x, y) for p, s, x, y in rows], dtype=bool
            )
            yield from _flag(
                "variants/alid", bad, lo, "not the engine's ALID of its position, a1 and a2"
            )


@_rule("variants.frequency")
def _variants_frequency(store: _Store) -> _Problems:
    for lo, hi in windows(0, store.array("variants/a1_frequency").shape[0], _ROWS):
        f = store.read("variants/a1_frequency", lo, hi).astype(np.float64)
        with np.errstate(invalid="ignore"):
            bad = ~(np.isfinite(f) & (f > 0) & (f < 1))
        yield from _flag("variants/a1_frequency", bad, lo, "not finite and in (0, 1)")


# ------------------------------------------------------------------ the ALID index


def _index_rows(store: _Store) -> npt.NDArray[np.int64]:
    """The Variant Set row each entry of the ALID index names, whole."""
    return store.read(layout.ALID_INDEX_ROW).astype(np.int64)


def _index_keys(store: _Store) -> npt.NDArray[Any]:
    """The ALID index's keys, whole, as numpy's fixed-width bytes: 32 bytes an entry, sorted.

    Whether the two arrays agree — one key an entry — is `alid_index.count`'s rule, so a key array
    that does not is a finding there, and every rule that reads the keys says it cannot be checked
    rather than failing on an array too short for the entries it names (spec §12; ogld#42's fix
    round 3)."""
    width = layout.ALID_INDEX_WIDTH
    values = store.read(layout.ALID_INDEX_ALID)
    entries = int(store.array(layout.ALID_INDEX_ROW).shape[0])
    if values.size != entries * width:
        raise _Unchecked(
            layout.ALID_INDEX_ALID,
            f"holds {values.size:,} bytes, not {entries:,} keys of {width}: one key an entry "
            "(alid_index.count)",
        )
    return values.reshape(-1, width).view(f"|S{width}").ravel()


def _entries_by_row(store: _Store) -> npt.NDArray[np.int64]:
    """Where each Variant Set row's entry lies in the index, or −1 where no entry names it: the
    `row` array inverted, so a rule that walks the table in blocks compares each row with the one
    entry the index has for it without a scattered read. An entry outside the Variant Set, or one
    of several naming a row, is `alid_index.rows`' finding; here the last such entry wins."""
    size = store.array("variants/position").shape[0]
    rows = _index_rows(store)
    at = np.full(size, -1, dtype=np.int64)
    inside = np.flatnonzero((rows >= 0) & (rows < size))
    at[rows[inside]] = inside
    return at


def _table_alids(
    store: _Store, lo: int, hi: int
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_]]:
    """Rows [lo, hi)'s ALIDs, and whether each fits the index's slot (spec §2; ADR 0010)."""
    alids = _strings(store, "alid", lo, hi)
    return alids, layout.alid_fits(alids)


@_rule("alid_index.rows")
def _alid_index_rows(store: _Store) -> _Problems:
    """One entry a variant: each names a row of the Variant Set, none of them twice, and every
    ALID that fits the slot is named by one (spec §2, §6; ADR 0010)."""
    path = layout.ALID_INDEX_ROW
    rows = _index_rows(store)
    size = store.array("variants/position").shape[0]
    yield from _flag(path, (rows < 0) | (rows >= size), 0, "is not a row of the Variant Set")
    at = np.argsort(rows, kind="stable")
    twice = rows[at[1:]][rows[at[1:]] == rows[at[:-1]]]
    if len(twice):
        named = ", ".join(str(row) for row in twice[:_NAMED])
        more = f" and {len(twice) - _NAMED:,} more" if len(twice) > _NAMED else ""
        yield path, f"names row(s) {named}{more} more than once: one entry a variant (spec §2)"
    by_row = _entries_by_row(store)
    for lo, hi in windows(0, size, _ROWS):
        _, fits = _table_alids(store, lo, hi)
        yield from _flag(
            "variants/alid", fits & (by_row[lo:hi] < 0), lo, "has no entry in the ALID index"
        )


@_rule("alid_index.order")
def _alid_index_order(store: _Store) -> _Problems:
    """The keys ascend bytewise: the order every search over them depends on (spec §2; ADR 0010)."""
    path = layout.ALID_INDEX_ALID
    width = layout.ALID_INDEX_WIDTH
    count = int(store.array(layout.ALID_INDEX_ALID).shape[0]) // width
    carry = np.empty(0, dtype=f"|S{width}")
    for lo, hi in windows(0, count, _ROWS):
        keys = store.read(path, lo * width, hi * width).reshape(-1, width)
        both = np.concatenate([carry, keys.view(f"|S{width}").ravel()])
        yield from _flag(
            path, ~(both[1:] > both[:-1]), lo - len(carry) + 1, "out of bytewise order"
        )
        carry = both[-1:]


@_rule("alid_index.keys")
def _alid_index_keys(store: _Store) -> _Problems:
    """Each entry's key is the full ALID of the row it names, NUL padded — so that row's ALID
    fits the slot, no key was truncated into it, and none is a key no variant has. One variant
    answering to another's name is what the width exists to prevent (spec §2; ADR 0010;
    opengwasdb issue #127)."""
    path = layout.ALID_INDEX_ALID
    width = layout.ALID_INDEX_WIDTH
    keys = _index_keys(store)
    by_row = _entries_by_row(store)
    size = store.array("variants/position").shape[0]
    for lo, hi in windows(0, size, _ROWS):
        alids, fits = _table_alids(store, lo, hi)
        entry = by_row[lo:hi]
        held = entry >= 0
        # A row whose ALID is wider than the slot has no key of its own: an entry naming it holds
        # a truncated key, or one that is another variant's.
        yield from _flag(path, held & ~fits, lo, "names a row whose ALID is wider than the slot")
        compare = held & fits
        expected, _ = layout.alid_keys(alids[compare])
        wrong = np.full(len(alids), False)
        wrong[compare] = keys[entry[compare]] != expected.view(f"|S{width}").ravel()
        yield from _flag(path, wrong, lo, "is not its row's own ALID, NUL padded")


@_rule("alid_index.count")
def _alid_index_count(store: _Store) -> _Problems:
    """The keys are 32 bytes an entry, and provenance's left-out count is the table's own count of
    ALIDs wider than the slot (spec §2, §6, §7; ADR 0010)."""
    width = layout.ALID_INDEX_WIDTH
    size = store.array("variants/position").shape[0]
    entries = int(store.array(layout.ALID_INDEX_ROW).shape[0])
    held = int(store.array(layout.ALID_INDEX_ALID).shape[0])
    if held != entries * width:
        yield (
            layout.ALID_INDEX_ALID,
            f"holds {held:,} bytes, not {entries:,} keys of {width}: one key an entry (spec §6)",
        )
    wide = sum(int((~_table_alids(store, lo, hi)[1]).sum()) for lo, hi in windows(0, size, _ROWS))
    recorded = store.integer("counts.alids_left_out")
    if recorded != wide:
        yield (
            "counts.alids_left_out",
            f"records {recorded:,} left out, but {wide:,} of the {size:,} ALIDs are wider than "
            "the slot: the count is the table's (spec §7)",
        )


# ------------------------------------------------------------------ the Near Band


@_rule("near.rows")
def _near_rows(store: _Store) -> _Problems:
    near_bp = store.integer("near_bp")
    for chrom, start, stop in store.chromosomes():
        path = f"near/{chrom}/indptr"
        indptr = store.read(path).astype(np.int64)
        problems = list(_indptr_problems(path, indptr, stop - start))
        yield from problems
        if not problems:
            expected = near_row_lengths(store.positions(chrom, start, stop), near_bp)
            bad = np.diff(indptr) != expected
            yield from _flag(path, bad, 0, "a row whose length is not spec §4's")


# ------------------------------------------------------------------ one pass over each chromosome


def _in_order[T, R](work: Callable[[T], R], items: Iterable[T], threads: int) -> Iterator[R]:
    """`work` of each item, in the items' order, on `threads` threads. At most `threads` items
    are begun and not yet taken, so a pass holds at most that many blocks at once."""
    if threads == 1:
        yield from map(work, items)
        return
    with ThreadPoolExecutor(threads, thread_name_prefix="ogld-validate") as pool:
        pending: deque[Future[R]] = deque()
        for item in items:
            if len(pending) == threads:
                yield pending.popleft().result()
            pending.append(pool.submit(work, item))
        while pending:
            yield pending.popleft().result()


#: The arrays the pass reads in blocks of rows, by the names its rules give them.
_PARTS: Final = {"band": ("near", "r")} | {
    part: (k.group, array)
    for k in _COMPONENTS
    for part, array in ((k.partner, "partner"), (k.code, "r"))
}


@dataclass(frozen=True)
class _Chromosome:
    """One chromosome's rows [start, stop), with the Near Band's, the Proxy Index's and the Far
    Pairs' `indptr` where each delimits them, and None where it cannot: a rule that needs it
    says why."""

    name: str
    start: int
    stop: int
    band: npt.NDArray[np.int64] | None
    index: npt.NDArray[np.int64] | None
    far: npt.NDArray[np.int64] | None

    @property
    def size(self) -> int:
        return self.stop - self.start

    def path(self, part: str) -> str:
        kind, name = _PARTS[part]
        return f"{kind}/{self.name}/{name}"

    def indptr(self, part: str) -> npt.NDArray[np.int64] | None:
        return {"near": self.band, "index": self.index, "far": self.far}[_PARTS[part][0]]

    def blocks(self) -> list[tuple[int, int]]:
        """The rows in blocks of at most `_WINDOW` band codes and an eighth as many entries of
        the index and of the Far Pairs."""
        entries = _WINDOW >> _ENTRY_SHIFT
        limits = [(self.band, _WINDOW), (self.index, entries), (self.far, entries)]
        delimiting = [(indptr, most) for indptr, most in limits if indptr is not None]
        return list(blocks(*delimiting)) if delimiting else []


def _chromosome(store: _Store, name: str, start: int, stop: int) -> _Chromosome:
    def delimiting(kind: str) -> npt.NDArray[np.int64] | None:
        try:
            return store.indptr(f"{kind}/{name}/indptr", stop - start)
        except _Unchecked:
            return None

    return _Chromosome(
        name, start, stop, delimiting("near"), delimiting("index"), delimiting("far")
    )


@dataclass(frozen=True)
class _Block:
    """Rows [first, last) of one chromosome as the pass read them: each part's elements, from
    offset `at[part]`, or why they cannot be read; with the index's, "owner", the row that holds
    each entry, and with the Far Pairs', "far_owner"."""

    first: int
    last: int
    at: dict[str, int]
    held: dict[str, npt.NDArray[Any] | _Unchecked]

    def get(self, part: str) -> npt.NDArray[Any]:
        found = self.held[part]
        if isinstance(found, _Unchecked):
            raise found
        return found


def _part(store: _Store, path: str, lo: int, hi: int) -> npt.NDArray[Any] | _Unchecked:
    """Elements [lo, hi) of the array at `path`, an index's or the Far Pairs' as int64; or why
    they cannot be read. An empty span is not read: most blocks hold no Far Pair, and a read of
    nothing still costs a Zarr call."""
    try:
        values = store.read(path, lo, hi) if hi > lo else np.empty(0, store.array(path).dtype)
    except _Unchecked as unread:
        return unread
    if len(values) != hi - lo:
        why = f"{path} holds {lo + len(values):,} elements, fewer than its indptr's {hi:,}"
        return _Unchecked(path, why)
    return values if path.startswith("near/") else values.astype(np.int64)


def _read_block(
    store: _Store, c: _Chromosome, parts: frozenset[str], rows: tuple[int, int]
) -> _Block:
    """Rows [a, b) of `c`, each of `parts` read once; this runs on the pass's threads."""
    a, b = rows
    at: dict[str, int] = {}
    held: dict[str, npt.NDArray[Any] | _Unchecked] = {}
    for part in parts:
        indptr = c.indptr(part)
        assert indptr is not None, "a rule takes only a part whose indptr delimits the rows"
        at[part], hi = int(indptr[a]), int(indptr[b])
        held[part] = _part(store, c.path(part), at[part], hi)
    for k in _COMPONENTS:
        entries = [held[part] for part in (k.partner, k.code) if part in parts]
        indptr = c.indptr(k.partner)
        if entries and indptr is not None:
            held[k.owner] = _owners(entries, indptr, a, b)
    return _Block(a, b, at, held)


def _owners(
    entries: list[npt.NDArray[Any] | _Unchecked], indptr: npt.NDArray[np.int64], a: int, b: int
) -> npt.NDArray[np.int64] | _Unchecked:
    """The row that holds each entry of rows [a, b) of a sparse component, or why its entries
    cannot be read. Built from entries read, never from `indptr` alone: an `indptr` that ends far
    past its array would otherwise ask for as many owners as it claims."""
    unread = [found for found in entries if isinstance(found, _Unchecked)]
    if unread:
        return unread[0]
    return owners(np.diff(indptr[a : b + 1]), np.arange(a, b, dtype=np.int64))


class _Scanned:
    """A rule the pass feeds, built for one chromosome. Building it reads what it needs before
    any block, and may raise `_Unchecked`. `parts` names what it takes from each block, none
    when it takes no block; `block` runs on the pass's threads, on several blocks at once."""

    parts: frozenset[str] = frozenset()

    def begin(self) -> _Problems:
        yield from ()

    def block(self, b: _Block) -> _Problems:
        yield from ()

    def end(self) -> _Problems:
        yield from ()


_SCANNED: dict[str, Callable[[_Store, _Chromosome], _Scanned]] = {}


def _scanned(name: str) -> Callable[[type[_Scanned]], type[_Scanned]]:
    def register(rule: type[_Scanned]) -> type[_Scanned]:
        _SCANNED[name] = rule
        return rule

    return register


_Outcome = tuple[list[tuple[str, str]], _Unchecked | None]


def _outcome(problems: Callable[[], _Problems]) -> _Outcome:
    """What `problems` yields, and what stopped it, if anything did."""
    found: list[tuple[str, str]] = []
    try:
        found.extend(problems())
    except _Unchecked as stopped:
        return found, stopped
    return found, None


def _record(found: dict[str, list[Finding]], name: str, outcome: _Outcome) -> bool:
    """Records a rule's outcome as findings; False when it stopped, so nothing more feeds it."""
    problems, stopped = outcome
    found[name].extend(Finding(name, where, message) for where, message in problems)
    if stopped is not None:
        found[name].append(_finding(name, stopped))
    return stopped is None


def _scan(store: _Store, names: list[str]) -> dict[str, list[Finding]]:
    """The scanned rules `names`, fed by one pass over each chromosome in blocks of rows."""
    try:
        ranges = store.chromosomes()
    except _Unchecked as unchecked:
        return {name: [_finding(name, unchecked)] for name in names}
    found: dict[str, list[Finding]] = {name: [] for name in names}
    live = list(names)
    for name, start, stop in ranges:
        ended = _pass(store, _chromosome(store, name, start, stop), live, found)
        live = [rule for rule in live if rule in ended]
    return found


def _pass(
    store: _Store, c: _Chromosome, names: list[str], found: dict[str, list[Finding]]
) -> set[str]:
    """One chromosome's pass: each rule built and begun, fed every block, and ended. Returns
    the rules that ended; the others stopped, and are fed nothing more."""
    begun = _begun(store, c, names, found)
    stopped = _feed(store, c, {name: rule for name, rule in begun.items() if rule.parts}, found)
    live = [(name, rule) for name, rule in begun.items() if name not in stopped]
    return {name for name, rule in live if _record(found, name, _outcome(rule.end))}


def _begun(
    store: _Store, c: _Chromosome, names: list[str], found: dict[str, list[Finding]]
) -> dict[str, _Scanned]:
    """Each rule of `names` built for `c` and begun; a rule that cannot be is recorded."""
    begun: dict[str, _Scanned] = {}
    for name in names:
        try:
            rule = _SCANNED[name](store, c)
        except _Unchecked as unchecked:
            found[name].append(_finding(name, unchecked))
            continue
        if _record(found, name, _outcome(rule.begin)):
            begun[name] = rule
    return begun


def _feed(
    store: _Store, c: _Chromosome, fed: dict[str, _Scanned], found: dict[str, list[Finding]]
) -> set[str]:
    """Each block of `c`'s rows, read once, to every rule in `fed`, on the store's threads.
    Returns the rules that stopped."""
    stopped: set[str] = set()
    parts = frozenset().union(*(rule.parts for rule in fed.values()))
    for part in parts:  # opened here, once, rather than by each thread
        with suppress(_Unchecked):
            store.find(c.path(part))

    def check(rows: tuple[int, int]) -> dict[str, _Outcome]:
        b = _read_block(store, c, parts, rows)
        live = [(name, rule) for name, rule in fed.items() if name not in stopped]
        return {name: _outcome(partial(rule.block, b)) for name, rule in live}

    for outcomes in _in_order(check, c.blocks() if fed else [], store.threads):
        for name, outcome in outcomes.items():
            if name not in stopped and not _record(found, name, outcome):
                stopped.add(name)
    return stopped


def _valid(partner: npt.NDArray[np.int64], size: int) -> npt.NDArray[np.bool_]:
    """Entries whose partner is a row of the chromosome: the others are index.rows'."""
    return (partner >= 0) & (partner < size)


def _sparse_indptr(store: _Store, c: _Chromosome, k: _Component) -> npt.NDArray[np.int64]:
    """The `indptr` of the sparse component `k`, which a rule reading its entries needs to
    delimit the rows."""
    return store.indptr(k.path(c.name, "indptr"), c.size)


# ------------------------------------------------------------------ the codes


class _Codes(_Scanned):
    """Every code of one `r` array is a code of the store's encoding, |code| ≤ r_scale: never
    the reserved one, and never one past the scale, which would decode to |r̂| > 1. The pass
    feeds it when its blocks tile the array; when its `indptr` cannot delimit them, it reads the
    array in windows of its own. Without an encoding it implements it cannot be checked."""

    def __init__(self, store: _Store, c: _Chromosome, part: str) -> None:
        self.encoding = store.encoding()
        self.store, self.part, self.path = store, part, c.path(part)
        self.length = store.array(self.path).shape[0]
        indptr = c.indptr(part)
        if indptr is not None and int(indptr[-1]) == self.length:
            self.parts = frozenset({part})

    def begin(self) -> _Problems:
        if self.parts:
            return

        def check(window: tuple[int, int]) -> list[tuple[str, str]]:
            return list(self._outside_encoding(self.store.read(self.path, *window), window[0]))

        tiles = windows(0, self.length, _WINDOW)
        for problems in _in_order(check, tiles, self.store.threads):
            yield from problems

    def block(self, b: _Block) -> _Problems:
        yield from self._outside_encoding(b.get(self.part), b.at[self.part])

    def _outside_encoding(self, codes: npt.NDArray[Any], first: int) -> _Problems:
        reserved, scale = self.encoding.reserved_code, self.encoding.scale
        yield from _flag(self.path, codes == reserved, first, f"the reserved code {reserved}")
        past = ~self.encoding.holds(codes) & (codes != reserved)
        yield from _flag(self.path, past, first, f"a code past the scale, |code| > {scale}")


@_scanned("near.codes")
class _NearCodes(_Codes):
    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c, "band")


@_scanned("index.codes")
class _IndexCodes(_Codes):
    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c, _INDEX.code)


@_scanned("far.codes")
class _FarCodes(_Codes):
    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c, _FAR.code)


# ------------------------------------------------------------------ the Proxy Index


def _sparse_ends(store: _Store, k: _Component, chrom: str) -> int | None:
    """The length `k`'s `partner` and `r` share, or None when they disagree."""
    lengths = {store.array(k.path(chrom, name)).shape[0] for name in ("partner", "r")}
    return lengths.pop() if len(lengths) == 1 else None


class _Rows(_Scanned):
    """A sparse component's `indptr` delimits the rows and ends at the length of `partner` and
    `r`; within a row, partners ascend and lie in [0, n_c), and each subclass says which other
    partner is misplaced. A chromosome whose `indptr` fails is not read further."""

    component: _Component
    #: What a misplaced partner is, as a finding names it.
    misplaced: str

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        k = self.component
        self.c, path, end = c, k.path(c.name, "indptr"), _sparse_ends(store, k, c.name)
        indptr = store.read(path).astype(np.int64)
        self.problems = list(_indptr_problems(path, indptr, c.size, end))
        if end is None:
            self.problems.append((f"{k.group}/{c.name}", "partner and r differ in length"))
        if not self.problems:
            self.parts = frozenset({k.partner})

    def is_misplaced(
        self, partner: npt.NDArray[np.int64], owner: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        raise NotImplementedError

    def begin(self) -> _Problems:
        yield from self.problems

    def block(self, b: _Block) -> _Problems:
        k = self.component
        partner, owner, at = b.get(k.partner), b.get(k.owner), b.at[k.partner]
        outside = ~_valid(partner, self.c.size) | self.is_misplaced(partner, owner)
        repeated = (np.diff(partner) <= 0) & (np.diff(owner) == 0)
        where = self.c.path(k.partner)
        yield from _flag(where, outside, at, f"a partner outside [0, n_c) or {self.misplaced}")
        yield from _flag(where, repeated, at + 1, "a partner not above the one before it")


@_scanned("index.rows")
class _IndexRows(_Rows):
    """The Proxy Index's partners are never the row itself: each pair is in both its rows."""

    component, misplaced = _INDEX, "the row itself"

    def is_misplaced(
        self, partner: npt.NDArray[np.int64], owner: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        return np.asarray(partner == owner)


@_scanned("far.rows")
class _FarRows(_Rows):
    """The Far Pairs' partners are each above their row: each pair is in its earlier row."""

    component, misplaced = _FAR, "not above its row"

    def is_misplaced(
        self, partner: npt.NDArray[np.int64], owner: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        return np.asarray(partner <= owner)


@_scanned("index.distance")
class _IndexDistance(_Scanned):
    """Every entry's partner lies within far_bp of its row's position."""

    parts = frozenset({_INDEX.partner})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.c, self.far_bp = c, store.integer("far_bp")
        _sparse_indptr(store, c, _INDEX)
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        partner, owner = b.get(_INDEX.partner), b.get(_INDEX.owner)
        partner = np.where(_valid(partner, self.c.size), partner, owner)
        beyond = ~within(self.pos[partner] - self.pos[owner], self.far_bp)
        what = f"a partner beyond {self.far_bp:,} bp"
        yield from _flag(self.c.path(_INDEX.partner), beyond, b.at[_INDEX.partner], what)


@_scanned("far.distance")
class _FarDistance(_Scanned):
    """Every Far Pair lies more than near_bp and at most far_bp apart. A partner outside the
    chromosome is far.rows' to say."""

    parts = frozenset({_FAR.partner})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.c, self.near_bp, self.far_bp = c, store.integer("near_bp"), store.integer("far_bp")
        _sparse_indptr(store, c, _FAR)
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        partner, owner = b.get(_FAR.partner), b.get(_FAR.owner)
        valid = _valid(partner, self.c.size)
        gap = self.pos[np.where(valid, partner, owner)] - self.pos[owner]
        outside = valid & (within(gap, self.near_bp) | ~within(gap, self.far_bp))
        what = f"a Far Pair not more than {self.near_bp:,} and at most {self.far_bp:,} bp apart"
        yield from _flag(self.c.path(_FAR.partner), outside, b.at[_FAR.partner], what)


class _Threshold(_Scanned):
    """Every code of a sparse component, the Proxy Index or the Far Pairs, reaches
    far_code_floor in magnitude."""

    component: _Component

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.c, self.floor = c, store.integer("far_code_floor")
        _sparse_indptr(store, c, self.component)
        self.code = self.component.code
        self.parts = frozenset({self.code})

    def block(self, b: _Block) -> _Problems:
        below = np.abs(b.get(self.code)) < self.floor
        what = f"|code| below far_code_floor {self.floor}"
        yield from _flag(self.c.path(self.code), below, b.at[self.code], what)


@_scanned("index.threshold")
class _IndexThreshold(_Threshold):
    component = _INDEX


@_scanned("far.threshold")
class _FarThreshold(_Threshold):
    component = _FAR


#: The bits a code takes in a key: the container's width, the same for every admitted encoding.
_CODE_BITS: Final = 8 * REncoding.dtype.itemsize


def _keys(
    lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64], code: npt.NDArray[np.int64], size: int
) -> npt.NDArray[np.int64]:
    """One integer per (pair, code), so two sets of entries compare with one set operation: any
    code of the container, less its smallest, the reserved code, fits `_CODE_BITS` bits."""
    return ((lo * size + hi) << _CODE_BITS) + (code - REncoding.reserved_code)


def _mixed(keys: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """splitmix64's finaliser of each key: a sum of these tells two multisets of keys apart
    unless they collide modulo 2⁶⁴, about one chance in 1.8×10¹⁹ for each difference."""
    z = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class _Sums:
    """Two sums of `_mixed` keys per row, added to from several threads: 0 is the empty sum."""

    def __init__(self, rows: int) -> None:
        self.sums = np.full((2, rows), 0, dtype=np.uint64)
        self.lock = threading.Lock()

    def add(self, which: int, rows: npt.NDArray[np.int64], keys: npt.NDArray[np.int64]) -> None:
        mixed = _mixed(keys)
        with self.lock:
            np.add.at(self.sums[which], rows, mixed)

    def differ(self) -> npt.NDArray[np.bool_]:
        differ: npt.NDArray[np.bool_] = self.sums[0] != self.sums[1]
        return differ


class _Named:
    """Per row, the first and the last row whose entries name it as their partner, kept from
    several threads: so a row can be checked again by reading only the blocks between them."""

    def __init__(self, rows: int) -> None:
        self.first = np.full(rows, rows, dtype=np.int64)
        self.last = np.full(rows, -1, dtype=np.int64)
        self.lock = threading.Lock()

    def add(self, named: npt.NDArray[np.int64], by: npt.NDArray[np.int64]) -> None:
        with self.lock:
            np.minimum.at(self.first, named, by)
            np.maximum.at(self.last, named, by)

    def blocks(
        self, rows: npt.NDArray[np.bool_], blocks: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """The `blocks` holding an entry that names one of `rows`."""
        named = rows & (self.last >= 0)
        size = len(self.first)
        opened = np.bincount(self.first[named], minlength=size + 1)
        closed = np.bincount(self.last[named] + 1, minlength=size + 1)
        held = np.cumsum(opened - closed)[:size] > 0
        return [(a, b) for a, b in blocks if held[a:b].any()]


def _odd_pairs(
    chrom: str, odd: npt.NDArray[np.int64], size: int, what: str, k: _Component
) -> _Problems:
    """One finding naming the pairs in `odd` (keys of `_keys`), if there are any."""
    pair = odd >> _CODE_BITS
    names = [f"({p // size}, {p % size})" for p in pair[:_NAMED].tolist()]
    if len(odd):
        more = f" and {len(odd) - _NAMED:,} more" if len(odd) > _NAMED else ""
        yield f"{k.group}/{chrom}", f"{what}: {', '.join(names)}{more}"


@_scanned("index.symmetric")
class _IndexSymmetric(_Scanned):
    """Each pair's two entries, matched in one bounded pass. Every entry adds a hash of
    (earlier row, later row, code) to its pair's earlier row, the entry in that row to one
    sum and the entry in the later row to the other; a row whose two sums differ is the
    earlier row of a pair held one way only, or with two codes."""

    parts = frozenset({_INDEX.partner, _INDEX.code})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        _sparse_indptr(store, c, _INDEX)
        self.c, self.sums = c, _Sums(c.size)

    def block(self, b: _Block) -> _Problems:
        partner, code, owner = b.get(_INDEX.partner), b.get(_INDEX.code), b.get(_INDEX.owner)
        ok = _valid(partner, self.c.size) & (partner != owner)
        lo, hi = np.minimum(owner, partner)[ok], np.maximum(owner, partner)[ok]
        in_later = (owner > partner)[ok]
        keys = _keys(lo, hi, code[ok], self.c.size)
        self.sums.add(0, lo[~in_later], keys[~in_later])
        self.sums.add(1, lo[in_later], keys[in_later])
        yield from ()

    def end(self) -> _Problems:
        what = "the earlier row of a pair held one way only, or with two codes"
        yield from _flag(f"index/{self.c.name}", self.sums.differ(), 0, what)


@_scanned("index.near")
class _IndexNear(_Scanned):
    """The index's near half is the band, thresholded (spec §5). Each block's entries in a
    pair's earlier row are compared with those Near Band rows at the floor, both ways. An entry
    in a pair's later row must carry the band's code: its key goes to one sum of its pair's
    earlier row, and that row's band pairs at the floor to the other. At the chromosome's end,
    only the blocks holding an entry that names a row whose sums differ are read again, and
    only those entries are checked, each against the band's code for its pair. That such an
    entry exists at all is index.symmetric's to say."""

    parts = frozenset({"band", _INDEX.partner, _INDEX.code})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.store, self.c = store, c
        self.pos = store.positions(c.name, c.start, c.stop)
        self.near_bp, self.floor = store.integer("near_bp"), store.integer("far_code_floor")
        self.band = store.indptr(f"near/{c.name}/indptr", c.size)
        self.index = _sparse_indptr(store, c, _INDEX)
        self.sums, self.named = _Sums(c.size), _Named(c.size)

    def _halves(
        self, owner: npt.NDArray[np.int64], partner: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """The entries within near_bp held in a pair's earlier row, and in its later row."""
        ok = _valid(partner, self.c.size)
        apart = self.pos[np.where(ok, partner, owner)] - self.pos[owner]
        close = ok & within(apart, self.near_bp)
        return close & (partner > owner), close & (partner < owner)

    def _kept(self, b: _Block) -> tuple[npt.NDArray[np.int64], ...]:
        """The block's Near Band pairs with |code| ≥ the floor: earlier row, later row, code."""
        codes = b.get("band")
        at = np.flatnonzero((codes >= self.floor) | (codes <= -self.floor))
        rows = np.arange(b.first, b.last, dtype=np.int64)
        row, later = near_pair_of(self.band[b.first : b.last + 1], at + b.at["band"])
        return rows[row], later + b.first, codes[at].astype(np.int64)

    def block(self, b: _Block) -> _Problems:
        lo, hi, code = self._kept(b)
        partner, held, owner = b.get(_INDEX.partner), b.get(_INDEX.code), b.get(_INDEX.owner)
        in_earlier, in_later = self._halves(owner, partner)
        size = self.c.size
        band = _keys(lo, hi, code, size)
        entries = _keys(owner[in_earlier], partner[in_earlier], held[in_earlier], size)
        if not np.array_equal(band, entries):  # both ascend in a store that passes
            odd = np.setxor1d(band, entries)
            what = "not the Near Band's pairs at the far code floor"
            yield from _odd_pairs(self.c.name, odd, size, what, _INDEX)
        self.sums.add(0, lo, band)
        up, down = partner[in_later], owner[in_later]
        self.sums.add(1, up, _keys(up, down, held[in_later], size))
        self.named.add(up, down)

    def end(self) -> _Problems:
        rows = self.sums.differ()
        if not rows.any():
            return

        def recheck(block: tuple[int, int]) -> list[tuple[str, str]]:
            b = _read_block(self.store, self.c, frozenset({_INDEX.partner, _INDEX.code}), block)
            return list(self._recheck(rows, b))

        blocks = self.named.blocks(rows, self.c.blocks())
        for problems in _in_order(recheck, blocks, self.store.threads):
            yield from problems

    def _recheck(self, rows: npt.NDArray[np.bool_], b: _Block) -> _Problems:
        """The block's entries in a pair's later row whose earlier row is one of `rows`,
        each against the band's code for its pair."""
        partner, held, owner = b.get(_INDEX.partner), b.get(_INDEX.code), b.get(_INDEX.owner)
        _, in_later = self._halves(owner, partner)
        in_later[in_later] = rows[partner[in_later]]
        up, down, code = partner[in_later], owner[in_later], held[in_later]
        band = self.store.pick(f"near/{self.c.name}/r", near_address(self.band, up, down))
        differ = band.astype(np.int64) != code
        odd = _keys(up[differ], down[differ], code[differ], self.c.size)
        what = "entries whose code is not the Near Band's"
        yield from _odd_pairs(self.c.name, odd, self.c.size, what, _INDEX)


# ------------------------------------------------------------------ the Far Pairs


@_scanned("far.index")
class _FarIndex(_Scanned):
    """The Far Pairs are the index's entries beyond near_bp in their pairs' earlier rows, pair
    for pair and code for code (spec §5; ADR 0007). Both are read by the same block of rows, so
    each block's Far Pairs are compared with that block's entries, both ways, and nothing is
    carried. That the index holds each such pair in its later row too is index.symmetric's to
    say."""

    parts = frozenset({_INDEX.partner, _INDEX.code, _FAR.partner, _FAR.code})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.c, self.near_bp = c, store.integer("near_bp")
        _sparse_indptr(store, c, _INDEX)
        _sparse_indptr(store, c, _FAR)
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        partner, code, owner = b.get(_INDEX.partner), b.get(_INDEX.code), b.get(_INDEX.owner)
        size = self.c.size
        later = _valid(partner, size) & (partner > owner)
        gap = self.pos[np.where(later, partner, owner)] - self.pos[owner]
        beyond = later & ~within(gap, self.near_bp)
        entries = _keys(owner[beyond], partner[beyond], code[beyond], size)
        held = _keys(b.get(_FAR.owner), b.get(_FAR.partner), b.get(_FAR.code), size)
        if not np.array_equal(entries, held):  # both ascend in a store that passes
            odd = np.setxor1d(entries, held)
            what = "not the Proxy Index's pairs beyond near_bp"
            yield from _odd_pairs(self.c.name, odd, size, what, _FAR)

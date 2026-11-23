"""The spec's validation rules, by name, on write and on demand (ld-store §12; ogld#4).

Each rule yields where and how one opened hierarchy and its provenance break it. `validate`
runs all 30, in the spec's order, on a store at a path or URL; `StoreWriter.finish` runs them
on the store it wrote, before the provenance that makes it readable goes in. A rule never
trusts what it does not check: when what it needs is missing or malformed it reports that it
cannot be checked, and when an array will not decode it reports that it cannot be read, each
as a finding, rather than pass or stop.

Every rule reads in bounded windows, through Zarr directly: a chromosome's positions and
`indptr` arrays whole, as a reader keeps them (ld-store §10); a Near Band `_WINDOW` codes, the Far
Pairs an eighth as many entries, and the variant columns `_ROWS` rows at a time. Seventeen rules
are generators over the whole store. The fifteen that read the band, the Far Pairs and the
Partner Rows element by element (`_SCANNED`) share one pass over each chromosome instead: each
block of rows is read once, on the threads the caller gives, and handed to all fifteen. The
Partner Rows are frames, not `indptr`-delimited arrays, so a block they join is whole frames —
at most `_FRAMES_PER_BLOCK` of them — and its entries are one `decode_frame` of each
(partner-rows §5.2, §7).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, TypeGuard

import numpy as np
import numpy.typing as npt
import zarr
from zarr.core.sync import sync
from zarr.storage import LocalStore

from opengwasdb.ld.format.encoding import FAR_BP, NEAR_BP, T_FAR_Q, Codes, REncoding, t_far
from opengwasdb.ld.format.geometry import (
    blocks,
    near_address,
    near_pair_of,
    near_row_lengths,
    owners,
    windows,
    within,
)
from opengwasdb.ld.format.provenance import (
    READABLE_FORMATS,
    VALIDATION_RULES,
    claims_partner_rows,
    release_of,
)
from opengwasdb.ld.format.semantics import InputError
from opengwasdb.ld.identity import IdentityError, alid_locus, chromosome_rank, orient
from opengwasdb.ld.store import layout
from opengwasdb.ld.store.frames import Frame, FrameCache, PartnerFrames, open_partner_frames
from opengwasdb.ld.store.layout import chromosomes_listed, never_data, shown_fill
from opengwasdb.ld.store.location import frame_source, open_group

#: Elements of a Near Band a rule reads at once: 8 Mi codes are 16 MiB.
_WINDOW: Final = 1 << 23
#: A block of rows holds at most `_WINDOW >> _ENTRY_SHIFT` Far Pairs, 1 Mi. The rules work on an
#: entry through about 100 bytes of int64 arrays and on a band code through about 6, so a
#: block's working arrays stay near 150 MiB however dense the component is (ld-store §12).
_ENTRY_SHIFT: Final = 3
#: Rows of the variant columns a rule reads at once.
_ROWS: Final = 1 << 20
#: Frames of one chromosome's Partner Rows a block holds at most: 4,096 rows at G = 16. A frame
#: is the component's unit of reading — one range request and one decompress (partner-rows §5.4)
#: — so a block's Partner Rows payload is bounded by the frames it holds, whatever the store's
#: rows look like. The band's limit binds first on a chromosome whose rows are dense.
_FRAMES_PER_BLOCK: Final = 256
#: Partner Rows entries a block holds at most. A later row holds an entry for every earlier
#: partner within far_bp, so the entries a block carries are not bounded by its own band codes:
#: a few thousand rows at the end of a chromosome whose pairs are all in LD carry every pair
#: before them (ld-store §12). A frame is the component's unit of reading, so a block holds one frame
#: at least. 524,288 entries cost the ten rules about 42 MB at their worst (measured), which keeps
#: a block's working arrays near the 150 MiB the band's own limit is sized for.
_PARTNER_ENTRIES: Final = 1 << 19
#: Offending elements a finding names; it counts the rest.
_NAMED: Final = 5

#: The rules a store's Partner Rows must pass before its provenance claims them (partner-rows
#: §10, §6.2): the nine `partners.*` rules and the threshold they are judged at. `ogld
#: derive-partners` runs these alone, because the band and the Far Pairs of a store it derives
#: from were judged when it was built and re-reading them would cost a whole validation per run
#: (ogld#69).
PARTNER_RULES: Final = tuple(
    rule
    for rule in VALIDATION_RULES
    if rule.startswith("partners.") or rule == "provenance.partner_threshold"
)


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
        super().__init__(f"the LD Store breaks {len(failing)} rule(s) of ld-store §12: {named}")
        self.findings = findings


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Every rule of `opengwasdb.ld.format.VALIDATION_RULES` against the store at `location`.

    The result has one key per rule, in the spec's order, whose list is empty when the
    store passes it. Each rule reads in bounded windows and never loads a whole Near
    Band (ld-store §12; ogld#4). A location with no Zarr group is an `InputError`.
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
    With `threads` above 1, the pass over the Near Band, the Far Pairs and the Partner Rows
    reads and checks that many blocks at once, and holds no more (ld-store §12)."""
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
        self._nodes: dict[str, zarr.Array[Any] | zarr.Group | None] = {}

    def value(self, key: str) -> Any:
        """The provenance value at dotted `key`, or `_Unchecked`."""
        if not isinstance(self.provenance, dict):
            raise _Unchecked("ld_store", "the root group has no ld_store provenance (ld-store §7)")
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
        """The node at `path`, or None; one whose metadata will not parse is `_Unreadable`.
        Each is opened once: a lookup is a metadata read, and the rules ask for most of them
        more than once."""
        if path not in self._nodes:
            try:
                self._nodes[path] = self.root.get(path)
            except _READ_ERRORS as error:
                raise _Unreadable(path, f"{type(error).__name__}: {error}") from error
        return self._nodes[path]

    def find(self, path: str) -> zarr.Array[Any] | None:
        """The array at `path`, or None."""
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

    def partners_judged(self) -> bool:
        """Whether this store's Partner Rows are judged: its provenance claims them, which a store
        of format 0.2.0 or later does and no other can (ld-store §13; partner-rows §6.2).

        A 0.1.0 store that carries the component anyway is an **unclaimed** one: a derivation
        killed between making `partners/` a node of the hierarchy and writing the provenance that
        claims it. Nothing may read those frames — a reader refuses a store whose provenance has
        no `t_partner` — so judging them here would fail a store no one can use and complete
        nothing; the device that completes or rebuilds it is `ogld derive-partners` (ogld#69).
        """
        return claims_partner_rows(self.provenance)

    def positions(self, name: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """One chromosome's positions, kept as a reader keeps them (ld-store §10)."""
        key = f"position/{name}"
        if key not in self._kept:
            self._kept[key] = self.read("variants/position", start, stop).astype(np.int64)
        return self._kept[key]

    def indptr(self, path: str, rows: int) -> npt.NDArray[np.int64]:
        """An `indptr` array that delimits `rows` rows; one that cannot is `_Unchecked`."""
        if path not in self._kept:
            values = self.read(path).astype(np.int64)
            for _, problem in _indptr_problems(path, values, rows):
                raise _Unchecked(path, f"{path} {problem} (near.rows, far.rows)")
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
    """A sparse component of each chromosome, `partner` and `r` delimited by an `indptr` (ld-store
    §5): its group, the names the pass gives its two parts, and the name it gives the row that
    holds each entry. A rule names the component it checks; none is assumed."""

    group: str
    partner: str
    code: str
    owner: str

    def path(self, chrom: str, array: str) -> str:
        return f"{self.group}/{chrom}/{array}"


_FAR: Final = _Component("far", "far_partner", "far_code", "far_owner")
#: The store's sparse components: the Far Pairs (ld-store §5). The Proxy Index was the other, and
#: #72 removed it. The Partner Rows are named for the findings that compare their entries with
#: the Near Band's and the Far Pairs' (partner-rows §10). Their entries are frames, not
#: `indptr`-delimited arrays, so they are no part of `_COMPONENTS`.
_COMPONENTS: Final = (_FAR,)
_PARTNERS: Final = _Component("partners", "partner", "code", "row")

_VARIANT_DTYPES: Final = {
    "position": "int32",
    "alid": "string",
    "a1": "string",
    "a2": "string",
    "a1_frequency": "float32",
}

#: The ALID search index's two arrays (ld-store §1, §6; ADR 0096). The keys are bytes, not the
#: variable-length strings `variants/alid` holds, so a search compares them as they lie.
_ALID_INDEX_DTYPES: Final = {
    "alid": "uint8",
    "row": "int32",
}


def _shape(store: _Store, path: str, dtype: str, length: int | None) -> _Problems:
    array = store.find(path)
    if array is None:
        yield path, "missing (ld-store §1)"
        return
    found = array.metadata.to_dict()["data_type"]
    if found != dtype:
        yield path, f"its dtype is {found}, not {dtype} (ld-store §1)"
    if length is not None and array.shape != (length,):
        yield path, f"its shape is {array.shape}, not ({length},) (ld-store §2, §4, §5)"


def _last(store: _Store, path: str) -> int | None:
    array = store.find(path)
    if array is None or not array.shape or not array.shape[0]:
        return None
    return int(store.read(path, array.shape[0] - 1)[0])


def _arrays(store: _Store) -> Iterator[str]:
    """The path of every array ld-store §1 names, for the store's chromosomes."""
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
    """The ALID index's two arrays (ld-store §6, §1): its keys are 32 bytes an entry, and it has as
    many entries as its `row` array holds. `alid_index.count` holds their lengths to the Variant
    Set's."""
    held = store.find(layout.ALID_INDEX_ROW)
    keys = held.shape[0] * layout.ALID_INDEX_WIDTH if held is not None else None
    yield from _shape(store, f"{layout.ALID_INDEX}/alid", _ALID_INDEX_DTYPES["alid"], keys)
    yield from _shape(store, f"{layout.ALID_INDEX}/row", _ALID_INDEX_DTYPES["row"], None)


@_rule("envelope.arrays")
def _envelope_arrays(store: _Store) -> _Problems:
    if store.root.metadata.zarr_format != 3:
        yield "/", f"the hierarchy is Zarr v{store.root.metadata.zarr_format}, not v3 (ld-store §1)"
        return
    ranges = store.chromosomes()
    size = ranges[-1][2] if ranges else 0
    for name, dtype in _VARIANT_DTYPES.items():
        yield from _shape(store, f"variants/{name}", dtype, size)
    yield from _index_shapes(store)
    yield from _counted(store, size)
    # The container's dtype: the same for every admitted encoding (ld-store §1, §3; ADR 0090).
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
                    "chunk it lost would read as data (ld-store §6)"
                ),
            )


# ------------------------------------------------------------------ provenance

#: Every key ld-store §7 names, required ones and the keys inside them, with their types.
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
    "counts.alids_left_out": "int",
    "software": "object",
    "software.ogld": "str",
    "software.numpy": "str",
    "software.zarr": "str",
    "software.zstd": "str",
    "build.created_at": "str",
    "build.wall_seconds": "number",
    "build.peak_memory_bytes": "int",
    "build.threads": "int",
}

#: The keys a format version adds, and the version that added them: required from it on, and
#: checked whenever they are there. A 0.1.0 store has no Partner Rows, so it has none of their
#: keys, and an older one that carries one anyway is still held to its type (ld-store §7, §13).
_VERSIONED: Final = {
    "t_partner": "0.2.0",
    "partner_code_floor": "0.2.0",
    "counts.partner_entries": "0.2.0",
    "software.zstd": "0.2.0",
}

#: Each entry of the PC Check's `scan`, one per PC of the PC File (ld-store §7).
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

#: The quantiles `pc_diagnostic.near_100kb_change` holds, by name (ld-store §7).
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
        return "null, which ld-store §7 allows nowhere for this key"
    if only == "import" and context.source != "ld_matrix":
        return "null, which ld-store §7 allows only for an ld_matrix source"
    if only == "unmeasured" and context.n_pairs != 0:
        return "null, which ld-store §7 allows only when pc_diagnostic.pairs.n_pairs is 0"
    return None


def _key_problem(kind: str, value: object, context: _Context) -> str | None:
    base, nullable, only = kind.partition("?")
    if value is _MISSING:
        return "missing (ld-store §7)"
    if value is None:
        return _null_problem(nullable, only, context)
    if not _TYPES[base](value):
        return f"{value!r:.80} is not of type {base} (ld-store §7)"
    if only == "unmeasured" and context.n_pairs == 0:
        return (
            f"{value!r}, but pc_diagnostic.pairs.n_pairs is 0, so nothing was measured: it is "
            "null (ld-store §7)"
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


def _release(version: object) -> tuple[int, ...] | None:
    """A `MAJOR.MINOR.PATCH` version as numbers, or None for one that is not that (§13)."""
    return release_of(version)


def _added_later(key: str, version: object) -> bool:
    """Whether `key` joined §7 after the recorded `version`, so a store of that version has none.

    A version this package cannot read is not excused a key: the keys are all required, and
    `provenance.encoding` reports the version itself (ld-store §13).
    """
    added, told = _VERSIONED.get(key), _release(version)
    return added is not None and told is not None and told < (_release(added) or told)


@_rule("provenance.complete")
def _provenance_complete(store: _Store) -> _Problems:
    provenance = store.provenance
    if not isinstance(provenance, dict):
        yield "ld_store", "the root group has no ld_store provenance object (ld-store §7)"
        return
    context = _Context(
        _lookup(provenance, "source.kind"), _lookup(provenance, "pc_diagnostic.pairs.n_pairs")
    )
    version = _lookup(provenance, "format_version")
    for key, kind in _SCHEMA.items():
        if _lookup(provenance, key) is _MISSING and _added_later(key, version):
            continue
        if not _within_null(provenance, key):
            problem = _key_problem(kind, _lookup(provenance, key), context)
            if problem is not None:
                yield key, problem
    yield from _scan_problems(provenance)
    yield from _quantile_problems(provenance)


@_rule("provenance.encoding")
def _provenance_encoding(store: _Store) -> _Problems:
    """The format version is the one this package reads, and the encoding one it implements,
    `REncoding.of` (ld-store §3, §13). `ogld info` and every reader refuse a store with any other."""
    version = store.value("format_version")
    if version not in READABLE_FORMATS:
        yield (
            "format_version",
            f"{version!r}: this ogld reads {' and '.join(READABLE_FORMATS)} only (ld-store §13)",
        )
    try:
        REncoding.of(store.value("encoding"))
    except InputError as unimplemented:
        yield "encoding", str(unimplemented)


@_rule("provenance.distances")
def _provenance_distances(store: _Store) -> _Problems:
    near, far = store.integer("near_bp"), store.integer("far_bp")
    if near != NEAR_BP:
        yield "near_bp", f"{near:,}, not {NEAR_BP:,} (ADR 0087)"
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
                f"k_target {k_target} (ld-store §7)"
            ),
        )


def _k_without_pc_file(adjustment: dict[str, Any], k: int, source: object) -> _Problems:
    """No PC File was read: k_min applies to nothing, and a build is unadjusted."""
    if adjustment.get("k_min") is not None:
        yield (
            "adjustment.k_min",
            (
                f"{adjustment['k_min']}, but no PC File was read, so k_min applies to nothing and "
                "is null (ld-store §7)"
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
        yield "t_far_q", f"{q!r}, not {T_FAR_Q} (ADR 0087)"
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


@_rule("provenance.partner_threshold")
def _provenance_partner_threshold(store: _Store) -> _Problems:
    """t_partner is never above t_far, and `partner_code_floor` is its floor at the recorded
    scale (partner-rows §5.3, §10; ld-store §3, §7). A store of a version without Partner Rows has
    no such keys, and nothing to judge (ld-store §13)."""
    if not store.partners_judged():
        return
    t_partner, far = store.value("t_partner"), store.value("t_far")
    if not _is_number(t_partner):
        raise _Unchecked("t_partner", f"provenance t_partner is {t_partner!r}, not a number")
    if not _is_number(far):
        raise _Unchecked("t_far", f"provenance t_far is {far!r}, not a number")
    encoding = store.encoding()
    if not 0.0 <= t_partner <= 1.0:
        yield "t_partner", f"{t_partner!r} is not an r² in [0, 1] (partner-rows §3)"
        return
    if t_partner > far:
        yield (
            "t_partner",
            (
                f"{t_partner!r} is above t_far {far!r}: the Partner Threshold is never above the "
                "Far Threshold, or a pair between the two within near_bp is in neither component "
                "(ld-store §3; partner-rows §3)"
            ),
        )
    floor = store.integer("partner_code_floor")
    expected = encoding.far_code_floor(t_partner)
    if floor != expected:
        yield (
            "partner_code_floor",
            f"{floor}, but t_partner {t_partner!r} at × {encoding.scale} gives {expected}",
        )


# ------------------------------------------------------------------ variants


@_rule("variants.chromosomes")
def _variants_chromosomes(store: _Store) -> _Problems:
    ranges = store.chromosomes()
    expected, previous = 0, 0
    for name, start, stop in ranges:
        if chromosome_rank(name) <= previous:
            yield "variants", f"chromosome {name} is out of chromosome order (ld-store §2)"
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
    rather than failing on an array too short for the entries it names (ld-store §12; ogld#42's fix
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
    """Rows [lo, hi)'s ALIDs, and whether each fits the index's slot (ld-store §2; ADR 0096)."""
    alids = _strings(store, "alid", lo, hi)
    return alids, layout.alid_fits(alids)


@_rule("alid_index.rows")
def _alid_index_rows(store: _Store) -> _Problems:
    """One entry a variant: each names a row of the Variant Set, none of them twice, and every
    ALID that fits the slot is named by one (ld-store §2, §6; ADR 0096)."""
    path = layout.ALID_INDEX_ROW
    rows = _index_rows(store)
    size = store.array("variants/position").shape[0]
    yield from _flag(path, (rows < 0) | (rows >= size), 0, "is not a row of the Variant Set")
    at = np.argsort(rows, kind="stable")
    twice = rows[at[1:]][rows[at[1:]] == rows[at[:-1]]]
    if len(twice):
        named = ", ".join(str(row) for row in twice[:_NAMED])
        more = f" and {len(twice) - _NAMED:,} more" if len(twice) > _NAMED else ""
        yield path, f"names row(s) {named}{more} more than once: one entry a variant (ld-store §2)"
    by_row = _entries_by_row(store)
    for lo, hi in windows(0, size, _ROWS):
        _, fits = _table_alids(store, lo, hi)
        yield from _flag(
            "variants/alid", fits & (by_row[lo:hi] < 0), lo, "has no entry in the ALID index"
        )


@_rule("alid_index.order")
def _alid_index_order(store: _Store) -> _Problems:
    """The keys ascend bytewise: the order every search over them depends on (ld-store §2; ADR 0096)."""
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
    answering to another's name is what the width exists to prevent (ld-store §2; ADR 0096;
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
    ALIDs wider than the slot (ld-store §2, §6, §7; ADR 0096)."""
    width = layout.ALID_INDEX_WIDTH
    size = store.array("variants/position").shape[0]
    entries = int(store.array(layout.ALID_INDEX_ROW).shape[0])
    held = int(store.array(layout.ALID_INDEX_ALID).shape[0])
    if held != entries * width:
        yield (
            layout.ALID_INDEX_ALID,
            f"holds {held:,} bytes, not {entries:,} keys of {width}: one key an entry (ld-store §6)",
        )
    wide = sum(int((~_table_alids(store, lo, hi)[1]).sum()) for lo, hi in windows(0, size, _ROWS))
    recorded = store.integer("counts.alids_left_out")
    if recorded != wide:
        yield (
            "counts.alids_left_out",
            f"records {recorded:,} left out, but {wide:,} of the {size:,} ALIDs are wider than "
            "the slot: the count is the table's (ld-store §7)",
        )
    yield from _alid_mmap(store)


def _alid_mmap(store: _Store) -> _Problems:
    """Optional local bytes must exactly reproduce the canonical arrays (ADR 0098)."""
    try:
        companion = layout.alid_companion(store.root)
    except InputError as invalid:
        yield layout.ALID_MMAP, str(invalid)
        return
    if companion is None:
        return
    yield from _alid_mmap_contents(store, companion.path, companion.count)


def require_alid_mmap(root: zarr.Group, directory: Path, count: int) -> None:
    """Refuse publication unless these unpublished payloads exactly match canonical arrays.

    Publication and `alid_index.count` use one check; lookup deliberately checks integrity
    and content binding only, under the trusted-producer model (ADR 0098).
    """
    store = _Store(root, root.attrs.get("ld_store"))
    try:
        problems = list(_alid_mmap_contents(store, directory, count))
    except _Unchecked as stopped:
        raise layout.alid_mmap_error(
            directory / layout.ALID_MMAP_MANIFEST, f"{stopped.where}: {stopped}"
        ) from stopped
    if problems:
        raise InputError(problems[0][1])


def _alid_mmap_contents(store: _Store, directory: Path, count: int) -> _Problems:
    entries = int(store.array(layout.ALID_INDEX_ROW).shape[0])
    if count != entries:
        yield (
            layout.ALID_MMAP,
            str(
                layout.alid_mmap_error(
                    directory / layout.ALID_MMAP_MANIFEST,
                    "entry count differs from canonical index",
                )
            ),
        )
        return
    for payload in layout.ALID_MMAP_PAYLOADS:
        with (directory / payload.name).open("rb") as stream:
            for lo, hi in windows(0, entries, _ROWS):
                canonical = store.read(
                    payload.source, lo * payload.width, hi * payload.width
                ).astype(payload.dtype, copy=False)
                if stream.read(canonical.nbytes) != canonical.tobytes():
                    yield (
                        layout.ALID_MMAP,
                        str(
                            layout.alid_mmap_error(
                                directory / payload.name,
                                f"differs from canonical index at entry {lo}",
                            )
                        ),
                    )
                    break
            else:
                if stream.read(1):
                    yield (
                        layout.ALID_MMAP,
                        str(
                            layout.alid_mmap_error(
                                directory / payload.name, "trailing payload bytes"
                            )
                        ),
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
            yield from _flag(path, bad, 0, "a row whose length is not ld-store §4's")


# ------------------------------------------------------------------ one pass over each chromosome


def _in_order[T, R](work: Callable[[T], R], items: Iterable[T], threads: int) -> Iterator[R]:
    """`work` of each item, in the items' order, on `threads` threads. At most `threads` items
    are begun and not yet taken: a full queue is drained before the next item is asked for, so a
    pass holds that many blocks at once and no more — the planner's own item included, which is
    why the item is taken only where there is room for it (ld-store §12)."""
    if threads == 1:
        yield from map(work, items)
        return
    with ThreadPoolExecutor(threads, thread_name_prefix="ogld-validate") as pool:
        pending: deque[Future[R]] = deque()
        source = iter(items)
        while True:
            if len(pending) == threads:
                yield pending.popleft().result()
                continue
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append(pool.submit(work, item))
        while pending:
            yield pending.popleft().result()


#: The arrays the pass reads in blocks of rows, by the names its rules give them.
_PARTS: Final = {"band": ("near", "r")} | {
    part: (k.group, array)
    for k in _COMPONENTS
    for part, array in ((k.partner, "partner"), (k.code, "r"))
}


class _Frames:
    """One chromosome's Partner Rows as the pass reads them: the group's sizes, and `PartnerFrames`,
    the reader that turns a frame's number into one read and one decompress (partner-rows §7).

    No frame is kept: validation reads each frame once, in passing, so the reader's cache holds
    nothing and a frame is not decoded twice for want of it. `of` refuses a group a rule cannot
    read frames from, naming what it needed; `partners.files` judges the group itself.
    """

    def __init__(
        self,
        chrom: str,
        reader: PartnerFrames,
        rows: int,
        rows_per_frame: int,
        entries: int | None,
    ) -> None:
        self.chrom, self.reader = chrom, reader
        self.rows, self.rows_per_frame, self.entries = rows, rows_per_frame, entries

    def decode(self, number: int) -> Frame:
        """Frame `number`: one read through `location`'s frame source — `os.pread` on a local chunk
        file's kept descriptor — one decompress, and one unshuffle of each plane (partner-rows
        §5.2, §7). Its deltas are decoded here too, so a frame whose rows decode to something no
        row holds is `partners.frames`' finding rather than a fault of the rule that read it."""
        return self.reader.frame(number).decode()

    def close(self) -> None:
        """Release the source the frames were read through, a local file's descriptor included."""
        self.reader.close()

    @classmethod
    def of(cls, store: _Store, chrom: str, rows: int) -> _Frames:
        """The group of `chrom`, whose Variant Set has `rows` rows, or `_Unchecked` naming what a
        reader of its frames needs and cannot find. The group's own checks are `PartnerFrames`',
        so a group that cannot be read is refused with the reason it gives."""
        path = layout.partners_path(chrom)
        group = store.node(path)
        if not isinstance(group, zarr.Group):
            raise _Unchecked(path, f"{path} is missing (ld-store §5.1; partners.files)")
        try:
            reader = open_partner_frames(store.root, chrom, FrameCache(0))
        except ValueError as unreadable:
            raise _Unchecked(path, str(unreadable)) from unreadable
        attrs = dict(group.attrs)
        per_frame = attrs.get("rows_per_frame")
        entries = attrs.get("entries")
        assert _is_int(per_frame), "PartnerFrames refuses a group without rows_per_frame"
        assert per_frame >= 1, "PartnerFrames refuses a frame that holds no rows"
        return cls(chrom, reader, rows, int(per_frame), int(entries) if _is_int(entries) else None)


def _frames(store: _Store, chrom: str, rows: int) -> tuple[_Frames | None, _Unchecked | None]:
    """The chromosome's Partner Rows, or None and why a rule cannot read frames from them."""
    try:
        return _Frames.of(store, chrom, rows), None
    except _Unchecked as untold:
        return None, untold


@dataclass(frozen=True)
class _Chromosome:
    """One chromosome's rows [start, stop), with the Near Band's and the Far Pairs' `indptr`
    where each delimits them, and None where it cannot: a rule that needs it says why. `frames`
    is its Partner Rows, or None with `untold` saying why."""

    name: str
    start: int
    stop: int
    band: npt.NDArray[np.int64] | None
    far: npt.NDArray[np.int64] | None
    frames: _Frames | None = None
    untold: _Unchecked | None = None

    @property
    def size(self) -> int:
        return self.stop - self.start

    def path(self, part: str) -> str:
        kind, name = _PARTS[part]
        return f"{kind}/{self.name}/{name}"

    def indptr(self, part: str) -> npt.NDArray[np.int64] | None:
        return {"near": self.band, "far": self.far}[_PARTS[part][0]]

    def blocks(self) -> list[tuple[int, int]]:
        """The rows in blocks of at most `_WINDOW` band codes and an eighth as many Far Pairs."""
        entries = _WINDOW >> _ENTRY_SHIFT
        limits = [(self.band, _WINDOW), (self.far, entries)]
        delimiting = [(indptr, most) for indptr, most in limits if indptr is not None]
        return list(blocks(*delimiting)) if delimiting else []


def _chromosome(
    store: _Store, name: str, start: int, stop: int, partners: bool = False
) -> _Chromosome:
    def delimiting(kind: str) -> npt.NDArray[np.int64] | None:
        try:
            return store.indptr(f"{kind}/{name}/indptr", stop - start)
        except _Unchecked:
            return None

    frames, untold = (
        _frames(store, name, stop - start) if partners and store.partners_judged() else (None, None)
    )
    return _Chromosome(
        name,
        start,
        stop,
        delimiting("near"),
        delimiting("far"),
        frames,
        untold,
    )


@dataclass
class _Decoded:
    """One frame of a block: read and decompressed once, or why it cannot be (partner-rows §5.4).
    Its payload is released once its entries are copied into the block's span, so a block holds
    its entries once and not twice while the span is built."""

    first: int
    frame: Frame | None
    untold: _Unreadable | None

    @property
    def entries(self) -> int:
        """The entries it holds, E; 0 for a frame that could not be read."""
        return self.frame.entries if self.frame is not None else 0


@dataclass(frozen=True)
class _Span:
    """The Partner Rows entries a block holds, one frame after another: each entry's partner, its
    code, the chromosome-local row that holds it, and how many rows a frame holds — an entry's
    frame is its row's, so no frame is kept for each of them (§5.2)."""

    partner: npt.NDArray[np.int32]
    code: Codes
    row: npt.NDArray[np.int64]
    per_frame: int

    def __len__(self) -> int:
        return len(self.partner)


def _decode(c: _Chromosome, number: int) -> _Decoded:
    """Frame `number` of `c`, read and decompressed once, or why it cannot be. Its refusal names
    the chromosome and the frame already, from `frames.py`'s own parse (partner-rows §5.4, §7)."""
    frames = c.frames
    assert frames is not None
    first = number * frames.rows_per_frame
    try:
        return _Decoded(first, frames.decode(number), None)
    except ValueError as unreadable:
        named = f"frame {number} of chromosome {c.name}"
        why = str(unreadable)
        if named not in why:
            why = f"{named}: {why}"
        where = f"{layout.partners_path(c.name)}/{layout.FRAMES}"
        return _Decoded(first, None, _Unreadable(where, why))


def _decoded(c: _Chromosome, rows: tuple[int, int]) -> list[_Decoded]:
    """The frames covering rows [a, b), each decoded once."""
    frames = c.frames
    assert frames is not None
    a, b = rows
    per_frame = frames.rows_per_frame
    return [_decode(c, number) for number in range(a // per_frame, -(-b // per_frame))]


def _plans(
    c: _Chromosome, partners: bool
) -> Iterator[tuple[tuple[int, int], list[_Decoded] | None]]:
    """The pass's blocks: rows, and the frames each holds when `partners`.

    Without the Partner Rows a block is a range of `c.blocks()`. With them a block is planned by
    decoding its frames, because what bounds it is its entries and not its rows: a later row
    holds an entry for every earlier partner within far_bp, so a few thousand rows near a
    chromosome's end carry every pair before them, which the block's own band codes say nothing
    about (ld-store §12). Frames are decoded here, once, on the pass's own thread and handed to the
    rules, so none is decompressed twice; a block holds `_PARTNER_ENTRIES` at most — one frame
    where a frame alone holds more — and ends at the first frame it cannot read.
    """
    if not partners:
        for rows in c.blocks():
            yield rows, None
        return
    frames = c.frames
    assert frames is not None
    per_frame = frames.rows_per_frame
    a: int = 0
    pending: _Decoded | None = None
    for _, stop in c.blocks() or [(0, c.size)]:
        edge = min(c.size, -(-stop // per_frame) * per_frame)
        while a < edge:
            rows, decoded, pending = _frame_block(c, a, edge, pending)
            yield rows, decoded
            a = rows[1]


def _frame_block(
    c: _Chromosome, start: int, edge: int, pending: _Decoded | None
) -> tuple[tuple[int, int], list[_Decoded], _Decoded | None]:
    """One block of whole frames from row `start` (a frame's edge) up to `edge`: it ends at
    `_PARTNER_ENTRIES` entries, at `_FRAMES_PER_BLOCK` frames, or at the first frame it cannot
    read. The frame it stopped before is returned, not dropped, so the next block decodes it
    once."""
    frames = c.frames
    assert frames is not None
    per_frame = frames.rows_per_frame
    a, entries = start, 0
    decoded: list[_Decoded] = []
    while a < edge and len(decoded) < _FRAMES_PER_BLOCK:
        one = pending if pending is not None else _decode(c, a // per_frame)
        pending = None
        if decoded and entries + one.entries > _PARTNER_ENTRIES:
            return (start, a), decoded, one
        decoded.append(one)
        entries += one.entries
        a = min(c.size, a + per_frame)
        if one.untold is not None:
            break
    return (start, a), decoded, None


def _span(frames: list[_Decoded], per_frame: int) -> _Span | _Unchecked:
    """The entries of `frames` back to back, or why the block's frames cannot be read: a rule
    that needs the entries says it cannot be checked rather than read a block short."""
    unread = [one.untold for one in frames if one.untold is not None]
    if unread:
        return unread[0]
    return _entries(frames, per_frame)


def _entries(frames: list[_Decoded], per_frame: int) -> _Span:
    """The frames' entries, one frame after another: each frame's arrays are copied into the span
    and the frame released, so a block holds its entries once, not once as frames beside their
    span — at 20 bytes an entry while it is built rather than 34."""
    partner = np.empty(sum(one.entries for one in frames), np.int32)
    code = np.empty(len(partner), REncoding.dtype)
    row = np.empty(len(partner), np.int64)
    at = 0
    for one in frames:
        frame = one.frame
        if frame is None:
            continue
        stop = at + frame.entries
        partner[at:stop] = frame.partners
        code[at:stop] = frame.codes
        lengths = np.diff(frame.indptr)
        row[at:stop] = owners(lengths, np.arange(one.first, one.first + frame.rows, dtype=np.int64))
        one.frame = None
        at = stop
    return _Span(partner, code, row, per_frame)


@dataclass(frozen=True)
class _Block:
    """Rows [first, last) of one chromosome as the pass read them: each part's elements, from
    offset `at[part]`, or why they cannot be read; with the Far Pairs', "far_owner", the row
    that holds each entry. `frames` is the block's Partner Rows
    frames as decoded (the rule that judges frames takes them), and `entries` the whole span's
    entries, or why they cannot be read."""

    first: int
    last: int
    at: dict[str, int]
    held: dict[str, npt.NDArray[Any] | _Unchecked]
    frames: list[_Decoded] | None = None
    entries: _Span | _Unchecked | None = None

    def get(self, part: str) -> npt.NDArray[Any]:
        found = self.held[part]
        if isinstance(found, _Unchecked):
            raise found
        return found

    def span(self) -> _Span:
        """The block's Partner Rows entries, or why they cannot be read."""
        found = self.entries
        assert found is not None, "a rule takes the Partner Rows only where the block holds them"
        if isinstance(found, _Unchecked):
            raise found
        return found


def _part(store: _Store, path: str, lo: int, hi: int) -> npt.NDArray[Any] | _Unchecked:
    """Elements [lo, hi) of the array at `path`, the Far Pairs' as int64; or why
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
    store: _Store,
    c: _Chromosome,
    parts: frozenset[str],
    rows: tuple[int, int],
    partners: bool = False,
    decoded: list[_Decoded] | None = None,
) -> _Block:
    """Rows [a, b) of `c`, each of `parts` read once, and the block's Partner Rows frames —
    `decoded` when the pass planned them, and decoded here otherwise — when `partners`; this
    runs on the pass's threads."""
    a, b = rows
    at, held = _parts_read(store, c, parts, rows)
    if not partners:
        return _Block(a, b, at, held)
    frames = _decoded(c, rows) if decoded is None else decoded
    assert c.frames is not None
    return _Block(a, b, at, held, frames, _span(frames, c.frames.rows_per_frame))


def _parts_read(
    store: _Store, c: _Chromosome, parts: frozenset[str], rows: tuple[int, int]
) -> tuple[dict[str, int], dict[str, npt.NDArray[Any] | _Unchecked]]:
    """Each of `parts` of rows [a, b) read once, from the offset its `indptr` gives, with the row
    that holds each of a component's entries."""
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
    return at, held


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
    when it takes no block; `block` runs on the pass's threads, on several blocks at once.
    `partners` says it takes the block's Partner Rows: its frames always, and its entries
    through `_Block.span`."""

    parts: frozenset[str] = frozenset()
    partners: bool = False

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
    wants = any(getattr(_SCANNED[name], "partners", False) for name in names)
    for name, start, stop in ranges:
        ended = _chromosome_pass(store, name, start, stop, live, found, wants)
        live = [rule for rule in live if rule in ended]
    return found


def _chromosome_pass(
    store: _Store,
    name: str,
    start: int,
    stop: int,
    live: list[str],
    found: dict[str, list[Finding]],
    wants: bool,
) -> set[str]:
    """One chromosome's pass, with the source its frames were read through released after it:
    the descriptor a local store keeps open is not left to a later chromosome."""
    c = _chromosome(store, name, start, stop, wants)
    try:
        return _pass(store, c, live, found)
    finally:
        if c.frames is not None:
            c.frames.close()


def _pass(
    store: _Store, c: _Chromosome, names: list[str], found: dict[str, list[Finding]]
) -> set[str]:
    """One chromosome's pass: each rule built and begun, fed every block, and ended. Returns
    the rules that ended; the others stopped, and are fed nothing more."""
    begun = _begun(store, c, names, found)
    fed = {name: rule for name, rule in begun.items() if rule.parts or rule.partners}
    stopped = _feed(store, c, fed, found)
    live = [(name, rule) for name, rule in begun.items() if name not in stopped]
    return {name for name, rule in live if _record(found, name, _outcome(rule.end))}


def _begun(
    store: _Store, c: _Chromosome, names: list[str], found: dict[str, list[Finding]]
) -> dict[str, _Scanned]:
    """Each rule of `names` built for `c` and begun; a rule that cannot be is recorded. A rule
    that reads the Partner Rows' frames is not built for a chromosome whose frames cannot be
    read: it says why it cannot be checked, as one whose prerequisite is missing does."""
    begun: dict[str, _Scanned] = {}
    for name in names:
        reads_frames = getattr(_SCANNED[name], "partners", False)
        if reads_frames and store.partners_judged() and c.frames is None:
            assert c.untold is not None
            found[name].append(_finding(name, c.untold))
            continue
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
    partners = any(rule.partners for rule in fed.values())
    for part in parts:  # opened here, once, rather than by each thread
        with suppress(_Unchecked):
            store.find(c.path(part))

    work = partial(_blocked, store, c, parts, partners, fed, stopped)
    for outcomes in _in_order(work, _plans(c, partners) if fed else (), store.threads):
        stopped |= _recorded(outcomes, found, stopped)
    return stopped


def _recorded(
    outcomes: dict[str, _Outcome], found: dict[str, list[Finding]], stopped: set[str]
) -> set[str]:
    """Records each rule's outcome, and returns the rules that stopped over it."""
    halted: set[str] = set()
    for name, outcome in outcomes.items():
        if name not in stopped and not _record(found, name, outcome):
            halted.add(name)
    return halted


def _blocked(
    store: _Store,
    c: _Chromosome,
    parts: frozenset[str],
    partners: bool,
    fed: dict[str, _Scanned],
    stopped: set[str],
    plan: tuple[tuple[int, int], list[_Decoded] | None],
) -> dict[str, _Outcome]:
    """One block of rows read once and checked by every rule that is still live; on the pass's
    threads, on several blocks at once."""
    rows, decoded = plan
    b = _read_block(store, c, parts, rows, partners, decoded)
    live = [(name, rule) for name, rule in fed.items() if name not in stopped]
    return {name: _outcome(partial(rule.block, b)) for name, rule in live}


def _valid(partner: npt.NDArray[Any], size: int) -> npt.NDArray[np.bool_]:
    """Entries whose partner is a row of the chromosome: the others are far.rows'."""
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


@_scanned("far.codes")
class _FarCodes(_Codes):
    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c, _FAR.code)


# ------------------------------------------------------------------ the Far Pairs' rows


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


@_scanned("far.rows")
class _FarRows(_Rows):
    """The Far Pairs' partners are each above their row: each pair is in its earlier row."""

    component, misplaced = _FAR, "not above its row"

    def is_misplaced(
        self, partner: npt.NDArray[np.int64], owner: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.bool_]:
        return np.asarray(partner <= owner)


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
    """Every code of a sparse component, the Far Pairs, reaches far_code_floor in
    magnitude."""

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


class _Counted:
    """Per row, how many Partner Rows entries name it as their partner, added to from several
    threads: a row of the Near Band at the Partner Floor has that many entries when the component
    is whole, one in each later row (partner-rows §3)."""

    def __init__(self, rows: int) -> None:
        # A count of none is a count, not an absence: every row has an entry count, and it is 0
        # until an entry names it (ld-store §8's NaN is for an unstored pair's r, which this is not).
        self.counts = np.full(rows, 0, dtype=np.int64)
        self.lock = threading.Lock()

    def add(self, named: npt.NDArray[np.int64]) -> None:
        with self.lock:
            np.add.at(self.counts, named, 1)


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


def _near_halves(
    pos: npt.NDArray[np.int64],
    near_bp: int,
    owner: npt.NDArray[np.int64],
    partner: npt.NDArray[np.int64],
    size: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """The entries within near_bp held in a pair's earlier row, and in its later row."""
    ok = _valid(partner, size)
    apart = pos[np.where(ok, partner, owner)] - pos[owner]
    close = ok & within(apart, near_bp)
    return close & (partner > owner), close & (partner < owner)


def _band_at_floor(
    b: _Block, band: npt.NDArray[np.int64], floor: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The block's Near Band pairs with |code| ≥ `floor`: earlier row, later row, code."""
    codes = b.get("band")
    at = np.flatnonzero((codes >= floor) | (codes <= -floor))
    rows = np.arange(b.first, b.last, dtype=np.int64)
    row, later = near_pair_of(band[b.first : b.last + 1], at + b.at["band"])
    return rows[row], later + b.first, codes[at].astype(np.int64)


# ------------------------------------------------------------------ the Partner Rows
#
# The Partner Rows are frames, not `indptr`-delimited arrays (partner-rows §5.1): the pass decodes
# whole frames per block, and every rule here reads the block's entries — each entry's partner, its
# code, the row that holds it and its frame — through `_Block.span`. A finding names the row and
# the frame, because an entry has no address of its own in an array a reader could look up.


def _flagged(
    where: str,
    bad: npt.NDArray[np.bool_],
    rows: npt.NDArray[np.int64],
    per_frame: int,
    what: str,
) -> _Problems:
    """One finding naming the rows and frames of the entries `bad` marks, if there is any."""
    at = np.flatnonzero(bad)
    if len(at):
        named = ", ".join(
            f"row {row} of frame {row // per_frame}" for row in rows[at[:_NAMED]].tolist()
        )
        more = f" and {len(at) - _NAMED:,} more" if len(at) > _NAMED else ""
        yield where, f"{what} at {named}{more}"


#: The Partner Rows group's attributes, and the type each is recorded as (partner-rows §5.3).
_PARTNER_ATTRS: Final = {
    "rows": "int",
    "rows_per_frame": "int",
    "frames": "int",
    "entries": "int",
    "frames_bytes": "int",
    "frames_sha256": "str",
    "frames_idx_sha256": "str",
    "zstd_level": "int",
}


@_rule("partners.files")
def _partners_files(store: _Store) -> _Problems:
    """Every chromosome of the Variant Set has a `partners/<c>/` group with both arrays, its
    attributes are the sizes §5.3 names, its `frames_idx` is F + 1 ascending offsets from 0 to
    `frames_bytes`, and both chunk files are the ones its hashes name (partner-rows §5.1, §5.3,
    §10). A store whose version has no Partner Rows has nothing to judge (ld-store §13).

    `partners/` is a group of the store's one hierarchy, never a directory beside it (§1, §5.1):
    a 0.2.0 store whose root does not list it is a damaged one even when every chromosome's group
    still resolves by its own path, because a listing of the store would not show its component.
    """
    if not store.partners_judged():
        return
    group = store.node(layout.PARTNERS)
    if isinstance(group, zarr.Group):
        yield from _temporary_entries(layout.PARTNERS, group)
    else:
        yield (
            layout.PARTNERS,
            "is not a group of the store's hierarchy: a 0.2.0 store's Partner Rows are a node of "
            "it, listed by `members()` with both arrays of every chromosome (ld-store §1, §5.1; "
            "partner-rows §5.1)",
        )
    for chrom, start, stop in store.chromosomes():
        yield from _partner_files(store, chrom, stop - start)


async def _children(store: Any, prefix: str) -> list[str]:
    return [name async for name in store.list_dir(prefix)]


def _temporary_entries(path: str, group: zarr.Group) -> _Problems:
    """No entry of `partners/` begins with a dot. A chromosome's group is built under one and
    renamed into place in one step (partner-rows §5.1; `FramesGroup`), so a dot-prefixed entry is
    a group a writer that crashed left behind: half a component, which no store may be read as
    whole beside. The entries are listed by the store, metadata or not: a directory zarr never
    wrote is not a node, but it is still there."""
    try:
        names = sync(_children(group.store_path.store, group.store_path.path))
    except _READ_ERRORS as unread:
        raise _Unreadable(path, f"{type(unread).__name__}: {unread}") from unread
    for name in sorted(names):
        if name.startswith("."):
            yield (
                f"{path}/{name}",
                "is a temporary group a writer left behind, not part of the component: a "
                "chromosome's Partner Rows are published by one rename (partner-rows §5.1)",
            )


def _partner_files(store: _Store, chrom: str, rows: int) -> _Problems:
    path = layout.partners_path(chrom)
    node = store.node(path)
    if not isinstance(node, zarr.Group):
        missing = "missing: every chromosome of the Variant Set has one (ld-store §5.1)"
        yield path, "is not a group (ld-store §5.1)" if node is not None else missing
        return
    attrs = dict(node.attrs)
    yield from _partner_attrs(path, attrs)
    yield from _partner_sizes(path, attrs, rows)
    frames, index = (
        store.find(f"{path}/{layout.FRAMES}"),
        store.find(f"{path}/{layout.FRAMES_INDEX}"),
    )
    length = attrs.get("frames_bytes")
    yield from _frames_array(path, frames, length)
    if index is None:
        yield f"{path}/{layout.FRAMES_INDEX}", "missing (ld-store §5.1)"
    elif (yield from _frames_index(path, index, attrs.get("frames"))):
        where = f"{path}/{layout.FRAMES_INDEX}"
        yield from _chunk_ends(where, index, index.dtype.itemsize * index.shape[0])
        offsets = store.read(where).astype(np.int64)
        yield from _offsets_judged(where, offsets, length)
        yield from _frames_index_hash(path, offsets, attrs.get("frames_idx_sha256"))
    if frames is not None and _is_int(length):
        yield from _chunk_ends(f"{path}/{layout.FRAMES}", frames, length)
        yield from _frames_file_judged(path, frames, length, attrs.get("frames_sha256"))


def _chunk_ends(where: str, array: zarr.Array[Any], expected: int) -> _Problems:
    """The chunk file `array` addresses holds exactly `expected` bytes. The hash checks read the
    declared prefix, so a store that appended bytes to a chunk keeps both its metadata and its
    hash right; the file's own size says otherwise, and a store that cannot be measured is asked
    for one byte past the end (ld-store §5.1, §5.3)."""
    if isinstance(array.store_path.store, LocalStore):
        try:
            size = layout.chunk_file(array).stat().st_size
        except OSError as lost:
            raise _Unreadable(where, f"{type(lost).__name__}: {lost}") from lost
        if size != expected:
            yield (
                where,
                (
                    f"its chunk file holds {size:,} bytes, not the {expected:,} its metadata and "
                    "hash name (ld-store §5.1, §5.3)"
                ),
            )
        return
    source = frame_source(array)
    try:
        beyond = source.read(expected, expected + 1)
    except ValueError as unread:
        raise _Unreadable(where, str(unread)) from unread
    finally:
        source.close()
    if beyond:
        yield (
            where,
            (f"holds bytes past the {expected:,} its metadata and hash name (ld-store §5.1, §5.3)"),
        )


def _partner_attrs(path: str, attrs: dict[str, Any]) -> _Problems:
    """The group records each of §5.3's attributes, of its type."""
    for name, kind in _PARTNER_ATTRS.items():
        if name not in attrs:
            yield path, f"records no {name} (ld-store §5.3)"
        elif not _TYPES[kind](attrs[name]):
            yield path, f"records {name} as {attrs[name]!r}, not of type {kind} (ld-store §5.3)"


def _partner_sizes(path: str, attrs: dict[str, Any], rows: int) -> _Problems:
    """`rows` is n_c and `frames` is ⌈n_c / G⌉, G being `rows_per_frame` (§5.2, §5.3)."""
    if attrs.get("rows") != rows:
        yield path, f"records {attrs.get('rows')!r} rows, but the chromosome has {rows:,} (§5.3)"
    per_frame = attrs.get("rows_per_frame")
    if not (_is_int(per_frame) and per_frame >= 1):
        yield path, f"records rows_per_frame {per_frame!r}, not whole rows of a frame (§5.2)"
        return
    due = -(-rows // per_frame)
    if attrs.get("frames") != due:
        yield (
            path,
            f"records {attrs.get('frames')!r} frames, but {rows:,} rows at {per_frame} a frame "
            f"are {due:,} (ld-store §5.2)",
        )


def _frames_array(path: str, frames: zarr.Array[Any] | None, length: object) -> _Problems:
    """The `frames` array is uint8, one chunk of `frames_bytes` bytes (§5.1, §5.3)."""
    where = f"{path}/{layout.FRAMES}"
    if frames is None:
        yield where, "missing (ld-store §5.1)"
        return
    found = frames.metadata.to_dict()["data_type"]
    if found != "uint8":
        yield where, f"its dtype is {found}, not uint8 (ld-store §5.1)"
    if not _is_int(length):
        yield where, "has no frames_bytes to be as long as (ld-store §5.3)"
    elif frames.shape != (length,):
        yield (
            where,
            f"its shape is {frames.shape}, not ({length:,},), the frames' bytes (§5.1, §5.3)",
        )
    elif frames.chunks != frames.shape:
        yield (
            where,
            f"its chunks are {frames.chunks}, not one chunk of every byte: a frame is one "
            "positional read of the chunk file (ld-store §5.1)",
        )


def _frames_index(
    path: str, index: zarr.Array[Any], count: object
) -> Generator[tuple[str, str], None, bool]:
    """The `frames_idx` array is uint64, one chunk of the F + 1 offsets (§5.1, §5.3), and it
    holds the offsets' own length: whether the rest of the rule can judge them."""
    where = f"{path}/{layout.FRAMES_INDEX}"
    found = index.metadata.to_dict()["data_type"]
    if found != "uint64":
        yield where, f"its dtype is {found}, not uint64 (ld-store §5.1)"
    if not _is_int(count):
        return False
    if index.shape != (count + 1,):
        yield where, f"its shape is {index.shape}, not ({count + 1:,},), F + 1 offsets (§5.3)"
        return False
    if index.chunks != index.shape:
        yield where, f"its chunks are {index.chunks}, not one of its whole offsets (§5.1)"
        return False
    return True


def _frames_index_hash(path: str, offsets: npt.NDArray[np.int64], recorded: object) -> _Problems:
    """The `frames_idx` chunk is the little-endian offsets' bytes, as its hash records them."""
    digest = hashlib.sha256(offsets.astype(layout.FRAMES_INDEX_DTYPE).tobytes()).hexdigest()
    if digest != recorded:
        yield (
            f"{path}/{layout.FRAMES_INDEX}",
            f"hashes to {digest}, not the recorded {recorded!r} (ld-store §5.3)",
        )


def _offsets_judged(where: str, offsets: npt.NDArray[np.int64], length: object) -> _Problems:
    """`offsets` start at 0, ascend strictly and end at the frames' `length` bytes (§5.2, §5.3)."""
    if offsets[0] != 0:
        yield where, f"starts at {offsets[0]:,}, not 0 (ld-store §5.2)"
    descending = np.flatnonzero(np.diff(offsets) <= 0)
    if len(descending):
        more = f" and {len(descending) - _NAMED:,} more" if len(descending) > _NAMED else ""
        named = ", ".join(str(int(at) + 1) for at in descending[:_NAMED])
        yield where, f"does not ascend strictly at offset(s) {named}{more} (ld-store §5.2)"
    if _is_int(length) and offsets[-1] != length:
        yield where, f"ends at {offsets[-1]:,}, but frames_bytes is {length:,} (ld-store §5.3)"


def _frames_file_judged(
    path: str, frames: zarr.Array[Any], length: int, recorded: object
) -> _Problems:
    """The `frames` chunk file is the bytes its recorded hash names, streamed in `_WINDOW`-byte
    reads through the frame source — `os.pread` on a local file's kept descriptor — and ending
    exactly where they do."""
    where = f"{path}/{layout.FRAMES}"
    source = frame_source(frames)
    digest = hashlib.sha256()
    at = 0
    try:
        while at < length:
            hi = min(length, at + _WINDOW)
            raw = source.read(at, hi)
            if len(raw) != hi - at:
                yield (
                    where,
                    (
                        f"holds {at + len(raw):,} bytes, fewer than the {length:,} its metadata "
                        "and hash name (ld-store §5.1, §5.3)"
                    ),
                )
                return
            digest.update(raw)
            at = hi
    except ValueError as unread:
        yield where, f"cannot be read: {unread}"
        return
    finally:
        source.close()
    found = digest.hexdigest()
    if found != recorded:
        yield where, f"hashes to {found}, not the recorded {recorded!r} (ld-store §5.3)"


class _PartnerRule(_Scanned):
    """A rule that reads a chromosome's Partner Rows. A store of a version that has none — a
    0.1.0 store without the component — passes it with nothing to judge (ld-store §13), and takes no
    part of any block; a store that has them is fed the block's frames, and says which one it
    could not read."""

    partners = True

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        self.store, self.c = store, c
        self.judged = store.partners_judged()
        if not self.judged:
            self.partners = False

    @property
    def where(self) -> str:
        """The group a finding about one of its entries names."""
        return layout.partners_path(self.c.name)


@_scanned("partners.frames")
class _PartnerFrames(_PartnerRule):
    """Every frame decompresses with a valid checksum, its k is the rows it is due and its
    payload is exactly 4 + 4k + 6E bytes (partner-rows §5.2, §10). A frame that fails is one
    finding, naming the chromosome and the frame; the other rules that would read it say they
    cannot be read rather than read it short."""

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        for one in b.frames or ():
            if one.untold is not None:
                yield one.untold.where, one.untold.why


@_scanned("partners.rows")
class _PartnerRows(_PartnerRule):
    """Within a row partners ascend strictly, lie in [0, n_c) and are never the row; and the
    frames hold the group's recorded `entries` (partner-rows §5.3, §10)."""

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        self.seen = 0

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        span = b.span()
        self.seen += len(span)
        partner = span.partner.astype(np.int64)
        misplaced = ~_valid(span.partner, self.c.size) | (partner == span.row)
        yield from _flagged(
            self.where,
            misplaced,
            span.row,
            span.per_frame,
            "a partner outside [0, n_c) or the row",
        )
        ascending = (span.row[1:] == span.row[:-1]) & ~(partner[1:] > partner[:-1])
        yield from _flagged(
            self.where,
            ascending,
            span.row[1:],
            span.per_frame,
            "a partner not above the one before it in its row",
        )

    def end(self) -> _Problems:
        if not self.judged:
            return
        frames = self.c.frames
        assert frames is not None
        if frames.entries is None:
            raise _Unchecked(f"{self.where}", "records no entries (ld-store §5.3)")
        if frames.entries != self.seen:
            yield (
                self.where,
                f"records {frames.entries:,} entries, but its frames hold {self.seen:,} "
                "(ld-store §5.3, §10)",
            )


@_scanned("partners.codes")
class _PartnerCodes(_PartnerRule):
    """Every code is one of the store's encoding: none is the reserved code, and |code| ≤
    r_scale, so a dequantised r̂ never leaves [−1, 1] (ld-store §12; partner-rows §10)."""

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        if self.judged:
            self.encoding = store.encoding()

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        span = b.span()
        reserved, scale = self.encoding.reserved_code, self.encoding.scale
        yield from _flagged(
            self.where,
            span.code == reserved,
            span.row,
            span.per_frame,
            f"the reserved code {reserved}",
        )
        past = ~self.encoding.holds(span.code) & (span.code != reserved)
        yield from _flagged(
            self.where, past, span.row, span.per_frame, f"a code past the scale, |code| > {scale}"
        )


def _split_by_distance(
    span: _Span, pos: npt.NDArray[np.int64], size: int, distance: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Each entry whose partner is a row of the chromosome, split into those within `distance`
    bp of their row's own position and those beyond it. An entry whose partner is not a row is
    in neither: that is `partners.rows`' finding."""
    partner = span.partner.astype(np.int64)
    valid = _valid(span.partner, size)
    gap = pos[np.where(valid, partner, span.row)] - pos[span.row]
    close = valid & within(gap, distance)
    return close, valid & ~close


@_scanned("partners.distance")
class _PartnerDistance(_PartnerRule):
    """Every entry's partner lies within far_bp of its row's position (partner-rows §10)."""

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        if not self.judged:
            return
        self.far_bp = store.integer("far_bp")
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        span = b.span()
        _, beyond = _split_by_distance(span, self.pos, self.c.size, self.far_bp)
        yield from _flagged(
            self.where, beyond, span.row, span.per_frame, f"a partner beyond {self.far_bp:,} bp"
        )


@_scanned("partners.threshold")
class _PartnerThreshold(_PartnerRule):
    """Every entry reaches the floor its distance has: |code| ≥ partner_code_floor within
    near_bp and ≥ far_code_floor beyond it (partner-rows §3, §10)."""

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        if not self.judged:
            return
        self.near_bp = store.integer("near_bp")
        self.partner_floor, self.far_floor = (
            store.integer("partner_code_floor"),
            store.integer("far_code_floor"),
        )
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        span = b.span()
        close, _ = _split_by_distance(span, self.pos, self.c.size, self.near_bp)
        floor = np.where(close, self.partner_floor, self.far_floor)
        below = _valid(span.partner, self.c.size) & (np.abs(span.code.astype(np.int64)) < floor)
        what = (
            f"|code| below the floor its distance has: {self.partner_floor} within "
            f"{self.near_bp:,} bp, {self.far_floor} beyond it"
        )
        yield from _flagged(self.where, below, span.row, span.per_frame, what)


@_scanned("partners.symmetric")
class _PartnerSymmetric(_PartnerRule):
    """(l, m, c) is an entry exactly when (m, l, c) is: the 64-bit hash-sum method the retired
    index.symmetric used, every entry adding its pair's hash to the earlier row's sum for its own
    side, and a row whose two sums differ the earlier row of a pair held one way only or with
    two codes (ld-store §12; partner-rows §10)."""

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        self.sums = _Sums(c.size) if self.judged else None

    def block(self, b: _Block) -> _Problems:
        if self.sums is None:
            return
        span = b.span()
        partner = span.partner.astype(np.int64)
        ok = _valid(span.partner, self.c.size) & (partner != span.row)
        lo = np.minimum(span.row, partner)[ok]
        hi = np.maximum(span.row, partner)[ok]
        later = (span.row > partner)[ok]
        keys = _keys(lo, hi, span.code[ok].astype(np.int64), self.c.size)
        self.sums.add(0, lo[~later], keys[~later])
        self.sums.add(1, lo[later], keys[later])
        yield from ()

    def end(self) -> _Problems:
        if self.sums is None:
            return
        what = "the earlier row of a pair held one way only, or with two codes"
        yield from _flag(self.where, self.sums.differ(), 0, what)


@_scanned("partners.near")
class _PartnerNear(_PartnerRule):
    """The Partner Rows' near half is the Near Band at the Partner Floor, both ways: every entry
    within near_bp carries the band's code, and every band code at the Partner Floor is an
    entry — the retired index.near's method at the Partner Threshold's floor (ld-store §12;
    partner-rows §10).
    The recheck at the chromosome's end reads again only the blocks holding an entry that names
    a row whose sums differ, each against the band's code for its pair."""

    parts = frozenset({"band"})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        if not self.judged:
            self.parts = frozenset()
            return
        self.pos = store.positions(c.name, c.start, c.stop)
        self.near_bp = store.integer("near_bp")
        self.floor = store.integer("partner_code_floor")
        self.band = store.indptr(f"near/{c.name}/indptr", c.size)
        self.sums, self.named = _Sums(c.size), _Named(c.size)
        #: The blocks the pass feeds, which the end re-reads: they are the pass's own, so a
        #: re-read holds the entries a block holds and no more.
        self.seen: list[tuple[int, int]] = []

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        self.seen.append((b.first, b.last))
        lo, hi, code = _band_at_floor(b, self.band, self.floor)
        span = b.span()
        size = self.c.size
        in_earlier, in_later = _near_halves(
            self.pos, self.near_bp, span.row, span.partner.astype(np.int64), size
        )
        band = _keys(lo, hi, code, size)
        entries = _keys(
            span.row[in_earlier],
            span.partner[in_earlier].astype(np.int64),
            span.code[in_earlier].astype(np.int64),
            size,
        )
        if not np.array_equal(band, entries):  # both ascend in a store that passes
            odd = np.setxor1d(band, entries)
            what = "not the Near Band's pairs at the partner code floor"
            yield from _odd_pairs(self.c.name, odd, size, what, _PARTNERS)
        self.sums.add(0, lo, band)
        up, down = span.partner[in_later].astype(np.int64), span.row[in_later]
        held = span.code[in_later].astype(np.int64)
        self.sums.add(1, up, _keys(up, down, held, size))
        self.named.add(up, down)

    def end(self) -> _Problems:
        if not self.judged:
            return
        differ = self.sums.differ()
        if not differ.any():
            return
        counted = _Counted(self.c.size)

        def recheck(block: tuple[int, int]) -> list[tuple[str, str]]:
            b = _read_block(self.store, self.c, frozenset(), block, partners=True)
            return list(self._recheck(differ, b, counted))

        for problems in _in_order(
            recheck, self.named.blocks(differ, sorted(self.seen)), self.store.threads
        ):
            yield from problems
        yield from self._lacking(differ, counted)

    def _recheck(self, rows: npt.NDArray[np.bool_], b: _Block, counted: _Counted) -> _Problems:
        """The block's entries in a pair's later row whose earlier row is one of `rows`, each
        against the band's code for its pair, and counted for that earlier row."""
        span = b.span()
        _, in_later = _near_halves(
            self.pos, self.near_bp, span.row, span.partner.astype(np.int64), self.c.size
        )
        in_later[in_later] = rows[span.partner[in_later].astype(np.int64)]
        up, down = span.partner[in_later].astype(np.int64), span.row[in_later]
        counted.add(up)
        code = span.code[in_later].astype(np.int64)
        band = self.store.pick(f"near/{self.c.name}/r", near_address(self.band, up, down))
        differ = band.astype(np.int64) != code
        odd = _keys(up[differ], down[differ], code[differ], self.c.size)
        what = "entries whose code is not the Near Band's"
        yield from _odd_pairs(self.c.name, odd, self.c.size, what, _PARTNERS)

    def _lacking(self, differ: npt.NDArray[np.bool_], counted: _Counted) -> _Problems:
        """Each differing row's Near Band pairs at the Partner Floor against the entries its later
        rows hold: a row with fewer entries than pairs is a pair the component lacks, which the
        sums alone cannot name (partner-rows §3, §10)."""
        lacking = []
        for row in np.flatnonzero(differ).tolist():
            codes = self.store.read(
                f"near/{self.c.name}/r", int(self.band[row]), int(self.band[row + 1])
            )
            pairs = int((np.abs(codes) >= self.floor).sum())
            if counted.counts[row] != pairs:
                lacking.append((row, pairs, int(counted.counts[row])))
        if not lacking:
            return
        named = ", ".join(
            f"row {row}: {pairs:,} pair(s) at the Partner Floor, {entries:,} entry(ies)"
            for row, pairs, entries in lacking[:_NAMED]
        )
        more = f" and {len(lacking) - _NAMED:,} more" if len(lacking) > _NAMED else ""
        yield (
            f"partners/{self.c.name}",
            f"a row whose later rows do not hold its Near Band pairs at the Partner Floor: "
            f"{named}{more}",
        )


@_scanned("partners.far")
class _PartnerFar(_PartnerRule):
    """The entries beyond near_bp, in each pair's earlier row, are exactly the Far Pairs, pair
    for pair and code for code: the retired far.index's comparison, read against the Partner
    Rows (ld-store §12; partner-rows §10). Both are read by the same block of rows, so nothing is
    carried; that an entry beyond near_bp is in its later row too is partners.symmetric's to
    say."""

    parts = frozenset({_FAR.partner, _FAR.code})

    def __init__(self, store: _Store, c: _Chromosome) -> None:
        super().__init__(store, c)
        if not self.judged:
            self.parts = frozenset()
            return
        self.near_bp = store.integer("near_bp")
        _sparse_indptr(store, c, _FAR)
        self.pos = store.positions(c.name, c.start, c.stop)

    def block(self, b: _Block) -> _Problems:
        if not self.judged:
            return
        span = b.span()
        size = self.c.size
        partner = span.partner.astype(np.int64)
        _, beyond = _split_by_distance(span, self.pos, size, self.near_bp)
        beyond &= partner > span.row
        entries = _keys(span.row[beyond], partner[beyond], span.code[beyond].astype(np.int64), size)
        held = _keys(b.get(_FAR.owner), b.get(_FAR.partner), b.get(_FAR.code), size)
        if not np.array_equal(entries, held):  # both ascend in a store that passes
            odd = np.setxor1d(entries, held)
            what = "not the Far Pairs for the pairs beyond near_bp"
            yield from _odd_pairs(self.c.name, odd, size, what, _PARTNERS)

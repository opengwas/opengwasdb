"""The spec's validation rules, by name, on write and on demand (spec §12; ogld#4).

Each rule is a generator over one opened hierarchy and its provenance, yielding where and
how the store breaks it. `validate` runs all 18, in the spec's order, on a store at a path or
URL; `StoreWriter.finish` runs them on the store it wrote, before the provenance that makes
it readable goes in. A rule never trusts what it does not check: when what it needs is
missing or malformed it reports that it cannot be checked, and when an array will not
decode it reports that it cannot be read, each as a finding, rather than pass or stop.

Every rule reads in bounded windows, in one pass, through Zarr directly: a chromosome's
positions and `indptr` arrays whole, as a reader keeps them (spec §10), and a Near Band, a
Proxy Index and the variant columns at most `_WINDOW` elements or `_ROWS` rows at a time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeGuard

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format.encoding import (
    FAR_BP,
    NEAR_BP,
    RESERVED_CODE,
    T_FAR_Q,
    far_code_floor,
    near_row_lengths,
    t_far,
)
from ogld.format.provenance import ENCODING, FORMAT_VERSION, VALIDATION_RULES
from ogld.identity import IdentityError, alid_locus, orient, variant_order_key
from ogld.store.layout import never_data, spans
from ogld.store.location import open_group

#: Elements of a Near Band or a Proxy Index a rule reads at once: 8 Mi codes are 16 MiB.
_WINDOW: Final = 1 << 23
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
    root = open_group(location)
    return validate_group(root, root.attrs.get("ld_store"))


def validate_group(
    root: zarr.Group, provenance: object, rules: Sequence[str] = VALIDATION_RULES
) -> dict[str, list[Finding]]:
    """The `rules` against the hierarchy `root`, judged with `provenance` as its `ld_store`."""
    store = _Store(root, provenance)
    return {rule: _run(rule, store) for rule in rules}


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
    except _Unreadable as unreadable:
        found.append(Finding(rule, unreadable.where, f"cannot be read: {unreadable.why}"))
    except _Unchecked as unchecked:
        found.append(Finding(rule, unchecked.where, f"cannot be checked: {unchecked.why}"))
    return found


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

    def __init__(self, root: zarr.Group, provenance: object) -> None:
        self.root = root
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

    def integer(self, key: str) -> int:
        value = self.value(key)
        if not _is_int(value):
            raise _Unchecked(key, f"provenance {key} is {value!r}, not an integer")
        return int(value)

    def find(self, path: str) -> zarr.Array[Any] | None:
        """The array at `path`, or None; each is opened once, as opening reads its metadata."""
        if path not in self._arrays:
            member = self.root.get(path)
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
        listed = self.root.get("variants")
        entries = listed.attrs.get("chromosomes") if isinstance(listed, zarr.Group) else None
        if not isinstance(entries, list) or not all(_range_entry(e) for e in entries):
            raise _Unchecked(
                "variants", "the chromosomes ranges are malformed (variants.chromosomes)"
            )
        return [(str(e["name"]), int(e["row_start"]), int(e["row_stop"])) for e in entries]

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
                raise _Unchecked(path, f"{path} {problem} (near.rows, index.rows)")
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


def _range_entry(entry: object) -> bool:
    if not isinstance(entry, dict) or set(entry) != {"name", "row_start", "row_stop"}:
        return False
    canonical = isinstance(entry["name"], str) and alid_locus(f"{entry['name']}:1:A:C") is not None
    bounds = _is_int(entry["row_start"]) and _is_int(entry["row_stop"])
    return canonical and bounds and 0 <= entry["row_start"] <= entry["row_stop"]


def _flag(where: str, bad: npt.NDArray[np.bool_], first: int, what: str) -> _Problems:
    """One finding for every element `bad` marks, numbered from `first`, if there is any."""
    at = np.flatnonzero(bad) + first
    if len(at):
        more = f" and {len(at) - _NAMED:,} more" if len(at) > _NAMED else ""
        yield where, f"{what} at {', '.join(str(i) for i in at[:_NAMED])}{more}"


def _windows(start: int, stop: int, step: int) -> Iterator[tuple[int, int]]:
    for lo in range(start, stop, step):
        yield lo, min(stop, lo + step)


def _blocks(*indptrs: npt.NDArray[np.int64]) -> Iterator[tuple[int, int]]:
    """Row ranges [a, b) holding at most `_WINDOW` elements in each array `indptrs` delimit,
    or one row when that row alone holds more."""
    rows, a = len(indptrs[0]) - 1, 0
    while a < rows:
        b = min(int(np.searchsorted(p, p[a] + _WINDOW, side="right")) - 1 for p in indptrs)
        b = min(max(b, a + 1), rows)
        yield a, b
        a = b


# ------------------------------------------------------------------ envelope


_VARIANT_DTYPES: Final = {
    "position": "int32",
    "alid": "string",
    "a1": "string",
    "a2": "string",
    "a1_frequency": "float32",
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
    for chrom, _, _ in store.chromosomes():
        yield from (f"near/{chrom}/{name}" for name in ("indptr", "r"))
        yield from (f"index/{chrom}/{name}" for name in ("indptr", "partner", "r"))


def _counted(store: _Store, rows: int) -> _Problems:
    """`n_variants` against the variant table, when provenance records it as a number."""
    recorded = _lookup(store.provenance, "n_variants")
    if _is_int(recorded) and recorded != rows:
        yield "n_variants", f"provenance records {recorded:,}, but variants/ holds {rows:,} rows"


@_rule("envelope.arrays")
def _envelope_arrays(store: _Store) -> _Problems:
    if store.root.metadata.zarr_format != 3:
        yield "/", f"the hierarchy is Zarr v{store.root.metadata.zarr_format}, not v3 (spec §1)"
        return
    ranges = store.chromosomes()
    size = ranges[-1][2] if ranges else 0
    for name, dtype in _VARIANT_DTYPES.items():
        yield from _shape(store, f"variants/{name}", dtype, size)
    yield from _counted(store, size)
    r_dtype = ENCODING["r_dtype"]
    for chrom, start, stop in ranges:
        for kind in ("near", "index"):
            yield from _shape(store, f"{kind}/{chrom}/indptr", "int64", stop - start + 1)
        yield from _shape(store, f"near/{chrom}/r", r_dtype, _last(store, f"near/{chrom}/indptr"))
        entries = _last(store, f"index/{chrom}/indptr")
        yield from _shape(store, f"index/{chrom}/partner", "int32", entries)
        yield from _shape(store, f"index/{chrom}/r", r_dtype, entries)


@_rule("envelope.fill")
def _envelope_fill(store: _Store) -> _Problems:
    """Every array's fill value is one no valid store holds there, so a chunk it lost can
    never read as data: the reserved code in an r array, a negative indptr, and so on."""
    for path in _arrays(store):
        fill = store.array(path).fill_value
        if not never_data(path.rsplit("/", 1)[-1], fill).all():
            shown = json.dumps(fill) if isinstance(fill, str) else str(fill)
            yield (
                path,
                (
                    f"its fill value {shown} is one valid data can hold there: a chunk it "
                    "lost would read as data (spec §6)"
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
    "counts.index_entries": "int",
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
    """The format version and the encoding are the ones this package reads (spec §3, §13):
    `ogld info` and every reader refuse a store with any other."""
    version = store.value("format_version")
    if version != FORMAT_VERSION:
        yield "format_version", f"{version!r}: this ogld reads {FORMAT_VERSION!r} only (spec §13)"
    recorded = store.value("encoding")
    if recorded != ENCODING:
        yield "encoding", f"{recorded!r} is not spec §3's {ENCODING!r}"


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
    floor = store.integer("far_code_floor")
    if floor != far_code_floor(expected):
        yield "far_code_floor", f"{floor}, but t_far {expected!r} gives {far_code_floor(expected)}"


@_rule("provenance.threshold")
def _provenance_threshold(store: _Store) -> _Problems:
    yield from _k_records(store)
    yield from _t_far_records(store)


# ------------------------------------------------------------------ variants


def _rank(name: str) -> int:
    return variant_order_key(orient(name, 1, "A", "C"))[0]


@_rule("variants.chromosomes")
def _variants_chromosomes(store: _Store) -> _Problems:
    ranges = store.chromosomes()
    expected, previous = 0, 0
    for name, start, stop in ranges:
        if _rank(name) <= previous:
            yield "variants", f"chromosome {name} is out of chromosome order (spec §2)"
        if start != expected:
            yield "variants", f"chromosome {name} starts at row {start:,}, not {expected:,}"
        expected, previous = stop, _rank(name)
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
        for lo, hi in _windows(start, stop, _ROWS):
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
        for lo, hi in _windows(start, stop, _ROWS):
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
    for lo, hi in _windows(0, store.array("variants/a1_frequency").shape[0], _ROWS):
        f = store.read("variants/a1_frequency", lo, hi).astype(np.float64)
        with np.errstate(invalid="ignore"):
            bad = ~(np.isfinite(f) & (f > 0) & (f < 1))
        yield from _flag("variants/a1_frequency", bad, lo, "not finite and in (0, 1)")


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


def _scan_codes(store: _Store, kind: str) -> _Problems:
    for chrom, _, _ in store.chromosomes():
        path = f"{kind}/{chrom}/r"
        for lo, hi in _windows(0, store.array(path).shape[0], _WINDOW):
            bad = store.read(path, lo, hi) == RESERVED_CODE
            yield from _flag(path, bad, lo, f"the reserved code {RESERVED_CODE}")


@_rule("near.codes")
def _near_codes(store: _Store) -> _Problems:
    yield from _scan_codes(store, "near")


# ------------------------------------------------------------------ the Proxy Index


@dataclass(frozen=True)
class _IndexBlock:
    """The Proxy Index entries of one chromosome's rows [first, last): owner, partner, code,
    and `at`, the offset of the first entry in `partner` and `r`."""

    first: int
    last: int
    at: int
    owner: npt.NDArray[np.int64]
    partner: npt.NDArray[np.int64]
    code: npt.NDArray[np.int64]

    def valid(self, size: int) -> npt.NDArray[np.bool_]:
        """Entries whose partner is a row of the chromosome: the others are index.rows'."""
        return (self.partner >= 0) & (self.partner < size)


def _index_block(
    store: _Store, chrom: str, indptr: npt.NDArray[np.int64], first: int, last: int
) -> _IndexBlock:
    lo, hi = int(indptr[first]), int(indptr[last])
    owner = np.repeat(np.arange(first, last, dtype=np.int64), np.diff(indptr[first : last + 1]))
    partner = store.read(f"index/{chrom}/partner", lo, hi).astype(np.int64)
    code = store.read(f"index/{chrom}/r", lo, hi).astype(np.int64)
    return _IndexBlock(first, last, lo, owner, partner, code)


def _chromosome_blocks(store: _Store, chrom: str, size: int) -> Iterator[_IndexBlock]:
    """One chromosome's Proxy Index in blocks of rows."""
    indptr = store.indptr(f"index/{chrom}/indptr", size)
    for a, b in _blocks(indptr):
        yield _index_block(store, chrom, indptr, a, b)


def _index_blocks(store: _Store) -> Iterator[tuple[str, int, int, _IndexBlock]]:
    """Each chromosome's Proxy Index in blocks of rows, with its first row and its size."""
    for chrom, start, stop in store.chromosomes():
        for block in _chromosome_blocks(store, chrom, stop - start):
            yield chrom, start, stop - start, block


def _partner_problems(chrom: str, size: int, e: _IndexBlock) -> _Problems:
    outside = ~e.valid(size) | (e.partner == e.owner)
    repeated = (np.diff(e.partner) <= 0) & (np.diff(e.owner) == 0)
    where = f"index/{chrom}/partner"
    yield from _flag(where, outside, e.at, "a partner outside [0, n_c) or the row itself")
    yield from _flag(where, repeated, e.at + 1, "a partner not above the one before it")


def _index_ends(store: _Store, chrom: str) -> int | None:
    """The length `partner` and `r` share, or None when they disagree."""
    lengths = {store.array(f"index/{chrom}/{name}").shape[0] for name in ("partner", "r")}
    return lengths.pop() if len(lengths) == 1 else None


@_rule("index.rows")
def _index_rows(store: _Store) -> _Problems:
    for chrom, start, stop in store.chromosomes():
        path, size, end = f"index/{chrom}/indptr", stop - start, _index_ends(store, chrom)
        indptr = store.read(path).astype(np.int64)
        problems = list(_indptr_problems(path, indptr, size, end))
        if end is None:
            problems.append((f"index/{chrom}", "partner and r differ in length"))
        yield from problems
        for a, b in _blocks(indptr) if not problems else ():
            yield from _partner_problems(chrom, size, _index_block(store, chrom, indptr, a, b))


@_rule("index.codes")
def _index_codes(store: _Store) -> _Problems:
    yield from _scan_codes(store, "index")


@_rule("index.distance")
def _index_distance(store: _Store) -> _Problems:
    far_bp = store.integer("far_bp")
    for chrom, start, size, e in _index_blocks(store):
        pos = store.positions(chrom, start, start + size)
        partner = np.where(e.valid(size), e.partner, e.owner)
        beyond = np.abs(pos[partner] - pos[e.owner]) > far_bp
        yield from _flag(f"index/{chrom}/partner", beyond, e.at, f"a partner beyond {far_bp:,} bp")


@_rule("index.threshold")
def _index_threshold(store: _Store) -> _Problems:
    floor = store.integer("far_code_floor")
    for chrom, _, _, e in _index_blocks(store):
        below = np.abs(e.code) < floor
        yield from _flag(f"index/{chrom}/r", below, e.at, f"|code| below far_code_floor {floor}")


def _keys(
    lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64], code: npt.NDArray[np.int64], size: int
) -> npt.NDArray[np.int64]:
    """One integer per (pair, code), so two sets of entries compare with one set operation."""
    return ((lo * size + hi) << 16) + (code - RESERVED_CODE)


def _mixed(keys: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """splitmix64's finaliser of each key: a sum of these tells two multisets of keys apart
    unless they collide modulo 2⁶⁴, about one chance in 1.8×10¹⁹ for each difference."""
    z = keys.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _odd_pairs(chrom: str, odd: npt.NDArray[np.int64], size: int, what: str) -> _Problems:
    """One finding naming the pairs in `odd` (keys of `_keys`), if there are any."""
    pair = odd >> 16
    names = [f"({p // size}, {p % size})" for p in pair[:_NAMED].tolist()]
    if len(odd):
        more = f" and {len(odd) - _NAMED:,} more" if len(odd) > _NAMED else ""
        yield f"index/{chrom}", f"{what}: {', '.join(names)}{more}"


@_rule("index.symmetric")
def _index_symmetric(store: _Store) -> _Problems:
    """Each pair's two entries, matched in one bounded pass. Every entry adds a hash of
    (earlier row, later row, code) to its pair's earlier row, the entry in that row to one
    sum and the entry in the later row to the other; a row whose two sums differ is the
    earlier row of a pair held one way only, or with two codes."""
    for chrom, start, stop in store.chromosomes():
        size = stop - start
        # Two sums of pair hashes per row, not r values: 0 is the empty sum.
        sums = np.full((2, size), 0, dtype=np.uint64)
        for e in _chromosome_blocks(store, chrom, size):
            ok = e.valid(size) & (e.partner != e.owner)
            lo, hi = np.minimum(e.owner, e.partner)[ok], np.maximum(e.owner, e.partner)[ok]
            later = (e.owner > e.partner)[ok].astype(np.int64)
            np.add.at(sums, (later, lo), _mixed(_keys(lo, hi, e.code[ok], size)))
        what = "the earlier row of a pair held one way only, or with two codes"
        yield from _flag(f"index/{chrom}", sums[0] != sums[1], 0, what)


def _band_block(
    store: _Store, chrom: str, indptr: npt.NDArray[np.int64], a: int, b: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The Near Band's pairs of rows [a, b): earlier row, later row, code (spec §4)."""
    rows = np.arange(a, b, dtype=np.int64)
    length = np.diff(indptr[a : b + 1])
    code = store.read(f"near/{chrom}/r", int(indptr[a]), int(indptr[b])).astype(np.int64)
    return np.repeat(rows, length), spans(rows + 1, length), code


def _near_half(store: _Store, chrom: str, start: int, stop: int) -> _Problems:
    """Each block's index entries within near_bp: those in a pair's earlier row are the band's
    row, thresholded; those in its later row carry the band's code of their pair."""
    size, pos = stop - start, store.positions(chrom, start, stop)
    near_bp, floor = store.integer("near_bp"), store.integer("far_code_floor")
    band = store.indptr(f"near/{chrom}/indptr", size)
    index = store.indptr(f"index/{chrom}/indptr", size)
    for a, b in _blocks(band, index):
        lo, hi, code = _band_block(store, chrom, band, a, b)
        kept = np.abs(code) >= floor
        e = _index_block(store, chrom, index, a, b)
        ok = e.valid(size)
        within = ok & (np.abs(pos[np.where(ok, e.partner, e.owner)] - pos[e.owner]) <= near_bp)
        later, earlier = within & (e.partner > e.owner), within & (e.partner < e.owner)
        odd = np.setxor1d(
            _keys(lo[kept], hi[kept], code[kept], size),
            _keys(e.owner[later], e.partner[later], e.code[later], size),
        )
        yield from _odd_pairs(chrom, odd, size, "not the Near Band's pairs at the far code floor")
        up, down = e.partner[earlier], e.owner[earlier]
        held = store.pick(f"near/{chrom}/r", band[up] + (down - up - 1)).astype(np.int64)
        differ = held != e.code[earlier]
        yield from _odd_pairs(
            chrom, _keys(up[differ], down[differ], e.code[earlier][differ], size), size,
            "entries whose code is not the Near Band's",
        )  # fmt: skip


@_rule("index.near")
def _index_near(store: _Store) -> _Problems:
    """The index's near half is the band, thresholded (spec §5). The entries in each pair's
    earlier row are compared with that Near Band row, both ways; an entry in the later row
    must carry the band's code, and that it exists at all is index.symmetric's to say."""
    for chrom, start, stop in store.chromosomes():
        yield from _near_half(store, chrom, start, stop)

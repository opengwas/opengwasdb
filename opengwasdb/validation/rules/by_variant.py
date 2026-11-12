"""Rules for a sequence's variant-centric index, `ragged/by_variant/` (spec §11.1, ADR 0074).

The index holds every association of a Ragged component -- a Ragged release's
sequence or a Hybrid's Overflow -- again, sorted by variant, so a lookup by
variant reads one contiguous run (ADR 0060). It is a second copy, so each rule
here is one way the copy could disagree with the Analysis-sorted sequence while
reading as plausible: a run whose bounds are wrong, an Analysis it does not
have, a variant holding a different number of rows, rows out of order inside a
variant, a member present on one side only, a layout its role does not decide,
and codes that differ. The index's own arrays are read through `BY_VARIANT`, a
window at a time. The per-variant counts fold over the association scan, which
already holds every cell's variant, so the sequence is not read again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np

from opengwasdb import encoding as enc
from opengwasdb.store.arrays import (
    array_length,
    chunk_layout,
    inner_chunk_hint_for_path,
    inner_chunk_of,
    role_for_array_path,
    shard_layout,
)
from opengwasdb.store.open import parse_format_version
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Scan, Skip, Target
from opengwasdb.validation.rules.csr import ARRAYS, LENGTHS
from opengwasdb.validation.rules.encoding import PLAN
from opengwasdb.validation.scans import ASSOCIATIONS, Block, release_rows, sequence_offsets

#: The index's group, beside the sequence it indexes.
GROUP = "by_variant"
#: The rule every other rule here runs after: the index is there to be judged.
PRESENT = "by_variant.present"
#: The rule the index's runs are read after: they bound every row.
OFFSETS = "by_variant.offsets"
#: Index rows one window holds (issue #254).
_WINDOW = 1_000_000
#: What every message here names: the component the index belongs to.
_LABEL = "data.zarr/ragged"


def _block(target: Target) -> Any:
    """The manifest's record of the index: `provenance.ragged.by_variant`."""
    return target.store.manifest.provenance.get("ragged", {}).get("by_variant")


def _index(target: Target) -> Any:
    return target.group[GROUP]


def _n_assoc(target: Target) -> int:
    return int(sequence_offsets(target)[-1])


def index_offsets(target: Target) -> np.ndarray:
    """The index's per-variant run bounds, read once for the run."""
    return target.remember(
        "by_variant/offsets", lambda: np.asarray(_index(target)["offsets"][:], dtype=np.int64)
    )


#: The format from which every Ragged component carries the index (spec §11.1).
_REQUIRED_FROM = (0, 2, 0)


def present(target: Target) -> Iterator[str]:
    """Every Ragged component carries the variant-centric index, in format 0.2.0.

    A 0.1.0 component may lack it (spec §11.1, #380): the index did not exist
    when those releases were built. Its absence there is a skip, with that reason,
    not a pass, and one it does carry is judged like any other. A manifest that
    records an index the release does not carry is wrong in any format.
    """
    if GROUP in target.group:
        return
    if _block(target) is not None:
        yield (
            f"{_LABEL}: the manifest records provenance.ragged.by_variant but the release "
            f"carries no {GROUP} group; the index it describes is absent"
        )
        return
    version = target.store.manifest.format_version
    if parse_format_version(version) < _REQUIRED_FROM:
        raise Skip(
            f"a format {version} component need not carry the variant-centric index (spec §11.1)"
        )
    yield (
        f"{_LABEL}: carries no {GROUP} group; a Ragged component must "
        f"carry the variant-centric index (ADR 0060)"
    )


def provenance(target: Target) -> Iterator[str]:
    """The manifest records the index, and describes the index that is there."""
    block = _block(target)
    if block is None:
        yield (
            f"{_LABEL}/{GROUP}: the release carries a variant index but its "
            "manifest records no provenance.ragged.by_variant block"
        )
        return
    n_axis, n_assoc = release_rows(target), _n_assoc(target)
    if int(block.get("n_axis", -1)) != n_axis or int(block.get("n_rows", -1)) != n_assoc:
        yield (
            f"{_LABEL}: provenance.ragged.by_variant records "
            f"n_axis={block.get('n_axis')!r} n_rows={block.get('n_rows')!r} but the index "
            f"holds {n_axis} variants over {n_assoc} rows"
        )


def offsets(target: Target) -> Iterator[str]:
    """The runs span `[0, n_assoc)` over the variant axis, and every column is that long."""
    index = _index(target)
    bounds = index_offsets(target)
    n_axis, n_assoc = release_rows(target), _n_assoc(target)
    if len(bounds) != n_axis + 1:
        yield (
            f"{_LABEL}/{GROUP}/offsets has {len(bounds)} entries but the variant "
            f"axis is {n_axis} (expected {n_axis + 1})"
        )
    elif int(bounds[0]) != 0 or int(bounds[-1]) != n_assoc:
        yield (
            f"{_LABEL}/{GROUP}/offsets does not span [0, {n_assoc}) "
            f"(starts {int(bounds[0])}, ends {int(bounds[-1])})"
        )
    elif np.any(np.diff(bounds) < 0):
        yield f"{_LABEL}/{GROUP}/offsets is not non-decreasing"
    else:
        yield from _column_lengths(index, n_assoc)


def _column_lengths(index: Any, n_assoc: int) -> Iterator[str]:
    for name in ("z", "se", "analysis_index"):
        if array_length(index[name]) != n_assoc:
            yield (
                f"{_LABEL}/{GROUP}/{name} has {array_length(index[name])} "
                f"entries but the component has {n_assoc} associations"
            )
            return


def members(target: Target) -> Iterator[str]:
    """`eaf`, `imputed` and `n` are present on exactly both sides or neither."""
    index = _index(target)
    for name in ("eaf", "imputed", "n"):
        if (name in index) != (name in target.group):
            yield (
                f"{_LABEL}/{GROUP}/{name} is present on one side only; the "
                "component and its variant index must carry the same members"
            )


def layout(target: Target) -> Iterator[str]:
    """Every `by_variant/` array carries the role's decided inner chunk and shard.

    ADR 0060: the index adds no role and no shard policy, and the offsets array's
    1,000-element inner-chunk hint is part of its recorded layout. A conversion
    that reproduced a different chunk from a build would be a layout divergence a
    reader cannot see.
    """
    index = _index(target)
    errors: list[str] = []
    for name in sorted(index.array_keys()):
        _validate_one_index_array(index, name, errors, _LABEL)
    yield from errors


# --- the index's rows, a window at a time ---


@dataclasses.dataclass(frozen=True)
class IndexWindow:
    """One window of the index's rows: each row's Analysis and variant."""

    analysis: np.ndarray
    variant: np.ndarray
    #: The `(variant, analysis)` of the row before the window, when there is one.
    previous: tuple[int, int] | None


def _window_limits(n_rows: int) -> Iterator[tuple[int, int]]:
    """Every window of an index of `n_rows` rows: the one place its size is set."""
    for lo in range(0, n_rows, _WINDOW):
        yield lo, min(lo + _WINDOW, n_rows)


def _n_windows(target: Target) -> int:
    """How many windows `_index_windows` reads for `target`, before a row is read."""
    return sum(1 for _ in _window_limits(_n_assoc(target)))


def _index_windows(target: Target) -> Iterator[IndexWindow]:
    """The index's `analysis_index`, read `_WINDOW` rows at a time, never whole."""
    column = _index(target)["analysis_index"]
    bounds = index_offsets(target)
    previous: tuple[int, int] | None = None
    for lo, hi in _window_limits(_n_assoc(target)):
        analysis = np.asarray(column[lo:hi], dtype=np.int64)
        rows = np.arange(lo, hi, dtype=np.int64)
        variant = np.searchsorted(bounds, rows, side="right") - 1
        yield IndexWindow(analysis, variant, previous)
        previous = (int(variant[-1]), int(analysis[-1]))


BY_VARIANT: Scan[IndexWindow] = Scan("by_variant", _index_windows, total=_n_windows)


@dataclasses.dataclass(frozen=True)
class Span:
    """The lowest and highest Analysis named so far."""

    lowest: int = np.iinfo(np.int64).max
    highest: int = np.iinfo(np.int64).min

    def __add__(self, other: Span) -> Span:
        return Span(min(self.lowest, other.lowest), max(self.highest, other.highest))


def _span(_target: Target, window: IndexWindow) -> Span:
    return Span(int(window.analysis.min()), int(window.analysis.max()))


def _judge_span(target: Target, span: Span) -> Iterator[str]:
    """Every row names one of the component's Analyses."""
    n_analyses = len(sequence_offsets(target)) - 1
    if _n_assoc(target) and (span.lowest < 0 or span.highest >= n_analyses):
        yield f"{_LABEL}/{GROUP}/analysis_index is outside the component's {n_analyses} Analyses"


def _decreasing(_target: Target, window: IndexWindow) -> int:
    analysis, variant = window.analysis, window.variant
    found = int(np.count_nonzero((variant[1:] == variant[:-1]) & (analysis[1:] < analysis[:-1])))
    previous = window.previous
    if previous is not None and previous[0] == int(variant[0]) and int(analysis[0]) < previous[1]:
        found += 1
    return found


def _judge_order(_target: Target, n: int) -> Iterator[str]:
    """Within each variant's run, the rows' Analyses are non-decreasing."""
    if n:
        yield f"{_LABEL}/{GROUP}/analysis_index is not non-decreasing within a variant's block"


# --- each variant's run against the sequence, over the association scan ---


@dataclasses.dataclass(frozen=True)
class Histogram:
    """How many cells of the sequence each variant holds, added up window by window.

    A window's measure holds its variants and how many cells each has; the total
    holds one count per variant, filled in place as each window is added, so it
    never costs more than one entry per variant. The zero holds nothing, and is
    never written to.
    """

    counts: np.ndarray | None = None
    values: np.ndarray | None = None
    multiplicity: np.ndarray | None = None
    n_axis: int = 0

    def __add__(self, other: Histogram) -> Histogram:
        counts = self.counts if self.counts is not None else np.zeros(other.n_axis, np.int64)
        if other.values is not None and other.multiplicity is not None:
            np.add.at(counts, other.values, other.multiplicity)
        return Histogram(counts)


def _variant_counts(target: Target, block: Block) -> Histogram:
    """The window's cells per variant. A variant outside the axis is not counted: it
    holds no index run, so the disagreement is reported against the run it lacks."""
    assert block.variant is not None
    n_axis = release_rows(target)
    variant = block.variant[(block.variant >= 0) & (block.variant < n_axis)]
    values, multiplicity = np.unique(variant, return_counts=True)
    return Histogram(None, values.astype(np.int64), multiplicity.astype(np.int64), n_axis)


def _judge_counts(target: Target, total: Histogram) -> Iterator[str]:
    """Every variant holds as many index rows as the sequence holds cells."""
    bounds = index_offsets(target)
    held = np.diff(bounds)
    counts = total.counts if total.counts is not None else np.zeros(len(held), dtype=np.int64)
    if not np.array_equal(held, counts):
        differing = int(np.argmax(held != counts))
        yield (
            f"{_LABEL}/{GROUP}/offsets: variant {differing} holds "
            f"{int(bounds[differing + 1] - bounds[differing])} rows but the "
            f"Analysis-sorted planes hold {int(counts[differing])}"
        )


def sample(target: Target) -> Iterator[str]:
    """Sampled cells of the index hold the codes the sequence holds (spec §11.1)."""
    errors: list[str] = []
    _validate_variant_index_sample(
        target.group,
        _index(target),
        sequence_offsets(target),
        index_offsets(target),
        target.encoding,
        _n_assoc(target),
        errors,
        _LABEL,
    )
    yield from errors


def _code_differs(a: Any, b: Any) -> bool:
    """True when stored codes a and b differ, treating NaN == NaN."""
    if a == b:
        return False
    try:
        return not (np.isnan(a) and np.isnan(b))
    except (TypeError, ValueError):
        return True


def _sample_non_empty_variants(by_offsets: np.ndarray, sample_variants: int) -> np.ndarray:
    """Deterministic selection of non-empty variant indices for sampled validation."""
    widths = np.diff(by_offsets)
    non_empty = np.where(widths > 0)[0]
    if len(non_empty) <= sample_variants:
        return non_empty
    rng = np.random.default_rng(0)
    chosen = rng.choice(non_empty, size=sample_variants - 2, replace=False)
    return np.unique(np.concatenate([[non_empty[0], non_empty[-1]], chosen]))


def _check_z_overflow(
    root: Any, index: Any, b_z: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_z != enc.Z_OVERFLOW:
        return None
    idx_v = enc.ZOverflowTable.open(index).lookup(np.array([s_idx]))
    src_v = enc.ZOverflowTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} z overflow value != Analysis-sorted table" if idx_v != src_v else None


def _check_eaf_exception(
    root: Any, index: Any, b_eaf: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_eaf != enc.EAF_EXCEPTION:
        return None
    idx_v = enc.EafExceptionTable.open(index).lookup(np.array([s_idx]))
    src_v = enc.EafExceptionTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} eaf exception value != Analysis-sorted table" if idx_v != src_v else None


def _check_se_exception(
    root: Any, index: Any, b_se: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_se != enc.SE_EXCEPTION:
        return None
    idx_v = enc.SeExceptionTable.open(index).lookup(np.array([s_idx]))
    src_v = enc.SeExceptionTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} se exception value != Analysis-sorted table" if idx_v != src_v else None


def _check_cell_exceptions(
    root: Any,
    index: Any,
    encoding: enc.StoreEncoding,
    b_z: Any,
    b_se: Any,
    b_eaf: Any,
    tag: str,
    s_idx: int,
    src_idx: int,
) -> str | None:
    """Validate exception-table values at an exception cell; returns an error string or None."""
    if encoding.z.is_fixed_point:
        err = _check_z_overflow(root, index, b_z, tag, s_idx, src_idx)
        if err is not None:
            return err
    if encoding.eaf.is_residual and b_eaf is not None:
        err = _check_eaf_exception(root, index, b_eaf, tag, s_idx, src_idx)
        if err is not None:
            return err
    if encoding.se.is_residual:
        return _check_se_exception(root, index, b_se, tag, s_idx, src_idx)
    return None


def _check_cell_codes(
    root: Any,
    b_z: Any,
    b_se: Any,
    b_eaf: Any,
    b_imp: Any,
    tag: str,
    src_idx: int,
    b_n: Any = None,
) -> str | None:
    """Compare raw stored codes for equality, returning an error string on mismatch."""
    if _code_differs(b_z, root["z"][src_idx]):
        return f"{tag} z code {b_z!r} != source {root['z'][src_idx]!r}"
    if _code_differs(b_se, root["se"][src_idx]):
        return f"{tag} se code {b_se!r} != source {root['se'][src_idx]!r}"
    for name, code in (("eaf", b_eaf), ("n", b_n)):
        if code is not None and _code_differs(code, root[name][src_idx]):
            return f"{tag} {name} code {code!r} != source {root[name][src_idx]!r}"
    if b_imp is not None and b_imp != root["imputed"][src_idx]:
        return f"{tag} imputed {b_imp!r} != source {root['imputed'][src_idx]!r}"
    return None


@dataclasses.dataclass(frozen=True)
class _SampledCell:
    """One index row of a sampled variant, with the codes it holds."""

    variant: int
    analysis: int
    #: The row's position in the index, which its exception tables are keyed by.
    ordinal: int
    codes: tuple[Any, Any, Any, Any, Any]


#: The index columns a sampled cell's codes are read from, in its `codes` order.
_CODES = ("z", "se", "eaf", "imputed", "n")


def _index_columns(root: Any, index: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
    """The index's columns at `rows`, each read once. `eaf`, `imputed` and `n` are
    read only where both sides hold them, since only then are they compared."""
    shared = [name for name in ("eaf", "imputed", "n") if name in index and name in root]
    return {
        name: np.asarray(index[name].oindex[rows])
        for name in ("analysis_index", "z", "se", *shared)
    }


def _codes_at(columns: dict[str, np.ndarray], k: int) -> tuple[Any, Any, Any, Any, Any]:
    z, se, eaf, imputed, n = (columns[name][k] if name in columns else None for name in _CODES)
    return z, se, eaf, imputed, n


def _sampled_cells(
    root: Any, index: Any, by_offsets: np.ndarray, variants: list[int]
) -> list[_SampledCell]:
    """Every index row of each sampled variant, in sampled-variant order."""
    bounds = [(int(by_offsets[v]), int(by_offsets[v + 1])) for v in variants]
    rows = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in bounds])
    columns = _index_columns(root, index, rows)
    analyses = columns["analysis_index"].astype(np.int32)
    cells: list[_SampledCell] = []
    for v, (s, e) in zip(variants, bounds, strict=True):
        for row in range(s, e):
            k = len(cells)
            cells.append(_SampledCell(v, int(analyses[k]), row, _codes_at(columns, k)))
    return cells


def _source_positions(
    root: Any, offsets: np.ndarray, cells: list[_SampledCell]
) -> dict[tuple[int, int], int]:
    """Where each sampled cell sits in the Analysis-sorted planes, or -1 where it is not.

    Each Analysis's segment is read once, for every sampled variant it holds,
    rather than once for each.
    """
    wanted: dict[int, list[int]] = {}
    for cell in cells:
        wanted.setdefault(cell.analysis, []).append(cell.variant)
    found: dict[tuple[int, int], int] = {}
    for a, variants in wanted.items():
        cs, ce = int(offsets[a]), int(offsets[a + 1])
        src_vi = np.asarray(root["variant_index"][cs:ce], dtype=np.int32)
        for v in variants:
            pos = int(np.searchsorted(src_vi, v))
            held = pos < len(src_vi) and src_vi[pos] == v
            found[(v, a)] = cs + pos if held else -1
    return found


class _Gathered:
    """The sequence's codes at the sampled cells, each plane read once, not a cell at a time.

    Indexed as the sequence is, `gathered["z"][position]`, and each code comes back
    as the 0-d array a single-cell Zarr read returns, so a finding names a code in
    exactly the words it always has.
    """

    _PLANES = ("z", "se", "eaf", "n", "imputed")

    def __init__(self, root: Any, positions: list[int]) -> None:
        wanted = np.asarray(sorted(set(positions)), dtype=np.int64)
        self._where = {int(p): i for i, p in enumerate(wanted)}
        self._codes = {
            name: np.asarray(root[name].oindex[wanted])
            for name in self._PLANES
            if name in root and len(wanted)
        }

    def __getitem__(self, name: str) -> _GatheredPlane:
        return _GatheredPlane(self._codes[name], self._where)


class _GatheredPlane:
    def __init__(self, codes: np.ndarray, where: dict[int, int]) -> None:
        self._codes, self._where = codes, where

    def __getitem__(self, position: int) -> np.ndarray:
        return np.asarray(self._codes[self._where[position]])


def _validate_one_cell(
    root: Any,
    gathered: _Gathered,
    index: Any,
    encoding: enc.StoreEncoding,
    label: str,
    cell: _SampledCell,
    src_pos: int,
) -> str | None:
    v, a = cell.variant, cell.analysis
    if src_pos < 0:
        return (
            f"{label}/{GROUP}: variant {v} in Analysis {a} is missing "
            "from the Analysis-sorted planes"
        )
    tag = f"{label}/{GROUP}: (variant={v}, analysis={a})"
    b_z, b_se, b_eaf, b_imp, b_n = cell.codes
    err = _check_cell_codes(gathered, b_z, b_se, b_eaf, b_imp, tag, src_pos, b_n)
    if err is not None:
        return err
    return _check_cell_exceptions(
        root, index, encoding, b_z, b_se, b_eaf, tag, cell.ordinal, src_pos
    )


def _validate_variant_index_sample(
    root: Any,
    index: Any,
    offsets: np.ndarray,
    by_offsets: np.ndarray,
    encoding: enc.StoreEncoding,
    n_assoc: int,
    errors: list[str],
    label: str,
    *,
    sample_variants: int = 100,
) -> None:
    """Sampled cell comparison: cells in `by_variant` match the Analysis-sorted planes.

    The first disagreement, in sampled-variant order, is the one reported.
    """
    if n_assoc == 0:
        return
    variants = [int(v) for v in _sample_non_empty_variants(by_offsets, sample_variants)]
    cells = _sampled_cells(root, index, by_offsets, variants)
    positions = _source_positions(root, offsets, cells)
    gathered = _Gathered(root, [p for p in positions.values() if p >= 0])
    for cell in cells:
        err = _validate_one_cell(
            root, gathered, index, encoding, label, cell, positions[(cell.variant, cell.analysis)]
        )
        if err is not None:
            errors.append(err)
            return


def _validate_one_index_array(index: Any, name: str, errors: list[str], label: str) -> None:
    """One `by_variant/` array's inner chunk and shard against its role policy."""
    path = f"ragged/by_variant/{name}"
    role = role_for_array_path(path)
    if role is None:
        errors.append(f"{label}/{path}: no ArrayRole is registered for this path")
        return
    array = index[name]
    shape = tuple(int(size) for size in array.shape)
    expected = chunk_layout(role, shape, hint=inner_chunk_hint_for_path(path))
    if tuple(inner_chunk_of(array)) != expected:
        errors.append(
            f"{label}/{path}: inner chunk {tuple(inner_chunk_of(array))} != the "
            f"role policy's {expected}"
        )
    expected_shard = shard_layout(role, shape, inner_chunk=expected)
    if array.shards is not None and tuple(int(s) for s in array.shards) != expected_shard:
        errors.append(
            f"{label}/{path}: shard {tuple(int(s) for s in array.shards)} != the "
            f"role policy's {expected_shard}"
        )


_INDEXED = (PRESENT, OFFSETS)

#: The rules about the index's own arrays, which read nothing of the sequence's.
CHECKS: tuple[Rule, ...] = (
    Check(name=PRESENT, kinds=Kind.CSR, after=(ARRAYS,), run=present),
    Check(name="by_variant.provenance", kinds=Kind.CSR, after=(PRESENT, LENGTHS), run=provenance),
    Check(name=OFFSETS, kinds=Kind.CSR, after=(PRESENT, LENGTHS), run=offsets),
    Check(name="by_variant.members", kinds=Kind.CSR, after=(PRESENT,), run=members),
    Check(name="by_variant.layout", kinds=Kind.CSR, after=(PRESENT,), run=layout),
    Fold(
        name="by_variant.analyses",
        kinds=Kind.CSR,
        after=_INDEXED,
        scan=BY_VARIANT,
        zero=Span(),
        measure=_span,
        judge=_judge_span,
    ),
    Fold(
        name="by_variant.order",
        kinds=Kind.CSR,
        after=_INDEXED,
        scan=BY_VARIANT,
        zero=0,
        measure=_decreasing,
        judge=_judge_order,
    ),
)

#: The rule holding each variant's run to the sequence, over its one read.
COUNTS: Rule = Fold(
    name="by_variant.counts",
    kinds=Kind.CSR,
    after=(ARRAYS, LENGTHS, *PLAN, *_INDEXED),
    scan=ASSOCIATIONS,
    zero=Histogram(),
    measure=_variant_counts,
    judge=_judge_counts,
)

#: The sampled comparison, once every rule about the index's shape has passed, and
#: the sequence's exact-value tables, which it compares the index's against, agree
#: with the cells the sequence marks.
SAMPLE: Rule = Check(
    name="by_variant.sample",
    kinds=Kind.CSR,
    after=(
        *_INDEXED,
        "by_variant.provenance",
        "by_variant.members",
        "by_variant.layout",
        "by_variant.analyses",
        "by_variant.order",
        "by_variant.counts",
        "side_tables.z_overflow",
        "side_tables.eaf_exceptions",
        "side_tables.se_exceptions",
    ),
    run=sample,
)

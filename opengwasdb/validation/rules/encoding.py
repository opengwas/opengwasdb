"""Rules that hold a component's arrays to the encoding its manifest declares (ADR 0074).

The plan is authoritative: a reader decodes what the manifest says, so a plane
that is not in its declared encoding does not read as "the other encoding" -- it
reads as plausible, wrong values (issue #119, #106). Each rule here reads array
metadata and the small side arrays beside a plane: its dtype, its per-variant
chunking, the `eaf`, `n`, `se` and `z` side arrays the plan names, and those
arrays' own consistency. A Dense grid and an association sequence are judged by
the same rules; the label in a message is the group they read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionTable,
    StoreEncoding,
    ZOverflowTable,
    per_variant_chunk_size,
)
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.arrays import array_length, inner_chunk_of
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.csr import ARRAYS
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import per_variant

#: The rules a component's values are decoded under: a decode needs every one.
PLAN = (
    "encoding.dtypes",
    "encoding.per_variant_chunks",
    "encoding.eaf",
    "encoding.n",
    "encoding.se",
    "encoding.z",
)


SE_TABLE_NAMES = ("se_exception_index", "se_exception_value", "se_coefficients")

#: How an EAF plan rule reads a per-variant array: once, for the plan and the decode.
_Read = Callable[[str], np.ndarray]

#: The arrays each exact-value side table is stored in, as its reader names them.
EAF_EXCEPTION_INDEX = EafExceptionTable.index_name
EAF_EXCEPTION_VALUE = EafExceptionTable.value_name
Z_OVERFLOW_INDEX = ZOverflowTable.index_name
Z_OVERFLOW_VALUE = ZOverflowTable.value_name


def plan_label(target: Target) -> str:
    """The group a plan message names: `data.zarr` for a grid, `data.zarr/ragged` else."""
    return "data.zarr/ragged" if target.kind is Kind.CSR else "data.zarr"


def plane_label(target: Target, plane: str) -> str:
    """The plane a value message names, as each component has always named it.

    An Overflow's value messages say which component they are about; a Ragged
    release's and a grid's name the plane's path.
    """
    if target.kind is Kind.CSR and target.layout is PrimaryStorageLayout.HYBRID:
        return f"Ragged Overflow {plane}"
    return f"{plan_label(target)}/{plane}"


def dtypes(target: Target) -> list[str]:
    """Each statistic plane present has the dtype its declared encoding stores."""
    group, encoding, label = target.group, target.encoding, plan_label(target)
    found = []
    for name, declared_plane in (("z", encoding.z), ("se", encoding.se), ("eaf", encoding.eaf)):
        if name not in group:
            continue
        actual = str(group[name].dtype)
        if actual != declared_plane.dtype:
            found.append(
                f"{label}/{name} has dtype {actual} but the manifest declares "
                f"{declared_plane.kind} ({declared_plane.dtype})"
            )
    return found


def _baseline_in_unit_interval(array: Any) -> bool:
    values = np.asarray(array[:], dtype=np.float64)
    finite = np.isfinite(values)
    return not bool(np.any(finite & ((values <= 0.0) | (values >= 1.0))))


def _frequencies_in_unit_interval(array: Any) -> bool:
    values = np.asarray(array[:], dtype=np.float64)
    finite = np.isfinite(values)
    return not bool(np.any(finite & ((values < 0.0) | (values > 1.0))))


def _validate_per_variant_chunking(group: Any, errors: list[str], *, label: str) -> None:
    """Reject a per-variant array chunked as one block (issue #135).

    A whole-array chunk means every read of the frequency plane decompresses
    the entire baseline, whatever it asked for, so the chunking is part of the
    release's contract rather than a detail of how it was written.
    """
    for name in (EAF_BASELINE, EAF_REFERENCE):
        if name not in group:
            continue
        array = group[name]
        expected = per_variant_chunk_size(group, array_length(array))
        actual = inner_chunk_of(array)[0]
        if actual > expected:
            errors.append(
                f"{label}/{name} has inner chunk shape {inner_chunk_of(array)}; its "
                f"per-variant chunk must be no larger than {expected}, matching this "
                "component's variant/read axis rather than spanning the whole array. "
                "(A sharded array's *inner* chunk is the unit a query reads; the shard "
                "is not judged here.)"
            )


def _se_coefficients_errors(group: Any, label: str) -> list[str]:
    """`se_coefficients` is `float32`, `(n_analyses, 2)`, and wholly finite."""
    coefficients = group["se_coefficients"]
    n_analyses = (
        int(group["se"].shape[1])
        if group["se"].ndim == 2
        else array_length(group["offsets"]) - 1
    )
    if str(coefficients.dtype) != "float32" or tuple(coefficients.shape) != (n_analyses, 2):
        return [f"{label}/se_coefficients must have float32 shape ({n_analyses}, 2)"]
    if not np.all(np.isfinite(coefficients[:])):
        return [f"{label}/se_coefficients contains non-finite values"]
    return []


def _se_table_shape_errors(index: Any, value: Any, label: str) -> list[str]:
    """The two side arrays are parallel, one-dimensional and correctly typed."""
    if (
        index.ndim != 1
        or value.ndim != 1
        # zarr 3 removed ``len(Array)``; the leading axis is the length.
        or int(index.shape[0]) != int(value.shape[0])
    ):
        return [f"{label} se exception arrays must be parallel one-dimensional arrays"]
    if str(index.dtype) != "int64" or str(value.dtype) != "float32":
        return [f"{label} se exception arrays must use int64 positions and float32 values"]
    return []


def _se_table_content_errors(
    positions: np.ndarray, values: np.ndarray, plane_size: int, label: str
) -> list[str]:
    """The keys are sorted, unique and inside the plane; the values are finite."""
    found = []
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        found.append(f"{label} se exception positions are unsorted or contain duplicates")
    elif len(positions) and (int(positions[0]) < 0 or int(positions[-1]) >= plane_size):
        found.append(f"{label} se exception position is outside the se plane")
    if not np.all(np.isfinite(values)):
        found.append(f"{label} se exception values contain non-finite values")
    return found


def _se_exception_table_errors(group: Any, label: str) -> list[str]:
    """The side table is parallel, typed, sorted, unique, in range and finite."""
    index = group["se_exception_index"]
    value = group["se_exception_value"]
    shape_errors = _se_table_shape_errors(index, value, label)
    if shape_errors:
        return shape_errors
    return _se_table_content_errors(
        np.asarray(index[:], dtype=np.int64),
        np.asarray(value[:], dtype=np.float32),
        int(group["se"].size),
        label,
    )


def _se_side_array_errors(group: Any, encoding: StoreEncoding, label: str) -> list[str]:
    """What a residual plane needs beside it before its contents can be judged."""
    if missing := [name for name in SE_TABLE_NAMES if name not in group]:
        return [f"{label} residual se is missing required arrays: {missing}"]
    if encoding.eaf.is_absent or "eaf" not in group:
        return [f"{label} residual se requires a complete eaf plane beside it"]
    return []


def _validate_se_plan(
    group: Any, encoding: StoreEncoding, errors: list[str], *, label: str
) -> None:
    """store-format spec §20: the `se` plane and its side arrays agree with the plan.

    A residual plane is unreadable without its coefficients, its exception
    table *and* a complete `eaf` plane; a `float16` plane must carry none of
    them, so a leftover side array is a failure rather than a harmless relic.
    """
    if not encoding.se.is_residual:
        if any(name in group for name in SE_TABLE_NAMES):
            errors.append(f"{label} carries residual se side arrays but declares float16 se")
        return
    if present := _se_side_array_errors(group, encoding, label):
        errors.extend(present)
        return
    errors.extend(_se_coefficients_errors(group, label))
    errors.extend(_se_exception_table_errors(group, label))


def _validate_z_plan(
    group: Any, encoding: StoreEncoding, errors: list[str], *, label: str
) -> None:
    """Seam: the fixed-point `z` side table agrees with the declared plan.

    The overflow side table is only read under the fixed-point encoding the
    manifest declares, so a stray table under float z is a failure rather than
    a harmless relic, and a declared fixed-point z without its exact-value
    side table cannot be decoded faithfully at all (issue #119, issue #106).
    """
    declared = encoding.z
    has_table = Z_OVERFLOW_INDEX in group or Z_OVERFLOW_VALUE in group
    if not declared.is_fixed_point:
        if has_table:
            errors.append(
                f"{label} carries a z overflow table but declares no fixed-point z encoding"
            )
        return
    missing = [name for name in (Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE) if name not in group]
    if missing:
        errors.append(
            f"{label} declares fixed-point z but is missing "
            f"{Z_OVERFLOW_INDEX}/{Z_OVERFLOW_VALUE}, which hold the exact values of "
            "any cell outside the representable range"
        )
        return
    _validate_z_overflow_table(ZOverflowTable.read(group), errors, label=label)


def _validate_z_overflow_table(
    table: ZOverflowTable, errors: list[str], *, label: str
) -> None:
    """Seam: a present fixed-point `z` side table is self-consistent.

    Positions must pair one-to-one with values, be sorted and unrepeated, and
    every stored exact value must be finite -- the table is the authoritative
    decode for cells the fixed-point plane cannot represent.
    """
    if len(table.index) != len(table.value):
        errors.append(
            f"{label} z overflow table has {len(table.index)} positions but "
            f"{len(table.value)} values"
        )
        return
    if len(table.index) > 1 and np.any(table.index[1:] <= table.index[:-1]):
        errors.append(f"{label} z overflow table is not sorted by position, or repeats one")
        return
    if len(table.index) and not np.all(np.isfinite(table.value)):
        errors.append(f"{label} z overflow table contains non-finite values")


def _validate_n_plan(
    group: Any, encoding: StoreEncoding, errors: list[str], *, label: str
) -> None:
    """The `n` plane is present exactly when the plan declares it, in its dtype,
    and only beside a `float16` `se` plane (ADR 0067 §4)."""
    present = "n" in group
    if encoding.n is not None and encoding.se.is_residual:
        errors.append(
            f"{label} declares an n plane beside se {encoding.se.kind}, but an n plane "
            "requires float16 se (ADR 0067 §4)"
        )
    if encoding.n is None:
        if present:
            errors.append(
                f"{label} carries an n plane but the manifest declares none; a reader "
                "following the plan would never read those sample sizes (ADR 0067)"
            )
        return
    if not present:
        errors.append(f"{label} declares an n plane but carries none (ADR 0067)")
        return
    actual = str(group["n"].dtype)
    if actual != encoding.n.dtype:
        errors.append(
            f"{label}/n has dtype {actual} but the manifest declares "
            f"{encoding.n.kind} ({encoding.n.dtype})"
        )


def _eaf_plane(group: Any, declared: Any, label: str) -> Iterator[str]:
    """A plane exactly when the plan declares one."""
    present = "eaf" in group
    if declared.is_absent and present:
        yield f"{label} carries an eaf plane but the manifest declares eaf absent"
    if not present and declared.kind in ("float32", "int8_residual"):
        yield f"{label} declares an eaf encoding of {declared.kind!r} but carries no eaf plane"


def _eaf_stray_residual_arrays(group: Any, label: str) -> Iterator[str]:
    """No baseline and no exception table beside a plane that is not residual-coded."""
    if EAF_BASELINE in group:
        yield f"{label} carries {EAF_BASELINE} but declares no residual-coded eaf plane"
    if EAF_EXCEPTION_INDEX in group or EAF_EXCEPTION_VALUE in group:
        yield f"{label} carries an eaf exception table but declares no residual-coded eaf plane"


def _eaf_residual_arrays(group: Any, read: _Read, label: str) -> Iterator[str]:
    """A residual plane's baseline, in (0, 1), and its exception table, both present."""
    if EAF_BASELINE not in group:
        yield (
            f"{label} declares a residual-coded eaf plane but is missing {EAF_BASELINE}, "
            "without which the plane cannot be decoded"
        )
    elif not _baseline_in_unit_interval(read(EAF_BASELINE)):
        yield f"{label}/{EAF_BASELINE} holds a finite value outside (0, 1)"
    if EAF_EXCEPTION_INDEX not in group or EAF_EXCEPTION_VALUE not in group:
        yield (
            f"{label} declares a residual-coded eaf plane but is missing "
            f"{EAF_EXCEPTION_INDEX}/{EAF_EXCEPTION_VALUE}, which hold the exact values "
            "of any cell the residual coding cannot express"
        )
    else:
        yield from _eaf_exception_table(EafExceptionTable.read(group), label)


def _eaf_exception_table(table: EafExceptionTable, label: str) -> Iterator[str]:
    """Positions pair with values, sorted and unrepeated; every value a frequency."""
    if len(table.index) != len(table.value):
        yield (
            f"{label} eaf exception table has {len(table.index)} positions but "
            f"{len(table.value)} values"
        )
    elif len(table.index) > 1 and np.any(table.index[1:] <= table.index[:-1]):
        yield f"{label} eaf exception table is not sorted by position, or repeats one"
    elif len(table.value) and not np.all(
        np.isfinite(table.value) & (table.value >= 0.0) & (table.value <= 1.0)
    ):
        yield f"{label} eaf exception table holds a value that is not a frequency in [0, 1]"


def _eaf_reference(group: Any, read: _Read, declared: Any, label: str) -> Iterator[str]:
    """Reference EAF exactly when declared, beside an imputed mask, and in [0, 1]."""
    has_reference = EAF_REFERENCE in group
    if declared.reference:
        yield from _declared_reference(group, read, has_reference, label)
    elif has_reference:
        yield (
            f"{label} carries {EAF_REFERENCE} but the manifest does not declare it; a "
            "reader following the plan would never read those frequencies"
        )


def _declared_reference(
    group: Any, read: _Read, has_reference: bool, label: str
) -> Iterator[str]:
    """A declared reference: present, beside an imputed mask, and holding frequencies.

    A panel frequency is returned to the caller as it stands, so it is checked as
    a frequency here rather than only where a plane is decoded -- a release that
    carries reference EAF and no plane of its own has no decode to check it in
    (issue #113). Inclusive, unlike the baseline: 0 and 1 have no logit but are
    perfectly good frequencies.
    """
    if not has_reference:
        yield (
            f"{label} declares reference EAF for its imputed cells but carries no "
            f"{EAF_REFERENCE} array"
        )
    if "imputed" not in group:
        yield (
            f"{label} declares reference EAF but carries no imputed mask to say which "
            "cells it describes (spec §9, §15)"
        )
    if has_reference and not _frequencies_in_unit_interval(read(EAF_REFERENCE)):
        yield f"{label}/{EAF_REFERENCE} holds a finite value outside [0, 1]"


def eaf_plan(target: Target) -> Iterator[str]:
    """The `eaf` plane, its baseline, its exception table and its reference.

    They are one artifact in up to four arrays, and a component holding some of
    them and not others cannot be decoded: an `int8` residual plane without its
    baseline is meaningless, and an exception cell with no table entry has lost
    its value entirely. The plan says which should be there; this is where that
    stops being a claim.
    """
    group, declared, label = target.group, target.encoding.eaf, plan_label(target)

    def read(name: str) -> np.ndarray:
        return per_variant(target, name)

    yield from _eaf_plane(group, declared, label)
    if declared.is_residual:
        yield from _eaf_residual_arrays(group, read, label)
    else:
        yield from _eaf_stray_residual_arrays(group, label)
    yield from _eaf_reference(group, read, declared, label)


def per_variant_chunks(target: Target) -> list[str]:
    errors: list[str] = []
    _validate_per_variant_chunking(target.group, errors, label=plan_label(target))
    return errors


def n_plan(target: Target) -> list[str]:
    errors: list[str] = []
    _validate_n_plan(target.group, target.encoding, errors, label=plan_label(target))
    return errors


def se_plan(target: Target) -> list[str]:
    errors: list[str] = []
    _validate_se_plan(target.group, target.encoding, errors, label=plan_label(target))
    return errors


def z_plan(target: Target) -> list[str]:
    errors: list[str] = []
    if "z" in target.group:
        _validate_z_plan(target.group, target.encoding, errors, label=plan_label(target))
    return errors


def _plan_rule(name: str, run: Callable[[Target], Iterable[str]]) -> Rule:
    return Check(name=name, kinds=Kind.GRID | Kind.CSR, after=(REQUIRED, ARRAYS), run=run)


RULES: tuple[Rule, ...] = tuple(
    _plan_rule(name, run)
    for name, run in zip(
        PLAN, (dtypes, per_variant_chunks, eaf_plan, n_plan, se_plan, z_plan), strict=True
    )
)

"""Generic HM3 extraction for the shared benchmark workload.

An **HM3 Indexed Variant Subset** (ADR 0061) is an optional, named, derived
Dense-store query index. It is a *default* derived artifact, not a Store
validity requirement, so a benchmark that measures one must also be able to
report a Store that has none without failing the run (issue #272). This module
is the one place that decides which of those states a Store is in, and -- when
it is measurable -- times one selected Analysis's extraction from it.

The three non-measured outcomes are deliberately distinct, because "absence and
zero are different" (CONTRIBUTING):

- ``missing``: the Store can carry a subset (it has a Dense Component) and
  publishes none named ``hm3``.
- ``not_applicable``: the Store is Ragged-only, and an Indexed Variant Subset is
  Dense-only (ADR 0061).
- ``unsupported``: the Store publishes a ``hm3`` group whose *declared* schema
  or encoding is one this build cannot read. A forward-compatibility state, not
  a measurement.

Detection is by the conventional subset name, not by a maintained HapMap3 ALID
list: the benchmark reports whether a Store publishes a usable ``hm3`` subset,
not whether that subset was built from the canonical list. Verifying the list is
the maintained-resource ticket's job, not the benchmark's.

The one failure class that must never be swallowed: a subset that is *present
and structurally valid* whose extraction then fails. That is a real defect in a
valid index, and it propagates rather than being downgraded to "missing" --
reporting a broken index as absent would turn a wrong answer into a plausible
one (#272).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from opengwasdb.encoding import StoreEncoding, UnsupportedEncoding
from opengwasdb.layouts.dense.indexed_subsets import (
    ATTR_ENCODING,
    ATTR_SCHEMA,
    INDEXED_SUBSET_SCHEMA_VERSION,
    indexed_subset_namespace_path,
    list_indexed_subsets,
    open_indexed_subset,
)
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.arrays import open_group

#: The conventional name of the HapMap3 subset. A default derived artifact, so
#: an operator may omit it; the name is the benchmark's only handle on it.
HM3_SUBSET_NAME = "hm3"

#: ``measured`` carries timings and counts; the other three are non-failing and
#: say why no measurement exists.
Hm3Status = Literal["measured", "missing", "unsupported", "not_applicable"]


@dataclass(frozen=True)
class Hm3State:
    """Which HM3 state a Store is in, with a human-readable reason."""

    status: Hm3Status
    detail: str


def _declared_contract_is_unsupported(store_path: Path, name: str) -> str | None:
    """The reason a published group declares a contract this build cannot read.

    Returns ``None`` when the declared schema and encoding are ones this build
    understands, or when the group is unreadable for some *other* reason (a
    damaged group is corruption, and ``open_indexed_subset`` must surface it
    loudly). The check is deliberately narrow -- schema and encoding kind only
    -- so it can never absorb a decoding or identity failure as "unsupported".
    """
    # The namespace resolves a Hybrid root to its nested Dense Component, the
    # same place the reader finds the group (issue #271).
    group_path = indexed_subset_namespace_path(store_path) / name
    try:
        attrs = dict(open_group(group_path).attrs)
    except Exception:
        return None
    schema = attrs.get(ATTR_SCHEMA)
    if (
        isinstance(schema, int)
        and not isinstance(schema, bool)
        and schema != INDEXED_SUBSET_SCHEMA_VERSION
    ):
        return f"declares subset schema {schema!r}, not {INDEXED_SUBSET_SCHEMA_VERSION}"
    try:
        StoreEncoding.from_manifest(attrs[ATTR_ENCODING])
    except UnsupportedEncoding as exc:
        return f"declares an encoding this build cannot read: {exc}"
    except (KeyError, TypeError, ValueError, AttributeError):
        # A malformed encoding block is a damaged group, not a future contract.
        return None
    return None


def classify_hm3(store_path: str | Path) -> Hm3State:
    """Which of the four HM3 states a Store Release is in.

    Ragged is decided from the layout before the filesystem is inspected: a
    Ragged release has no Dense Component, so a ``hm3`` directory under its
    ``data.zarr`` (a stray, unmanaged entry) is not a usable subset. A Dense or
    Hybrid release resolves to its Dense Component through the same
    ``list_indexed_subsets`` the reader uses, so a Hybrid's subset is found at
    the root the operator names (issue #271).
    """
    path = Path(store_path)
    manifest = StoreManifest.load(path)
    if manifest.primary_layout is PrimaryStorageLayout.RAGGED:
        return Hm3State(
            status="not_applicable",
            detail=(
                "Ragged layout carries no Dense Component; an Indexed Variant "
                "Subset is Dense-only (ADR 0061)"
            ),
        )
    if HM3_SUBSET_NAME not in list_indexed_subsets(path):
        return Hm3State(
            status="missing",
            detail=f"no Indexed Variant Subset named {HM3_SUBSET_NAME!r} is published",
        )
    unsupported = _declared_contract_is_unsupported(path, HM3_SUBSET_NAME)
    if unsupported is not None:
        return Hm3State(status="unsupported", detail=unsupported)
    # A damaged, stale or otherwise invalid subset is *present*, so it must
    # surface loudly rather than be reported as missing -- that downgrade is the
    # plausible-wrong-answer this project refuses (#272).
    open_indexed_subset(path, HM3_SUBSET_NAME)
    return Hm3State(status="measured", detail="")


def _percentile_ms(times: list[float], fraction: float) -> float:
    ordered = sorted(times)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def measure_hm3_extract(
    query: Any,
    analysis_id: str,
    *,
    store_path: str | Path | None = None,
    reps: int = 5,
) -> dict[str, Any]:
    """One Analysis's full HM3 extraction, or the non-failing reason it was not run.

    ``query`` is an opened query handle; ``store_path`` overrides the path used
    for detection when the handle is a nested component (defaults to
    ``query.store.path``). A measured record times a warm-up read and ``reps``
    timed reads of ``analysis(analysis_id, indexed_subset="hm3")`` and reports
    the result count alongside the subset's own recorded counts. A non-measured
    record carries the status and its reason and no timing keys, so a caller
    cannot mistake "not measured" for a zero-row measurement.
    """
    path = Path(store_path) if store_path is not None else Path(query.store.path)
    state = classify_hm3(path)
    record: dict[str, Any] = {
        "subset": HM3_SUBSET_NAME,
        "status": state.status,
        "detail": state.detail,
    }
    if state.status != "measured":
        return record

    # The subset is present and structurally valid. From here any failure is a
    # valid index failing to answer and must propagate (see module docstring).
    subset = open_indexed_subset(path, HM3_SUBSET_NAME)
    started = time.perf_counter()
    warm = query.analysis(analysis_id, indexed_subset=HM3_SUBSET_NAME)
    first_read_ms = (time.perf_counter() - started) * 1000.0
    times: list[float] = []
    result_count = 0
    for _ in range(max(1, int(reps))):
        started = time.perf_counter()
        result = query.analysis(analysis_id, indexed_subset=HM3_SUBSET_NAME)
        times.append((time.perf_counter() - started) * 1000.0)
        result_count = int(len(result["z"]))
    record.update(
        {
            "analysis_id": analysis_id,
            "first_read_ms": round(first_read_ms, 3),
            "median_ms": round(_percentile_ms(times, 0.5), 3),
            "p95_ms": round(_percentile_ms(times, 0.95), 3),
            "reps": len(times),
            "result_count": result_count,
            "warm_result_count": int(len(warm["z"])),
            "subset_n_variants": int(subset.n_subset_variants),
            "subset_n_analyses": int(subset.n_analyses),
            "subset_requested_count": int(subset.requested_count),
            "subset_resolved_count": int(subset.resolved_count),
            "subset_absent_count": int(subset.absent_count),
            "subset_reference_assembly": str(subset.reference_assembly),
            "subset_input_sha256": str(subset.input_sha256),
        }
    )
    return record

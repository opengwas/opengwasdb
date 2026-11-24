"""Rules about a Hybrid release's two components and the variant space they share (ADR 0074).

A Hybrid release is a Ragged Overflow and a nested Dense Component (spec §16).
The component must open as a Dense release, written under the release's own
plan: the two partition one Analysis's associations, so a result assembled
from both is coherent only if they were encoded the same way (issue #119).
`dense_to_shared.npy` maps each Dense Component row to its row of the shared
variant table. It is judged against both variant axes. Against the shared one
(`hybrid.dense_to_shared`): it names rows that exist, in strictly ascending order,
so no shared row twice and the map can be searched, before anything can be said
about which shared rows are on the panel. Against the Dense side
(`hybrid.coverage`): it names one shared row for each of the component's rows, and
each row it maps holds the component's own variant. From format 0.3.0 the Dense
side is only the panel, the rows of its grid: the component carries no variant of
its own (#472), so the map is the whole of its identity and nothing is compared by
ALID. Before 0.3.0 the component's own Variant Table is the Dense side, and its
ALIDs are compared with the shared rows the map names.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.hybrid import dense_to_shared_path
from opengwasdb.store_reader.release import release_arrays
from opengwasdb.store_reader.variant_axis import VariantAxis
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.rules.variants import representative_variant_indices
from opengwasdb.validation.scans import release_rows

#: The rule every rule reading the Dense Component's place in the shared space runs after.
DENSE_TO_SHARED = "hybrid.dense_to_shared"
#: The rule every rule reading the Dense Component runs after: it opens as Dense.
COMPONENT = "hybrid.component"


def is_hybrid(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.HYBRID and not target.is_component


def _release(target: Target) -> Target:
    """The outer release a Hybrid target belongs to."""
    while target.parent is not None:
        target = target.parent
    return target


def component(target: Target) -> Target:
    """The Dense Component's release target, which exists once `hybrid.component` passed."""
    release = _release(target)
    return next(child for child in release.children if child.kind is Kind.RELEASE)


def dense_to_shared(target: Target) -> np.ndarray:
    """The shared row of each Dense Component row, read once for the run."""
    release = _release(target)
    return release.remember(
        "dense_to_shared", lambda: np.load(dense_to_shared_path(release.store.path))
    )


def mapping(target: Target) -> Iterator[str]:
    """`dense_to_shared` is a strictly-increasing set of valid shared rows.

    Each fault is named by itself: a row outside the shared table, a shared row named
    twice, and an order that is not ascending, which `searchsorted` cannot search.
    """
    rows = np.asarray(dense_to_shared(target), dtype=np.int64)
    n_shared = release_rows(target)
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= n_shared):
        outside = int(np.flatnonzero((rows < 0) | (rows >= n_shared))[0])
        yield (
            "dense_to_shared references rows outside the shared variant table: Dense "
            f"Component row {outside} names shared row {int(rows[outside])} of {n_shared}"
        )
        return
    steps = np.diff(rows)
    if (steps == 0).any():
        twice = int(np.flatnonzero(steps == 0)[0])
        yield (
            "dense_to_shared maps two Dense Component rows to one shared row: rows "
            f"{twice} and {twice + 1} both name {int(rows[twice])}"
        )
    elif len(rows) != len(np.unique(rows)):
        yield "dense_to_shared maps two Dense Component rows to one shared row"
    if (steps < 0).any():
        down = int(np.flatnonzero(steps < 0)[0])
        yield (
            "dense_to_shared is not ascending: Dense Component row "
            f"{down + 1} names shared row {int(rows[down + 1])}, before row {down}'s "
            f"{int(rows[down])}; the map is searched, and must keep the panel's order"
        )


RULES: tuple[Rule, ...] = (
    Check(name=DENSE_TO_SHARED, kinds=Kind.RELEASE, when=is_hybrid, after=(REQUIRED,), run=mapping),
)


def plan(target: Target) -> Iterator[str]:
    """Both components declare one statistic encoding.

    Each carries its own manifest, so nothing but this rule couples them.
    `eaf_reference` is the one field that legitimately differs: it records a
    physical fact about the component that declares it, and only the Dense
    Component has imputed cells for a reference frequency to describe (ADR 0037
    §4). Compared with it normalised away, so a real disagreement about the
    *encoding* still fails.
    """
    encoding = component(target).store.manifest.encoding
    release = target.store.manifest.encoding
    if encoding.with_eaf_reference(False) != release:
        yield (
            "the Dense Component and the Hybrid release declare different statistic "
            f"encodings ({encoding.to_manifest()} vs {release.to_manifest()}); both "
            "components of a Hybrid release are written under one plan"
        )


def coverage(target: Target) -> Iterator[str]:
    """The map names one shared row for each Dense Component row, and the shared row
    each maps to holds that row's variant.

    The Dense side's rows are its grid's from 0.3.0, where the component carries no
    variant of its own (#472), and its Variant Table's before: there the ALIDs of
    representative rows are compared too, as they always were.
    """
    rows = dense_to_shared(target)
    dense_target = component(target)
    if dense_target.shares_variant_axis:
        yield from _panel_coverage(rows, dense_target)
        return
    n_dense = dense_target.variant_axis.n_variants
    if len(rows) != n_dense:
        yield (
            f"dense_to_shared maps {len(rows)} Dense Component rows, but its Variant Table "
            f"holds {n_dense}"
        )
        return
    picked = list(representative_variant_indices(len(rows)))
    dense = _alids(dense_target.variant_axis, picked)
    shared = _alids(target.variant_axis, [int(rows[r]) for r in picked])
    for r, dense_alid, shared_alid in zip(picked, dense, shared, strict=True):
        if dense_alid is None or shared_alid is None or dense_alid != shared_alid:
            yield (
                "dense_to_shared coverage violated: Dense Component row "
                f"{r} does not match its shared variant table row"
            )
            return


def _panel_coverage(rows: np.ndarray, dense_target: Target) -> Iterator[str]:
    """From 0.3.0: the map names one shared row for each row of the component's grid."""
    n_panel = int(release_arrays(dense_target.store.path)["z"].shape[0])
    if len(rows) != n_panel:
        yield (
            f"dense_to_shared maps {len(rows)} Dense Component rows, but its grid holds "
            f"{n_panel}: each grid row must name its shared variant"
        )


def _alids(axis: VariantAxis, rows: list[int]) -> list[str | None]:
    """Each row's ALID, None for a row the axis lacks: from the table's rows on the 0.2
    series, from the ALID sidecar on 0.3.0, whose Store Reader reads no table (#469)."""
    if axis.reads_table:
        records = [axis.by_index(row) for row in rows]
        return [None if record is None else record.alid for record in records]
    alids: list[str | None] = axis.alid_index.alids(np.asarray(rows, dtype=np.int64)).tolist()
    return alids


#: The rules about the Dense Component, once it has opened as one.
COMPONENT_RULES: tuple[Rule, ...] = (
    Check(name="hybrid.plan", kinds=Kind.RELEASE, when=is_hybrid, after=(COMPONENT,), run=plan),
    Check(
        name="hybrid.coverage",
        kinds=Kind.RELEASE,
        when=is_hybrid,
        after=(COMPONENT, DENSE_TO_SHARED),
        run=coverage,
    ),
)

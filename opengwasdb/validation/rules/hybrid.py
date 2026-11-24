"""Rules about a Hybrid release's two components and the variant space they share (ADR 0074).

A Hybrid release is a Ragged Overflow and a nested Dense Component (spec §16).
The component must open as a Dense release, written under the release's own
plan: the two partition one Analysis's associations, so a result assembled
from both is coherent only if they were encoded the same way (issue #119).
`dense_to_shared.npy` maps each Dense Component row to its row of the shared
variant table. It must name rows that exist, and no shared row twice, before
anything can be said about which shared rows are on the panel, and each row it
maps must hold the component's own variant.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.hybrid import dense_to_shared_path
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
    """`dense_to_shared` is a strictly-increasing set of valid shared rows."""
    rows = dense_to_shared(target)
    n_shared = release_rows(target)
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= n_shared):
        yield "dense_to_shared references rows outside the shared variant table"
    elif len(rows) != len(np.unique(rows)):
        yield "dense_to_shared maps two Dense Component rows to one shared row"


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
    """The shared row each Dense Component row maps to holds that row's variant."""
    rows = dense_to_shared(target)
    picked = list(representative_variant_indices(len(rows)))
    dense = _alids(component(target).variant_axis, picked)
    shared = _alids(target.variant_axis, [int(rows[r]) for r in picked])
    for r, dense_alid, shared_alid in zip(picked, dense, shared, strict=True):
        if dense_alid is None or shared_alid is None or dense_alid != shared_alid:
            yield (
                "dense_to_shared coverage violated: Dense Component row "
                f"{r} does not match its shared variant table row"
            )
            return


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

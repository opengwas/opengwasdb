"""Shared pieces of Reference-Completed manifest construction, used by the
Dense and Ragged builders.

``build_completion_provenance`` records the parameters every Reference
Completion run reports, regardless of layout; each layout passes its own
completion counters as ``extra``. ``completed_release_manifest`` builds the
completed ``StoreManifest`` itself -- the fields a completion preserves or
re-stamps from its source are identical across the two layouts, as is the
provenance merge recording which release was completed and with what
parameters.

Hybrid's ``provenance["completion"]`` key is structurally different -- it
records *which component* was completed, not imputation parameters -- so it
is not built from these helpers.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from opengwasdb.completion.block import COMPLETION_METHOD
from opengwasdb.encoding import StoreEncoding
from opengwasdb.model.enums import CompletionState
from opengwasdb.model.manifest import StoreManifest


def build_completion_provenance(
    *,
    ld_panel_id: str,
    ancestry: str,
    min_cor: float,
    thresh: float,
    n_variants_total: int,
    n_variants_new: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "method": COMPLETION_METHOD,
        "ld_panel_id": ld_panel_id,
        "ancestry": ancestry,
        "min_cor": min_cor,
        "pca_thresh": thresh,
        "n_variants_total": n_variants_total,
        "n_variants_new": n_variants_new,
        **extra,
    }


def completed_release_manifest(
    source_manifest: StoreManifest,
    *,
    encoding: StoreEncoding,
    release_id: str | None,
    source_format_version: str,
    completion_provenance: dict[str, Any],
) -> StoreManifest:
    """The Reference-Completed release's manifest, shared by the Dense and
    Ragged builders. A completion writes into its source's arrays, so the
    completed release keeps its source's ``store_id``, ``primary_layout``,
    ``association_coverage``, ``reference_assembly`` and ``encoding``, and
    preserves -- not re-stamps -- its ``format_version`` (ADR 0038 §4); the
    one encoding addition, ``eaf_reference``, is recorded in the encoding
    itself rather than here (ADR 0037 §4). ``release_id`` defaults to the
    source's with ``-completed`` appended when none is supplied, exactly as
    each builder used to compute it. ``completion_provenance`` is the
    caller's own ``build_completion_provenance`` result: the completion
    counters a layout reports differ, so each layout builds its own."""
    return StoreManifest(
        encoding=encoding,
        store_id=source_manifest.store_id,
        release_id=release_id or f"{source_manifest.release_id}-completed",
        format_version=source_format_version,
        primary_layout=source_manifest.primary_layout,
        association_coverage=source_manifest.association_coverage,
        completion_state=CompletionState.REFERENCE_COMPLETED,
        reference_assembly=source_manifest.reference_assembly,
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            **source_manifest.provenance,
            "source_release_id": source_manifest.release_id,
            "completion": completion_provenance,
        },
    )

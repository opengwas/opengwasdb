"""Residual-SE Store Releases for the association-read tests (#253, ADR 0070).

Residual SE (ADR 0037 §3) is decoded through the frequency plane, so these are
the releases on which a reader's shared EAF read, and the panel substitution on
imputed cells, actually decide what SE and EAF read back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr
from residual_fixtures import residual_eligible_records
from test_hybrid_completion import (
    _residual_hybrid_crossover_source,
    _residual_ld_panel_with_crossover,
)
from test_se_residual_encoding import _residual_source_and_panel

from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

#: The panel frequency shift the completed Dense release applies, so an observed
#: cell's own frequency differs from the panel's and an unmasked substitution is
#: visible. Well inside the residual coding's range and the panel's [0, 1].
PANEL_SHIFT = 0.15

#: The variant row `dense_observed_missing` leaves Analysis `b` unobserved at, in
#: the middle of the axis so that a contiguous (mask-free) slice of the shared
#: array cannot coincide with the present mask.
MISSING_VARIANT = 300


def dense_completed(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A Reference-Completed residual Dense release whose panel differs.

    Returns the store, the observed frequency of each variant row, and the
    panel frequency now held in `eaf_reference`. The panel is overwritten so it
    is *not* the observed frequency: otherwise a masked substitution and an
    unmasked one read the same number (ADR 0037 §4's 3000x error is exactly a
    panel value that differs from the cohort's). Analysis `a` observes every
    variant; `b` misses some, which completion imputes.
    """
    source, panel_path, _expected = _residual_source_and_panel(tmp_path)
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        source, completed, panel_path, ancestry="EUR", min_cor=0.0, release_id="comp"
    )
    root = zarr.open_group(str(completed / "data.zarr"), mode="r+", zarr_format=3)
    n_variants = int(root["eaf_reference"].shape[0])
    observed = np.linspace(0.05, 0.95, n_variants, dtype=np.float32)
    panel = np.clip(observed + PANEL_SHIFT, 0.02, 0.98).astype(np.float32)
    root["eaf_reference"][:] = panel
    return completed, observed, panel


def dense_observed_missing(tmp_path: Path) -> Path:
    """A residual Dense observed release with one genuinely missing cell."""
    records, _expected = residual_eligible_records(600)
    kept = [
        r
        for r in records
        if not (r.analysis_id == "b" and r.variant.position == MISSING_VARIANT)
    ]
    assert len(kept) == len(records) - 1, "the fixture must drop exactly one cell"
    store = tmp_path / "obs.opengwasdb"
    build_dense_observed_store(
        kept,
        store,
        store_id="s",
        release_id="obs",
        reference_assembly="GRCh38",
        chunk_shape=(100, 2),
    )
    return store


def hybrid_completed(tmp_path: Path) -> Path:
    """A Reference-Completed Hybrid with residual SE and imputed Dense cells."""
    src, crossover_alid, _crossover_se, crossover_eaf = _residual_hybrid_crossover_source(
        tmp_path
    )
    ld = _residual_ld_panel_with_crossover(tmp_path, crossover_alid, crossover_eaf)
    dst = tmp_path / "comp.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)
    return dst


def copied(store: Path, directory: Path) -> Path:
    """A private copy of a built store, for a test that writes into it."""
    return Path(shutil.copytree(store, directory / store.name))

"""Dense layout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.layouts.dense.build import AnalysisSource as AnalysisSource
    from opengwasdb.layouts.dense.build import DenseBuildResult as DenseBuildResult
    from opengwasdb.layouts.dense.build import (
        build_dense_observed_store as build_dense_observed_store,
    )
    from opengwasdb.layouts.dense.constants import DEFAULT_CHUNK_SHAPE as DEFAULT_CHUNK_SHAPE

#: Each re-exported name, by the module it comes from; imported on first use
#: (`opengwasdb.lazy_exports`, #464).
_EXPORTS = {
    "opengwasdb.layouts.dense.build": (
        "AnalysisSource",
        "DenseBuildResult",
        "build_dense_observed_store",
    ),
    "opengwasdb.layouts.dense.constants": (
        "DEFAULT_CHUNK_SHAPE",
    ),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)

"""Statistic array encodings: the plan, the codec, and the decoded planes.

ADR 0037 / issue #119. `StoreEncoding` is decided once at build time from
measured properties of the data and recorded in `manifest.json`; `StoreCodec`
is the only place a store's stored bytes become physical values.
`AssociationCodec` is how a component stores associations: what it may hold
(`storable`) and every plane's codes from one call (ADR 0077).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengwasdb.lazy_exports import exported_names, lazy_hooks

if TYPE_CHECKING:
    from opengwasdb.encoding.association_codec import AssociationCodec as AssociationCodec
    from opengwasdb.encoding.association_codec import CellValues as CellValues
    from opengwasdb.encoding.association_codec import Placement as Placement
    from opengwasdb.encoding.association_codec import storable as storable
    from opengwasdb.encoding.codec import (
        DEFAULT_BASELINE_CELL_BUDGET as DEFAULT_BASELINE_CELL_BUDGET,
    )
    from opengwasdb.encoding.codec import EafExceptionBuilder as EafExceptionBuilder
    from opengwasdb.encoding.codec import SeExceptionBuilder as SeExceptionBuilder
    from opengwasdb.encoding.codec import SparseExactBuilder as SparseExactBuilder
    from opengwasdb.encoding.codec import StoreCodec as StoreCodec
    from opengwasdb.encoding.codec import ZOverflowBuilder as ZOverflowBuilder
    from opengwasdb.encoding.codec import eaf_baseline_from_grid as eaf_baseline_from_grid
    from opengwasdb.encoding.codec import eaf_baseline_from_pairs as eaf_baseline_from_pairs
    from opengwasdb.encoding.codec import (
        eaf_baseline_from_sorted_runs as eaf_baseline_from_sorted_runs,
    )
    from opengwasdb.encoding.codec import eaf_exception_table as eaf_exception_table
    from opengwasdb.encoding.codec import encode_sample_sizes as encode_sample_sizes
    from opengwasdb.encoding.measure import combine_eaf_measurements as combine_eaf_measurements
    from opengwasdb.encoding.measure import fit_se as fit_se
    from opengwasdb.encoding.measure import fit_se_grid as fit_se_grid
    from opengwasdb.encoding.measure import measure_eaf as measure_eaf
    from opengwasdb.encoding.measure import measure_eaf_grid as measure_eaf_grid
    from opengwasdb.encoding.measure import measure_eaf_sample as measure_eaf_sample
    from opengwasdb.encoding.planes import write_dense_cells as write_dense_cells
    from opengwasdb.encoding.planes import write_eaf_baseline as write_eaf_baseline
    from opengwasdb.encoding.planes import write_eaf_reference as write_eaf_reference
    from opengwasdb.encoding.planes import write_encoded_planes as write_encoded_planes
    from opengwasdb.encoding.se import OverflowCellBatches as OverflowCellBatches
    from opengwasdb.encoding.se import OverflowCells as OverflowCells
    from opengwasdb.encoding.se import optimise_dense_se as optimise_dense_se
    from opengwasdb.encoding.se import optimise_dense_se_joint as optimise_dense_se_joint
    from opengwasdb.encoding.se import write_dense_se_side_arrays as write_dense_se_side_arrays
    from opengwasdb.store.arrays import component_variant_chunk as component_variant_chunk
    from opengwasdb.store.arrays import per_variant_chunk_size as per_variant_chunk_size
    from opengwasdb.store_reader.encoding.codec import EAF_BASELINE as EAF_BASELINE
    from opengwasdb.store_reader.encoding.codec import EAF_EXCEPTION_INDEX as EAF_EXCEPTION_INDEX
    from opengwasdb.store_reader.encoding.codec import EAF_EXCEPTION_VALUE as EAF_EXCEPTION_VALUE
    from opengwasdb.store_reader.encoding.codec import EAF_REFERENCE as EAF_REFERENCE
    from opengwasdb.store_reader.encoding.codec import Z_OVERFLOW_INDEX as Z_OVERFLOW_INDEX
    from opengwasdb.store_reader.encoding.codec import Z_OVERFLOW_VALUE as Z_OVERFLOW_VALUE
    from opengwasdb.store_reader.encoding.codec import EafExceptionTable as EafExceptionTable
    from opengwasdb.store_reader.encoding.codec import SeExceptionTable as SeExceptionTable
    from opengwasdb.store_reader.encoding.codec import SparseExactTable as SparseExactTable
    from opengwasdb.store_reader.encoding.codec import ZOverflowTable as ZOverflowTable
    from opengwasdb.store_reader.encoding.codec import expit as expit
    from opengwasdb.store_reader.encoding.codec import logit as logit
    from opengwasdb.store_reader.encoding.codec import positions_at as positions_at
    from opengwasdb.store_reader.encoding.codec import positions_flat as positions_flat
    from opengwasdb.store_reader.encoding.codec import positions_pairs as positions_pairs
    from opengwasdb.store_reader.encoding.codec import positions_row_band as positions_row_band
    from opengwasdb.store_reader.encoding.codec import positions_rows_cols as positions_rows_cols
    from opengwasdb.store_reader.encoding.plan import DEFAULT_Z_SCALE as DEFAULT_Z_SCALE
    from opengwasdb.store_reader.encoding.plan import EAF_ABSENT as EAF_ABSENT
    from opengwasdb.store_reader.encoding.plan import EAF_CODE_MAX as EAF_CODE_MAX
    from opengwasdb.store_reader.encoding.plan import EAF_CODE_MIN as EAF_CODE_MIN
    from opengwasdb.store_reader.encoding.plan import EAF_EXCEPTION as EAF_EXCEPTION
    from opengwasdb.store_reader.encoding.plan import EAF_EXCEPTION_BUDGET as EAF_EXCEPTION_BUDGET
    from opengwasdb.store_reader.encoding.plan import EAF_RANGE_CANDIDATES as EAF_RANGE_CANDIDATES
    from opengwasdb.store_reader.encoding.plan import ENCODING_VERSION as ENCODING_VERSION
    from opengwasdb.store_reader.encoding.plan import N_MAX as N_MAX
    from opengwasdb.store_reader.encoding.plan import N_MISSING as N_MISSING
    from opengwasdb.store_reader.encoding.plan import SE_EXCEPTION as SE_EXCEPTION
    from opengwasdb.store_reader.encoding.plan import SE_EXCEPTION_BUDGET as SE_EXCEPTION_BUDGET
    from opengwasdb.store_reader.encoding.plan import SE_MISSING as SE_MISSING
    from opengwasdb.store_reader.encoding.plan import SE_RANGE_CANDIDATES as SE_RANGE_CANDIDATES
    from opengwasdb.store_reader.encoding.plan import Z_MISSING as Z_MISSING
    from opengwasdb.store_reader.encoding.plan import Z_OVERFLOW as Z_OVERFLOW
    from opengwasdb.store_reader.encoding.plan import EafBaselineError as EafBaselineError
    from opengwasdb.store_reader.encoding.plan import EafEncoding as EafEncoding
    from opengwasdb.store_reader.encoding.plan import EafMeasurements as EafMeasurements
    from opengwasdb.store_reader.encoding.plan import EncodingMeasurements as EncodingMeasurements
    from opengwasdb.store_reader.encoding.plan import NEncoding as NEncoding
    from opengwasdb.store_reader.encoding.plan import SeEncoding as SeEncoding
    from opengwasdb.store_reader.encoding.plan import SeMeasurements as SeMeasurements
    from opengwasdb.store_reader.encoding.plan import StoreEncoding as StoreEncoding
    from opengwasdb.store_reader.encoding.plan import UnsupportedEncoding as UnsupportedEncoding
    from opengwasdb.store_reader.encoding.plan import ZEncoding as ZEncoding
    from opengwasdb.store_reader.encoding.planes import DenseEafPlane as DenseEafPlane
    from opengwasdb.store_reader.encoding.planes import DenseSePlane as DenseSePlane
    from opengwasdb.store_reader.encoding.planes import DenseZPlane as DenseZPlane
    from opengwasdb.store_reader.encoding.planes import EafRead as EafRead
    from opengwasdb.store_reader.encoding.planes import RaggedEafPlane as RaggedEafPlane
    from opengwasdb.store_reader.encoding.planes import RaggedSePlane as RaggedSePlane
    from opengwasdb.store_reader.layout import component_chunk_size as component_chunk_size

#: Each re-exported name, by the module it comes from; imported on first use
#: (`opengwasdb.lazy_exports`, #464).
_EXPORTS = {
    "opengwasdb.encoding.association_codec": (
        "AssociationCodec",
        "CellValues",
        "Placement",
        "storable",
    ),
    "opengwasdb.encoding.codec": (
        "DEFAULT_BASELINE_CELL_BUDGET",
        "EafExceptionBuilder",
        "SeExceptionBuilder",
        "SparseExactBuilder",
        "StoreCodec",
        "ZOverflowBuilder",
        "eaf_baseline_from_grid",
        "eaf_baseline_from_pairs",
        "eaf_baseline_from_sorted_runs",
        "eaf_exception_table",
        "encode_sample_sizes",
    ),
    "opengwasdb.store_reader.encoding.codec": (
        "EAF_BASELINE",
        "EAF_EXCEPTION_INDEX",
        "EAF_EXCEPTION_VALUE",
        "EAF_REFERENCE",
        "Z_OVERFLOW_INDEX",
        "Z_OVERFLOW_VALUE",
        "EafExceptionTable",
        "SeExceptionTable",
        "SparseExactTable",
        "ZOverflowTable",
        "expit",
        "logit",
        "positions_at",
        "positions_flat",
        "positions_pairs",
        "positions_row_band",
        "positions_rows_cols",
    ),
    "opengwasdb.encoding.measure": (
        "combine_eaf_measurements",
        "fit_se",
        "fit_se_grid",
        "measure_eaf",
        "measure_eaf_grid",
        "measure_eaf_sample",
    ),
    "opengwasdb.store_reader.encoding.plan": (
        "DEFAULT_Z_SCALE",
        "EAF_ABSENT",
        "EAF_CODE_MAX",
        "EAF_CODE_MIN",
        "EAF_EXCEPTION",
        "EAF_EXCEPTION_BUDGET",
        "EAF_RANGE_CANDIDATES",
        "ENCODING_VERSION",
        "N_MAX",
        "N_MISSING",
        "SE_EXCEPTION",
        "SE_EXCEPTION_BUDGET",
        "SE_MISSING",
        "SE_RANGE_CANDIDATES",
        "Z_MISSING",
        "Z_OVERFLOW",
        "EafBaselineError",
        "EafEncoding",
        "EafMeasurements",
        "EncodingMeasurements",
        "NEncoding",
        "SeEncoding",
        "SeMeasurements",
        "StoreEncoding",
        "UnsupportedEncoding",
        "ZEncoding",
    ),
    "opengwasdb.store_reader.encoding.planes": (
        "DenseEafPlane",
        "DenseSePlane",
        "DenseZPlane",
        "EafRead",
        "RaggedEafPlane",
        "RaggedSePlane",
    ),
    "opengwasdb.encoding.planes": (
        "write_dense_cells",
        "write_eaf_baseline",
        "write_eaf_reference",
        "write_encoded_planes",
    ),
    "opengwasdb.encoding.se": (
        "OverflowCellBatches",
        "OverflowCells",
        "optimise_dense_se",
        "optimise_dense_se_joint",
        "write_dense_se_side_arrays",
    ),
    "opengwasdb.store.arrays": (
        "PER_VARIANT_CHUNK as DEFAULT_PER_VARIANT_CHUNK",
        "component_chunk_size",
        "component_variant_chunk",
        "per_variant_chunk_size",
    ),
}

__all__ = exported_names(_EXPORTS)
__getattr__, __dir__ = lazy_hooks(__name__, _EXPORTS)

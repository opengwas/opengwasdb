"""Hybrid Layout — Dense Component (reference panel) + Ragged Overflow (off-panel observed).

A Hybrid store is a thin integration layer over the existing dense and ragged
components (ADR 0026):

- The **Dense Component** is a Dense store nested at ``<store>/dense`` whose variant
  axis is the completion reference panel. It is built, completed, top-hit-indexed,
  and validated by the *unchanged* dense machinery. From format 0.3.0 it carries no
  variant table or ``analyses.tsv`` of its own: opened as a release, its axis is the
  shared one seen through ``dense_to_shared.npy`` (#472).
- The **Ragged Overflow Component** is a CSR at ``<store>/data.zarr/ragged`` holding
  each Analysis's off-panel observed associations. Its ``variant_index`` values
  reference the **shared** Store Variant Table at ``<store>/variants.tsv.gz`` (the
  union of panel ∪ off-panel-observed variants).

The two components partition each Analysis's associations disjointly: a variant is
on-panel (a Dense Component row) xor off-panel (a Ragged Overflow entry).
"""

from opengwasdb.store_reader.hybrid import (
    DENSE_SUBDIR,
    dense_component_path,
    dense_to_shared_path,
)

__all__ = [
    "DENSE_SUBDIR",
    "dense_component_path",
    "dense_to_shared_path",
]

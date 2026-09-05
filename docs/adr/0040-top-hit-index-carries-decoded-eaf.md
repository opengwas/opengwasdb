# Top-hit indexes carry decoded effect allele frequency

Refines ADR 0036 (per-association effect allele frequency) and ADR 0037
(statistic array encodings). Neither is superseded: this answers what the
Top-Hit Index must materialise now that `eaf` is part of the association
result contract, which ADR 0036 added to the contract without extending the
derived index that answers `top_hits()`. Issues #131-#134.

## Context

ADR 0036 added `eaf` to the shared association result contract, but the
layout-specific Top-Hit Indexes continued to materialise only `z`, `se`, and
completion status. A `top_hits()` query therefore gathered frequency values
from the much larger association plane after reading every other result field
from the index. On ukb-b this moved a selected-Analysis query from 1.17 ms to
86.6 ms. Dense, Ragged, and Hybrid adapters also no longer had one equivalent
fast path.

## Decision

Each threshold tier may carry decoded `eaf` as `float32`, parallel to and in
the identical `(analysis_index, variant_index)` order as `z` and `se`. Writers
include it whenever that component can report frequencies, including reference
panel frequencies resolved for imputed cells. A component declaring no
frequency plane writes no EAF index array.

Readers prefer the indexed array. If it is absent, they gather from the
component's frequency plane, preserving compatibility with every existing
Store Release. The Top-Hit Index remains a rebuildable derived artifact, so
adding this array does not change association data or the Store format version.

Post-hoc rebuilds decode EAF while scanning the stored Z plane in variant-row
bands. Dense and Hybrid VCF builds already hold harvested candidate coordinates;
after encoding the frequency plane they revisit candidate-containing row chunks
in ascending order and populate the index without a scattered gather. Ragged
builders decode their association-aligned CSR frequencies once and apply each
tier's existing permutation.

## Consequences

Top-hit queries pay no EAF-plane I/O when the new array is present. Index size
increases by one compressible `float32` per hit per tier; the measured ukb-b
index is 905,110,866 bytes at 16,384-entry chunks. Older indexes remain correct
but retain the slower gather until rebuilt. Validation treats `eaf` as optional
for compatibility, but checks its length and decoded value when present.

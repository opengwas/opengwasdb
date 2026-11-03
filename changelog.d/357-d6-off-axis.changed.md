- **The rows routing drops because no axis holds them are counted and logged
  (`off_axis`; issue #357, D6, ADR 0072).** Before, these were counted nowhere:
  - a `build-dense-vcf` row whose variant failed the hg19 lift, or that a
    Variant Reference omits;
  - a two-pass `build-hybrid` row no axis holds, which was spilled and then
    deleted unread;
  - a `build-ragged-besd` row whose ESI entry never made the axis.

  Each is now in the Admission record and its log, by source key. A single-pass
  Hybrid still folds such rows into its Ragged Overflow.

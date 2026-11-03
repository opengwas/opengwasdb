- **The rows routing drops because no axis holds them are counted and logged
  (`off_axis`; issue #357, D6, ADR 0072).** Before, these were counted nowhere:
  - a `build-dense-vcf` row whose variant failed the hg19 lift, or that a
    Variant Reference omits;
  - a two-pass `build-hybrid` row no axis holds, which was spilled and then
    deleted unread;
  - a single-pass `build-hybrid` row whose off-reference key the fold could not
    resolve, dropped after Pass 2's record was written;
  - a `build-ragged-besd` row whose ESI entry never made the axis.

  Each is now in the Admission record and its log, by source key; the fold's,
  in its `fold` stage. A single-pass Hybrid still folds the rows it resolves
  into its Ragged Overflow. A two-pass build that finds an off-reference spill
  refuses it, since nothing would fold it.

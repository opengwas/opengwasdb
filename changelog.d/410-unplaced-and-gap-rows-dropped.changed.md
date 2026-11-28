- **A GWAS Catalog row its harmoniser could not place, and a GRCh38 row where
  GRCh38 has no sequence, are now dropped and counted (issue #410, ADR 0072).**
  - **Rows the harmoniser placed nowhere.** In the pre-GWAS-SSF layout, which
    keeps the study's alleles in the main columns, a row whose `hm_code` is 14,
    15 or 16 was read at the submitter's position. The codes mean "Required
    fields are not known", "No matching variants in reference VCF" and
    "Multiple matching variants in reference VCF". On real files that position
    was often wrong: a GRCh37 position never lifted, or a position on an alt or
    random scaffold under the primary chromosome's label. OGS-00011 stored
    `1:5:G:T`, which a lift had put on `chr4_GL000008v2_random:5`. Every reader
    now drops such a row as `hm_unplaced`.
  - **How many rows.** OGS-00011's 731 sources of that layout hold 4,360,634
    rows with code 14, 53,177,527 with code 15 and 11,123 with code 16. About
    3.19M of the code-15 SNVs sat at a wrong position.
  - **Gap and past-end rows.** Every builder drops a row of a GRCh38 source
    placed in an assembly gap (a run of two or more `N`), or past its contig's
    end, as `in_assembly_gap` or `past_contig_end`, and holds its variant on no
    axis. The rule set's version is `admission/16`.
  - **Lifted rows are judged after the lift.** A row whose lift lands in a gap
    or past its contig's end is dropped and counted under the same two reasons,
    by every path that lifts (`place_union`, the Hybrid fold, extraction,
    `build-ragged-ssf`, `build-ragged-besd`). The hg19-to-GRCh38 chain aligns
    2,372 bases, in 14 runs of 9 primary-contig blocks, into gaps. A lifted
    Ragged SSF or BESD Store's `provenance.lift` counts them as
    `n_outside_sequence`, beside `n_off_primary`. A Hybrid
    checkpoint records each such key with its reason, so its format moves to 7
    and a format-6 checkpoint is refused.
  - **New `validate` rule, `variant_table.assembly_gap`.** It fails a GRCh38
    release that holds such a variant: an error from format 0.2.2, a warning
    before it. It stays the backstop for a release built before, and for a
    Variant Reference extracted before, lifted rows were judged.
  - **Published Stores this rule fails:** OGS-00011 (1,325 variants in gaps and
    373 past an end), OGS-00032 (86 and 5) and OGS-00033 (232 and 18). The 0.3.0
    rebuild clears them.
  - **The gap table.** `opengwasdb/variants/grch38-assembly-gaps.tsv` holds
    786 gaps and 25 contig lengths, derived from UCSC's `hg38.fa.gz` by
    `scripts/derive_grch38_assembly_gaps.py`. A single `N` base is an IUPAC
    code, not a gap: real variants sit there.

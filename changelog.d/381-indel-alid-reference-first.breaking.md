- **An indel's ALID spells its reference allele first, and every builder writes
  `format_version` 0.2.2 (issue #381, ADR 0079).** A deletion that 0.2.0 spelt
  like the insertion of the same sequence is now its own variant:
  FinnGen R13 BMI_IRN's `GGGGCC>G` is `1:827587:GGGGCC:G`, beside the insertion
  `1:827587:G:GGGGCC`, and its statistic is signed for its reference allele.
  SNVs, multi-base substitutions and insertions keep their spelling. What a
  reader sees:
  - **A 0.2.0 or 0.2.1 Store** (0.2.1 is package 0.9.0's Ragged release carrying
    `regions.tsv`) is read as it was written. Its deletions keep the
    insertion's spelling, so a query by `1:827587:G:GGGGCC` there finds
    whichever row its build kept, and a lone deletion's z has the opposite sign
    to the same deletion in a 0.2.2 Store. Nothing rewrites it; the plan for the
    published Releases is #367's.
  - **A 0.2.2 Store** holds both variants of such a pair. Rebuilt from source,
    the FinnGen R13 pilot (OGS-00003) holds 96,428 more variants.
  - **An indel whose source declares no reference allele is dropped**, and logged
    per Analysis in the Admission record (`admission/13`). GWAS-VCF, FinnGen and
    `ref_alt_effect_is_alt` sources declare it. GWAS-SSF declares it by layout:
    in a file whose harmonised alleles are its main columns, a row whose
    `hm_code` is 10 or 11 declares its `other_allele` the reference, and other
    codes' indels are dropped as `indel_ssf_hm_code`; a file keeping harmonised
    alleles in `hm_*` columns drops every indel (`indel_ssf_hm_columns`), and so
    does one with no harmoniser column (`indel_ssf_unharmonised`); any other
    mix of harmoniser columns is refused. The record counts the indels the code
    kept, `indel_rows_kept_by_hm_code`. A BESD `.esi`, a `build-dense` association
    TSV, and an `effect_and_other` or `ref_alt_and_effect_allele` Column Map
    declare none (`indel_reference_undeclared`).
  - **Reference Completion places a panel indel by the panel's `REF` column, or
    by an indel-reference table beside its blocks** that
    `scripts/build_panel_indel_reference.py` settles from GRCh38 evidence, and
    re-signs a panel deletion's LD and frequency for its reference allele. A
    panel with neither places no indel, and its log and
    `provenance.completion.panel_indels` say so. A table is read only beside
    its `indel_reference.json`, whose checksum it must match. A panel row with
    no `EA`, in its cell or its block, or with an `EA` that is neither of its
    alleles, is not placed, SNV or indel; `panel_indels.unplaced` and
    `panel_snvs_unplaced` count them. Completion **refuses a 0.2.1, 0.2.0
    or 0.1.0 source**, each of which sorts its indels: rebuild it from source with
    this package, which writes 0.2.2, and complete the 0.2.2 rebuild. That covers
    package 0.9.0's 0.2.1 Stores: OGS-00006 (completed), OGS-00007, OGS-00024,
    OGS-00025 and OGS-00026 have been published at 0.2.1 since 11 Nov 2026, and
    each is completed from a 0.2.2 rebuild. The converter now writes 0.2.0, not
    the current version.
  - **Package 0.9.0 reads a 0.2.2 Store, warned that it is newer, and refuses to
    complete it, and its `validate` fails it** on its stamp rule ("release is
    stamped 0.2.2, but a release carrying … is 0.2.0" or "0.2.1"). Upgrade the
    package; do not re-stamp the Store.
  - **`validate` adds `variant_table.indel_order`**, which fails a Store from
    0.2.2 on whose indel ALID contradicts its row's `effect_allele`/`other_allele`
    or `source_alid`.
  - Python: `orient_to_canonical`, `oriented_alid`, `identity_fault`,
    `project_source_variant` and `oriented_variant` take a required
    `other_is_reference`, `orient_or_drop` a required `declaration` and `row`,
    and the tabular projection columns a required `reference`. `identity_fault`
    takes exactly two alleles; `chromosome_fault` judges a chromosome alone.

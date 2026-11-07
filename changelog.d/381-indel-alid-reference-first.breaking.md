- **An indel's ALID spells its reference allele first, and every builder writes
  `format_version` 0.2.1 (issue #381, ADR 0079).** A deletion that 0.2.0 spelt
  like the insertion of the same sequence is now its own variant:
  FinnGen R13 BMI_IRN's `GGGGCC>G` is `1:827587:GGGGCC:G`, beside the insertion
  `1:827587:G:GGGGCC`, and its statistic is signed for its reference allele.
  SNVs, multi-base substitutions and insertions keep their spelling. What a
  reader sees:
  - **A 0.2.0 Store** is read as it was written. Its deletions keep the
    insertion's spelling, so a query by `1:827587:G:GGGGCC` there finds
    whichever row its build kept, and a lone deletion's z has the opposite sign
    to the same deletion in a 0.2.1 Store. Nothing rewrites it; the plan for the
    published Releases is #367's.
  - **A 0.2.1 Store** holds both variants of such a pair. Rebuilt from source,
    the FinnGen R13 pilot (OGS-00003) holds 96,428 more variants.
  - **An indel whose source declares no reference allele is dropped**, and logged
    per Analysis in the Admission record as `indel_reference_undeclared`
    (`admission/12`): every indel of a GWAS-SSF file, harmonised or not, of a
    BESD `.esi`, of a `build-dense` association TSV, and of an
    `effect_and_other` or `ref_alt_and_effect_allele` Column Map. GWAS-VCF,
    FinnGen and `ref_alt_effect_is_alt` sources declare it.
  - **Reference Completion places no panel indel**, since no LD Reference Panel
    names one by its reference allele, and **refuses a 0.2.0 or 0.1.0 source**:
    rebuild it, then complete it. The converter now writes 0.2.0, not the
    current version.
  - **`validate` adds `variant_table.indel_order`**, which fails a Store from
    0.2.1 on whose indel ALID contradicts its row's `effect_allele`/`other_allele`
    or `source_alid`.
  - Python: `orient_to_canonical`, `oriented_alid`, `orient_or_drop`,
    `identity_fault`, `project_source_variant` and `oriented_variant` take a
    required `other_is_reference`, and the tabular projection columns a required
    `reference`.

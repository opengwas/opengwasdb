- **Every builder resolves a duplicate (variant, Analysis) the same way (issue
  #357, D5, ADR 0072).** Rows that agree on z and SE are one association: they
  collapse to the first, frequency and all (`duplicate_collapsed`, counted).
  Rows that disagree are all dropped, since nothing says which is right, and are
  logged by ALID (`duplicate_conflicting`). Before, `build-dense-vcf` and
  `build-hybrid` stored the last row in silence, including the last of
  GCST90295971's seven conflicting pairs, one of them of opposite sign.
  `build-dense` refused the build, and `build-ragged-ssf` dropped conflicting
  rows with a line on stdout. The rule is Variant Routing's
  `collapse_duplicates`; `Placed` reports what it collapsed and dropped.
- **An insertion and a deletion that share an ALID keep the insertion, where
  the source declares the reference allele.** The ALID sorts its alleles, so
  FinnGen's `1:827587` `G>GGGGCC` and `GGGGCC>G` are both `1:827587:G:GGGGCC`:
  two alleles of a repeat, not one variant. When a GWAS-VCF or FinnGen source
  says which allele is the reference, the row whose reference allele the ALID
  spells first is kept, and the other is dropped and logged
  (`duplicate_reversed_indel`). A source naming only an effect and an other
  allele (GWAS-SSF, BESD) declares no reference, so its pair is dropped as
  conflicting, as is any other conflicting duplicate. Each `ReaderAssociation`
  says what its source declares (`reference`). The ALID still spells the pair
  as one variant: #381 gives indels a ref-first ALID, and #382 corrects the
  published cells.

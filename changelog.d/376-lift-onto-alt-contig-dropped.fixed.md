- **A variant an hg19 → GRCh38 lift places on an alt or random contig is left
  out of the Store, and counted (issue #376).** The lift put such a variant at
  `11_KI270927v1_alt:…`, a contig the Store's queries and LD panels do not use,
  and stored it. Every builder (Dense and Hybrid GWAS-VCF, GWAS-SSF, BESD) and
  a Variant Reference's extraction now leave it out. Its rows are dropped and
  logged per Analysis in the Admission record as `lift_off_primary`, which moves
  `ADMISSION_RULES` to `admission/14`. A source already on GRCh38 is
  unchanged: an alt contig there is still a chromosome. Stores built before
  this change are unchanged; the published Stores' rebuilds are #367's.

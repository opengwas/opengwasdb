- **A variant an hg19 → GRCh38 lift places on an alt or random contig is left
  out of the Store, and counted (issue #376).** The lift put such a variant at
  `11_KI270927v1_alt:…`, a contig the Store's queries and LD panels do not use,
  and stored it. Every builder (Dense and Hybrid GWAS-VCF, GWAS-SSF, BESD) and
  a Variant Reference's extraction now leave it out. Its rows are dropped and
  logged per Analysis in the Admission record as `lift_off_primary`, which moves
  `ADMISSION_RULES` to `admission/14`. A source already on GRCh38 is
  unchanged: an alt contig there is still a chromosome. Stores built before
  this change are unchanged; the published Stores' rebuilds are #367's.

  A Variant Reference extracted before this change, which still routes an hg19
  row onto an alt contig, has that row dropped and counted at build time, and a
  build routed by a reference counts a lift's off-primary rows as
  `lift_off_primary`, not `off_axis`. A Hybrid checkpoint written before this
  change is refused on resume (`CHECKPOINT_FORMAT_VERSION` 4): start the build
  again. `validate` gains `variant_table.lifted_contig`, which flags a release
  whose variant table holds a variant lifted onto an alt, random or unplaced
  contig. It is an error from format 0.2.2 and a warning on earlier releases, so
  OGS-00008, -09 and -27 (0.1.0/0.2.0), and any 0.9.0 release, now warn on it until rebuilt (#367); it does not
  fail them.
  `provenance.lift.n_lifted` still counts the sites the chain mapped, and
  `n_off_primary` is added beside it.

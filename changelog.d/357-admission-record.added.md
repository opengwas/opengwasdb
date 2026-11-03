- **Every Observed-Only build writes an Association Admission record beside the
  Store (`--admission-record DIR`, required on the command line; issue #357).**
  `build-dense`, `build-dense-vcf`, `build-hybrid`,
  `build-hybrid-from-catalogue`, `build-ragged-ssf` and `build-ragged-besd` all
  write it. Called from Python, a builder writes it to `OUTPUT_PATH.admission`
  unless `admission_record=` names another directory. `admission.json` lists, for each Analysis in
  Manifest Row order, the rows it examined and admitted and the rows it dropped,
  by reason, under the rule set it names (`"rules": "admission/10"`). A row
  dropped by a declared INFO or MAF filter, or collapsed into an agreeing
  duplicate, is counted only; every other dropped row is also logged, by source
  key, in `dropped/<analysis_index>.tsv.gz` (and `<analysis_index>.fold.tsv.gz`
  for what the Hybrid's off-reference fold drops). The reasons include
  `duplicate_reversed_indel`: an indel row dropped because its declared
  reference allele is the one its ALID spells second (ADR 0072; #381 and #382
  follow up). The record is not part of the Store Release, so the Store's
  bytes and its `format_version` are unchanged. Pass 2 workers write it one
  part per Analysis, so it is the same at any `--n-workers`. A resumed
  `--checkpoint` build reads its counts back from those parts, and refuses a
  record that has lost one.
- **The Ragged builders' unusable-SE drops are counted and logged
  (`se_unusable`).** `build-ragged-ssf` and `build-ragged-besd` already dropped
  a row whose SE was missing, not positive or not finite, but left no trace of
  it. The Stores are unchanged.

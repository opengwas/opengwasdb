- **#357's reversed-indel rule is removed, with everything that fed it (issue
  #316, ADRs 0072 and 0079).** An indel's ALID spells its reference allele
  first, so an insertion and a deletion at one position are two variants, and
  the rule had not fired since 0.2.2. Admission's duplicate rule is now only:
  rows that agree collapse to the first, and rows that disagree are all
  dropped. `duplicate_reversed_indel` is retired in place -- a reason's code is
  its position, so it keeps its place, and records written before still read
  -- and no build gives it. Breaking for Python callers:
  `collapse_duplicates` takes no `reference`; `Placed`, `Duplicates` and
  `Unmatched` lose `reference` and `reversed_indels`; `Gate.duplicates` takes
  no reversed rows; `ReaderBatch` loses `ref_first` and `reference_codes`;
  `ReferenceAllele.ALT` and routing's `REFERENCE_*`, `allele_order` and
  `reference_codes` are gone. A Hybrid's Overflow and off-reference spills no
  longer carry a `reference` array; a resumed build ignores one written
  before. The spill format is internal to a build and has changed; the Store
  has not. Rebuilt from its sources before and after, the Hybrid pilot
  (OGS-00004) has identical arrays (75), `analyses.tsv`, `variants.tsv.gz` and
  Admission record, as does the FinnGen R13 pilot (OGS-00003).

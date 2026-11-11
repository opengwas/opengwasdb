- **`completion_quality` counts only rows the Store holds (the owner's ruling of 11
  Nov 2026, ADRs 0039 and 0079).** A panel indel the panel declares no reference
  for has no ALID (ADR 0079), so completion neither fills it nor writes it as a
  row. It was still counted missing, by the block that holds it, and at a shared
  endpoint by both blocks. Now no block counts it, so for each Analysis and block
  `n_imputed + n_missing` is the number of rows completion added that the block
  owns. That holds in Dense and Ragged alike, and Ragged's
  `completion_n_missing_total` follows. In a new 0.2.2 Store, a missing count
  drops by the unnamed indels completion used to count. Measured on the
  chromosome 15-22 EUR panel, the Ragged total went from 2,876 to 0 on 100
  OGS-00006 Analyses, and from 533 to 0 on GCST90296697. On the full EUR panel,
  Dense OGS-00003's quality rows went from 565,881 missing to 2,161. Published
  Stores are untouched. The unnamed indels stay counted in
  `provenance.completion.panel_indels`: `unplaced.no_reference`, and `n_indels`
  less `n_placed`.

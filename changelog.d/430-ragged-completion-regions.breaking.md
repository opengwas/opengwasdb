- **`complete-ragged` completes each Analysis only inside its regions, taken from the
  build's region table with `--regions` (issue #430, ADR 0039 as amended).** An
  Analysis's regions are its cis region, where it has a Trait position, and its
  significant trans regions; each is filled only within its own `[start, end]`. The
  cis boundary is the table's `cis` row, or else the Trait position ± `--cis-window-bp`;
  `suggestive_trans` and `qc_panel` rows never make a region, and an Analysis with no
  region is carried observed-only. It was completed against whole LD blocks: every
  block touching a cis window, past the cis boundary and never in a significant trans
  region, and, with no Trait position, every block holding four of its observations,
  which a genome-wide QC panel made almost every block. OGS-00006 imputed 10,001,552,498
  cells that way; under this rule its full completion adds a projected 1,038,245, every
  one inside a significant region on the chr15-22 subsets measured. Completing OGS-00007
  (rebuilt at 0.2.0) imputes 18,352 cells, 12,442 of them in significant trans regions,
  where it imputed 101,308, 95,398 outside every region. **Breaking:** `--regions` is
  required unless the source's Association Coverage is `cis`, so a `cis_and_signals`,
  `signals_only` or `gwas` Store is now refused without it (it used to complete), as is
  a table row of an unknown kind, another assembly or an Analysis the Store lacks, and a
  resume of a checkpoint written before this release. `check-observed-view` takes the same `--regions`, needs
  it for such a Store, and fails a missing row completion added outside every region
  (`missing_outside_retained`); its verdict records the table. The `complete-ragged`
  summary gains `n_regions_without_block` and `regions_without_block`, the `region_id`
  of each region no LD block of the panel overlaps, which is completed nowhere.
